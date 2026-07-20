"""
Điều phối pipeline ghép panorama đầy đủ và thu thập kết quả trung gian.

Chiến lược nhiều ảnh (2–5): lấy ảnh giữa làm tham chiếu để giảm biến dạng
phối cảnh, rồi lần lượt ghép về trái rồi về phải theo thứ tự người dùng chọn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import cv2
import numpy as np

from .features import (
    DetectorName,
    detect_features,
    match_features,
    matches_to_points,
)
from .homography import InsufficientMatchesError, estimate_homography
from .warp_blend import crop_black_borders, warp_and_blend

ProgressCallback = Callable[[float, str], None]


class StitchError(Exception):
    """Lỗi có thông điệp tiếng Việt dành cho giao diện người dùng."""


@dataclass(frozen=True)
class StitchConfig:
    """Tham số cấu hình pipeline."""

    detector: DetectorName = "ORB"
    nfeatures: int = 3000
    ratio_threshold: float = 0.75
    ransac_thresh: float = 5.0
    use_feathering: bool = True


@dataclass
class PairStageResult:
    """Kết quả trung gian khi ghép một cặp ảnh liên tiếp."""

    pair_label: str
    match_visualization: np.ndarray
    match_count: int
    inlier_ratio: float
    inlier_count: int
    warped_intermediate: np.ndarray
    panorama_after_pair: np.ndarray


@dataclass
class StitchResult:
    """Toàn bộ kết quả để hiển thị trên Streamlit."""

    keypoint_visualizations: list[np.ndarray] = field(default_factory=list)
    keypoint_counts: list[int] = field(default_factory=list)
    pair_results: list[PairStageResult] = field(default_factory=list)
    panorama_before_crop: np.ndarray | None = None
    panorama_final: np.ndarray | None = None
    reference_index: int = 0


def stitch_pair(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    config: StitchConfig,
    pair_label: str,
) -> PairStageResult:
    """
    Ghép một ảnh nguồn vào ảnh đích (dst là tham chiếu hiện tại).

    image_src sẽ được warp theo homography tới hệ tọa độ của image_dst.
    """
    feat_src = detect_features(image_src, config.detector, config.nfeatures)
    feat_dst = detect_features(image_dst, config.detector, config.nfeatures)

    match = match_features(
        feat_src.descriptors,
        feat_dst.descriptors,
        feat_src.keypoints,
        feat_dst.keypoints,
        image_src,
        image_dst,
        detector_name=config.detector,
        ratio_threshold=config.ratio_threshold,
    )

    if match.match_count < 4:
        raise StitchError(
            f"Không đủ điểm khớp giữa {pair_label} "
            f"(chỉ có {match.match_count} khớp, cần ≥ 4). "
            "Hãy chụp ảnh chồng lấn nhiều hơn (khoảng 30–50%) "
            "hoặc chọn cảnh có nhiều kết cấu/chi tiết hơn."
        )

    src_pts, dst_pts = matches_to_points(
        match.matches, feat_src.keypoints, feat_dst.keypoints
    )

    try:
        homo = estimate_homography(src_pts, dst_pts, ransac_thresh=config.ransac_thresh)
    except InsufficientMatchesError as exc:
        raise StitchError(str(exc)) from exc

    blend = warp_and_blend(
        image_src,
        image_dst,
        homo.matrix,
        use_feathering=config.use_feathering,
    )

    return PairStageResult(
        pair_label=pair_label,
        match_visualization=match.visualization,
        match_count=match.match_count,
        inlier_ratio=homo.inlier_ratio,
        inlier_count=homo.inlier_count,
        warped_intermediate=blend.warped_source,
        panorama_after_pair=blend.panorama,
    )


def stitch_panorama(
    images: list[np.ndarray],
    config: StitchConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StitchResult:
    """
    Ghép tuần tự 2–5 ảnh theo thứ tự trái → phải.

    Ảnh giữa được chọn làm gốc tham chiếu. Sau đó:
    - Ghép lần lượt các ảnh bên trái (từ gần gốc ra ngoài)
    - Ghép lần lượt các ảnh bên phải
    """
    if config is None:
        config = StitchConfig()

    n = len(images)
    if n < 2 or n > 5:
        raise StitchError("Cần tải lên từ 2 đến 5 ảnh để ghép panorama.")

    for i, img in enumerate(images):
        if img is None or img.size == 0:
            raise StitchError(f"Ảnh số {i + 1} không hợp lệ hoặc bị lỗi khi đọc.")

    def report(p: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(p, msg)

    result = StitchResult()
    result.reference_index = n // 2

    report(0.05, "Đang phát hiện keypoints trên từng ảnh...")
    for img in images:
        feat = detect_features(img, config.detector, config.nfeatures)
        result.keypoint_visualizations.append(feat.visualization)
        result.keypoint_counts.append(feat.count)

    # Bắt đầu với ảnh giữa
    panorama = images[result.reference_index].copy()
    left_indices = list(range(result.reference_index - 1, -1, -1))
    right_indices = list(range(result.reference_index + 1, n))
    ordered_src_indices = left_indices + right_indices

    total_pairs = len(ordered_src_indices)
    if total_pairs == 0:
        raise StitchError("Cần ít nhất 2 ảnh.")

    report(0.15, f"Ảnh tham chiếu: ảnh #{result.reference_index + 1}")

    for step, src_idx in enumerate(ordered_src_indices):
        side: Literal["trái", "phải"] = (
            "trái" if src_idx < result.reference_index else "phải"
        )
        pair_label = (
            f"Ảnh #{src_idx + 1} ({side}) → panorama "
            f"(gốc #{result.reference_index + 1})"
        )
        frac = 0.15 + 0.75 * (step / total_pairs)
        report(frac, f"Đang ghép {pair_label}...")

        try:
            pair = stitch_pair(images[src_idx], panorama, config, pair_label)
        except StitchError:
            raise
        except Exception as exc:  # noqa: BLE001 — chuyển thành thông báo UI
            raise StitchError(
                f"Lỗi khi ghép {pair_label}: {exc}. "
                "Thử tăng chồng lấn giữa các ảnh hoặc đổi sang SIFT."
            ) from exc

        result.pair_results.append(pair)
        panorama = pair.panorama_after_pair

    report(0.95, "Đang cắt viền đen...")
    result.panorama_before_crop = panorama
    result.panorama_final = crop_black_borders(panorama)
    report(1.0, "Hoàn tất!")
    return result


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Chuyển BGR (OpenCV) sang RGB (Streamlit / PIL)."""
    if image is None:
        raise ValueError("Ảnh None")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
