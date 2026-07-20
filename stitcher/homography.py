"""
Ước lượng homography bằng RANSAC.

Homography là ma trận 3x3 biến đổi phối cảnh giữa hai mặt phẳng ảnh.
RANSAC loại bỏ outlier trong tập điểm khớp để ước lượng ổn định.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class InsufficientMatchesError(ValueError):
    """Không đủ điểm khớp tốt để ước lượng homography (cần ≥ 4)."""


@dataclass(frozen=True)
class HomographyResult:
    """Kết quả ước lượng homography."""

    matrix: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    match_count: int
    inlier_ratio: float


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = 5.0,
) -> HomographyResult:
    """
    Ước lượng ma trận homography H sao cho x' ~ H x bằng RANSAC.

    RANSAC lặp lại:
    1) chọn ngẫu nhiên 4 cặp điểm
    2) tính H
    3) đếm inlier (sai số tái chiếu < ngưỡng)
    và giữ mô hình có nhiều inlier nhất.

    Ngưỡng mặc định 5.0 (pixel) cân bằng giữa độ cứng và khả năng hội tụ.
    """
    n = int(src_pts.shape[0]) if src_pts is not None else 0
    if n < 4:
        raise InsufficientMatchesError(
            "Không đủ điểm khớp để tính homography "
            f"(cần ít nhất 4, hiện có {n}). "
            "Hãy chụp ảnh chồng lấn nhiều hơn hoặc chọn cảnh có nhiều kết cấu."
        )

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
    )

    if H is None or mask is None:
        raise InsufficientMatchesError(
            "RANSAC không ước lượng được homography. "
            "Hãy tăng chồng lấn giữa các ảnh hoặc dùng ảnh có nhiều đặc trưng hơn."
        )

    mask = mask.ravel().astype(bool)
    inlier_count = int(mask.sum())
    if inlier_count < 4:
        raise InsufficientMatchesError(
            f"Quá ít inlier sau RANSAC ({inlier_count}/ {n}). "
            "Ảnh có thể thiếu chồng lấn hoặc bị nhiễu khớp quá nhiều."
        )

    return HomographyResult(
        matrix=H.astype(np.float64),
        inlier_mask=mask,
        inlier_count=inlier_count,
        match_count=n,
        inlier_ratio=inlier_count / float(n),
    )
