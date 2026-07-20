"""
Phát hiện và khớp đặc trưng (feature detection & matching).

ORB / SIFT tìm các điểm đặc trưng ổn định giữa các ảnh chồng lấn.
BFMatcher + kiểm tra tỉ lệ Lowe loại bỏ các khớp mơ hồ trước khi ước lượng
homography.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

DetectorName = Literal["ORB", "SIFT"]


@dataclass(frozen=True)
class FeatureResult:
    """Kết quả phát hiện đặc trưng trên một ảnh."""

    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    visualization: np.ndarray
    count: int


@dataclass(frozen=True)
class MatchResult:
    """Kết quả khớp đặc trưng giữa một cặp ảnh."""

    matches: list[cv2.DMatch]
    visualization: np.ndarray
    match_count: int


def create_detector(name: DetectorName = "ORB", nfeatures: int = 3000) -> cv2.Feature2D:
    """
    Tạo bộ phát hiện đặc trưng theo tên.

    ORB nhanh, không bản quyền, dựa trên FAST + BRIEF đã hướng (oriented).
    SIFT chậm hơn nhưng thường ổn định hơn với thay đổi tỉ lệ và góc nhìn.
    """
    if name == "ORB":
        return cv2.ORB_create(nfeatures=nfeatures)
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=nfeatures)
    raise ValueError(f"Bộ phát hiện không hỗ trợ: {name}")


def detect_features(
    image: np.ndarray,
    detector_name: DetectorName = "ORB",
    nfeatures: int = 3000,
) -> FeatureResult:
    """
    Phát hiện keypoints và mô tả (descriptors) trên một ảnh màu BGR.

    Ảnh được chuyển sang xám vì hầu hết bộ mô tả hoạt động trên cường độ sáng.
    Bản visualization vẽ keypoints để giải thích trong báo cáo / demo.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = create_detector(detector_name, nfeatures=nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoints = list(keypoints or [])

    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return FeatureResult(
        keypoints=keypoints,
        descriptors=descriptors,
        visualization=vis,
        count=len(keypoints),
    )


def match_features(
    desc1: np.ndarray | None,
    desc2: np.ndarray | None,
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    image1: np.ndarray,
    image2: np.ndarray,
    detector_name: DetectorName = "ORB",
    ratio_threshold: float = 0.75,
    max_draw: int = 50,
) -> MatchResult:
    """
    Khớp descriptors bằng BFMatcher + knnMatch(k=2) và kiểm tra tỉ lệ Lowe.

    Ý tưởng Lowe's ratio test:
    nếu khoảng cách tới hàng xóm gần nhất gần bằng hàng xóm thứ hai thì khớp
    bị coi là mơ hồ và bị loại. Ngưỡng thường dùng ~0.75.

    ORB dùng Hamming; SIFT dùng L2 (NORM_L2).
    """
    if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
        empty = _side_by_side(image1, image2)
        return MatchResult(matches=[], visualization=empty, match_count=0)

    if len(desc1) < 2 or len(desc2) < 2:
        empty = _side_by_side(image1, image2)
        return MatchResult(matches=[], visualization=empty, match_count=0)

    norm = cv2.NORM_HAMMING if detector_name == "ORB" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)

    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)
    draw_matches = good[:max_draw]
    vis = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return MatchResult(matches=good, visualization=vis, match_count=len(good))


def matches_to_points(
    matches: list[cv2.DMatch],
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chuyển danh sách DMatch thành hai mảng điểm tương ứng (N, 1, 2).

    src_pts: điểm trên ảnh nguồn (ảnh sẽ được warp).
    dst_pts: điểm trên ảnh đích (ảnh tham chiếu).
    """
    if not matches:
        return (
            np.empty((0, 1, 2), dtype=np.float32),
            np.empty((0, 1, 2), dtype=np.float32),
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def _side_by_side(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Ghép ngang hai ảnh (resize cùng chiều cao) khi không có khớp."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1 : w1 + w2] = image2
    return canvas
