"""
Warp phối cảnh, trộn feathering và cắt viền đen.

Ảnh nguồn được đưa lên hệ tọa độ ảnh tham chiếu bằng warpPerspective.
Trong vùng chồng nhau, feathering dùng gradient alpha theo khoảng cách tới biên
để làm mềm đường nối thay vì dán đè cứng.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WarpBlendResult:
    """Kết quả warp + blend một cặp ảnh."""

    panorama: np.ndarray
    warped_source: np.ndarray
    offset: tuple[int, int]
    canvas_size: tuple[int, int]


def compute_canvas_geometry(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Tính kích thước canvas chứa cả ảnh đích và ảnh nguồn đã biến đổi bởi H.

    Trả về:
    - H_translated: H đã dịch để mọi góc nằm trong canvas không âm
    - offset (x_min, y_min)
    - (width, height) của canvas
    """
    h_src, w_src = image_src.shape[:2]
    h_dst, w_dst = image_dst.shape[:2]

    corners_src = np.float32(
        [[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]
    ).reshape(-1, 1, 2)
    corners_dst = np.float32(
        [[0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]]
    ).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_src, H)
    all_corners = np.concatenate([warped_corners, corners_dst], axis=0)

    xs = all_corners[:, 0, 0]
    ys = all_corners[:, 0, 1]
    x_min, x_max = np.floor(xs.min()), np.ceil(xs.max())
    y_min, y_max = np.floor(ys.min()), np.ceil(ys.max())

    # Giới hạn canvas quá lớn (tránh tràn bộ nhớ khi H kém)
    max_dim = 12000
    width = int(min(max_dim, x_max - x_min))
    height = int(min(max_dim, y_max - y_min))
    if width <= 0 or height <= 0:
        raise ValueError("Kích thước canvas không hợp lệ sau khi ước lượng homography.")

    translation = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_translated = translation @ H
    offset = (int(-x_min), int(-y_min))
    return H_translated, offset, (width, height)


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Chiếu ảnh lên canvas kích thước (width, height) bằng warpPerspective."""
    width, height = canvas_size
    return cv2.warpPerspective(image, H, (width, height))


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """Mask boolean True nơi pixel có nội dung (không hoàn toàn đen)."""
    if image.ndim == 3:
        return np.any(image > 0, axis=2)
    return image > 0


def _feather_weights(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo trọng số feathering trên vùng chồng nhau.

    Dùng khoảng cách tới biên mask (distance transform): càng xa biên
    thì trọng số càng lớn. Trên vùng chỉ thuộc một ảnh, trọng số = 1.
    """
    mask_a_u8 = mask_a.astype(np.uint8) * 255
    mask_b_u8 = mask_b.astype(np.uint8) * 255

    dist_a = cv2.distanceTransform(mask_a_u8, cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform(mask_b_u8, cv2.DIST_L2, 5)

    overlap = mask_a & mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a

    weight_a = np.zeros(mask_a.shape, dtype=np.float32)
    weight_b = np.zeros(mask_b.shape, dtype=np.float32)

    weight_a[only_a] = 1.0
    weight_b[only_b] = 1.0

    if np.any(overlap):
        da = dist_a[overlap]
        db = dist_b[overlap]
        denom = da + db
        # Tránh chia 0 ở biên mỏng
        denom = np.where(denom < 1e-6, 1.0, denom)
        weight_a[overlap] = da / denom
        weight_b[overlap] = db / denom

    return weight_a, weight_b


def blend_images(
    warped_src: np.ndarray,
    dst_on_canvas: np.ndarray,
    use_feathering: bool = True,
) -> np.ndarray:
    """
    Trộn hai ảnh trên cùng canvas.

    Nếu feathering bật: alpha gradient theo khoảng cách trong vùng overlap.
    Nếu tắt: ưu tiên pixel không đen của warped_src khi chồng (paste-over).
    """
    mask_src = _binary_mask(warped_src)
    mask_dst = _binary_mask(dst_on_canvas)

    if not use_feathering:
        result = dst_on_canvas.copy()
        result[mask_src] = warped_src[mask_src]
        return result

    w_src, w_dst = _feather_weights(mask_src, mask_dst)
    w_src_3 = w_src[..., None]
    w_dst_3 = w_dst[..., None]

    src_f = warped_src.astype(np.float32)
    dst_f = dst_on_canvas.astype(np.float32)
    blended = src_f * w_src_3 + dst_f * w_dst_3

    union = mask_src | mask_dst
    out = np.zeros_like(dst_on_canvas)
    out[union] = np.clip(blended[union], 0, 255).astype(np.uint8)
    return out


def place_image_on_canvas(
    image: np.ndarray,
    canvas_size: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    """Đặt ảnh tham chiếu lên canvas tại offset (không warp)."""
    width, height = canvas_size
    ox, oy = offset
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    canvas[oy : oy + h, ox : ox + w] = image
    return canvas


def warp_and_blend(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    H: np.ndarray,
    use_feathering: bool = True,
) -> WarpBlendResult:
    """
    Warp ảnh nguồn theo H vào hệ tọa độ ảnh đích rồi blend.

    image_src được biến đổi bởi H để khớp với image_dst.
    """
    H_t, offset, canvas_size = compute_canvas_geometry(image_src, image_dst, H)
    warped_src = warp_image(image_src, H_t, canvas_size)
    dst_on_canvas = place_image_on_canvas(image_dst, canvas_size, offset)
    panorama = blend_images(warped_src, dst_on_canvas, use_feathering=use_feathering)
    return WarpBlendResult(
        panorama=panorama,
        warped_source=warped_src,
        offset=offset,
        canvas_size=canvas_size,
    )


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    Cắt viền đen bằng bounding box của contour lớn nhất trên mask nội dung.

    Giúp panorama gọn hơn sau khi warp tạo khoảng trống đen quanh biên.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return image
    return image[y : y + h, x : x + w]
