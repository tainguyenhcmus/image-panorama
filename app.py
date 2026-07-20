"""
Ứng dụng Streamlit demo ghép ảnh panorama.

Chạy: streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from stitcher import StitchConfig, StitchError, stitch_panorama
from stitcher.pipeline import bgr_to_rgb


st.set_page_config(
    page_title="Ghép ảnh Panorama",
    page_icon="🖼️",
    layout="wide",
)


def load_image_bgr(uploaded_file: Any) -> np.ndarray:
    """Đọc file upload thành ảnh BGR uint8."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {uploaded_file.name}")
    uploaded_file.seek(0)
    return image


def encode_png_bytes(image_bgr: np.ndarray) -> bytes:
    """Mã hóa ảnh BGR thành PNG bytes để tải xuống."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh PNG.")
    return buf.tobytes()


def move_item(items: list[Any], index: int, direction: int) -> list[Any]:
    """Đổi chỗ phần tử trong danh sách theo hướng -1 (lên) hoặc +1 (xuống)."""
    new_index = index + direction
    if new_index < 0 or new_index >= len(items):
        return items
    items = list(items)
    items[index], items[new_index] = items[new_index], items[index]
    return items


def main() -> None:
    st.title("Ghép ảnh Panorama")
    st.caption(
        "Pipeline cổ điển: ORB/SIFT → BFMatcher + Lowe → Homography RANSAC "
        "→ Warp → Feathering → Crop. Không dùng `cv2.Stitcher`."
    )

    with st.sidebar:
        st.header("Tham số")
        detector = st.selectbox("Bộ phát hiện đặc trưng", options=["ORB", "SIFT"], index=0)
        ratio_threshold = st.slider(
            "Ngưỡng tỉ lệ Lowe",
            min_value=0.60,
            max_value=0.90,
            value=0.75,
            step=0.01,
            help="Giữ khớp khi d1 < ratio * d2. Thấp hơn = chặt hơn.",
        )
        ransac_thresh = st.slider(
            "Ngưỡng RANSAC (pixel)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )
        use_feathering = st.toggle("Feathering (trộn mềm đường nối)", value=True)
        st.divider()
        st.markdown(
            "**Gợi ý chụp ảnh:** giữ máy nằm ngang, chồng lấn ~30–50%, "
            "tránh vật thể chuyển động mạnh."
        )

    uploaded = st.file_uploader(
        "Tải lên 2–5 ảnh (JPG/PNG), theo thứ tự trái → phải",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if "ordered_names" not in st.session_state:
        st.session_state.ordered_names = []
    if "file_map" not in st.session_state:
        st.session_state.file_map = {}

    if uploaded:
        # Đồng bộ map theo tên file; giữ thứ tự người dùng đã sắp nếu còn hợp lệ
        new_map = {f.name: f for f in uploaded}
        prev_order = [
            name for name in st.session_state.ordered_names if name in new_map
        ]
        for name in new_map:
            if name not in prev_order:
                prev_order.append(name)
        st.session_state.ordered_names = prev_order
        st.session_state.file_map = new_map

        if len(st.session_state.ordered_names) > 5:
            st.warning("Chỉ dùng tối đa 5 ảnh — bỏ bớt ảnh phía dưới.")
            st.session_state.ordered_names = st.session_state.ordered_names[:5]

        st.subheader("Thứ tự ảnh (trái → phải)")
        names = list(st.session_state.ordered_names)
        cols = st.columns(len(names))
        for i, name in enumerate(names):
            with cols[i]:
                f = st.session_state.file_map[name]
                try:
                    thumb = Image.open(f).convert("RGB")
                    f.seek(0)
                    st.image(thumb, caption=f"{i + 1}. {name}", use_container_width=True)
                except Exception:
                    st.write(name)
                c1, c2 = st.columns(2)
                if c1.button("◀", key=f"left_{i}", disabled=i == 0):
                    st.session_state.ordered_names = move_item(names, i, -1)
                    st.rerun()
                if c2.button("▶", key=f"right_{i}", disabled=i == len(names) - 1):
                    st.session_state.ordered_names = move_item(names, i, 1)
                    st.rerun()

        n_imgs = len(st.session_state.ordered_names)
        if n_imgs < 2:
            st.info("Cần ít nhất 2 ảnh để bắt đầu ghép.")
            return
        if n_imgs > 5:
            return

        stitch_clicked = st.button("Stitch", type="primary", use_container_width=True)

        if stitch_clicked:
            images_bgr: list[np.ndarray] = []
            try:
                for name in st.session_state.ordered_names:
                    images_bgr.append(load_image_bgr(st.session_state.file_map[name]))
            except Exception as exc:
                st.error(f"Lỗi đọc ảnh: {exc}")
                return

            config = StitchConfig(
                detector=detector,  # type: ignore[arg-type]
                ratio_threshold=float(ratio_threshold),
                ransac_thresh=float(ransac_thresh),
                use_feathering=bool(use_feathering),
            )

            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(p: float, msg: str) -> None:
                progress.progress(min(max(p, 0.0), 1.0))
                status.write(msg)

            try:
                with st.spinner("Đang chạy pipeline..."):
                    result = stitch_panorama(
                        images_bgr,
                        config=config,
                        progress_callback=on_progress,
                    )
            except StitchError as exc:
                progress.empty()
                status.empty()
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                progress.empty()
                status.empty()
                st.error(
                    f"Đã xảy ra lỗi không mong muốn: {exc}. "
                    "Thử ảnh khác hoặc điều chỉnh tham số."
                )
                return

            progress.empty()
            status.success(
                f"Ghép xong — ảnh tham chiếu: #{result.reference_index + 1} "
                f"(ảnh giữa trong dãy)."
            )

            # --- Stage 1: Keypoints ---
            with st.expander("1. Keypoints trên từng ảnh", expanded=True):
                kcols = st.columns(len(result.keypoint_visualizations))
                for i, (vis, count) in enumerate(
                    zip(result.keypoint_visualizations, result.keypoint_counts)
                ):
                    with kcols[i]:
                        st.image(
                            bgr_to_rgb(vis),
                            caption=f"Ảnh #{i + 1}: {count} keypoints",
                            use_container_width=True,
                        )

            # --- Stage 2 & 3: Matches + warped ---
            with st.expander("2. Khớp đặc trưng & tỉ lệ inlier RANSAC", expanded=True):
                for pair in result.pair_results:
                    st.markdown(f"**{pair.pair_label}**")
                    st.write(
                        f"Số khớp sau Lowe: **{pair.match_count}** · "
                        f"Inlier RANSAC: **{pair.inlier_count}** "
                        f"({pair.inlier_ratio:.1%})"
                    )
                    st.image(
                        bgr_to_rgb(pair.match_visualization),
                        caption="drawMatches (top 50 theo khoảng cách)",
                        use_container_width=True,
                    )

            with st.expander("3. Ảnh nguồn đã warp (trung gian)", expanded=False):
                for pair in result.pair_results:
                    st.markdown(f"**{pair.pair_label}**")
                    st.image(
                        bgr_to_rgb(pair.warped_intermediate),
                        caption="warped source trên canvas",
                        use_container_width=True,
                    )

            # --- Stage 4: Final ---
            with st.expander("4. Panorama trước / sau cắt viền", expanded=True):
                c_before, c_after = st.columns(2)
                with c_before:
                    st.image(
                        bgr_to_rgb(result.panorama_before_crop),
                        caption="Trước khi crop viền đen",
                        use_container_width=True,
                    )
                with c_after:
                    st.image(
                        bgr_to_rgb(result.panorama_final),
                        caption="Sau khi crop (kết quả cuối)",
                        use_container_width=True,
                    )

            st.download_button(
                label="Tải panorama cuối (PNG)",
                data=encode_png_bytes(result.panorama_final),
                file_name="panorama.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.info("Tải lên 2–5 ảnh JPG/PNG để bắt đầu.")


if __name__ == "__main__":
    main()
