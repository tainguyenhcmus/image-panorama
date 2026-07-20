# Ghép ảnh Panorama (Image Stitching)

Ứng dụng demo cho môn xử lý ảnh: pipeline computer vision cổ điển (không dùng deep learning, không dùng `cv2.Stitcher`).

## Cài đặt

Yêu cầu Python 3.10+.

```bash
cd panorama
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt theo địa chỉ Streamlit in ra (thường `http://localhost:8501`).

## Cách dùng nhanh

1. Tải 2–5 ảnh JPG/PNG theo thứ tự **trái → phải**.
2. Dùng nút ◀ / ▶ để sắp lại thứ tự nếu cần.
3. Chỉnh tham số ở sidebar (ORB/SIFT, Lowe ratio, RANSAC, feathering).
4. Bấm **Stitch** và xem từng giai đoạn pipeline trong các mục mở rộng.
5. Tải panorama cuối dạng PNG.

**Gợi ý chụp:** chồng lấn khoảng 30–50%, giữ máy nằm ngang, tránh cảnh quá ít kết cấu (tường trắng) hoặc vật chuyển động mạnh.

## Cấu trúc mã nguồn

```
panorama/
├── app.py                 # Giao diện Streamlit
├── stitcher/
│   ├── features.py        # ORB/SIFT + BFMatcher + Lowe
│   ├── homography.py      # findHomography + RANSAC
│   ├── warp_blend.py      # warpPerspective + feathering + crop
│   └── pipeline.py        # Điều phối ghép nhiều ảnh
├── requirements.txt
└── README.md
```

## Pipeline (tóm tắt)

1. **Phát hiện đặc trưng** — ORB (mặc định, `nfeatures=3000`) hoặc SIFT.
2. **Khớp đặc trưng** — `BFMatcher.knnMatch(k=2)` + kiểm tra tỉ lệ Lowe (mặc định 0.75).
3. **Homography** — `cv2.findHomography(..., RANSAC, threshold=5.0)`, báo cáo tỉ lệ inlier.
4. **Warp** — `cv2.warpPerspective` lên canvas đủ lớn cho cả hai ảnh.
5. **Blend** — feathering (gradient alpha theo distance transform trong vùng chồng nhau).
6. **Hậu xử lý** — cắt viền đen bằng bounding box của contour lớn nhất.

Với nhiều ảnh, **ảnh giữa** được chọn làm tham chiếu; các ảnh bên trái rồi bên phải được ghép tuần tự vào panorama hiện tại để giảm biến dạng phối cảnh.

---

## Ghi chú lý thuyết (cho báo cáo)

### ORB và SIFT khác nhau thế nào?

- **SIFT** (Scale-Invariant Feature Transform) xây pyramid tỉ lệ, phát hiện cực trị trong không gian scale-space, rồi mô tả bằng histogram hướng gradient quanh điểm. Ưu điểm: ổn định với thay đổi tỉ lệ, góc quay và một phần thay đổi ánh sáng. Nhược: chậm hơn, descriptor dạng số thực (L2).
- **ORB** (Oriented FAST and Rotated BRIEF) dùng FAST để tìm góc, gán hướng, rồi mô tả bằng BRIEF nhị phân đã xoay. Ưu điểm: rất nhanh, miễn phí bản quyền, phù hợp demo thời gian thực. Nhược: thường kém SIFT hơn trên thay đổi tỉ lệ mạnh / góc nhìn khó. Khớp ORB dùng khoảng cách Hamming.

Trong đồ án này mặc định ORB cho tốc độ; có thể chuyển SIFT khi ảnh khó khớp.

### Kiểm tra tỉ lệ Lowe (Lowe's ratio test)

Khi khớp descriptor, với mỗi điểm ta lấy **2 hàng xóm gần nhất**. Nếu khoảng cách tới hàng xóm 1 gần bằng hàng xóm 2 (`d1 ≈ d2`) thì khớp bị coi là **mơ hồ** (có thể khớp sai). Chỉ giữ khớp thỏa:

\[
d_1 < \tau \cdot d_2
\]

với \(\tau\) thường khoảng 0.75. Ngưỡng thấp hơn → ít khớp nhưng sạch hơn; cao hơn → nhiều khớp nhưng dễ lẫn outlier.

### Vì sao cần RANSAC khi ước lượng homography?

Ngay cả sau tỉ lệ Lowe vẫn còn **outlier** (khớp sai). Homography cần tối thiểu 4 cặp điểm đúng; nếu lẫn điểm sai, phương pháp least-squares trực tiếp sẽ lệch nặng. **RANSAC** lặp lại: chọn ngẫu nhiên 4 cặp → tính H → đếm inlier (sai số tái chiếu < ngưỡng) → giữ mô hình tốt nhất. Nhờ đó homography ổn định hơn trên dữ liệu thực.

### Homography biểu diễn điều gì?

Homography \(H\) là ma trận \(3\times 3\) ánh xạ điểm thuần nhất giữa hai mặt phẳng ảnh:

\[
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix}
\sim
H
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix},\quad
x'_{\text{euclid}} = x'/w',\ y'_{\text{euclid}} = y'/w'
\]

Nó mô tả quan hệ phối cảnh khi hai ảnh quan sát cùng một mặt phẳng (hoặc xấp xỉ khi camera xoay quanh tâm quang học — trường hợp panorama). `warpPerspective` dùng \(H\) để biến đổi toàn bộ ảnh nguồn sang hệ tọa độ ảnh tham chiếu.

### Vì sao feathering làm mềm đường nối?

Nếu chỉ **dán đè** (paste-over) ảnh đã warp lên ảnh kia, ranh giới chồng nhau thường lộ thành đường cắt cứng do sai số căn chỉnh nhẹ và khác biệt phơi sáng. **Feathering** gán trọng số alpha trong vùng overlap theo khoảng cách tới biên mask (distance transform): càng sâu trong một ảnh thì trọng số ảnh đó càng lớn. Kết quả là chuyển màu mượt, đường nối ít nhìn thấy hơn.

---

## Lỗi thường gặp

| Thông báo | Cách xử lý gợi ý |
|-----------|------------------|
| Không đủ điểm khớp (< 4) | Tăng chồng lấn, chọn cảnh nhiều chi tiết, thử SIFT |
| Ít inlier sau RANSAC | Nới ngưỡng RANSAC, kiểm tra thứ tự ảnh, giảm motion blur |
| Canvas / ảnh quá lớn | Ảnh quá phân giải cao — resize xuống trước khi upload |
