# 🎯 Giải pháp Logic cho Bài toán Dự đoán Vietlott 6/45

## 📋 Tổng quan

Hệ thống dự đoán số Vietlott 6/45 sử dụng Machine Learning với phương pháp Ensemble để kết hợp nhiều models và đưa ra dự đoán tốt nhất.

## 🧠 Logic Giải quyết Bài toán

### 1. **Feature Engineering (Tạo Đặc trưng)**

#### a) Tần suất & Gần đây (Gan)
- **Tần suất**: Đếm số lần mỗi số (1-45) xuất hiện trong N kỳ quay gần nhất
- **Gan (Last Seen)**: Số kỳ gần nhất mỗi số xuất hiện
- **Logic**: Số có tần suất cao hoặc "gan" (lâu không ra) có thể có xác suất cao hơn

#### b) FFT Signals (Phân tích Chu kỳ)
- Sử dụng **Biến đổi Fourier** để phát hiện chu kỳ ẩn trong chuỗi xuất hiện của mỗi số
- **Logic**: Nếu một số có chu kỳ xuất hiện đều đặn, có thể dự đoán được thời điểm tiếp theo

#### c) Delta & Skewness (Cấu trúc Bộ số)
- **Delta**: Khoảng cách giữa các số trong kỳ quay gần nhất (ví dụ: 5, 12, 18 → Delta: 7, 6)
- **Skewness**: Độ lệch phân phối số (số tập trung ở đầu/cuối hay rải đều)
- **Logic**: Phân tích pattern cấu trúc của bộ số để hiểu xu hướng

#### d) Poisson Probability (Xác suất Lý thuyết)
- So sánh xác suất thực tế vs lý thuyết (6/45 = 0.133)
- **Logic**: Số có xác suất thực tế lệch nhiều so với lý thuyết có thể "bù trừ" trong tương lai

### 2. **Multi-Model Ensemble**

#### Tại sao dùng Ensemble?
- **Giảm Bias**: Mỗi model có bias khác nhau, kết hợp giảm bias tổng thể
- **Giảm Variance**: Kết quả ổn định hơn khi kết hợp nhiều models
- **Tận dụng điểm mạnh**: Mỗi model có thế mạnh riêng

#### Các Models được sử dụng:
1. **XGBoost**: Gradient Boosting mạnh, xử lý non-linear tốt
2. **LightGBM**: Nhanh, hiệu quả với dữ liệu lớn
3. **CatBoost**: Xử lý categorical features tốt, ít overfitting
4. **TabNet**: Deep Learning cho tabular data, học được pattern phức tạp
5. **Random Forest**: Ensemble tree-based, robust
6. **Logistic Regression**: Baseline, đơn giản nhưng hiệu quả

### 3. **Probability Averaging**

- Mỗi model dự đoán xác suất cho 45 số (1-45)
- **Kết hợp**: Lấy trung bình xác suất từ tất cả models
- **Logic**: Xác suất trung bình từ nhiều models đáng tin cậy hơn xác suất từ 1 model

### 4. **Top-K Selection**

- Chọn K số (mặc định K=8) có xác suất cao nhất
- **Logic**: Không dự đoán chính xác 6 số, mà đề xuất K số có khả năng cao nhất

## 🔄 Quy trình Hoạt động

```
1. Load Data (lottery_data.npy)
   ↓
2. Feature Engineering (Tạo 300+ features từ lịch sử)
   ↓
3. Train/Load Models (6 models ensemble)
   ↓
4. Predict (Tính xác suất cho 45 số)
   ↓
5. Select Top-K (Chọn 8 số có xác suất cao nhất)
   ↓
6. Output (Đề xuất bộ số)
```

## 📊 Đánh giá Hiệu suất

- **Metric**: Số lượng số trúng trong Top-K
- **Test**: Dùng N kỳ cuối (mặc định 50) để kiểm tra
- **Kỳ vọng**: Trung bình 2-3 số trúng trong Top-8 (tỷ lệ ~25-37%)

## ⚠️ Lưu ý Quan trọng

1. **Xổ số là ngẫu nhiên**: Model không thể dự đoán chính xác 100%, chỉ phân tích pattern và đưa ra gợi ý
2. **Overfitting**: Cần cẩn thận với overfitting - model có thể "nhớ" dữ liệu train nhưng không generalize tốt
3. **Data Quality**: Chất lượng dữ liệu ảnh hưởng lớn đến kết quả
4. **Hyperparameter Tuning**: Tối ưu tham số giúp cải thiện hiệu suất đáng kể

## 🚀 Cải tiến Tiềm năng

1. **Time Series Features**: Thêm features về thời gian (ngày trong tuần, tháng, etc.)
2. **Pair/Triplet Analysis**: Phân tích cặp số, bộ 3 số thường xuất hiện cùng nhau
3. **Hot/Cold Numbers**: Phân loại số "nóng" (xuất hiện nhiều) và "lạnh" (ít xuất hiện)
4. **Sequence Patterns**: Phân tích chuỗi số liên tiếp, số chẵn/lẻ
5. **Cross-Validation**: Sử dụng time-series cross-validation thay vì random split

## 📈 Kết luận

Hệ thống sử dụng:
- **Feature Engineering** để trích xuất thông tin từ lịch sử
- **Ensemble Learning** để kết hợp sức mạnh của nhiều models
- **Probability-based Selection** để đưa ra gợi ý hợp lý

Mặc dù không thể đảm bảo trúng 100%, nhưng phương pháp này giúp phân tích dữ liệu một cách có hệ thống và đưa ra gợi ý dựa trên pattern lịch sử.

