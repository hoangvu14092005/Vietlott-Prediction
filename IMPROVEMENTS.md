# 🚀 Tài liệu Cải tiến Dự án Vietlott Prediction V7

## 📋 Tổng quan

Dự án đã được nâng cấp lên phiên bản V7 với nhiều cải tiến quan trọng về features, models và output format.

## ✨ Các Cải tiến Chính

### 1. 📊 Nâng cấp Features (300 → 600+ features)

#### Features mới được thêm vào:

1. **Hot/Cold Numbers** (90 features)
   - Phân loại số "nóng" (xuất hiện nhiều) và "lạnh" (ít xuất hiện)
   - Vector 45 số nóng + 45 số lạnh

2. **Pair Frequency** (45 features)
   - Tần suất xuất hiện của các cặp số phổ biến nhất
   - Giúp phát hiện pattern các số thường đi cùng nhau

3. **Gap Analysis** (45 features)
   - Phân tích khoảng cách giữa các số trong các kỳ quay gần đây
   - Giúp hiểu cấu trúc phân bố số

4. **Sum & Statistics** (5 features)
   - Tổng các số trong kỳ quay
   - Tỷ lệ số chẵn/lẻ
   - Số liên tiếp trong kỳ gần nhất

5. **Zone Distribution** (3 features)
   - Chia 45 số thành 3 vùng: 1-15, 16-30, 31-45
   - Phân tích phân bố số theo vùng

6. **Trend Analysis** (45 features)
   - Xu hướng tần suất của mỗi số trong 20 kỳ gần nhất
   - Phát hiện số đang "nóng lên" hoặc "nguội đi"

7. **Correlation Matrix** (45 features)
   - Ma trận tương quan giữa các số
   - Số nào thường xuất hiện cùng nhau

8. **Entropy & Variance** (2 features)
   - Entropy của phân phối tần suất
   - Variance của tần suất

### 2. 🧠 Cải thiện Models

#### Weighted Ensemble
- Thay vì trung bình đơn giản, giờ sử dụng **trọng số** cho từng model
- Models tốt hơn sẽ có trọng số cao hơn
- Có thể cập nhật trọng số dựa trên validation performance

#### Early Stopping
- LightGBM và CatBoost giờ hỗ trợ early stopping
- Tránh overfitting và giảm thời gian training

#### Regularization
- Thêm L1 và L2 regularization cho XGBoost
- Giúp model generalize tốt hơn

#### Model Weights Management
- Lưu trọng số cùng với models
- Có thể cập nhật động dựa trên performance

### 3. 📈 Output Format: Xác suất thay vì Số cụ thể

#### Trước đây:
- Chỉ output top-K số (ví dụ: [5, 12, 18, 23, 28, 35, 40, 42])
- Không biết độ tin cậy của từng số

#### Bây giờ:
- **Output xác suất cho TẤT CẢ 45 số**
- Hiển thị dạng bảng với:
  - Xác suất chi tiết (0.0000 - 1.0000)
  - Phần trăm (%)
  - Mức độ (Rất cao, Cao, Trung bình, Thấp, Rất thấp)
  - Biểu đồ bar chart

#### Tính năng mới:
- **Bảng xác suất đầy đủ**: Xem xác suất của tất cả 45 số
- **Top-K với xác suất**: Top K số kèm xác suất chi tiết
- **Phân tích theo vùng**: Tổng xác suất theo vùng 1-15, 16-30, 31-45
- **Phân tích chẵn/lẻ**: Tổng xác suất số chẵn và lẻ
- **Tóm tắt thống kê**: Mean, std, min, max của phân phối xác suất

### 4. 🎨 Module Visualization

Tạo module `src/visualizer.py` với các chức năng:

- `print_probability_table()`: In bảng xác suất đẹp
- `print_probability_summary()`: Tóm tắt thống kê
- `get_top_numbers()`: Lấy top-K số với xác suất
- `get_probability_by_zones()`: Phân tích theo vùng
- `get_probability_by_parity()`: Phân tích chẵn/lẻ
- `print_zone_analysis()`: In phân tích vùng và chẵn/lẻ
- `export_probabilities_to_dict()`: Xuất ra dictionary

## 📊 So sánh Trước và Sau

| Tiêu chí | V4 (Cũ) | V7 (Mới) |
|----------|---------|----------|
| Số lượng Features | ~300 | ~600+ |
| Ensemble Method | Simple Average | Weighted Average |
| Early Stopping | ❌ | ✅ |
| Output Format | Top-K số | Xác suất 45 số |
| Visualization | ❌ | ✅ |
| Zone Analysis | ❌ | ✅ |
| Pair Analysis | ❌ | ✅ |
| Trend Analysis | ❌ | ✅ |

## 🎯 Lợi ích

1. **Chính xác hơn**: Nhiều features hơn giúp model học được nhiều pattern hơn
2. **Linh hoạt hơn**: Output xác suất cho phép người dùng tự quyết định
3. **Trực quan hơn**: Visualization giúp hiểu rõ hơn về dự đoán
4. **Tin cậy hơn**: Weighted ensemble và early stopping giảm overfitting

## 📝 Cách sử dụng

### Chạy dự đoán:
```bash
python main.py
```

### Output sẽ bao gồm:
1. Bảng xác suất đầy đủ cho 45 số
2. Top-K số có xác suất cao nhất
3. Tóm tắt thống kê phân phối
4. Phân tích theo vùng và chẵn/lẻ

### Sử dụng xác suất trong code:
```python
from src.visualizer import ProbabilityVisualizer

# Lấy xác suất
probas = manager.predict_ensemble(X_input)

# Lấy top-K
visualizer = ProbabilityVisualizer()
top_numbers = visualizer.get_top_numbers(probas, k=8)

# Xuất dictionary
prob_dict = visualizer.export_probabilities_to_dict(probas)
```

## 🔮 Hướng phát triển tiếp theo

1. **Deep Learning**: Thử các kiến trúc neural network phức tạp hơn
2. **Time Series Models**: Áp dụng LSTM, Transformer cho dữ liệu chuỗi thời gian
3. **Feature Selection**: Tự động chọn features quan trọng nhất
4. **AutoML**: Tự động tìm kiếm kiến trúc model tốt nhất
5. **Web Interface**: Tạo web app để hiển thị xác suất trực quan

## ⚠️ Lưu ý

- Xổ số là ngẫu nhiên, model chỉ phân tích pattern lịch sử
- Xác suất cao không đảm bảo sẽ trúng
- Nên kết hợp với phân tích khác và quản lý rủi ro hợp lý

