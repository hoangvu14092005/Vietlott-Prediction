# 🎰 Vietlott 6/45 Prediction System

Hệ thống dự đoán số Vietlott 6/45 sử dụng Machine Learning với Ensemble Models.

## 📁 Cấu trúc dự án

```
Vietlott_Prediction/
├── main.py                 # File chính để chạy dự đoán
├── lottery_data.npy        # Dữ liệu lịch sử kết quả quay số (file input chính)
├── src/                    # Thư mục chứa các module chính
│   ├── data_loader.py      # Module tải và xử lý dữ liệu từ .npy
│   ├── features.py         # Feature Engineering (FFT, Delta, Poisson)
│   ├── models.py           # Quản lý các model ML (XGB, LGB, Cat, TabNet, RF, LR)
│   ├── tuner.py            # Hyperparameter tuning với Optuna
│   └── data_warehouse.py   # Database warehouse (tùy chọn)
├── data/                   # Thư mục chứa models và configs
│   ├── lottery.db          # SQLite database chứa dữ liệu đã xử lý
│   ├── ultra_ensemble_v4.pkl  # Model đã train sẵn
│   └── best_params.json    # Tham số tối ưu cho các models
├── scripts/                # Các script tiện ích
│   └── create_best_params.py  # Script tạo file best_params.json
└── README.md               # File này
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo file `lottery_data.npy` nằm ở thư mục gốc. File này chứa dữ liệu lịch sử kết quả quay số dạng numpy array.

### 2. Chạy dự đoán

```bash
python main.py
```

### 3. Cấu hình (trong main.py)

- `RUN_TUNING = False`: Bật/tắt hyperparameter tuning (chỉ cần chạy 1 lần)
- `FORCE_RETRAIN = True`: Bắt buộc train lại model (đặt False nếu đã có model tốt)
- `PAST_WINDOW = 100`: Số kỳ quay quá khứ để tính features
- `TEST_SIZE = 50`: Số kỳ cuối dùng để kiểm tra độ chính xác
- `TOP_K = 8`: Số lượng số đề xuất cho mỗi kỳ

## 🧠 Kiến trúc Model

### Ensemble Models
Hệ thống sử dụng 6 models và kết hợp kết quả:
1. **XGBoost** - Gradient Boosting
2. **LightGBM** - Gradient Boosting (nhanh hơn)
3. **CatBoost** - Gradient Boosting (xử lý categorical tốt)
4. **TabNet** - Deep Learning cho tabular data
5. **Random Forest** - Ensemble tree-based
6. **Logistic Regression** - Baseline model

Kết quả cuối cùng là trung bình xác suất từ tất cả models.

### Feature Engineering

1. **Tần suất & Gần đây (Gan)**: Tần suất xuất hiện và số kỳ gần nhất mỗi số xuất hiện
2. **FFT Signals**: Phân tích chu kỳ ẩn bằng biến đổi Fourier
3. **Delta & Skewness**: Phân tích cấu trúc bộ số (khoảng cách, độ lệch)
4. **Poisson Probability**: So sánh xác suất thực tế vs lý thuyết

## 📊 Đánh giá hiệu suất

Hệ thống sẽ:
- Train trên dữ liệu lịch sử (trừ TEST_SIZE kỳ cuối)
- Test trên TEST_SIZE kỳ cuối
- Hiển thị số lượng số trúng trung bình trên mỗi kỳ
- Dự đoán kỳ tiếp theo

## 🔧 Tuning Hyperparameters

Để tối ưu hóa tham số models:

1. Trong `main.py`, đặt `RUN_TUNING = True`
2. Chạy `python main.py`
3. Sau khi hoàn thành, đặt lại `RUN_TUNING = False`

Kết quả sẽ được lưu vào `data/best_params.json` và tự động được sử dụng trong các lần train tiếp theo.

## 📝 Lưu ý

- File `lottery_data.npy` phải có format: mỗi dòng là 1 kỳ quay, mỗi kỳ có 6 số (1-45)
- Model đã train sẽ được lưu vào `data/ultra_ensemble_v4.pkl`
- Dữ liệu sẽ được import vào SQLite database `data/lottery.db` để xử lý

## 🎯 Logic giải quyết bài toán

1. **Feature Engineering**: Tạo các đặc trưng toán học từ lịch sử (FFT, Poisson, Delta)
2. **Multi-model Ensemble**: Kết hợp nhiều models để giảm bias và variance
3. **Probability Averaging**: Lấy trung bình xác suất từ tất cả models
4. **Top-K Selection**: Chọn K số có xác suất cao nhất

Lưu ý: Xổ số là ngẫu nhiên, model chỉ giúp phân tích pattern và đưa ra gợi ý dựa trên dữ liệu lịch sử.

