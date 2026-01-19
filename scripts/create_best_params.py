import json
import os

# Đây là bộ tham số TỐT NHẤT trích xuất từ Log của bạn
# Mình đã lọc ra Trial 12 (XGB), Trial 7 (LGB) và Trial 2 (Cat)

best_params = {
    "xgb": {
        # Best Trial 12 từ log của bạn
        "n_estimators": 129,
        "max_depth": 8,
        "learning_rate": 0.2057,
        "subsample": 0.61,
        "colsample_bytree": 0.99,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0
    },
    "lgb": {
        # Best Trial 7 từ log của bạn
        "n_estimators": 123,
        "learning_rate": 0.0256,
        "num_leaves": 71,
        "feature_fraction": 0.66,
        "verbose": -1,
        "n_jobs": -1
    },
    "cat": {
        # Best Trial 2 từ log của bạn (Vừa chính xác vừa NHANH)
        "iterations": 312,
        "depth": 4,  # Depth 4 chạy siêu nhanh, tránh bị treo như Depth 10
        "learning_rate": 0.2667,
        "l2_leaf_reg": 4.3,
        "verbose": 0,
        "allow_writing_files": False
    }
}

output_path = "data/best_params.json"
os.makedirs("data", exist_ok=True)

with open(output_path, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"✅ ĐÃ CỨU HỘ THÀNH CÔNG!")
print(f"📁 Đã tạo file '{output_path}' với tham số chuẩn từ Log.")
print("👉 Bây giờ bạn có thể chạy main.py được rồi!")

