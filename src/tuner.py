# src/tuner.py
import optuna
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class UltraTuner:
    def __init__(self, X, y, output_path="data/best_params.json"):
        self.output_path = output_path
        
        # 1. Cắt dữ liệu cũ quá (giữ 2000 kỳ cuối) để model không học nhiễu
        limit = 2000
        if len(X) > limit:
            print(f"✂️ [Tuner] Giới hạn dữ liệu tuning: {limit} dòng gần nhất.")
            X_tune = X[-limit:]
            y_tune = y[-limit:]
        else:
            X_tune = X
            y_tune = y

        # 2. CHIA TẬP TRAIN/VAL
        self.X_train, self.X_val, self.y_train_full, self.y_val_full = train_test_split(
            X_tune, y_tune, test_size=0.2, random_state=42, shuffle=False
        )

        # 3. CHIẾN THUẬT ĐẠI DIỆN (PROXY TUNING)
        # Chỉ lấy 5 cột mục tiêu ngẫu nhiên (ví dụ: index 0, 10, 20, 30, 40) để Tuning
        # Giúp tốc độ nhanh gấp 9 lần (5 models vs 45 models)
        self.proxy_indices = [0, 10, 20, 30, 40] 
        
        # Chỉ lấy các cột đại diện
        self.y_train = self.y_train_full[:, self.proxy_indices]
        self.y_val = self.y_val_full[:, self.proxy_indices]
        
        print(f"⚡ [Tuner Strategy] Proxy Mode: Chỉ tuning trên {len(self.proxy_indices)}/45 số đại diện.")
        
        self.best_params = {}

    def _evaluate_proxy(self, model):
        """
        Hàm chấm điểm dựa trên Log Loss của 5 cột đại diện.
        Log Loss càng thấp -> Model càng tự tin và chính xác.
        """
        # Train trên 5 cột
        model.fit(self.X_train, self.y_train)
        
        # Predict proba trả về list (mỗi phần tử là proba cho 1 cột)
        probas_list = model.predict_proba(self.X_val)
        
        # Chuyển list of arrays -> mảng 2D (n_samples, 5)
        # Lấy cột index 1 (xác suất ra số đó)
        try:
            val_preds = np.array([p[:, 1] for p in probas_list]).T
        except IndexError:
            # Fallback cho trường hợp model dự đoán cứng (không ra xác suất)
             val_preds = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_list]).T

        # Tính Log Loss trung bình
        score = log_loss(self.y_val, val_preds)
        return score

    def tune_xgb(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist', # Tăng tốc
            'verbosity': 0,
            'random_state': 42
        }
        model = MultiOutputClassifier(XGBClassifier(**params))
        return self._evaluate_proxy(model)

    def tune_lgb(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'verbose': -1,
            'random_state': 42
        }
        model = MultiOutputClassifier(LGBMClassifier(**params))
        return self._evaluate_proxy(model)

    def tune_cat(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 400),
            # CatBoost rất nặng với depth lớn, giữ mức 4-7 là tối ưu cho xổ số
            'depth': trial.suggest_int('depth', 4, 7), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': 128, # Giảm độ chi tiết biên để tăng tốc
            'thread_count': -1,  # Dùng full CPU
            'verbose': 0,
            'random_state': 42,
            'allow_writing_files': False
        }
        model = MultiOutputClassifier(CatBoostClassifier(**params))
        return self._evaluate_proxy(model)

    def run_optimization(self, n_trials=20):
        print(f"🔥 [Tuner] Bắt đầu tối ưu hóa với {n_trials} trials (Proxy Mode)...")
        
        # 1. XGBoost
        print("   -> 🚀 Tuning XGBoost...")
        study_xgb = optuna.create_study(direction='minimize') # LogLoss -> Minimize
        study_xgb.optimize(self.tune_xgb, n_trials=n_trials)
        self.best_params['xgb'] = study_xgb.best_params
        print(f"      ✅ Best LogLoss: {study_xgb.best_value:.4f}")

        # 2. LightGBM
        print("   -> 🚀 Tuning LightGBM...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(self.tune_lgb, n_trials=n_trials)
        self.best_params['lgb'] = study_lgb.best_params
        print(f"      ✅ Best LogLoss: {study_lgb.best_value:.4f}")
        
        # 3. CatBoost
        print("   -> 🚀 Tuning CatBoost...")
        study_cat = optuna.create_study(direction='minimize')
        study_cat.optimize(self.tune_cat, n_trials=n_trials) # Giảm số trial của Cat nếu cần
        self.best_params['cat'] = study_cat.best_params
        print(f"      ✅ Best LogLoss: {study_cat.best_value:.4f}")

        self.save_best_params()

    def save_best_params(self):
        with open(self.output_path, "w") as f:
            json.dump(self.best_params, f, indent=4)
        print(f"💾 [Tuner] Đã lưu params vào '{self.output_path}'")