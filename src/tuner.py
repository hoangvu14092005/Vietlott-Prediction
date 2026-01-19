# src/tuner.py
import optuna
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Import Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class UltraTuner:
    def __init__(self, X, y, output_path="data/best_params.json"):
        self.X = X
        self.y = y
        self.output_path = output_path
        # Chia tập Validation riêng để Tuner đánh giá
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.best_params = {}

    def _evaluate_model(self, model):
        """Hàm chấm điểm Model: Dựa trên số lượng số trúng trong Top 8"""
        model.fit(self.X_train, self.y_train)
        
        # Dự báo
        probas_list = model.predict_proba(self.X_val)
        
        # Xử lý output của MultiOutputClassifier (list of arrays -> array)
        # Cách lấy xác suất class 1 (số được chọn)
        try:
            # Cho XGB, LGBM
            final_proba = np.array([p[:, 1] for p in probas_list]).T
        except:
            # Fallback nếu format khác
            final_proba = np.array([p[:, 1] for p in probas_list]).T

        total_hits = 0
        n_samples = len(self.X_val)
        
        for i in range(n_samples):
            # Lấy Top 8 số có xác suất cao nhất
            top_indices = np.argsort(final_proba[i])[-8:] 
            pred_nums = set(top_indices + 1)
            
            # Số thực tế
            true_indices = np.where(self.y_val[i] == 1)[0]
            true_nums = set(true_indices + 1)
            
            hits = len(pred_nums & true_nums)
            total_hits += hits
            
        return total_hits / n_samples # Trả về trung bình số hit/kỳ

    def tune_xgb(self, trial):
        """Không gian tìm kiếm cho XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0
        }
        model = MultiOutputClassifier(XGBClassifier(**params))
        return self._evaluate_model(model)

    def tune_lgb(self, trial):
        """Không gian tìm kiếm cho LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': -1,
            'random_state': 42
        }
        model = MultiOutputClassifier(LGBMClassifier(**params))
        return self._evaluate_model(model)

    def tune_cat(self, trial):
        """Không gian tìm kiếm cho CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 600),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'verbose': 0,
            'random_state': 42,
            'allow_writing_files': False
        }
        model = MultiOutputClassifier(CatBoostClassifier(**params))
        return self._evaluate_model(model)

    def run_optimization(self, n_trials=20):
        print(f"🔥 [Tuner] Bắt đầu tối ưu hóa với {n_trials} thử nghiệm mỗi model...")
        
        # 1. Tune XGBoost
        print("   -> Đang Tuning XGBoost...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(self.tune_xgb, n_trials=n_trials)
        self.best_params['xgb'] = study_xgb.best_params
        print(f"      ✅ XGB Best Score: {study_xgb.best_value:.4f}")

        # 2. Tune LightGBM
        print("   -> Đang Tuning LightGBM...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(self.tune_lgb, n_trials=n_trials)
        self.best_params['lgb'] = study_lgb.best_params
        print(f"      ✅ LGB Best Score: {study_lgb.best_value:.4f}")
        
        # 3. Tune CatBoost
        print("   -> Đang Tuning CatBoost (Có thể hơi lâu)...")
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(self.tune_cat, n_trials=n_trials)
        self.best_params['cat'] = study_cat.best_params
        print(f"      ✅ CAT Best Score: {study_cat.best_value:.4f}")

        # Lưu kết quả
        self.save_best_params()

    def save_best_params(self):
        with open(self.output_path, "w") as f:
            json.dump(self.best_params, f, indent=4)
        print(f"💾 [Tuner] Đã lưu bộ tham số tối ưu vào '{self.output_path}'")