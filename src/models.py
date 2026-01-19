# src/models.py
import os
import json
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# The Boosting Trio
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Deep Learning
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

class UltraModelManager:
    def __init__(self, model_path="data/ultra_ensemble_v4.pkl"):
        self.model_path = model_path
        self.models = {}
        self.model_weights = {}  # Trọng số cho từng model dựa trên performance
        
    def load_best_params(self):
        """Đọc file json chứa tham số tối ưu"""
        path = "data/best_params.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                print(f"💎 [Model Manager] Đã tìm thấy Best Params! Đang áp dụng...")
                return json.load(f)
        else:
            print("⚠️ [Model Manager] Không thấy Best Params. Sử dụng cấu hình mặc định.")
            return {}

    def build_models(self):
        """
        Defines the architecture of the ensemble.
        """
        print("🔨 [Model Factory] Đang khởi tạo Models...")
        
        # Tải tham số tối ưu (nếu có)
        best = self.load_best_params()
        
        # 1. LightGBM (Fix lỗi trùng verbose)
        p_lgb = best.get('lgb', {'n_estimators': 300, 'learning_rate': 0.05})
        # Gán cứng verbose vào dict để tránh lỗi, và xóa tham số trong hàm khởi tạo
        p_lgb['verbose'] = -1
        p_lgb['early_stopping_rounds'] = 50
        self.models['lgb'] = MultiOutputClassifier(
            LGBMClassifier(**p_lgb, random_state=42)
        )
        
        # 2. XGBoost (Fix lỗi trùng verbosity)
        p_xgb = best.get('xgb', {'n_estimators': 300, 'max_depth': 6})
        p_xgb['verbosity'] = 0 
        p_xgb['tree_method'] = 'hist'
        p_xgb['reg_alpha'] = 0.1  # L1 regularization
        p_xgb['reg_lambda'] = 0.1  # L2 regularization
        self.models['xgb'] = MultiOutputClassifier(
            XGBClassifier(**p_xgb, random_state=42)
        )
        
        # 3. CatBoost (Fix lỗi trùng verbose)
        p_cat = best.get('cat', {'iterations': 300, 'depth': 6, 'learning_rate': 0.05})
        p_cat['verbose'] = 0
        p_cat['allow_writing_files'] = False
        p_cat['early_stopping_rounds'] = 50
        self.models['cat'] = MultiOutputClassifier(
            CatBoostClassifier(**p_cat, random_state=42)
        )
        
        # 4. Random Forest
        self.models['rf'] = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)

        # 5. Logistic Regression
        self.models['lr'] = MultiOutputClassifier(
            LogisticRegression(solver='liblinear', random_state=42)
        )

        # 6. TabNet
        self.models['tab'] = TabNetMultiTaskClassifier(
            verbose=0, 
            optimizer_params=dict(lr=0.02),
            seed=42
        )

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Trains all models with optional validation set for early stopping."""
        if not self.models:
            self.build_models()
            
        print(f"🚀 [Training] Bắt đầu huấn luyện trên {X_train.shape[0]} mẫu...")
        
        # Tính trọng số mặc định (có thể cập nhật sau khi đánh giá)
        default_weight = 1.0 / len(self.models)
        
        for name, model in self.models.items():
            print(f"   ⮞ Training {name.upper()}...", end=" ")
            try:
                if name == 'tab':
                    model.fit(X_train, y_train, max_epochs=50, batch_size=128, virtual_batch_size=64)
                elif name == 'lgb' and X_val is not None:
                    # LightGBM với early stopping
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             verbose=False)
                elif name == 'cat' and X_val is not None:
                    # CatBoost với early stopping
                    model.fit(X_train, y_train,
                             eval_set=(X_val, y_val),
                             verbose=False)
                else:
                    model.fit(X_train, y_train)
                print("✅ Done.")
                self.model_weights[name] = default_weight
            except Exception as e:
                print(f"❌ Failed: {e}")
                self.model_weights[name] = 0.0  # Không dùng model lỗi

        # Chuẩn hóa trọng số
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight

        self.save_models()

    def predict_ensemble(self, X_input, return_individual=False):
        """
        Returns the weighted averaged probability from all models.
        
        Args:
            X_input: Input features
            return_individual: If True, returns individual model predictions too
        
        Returns:
            final_proba: Weighted average probability (45 numbers)
            individual_probs: Dict of individual model predictions (if return_individual=True)
        """
        if not self.models:
            self.load_models()
            
        final_proba = np.zeros((45,))
        individual_probs = {}
        
        for name, model in self.models.items():
            try:
                raw_preds = model.predict_proba(X_input)
                
                if name == 'tab':
                    task_probs = np.array([task[:, 1][0] for task in raw_preds])
                else:
                    task_probs = np.array([task[0][1] for task in raw_preds])
                
                # Sử dụng trọng số nếu có, nếu không thì dùng trung bình đơn giản
                weight = self.model_weights.get(name, 1.0 / len(self.models))
                final_proba += task_probs * weight
                
                if return_individual:
                    individual_probs[name] = task_probs
                
            except Exception as e:
                # print(f"⚠️ Prediction error in {name}: {e}")
                pass
        
        # Đảm bảo xác suất hợp lệ (0-1)
        final_proba = np.clip(final_proba, 0, 1)
        
        if return_individual:
            return final_proba, individual_probs
        else:
            return final_proba
    
    def update_model_weights(self, validation_scores):
        """
        Cập nhật trọng số models dựa trên performance trên validation set.
        
        Args:
            validation_scores: Dict {model_name: score} - score càng cao càng tốt
        """
        total_score = sum(validation_scores.values())
        if total_score > 0:
            for name in self.models:
                score = validation_scores.get(name, 0)
                self.model_weights[name] = score / total_score
        else:
            # Nếu không có score, dùng trọng số đều
            default_weight = 1.0 / len(self.models)
            for name in self.models:
                self.model_weights[name] = default_weight

    def save_models(self):
        save_data = {
            'models': self.models,
            'weights': self.model_weights
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"💾 [System] Models saved to {self.model_path}")

    def load_models(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                save_data = pickle.load(f)
                # Backward compatibility: nếu là dict cũ chỉ có models
                if isinstance(save_data, dict) and 'models' in save_data:
                    self.models = save_data['models']
                    self.model_weights = save_data.get('weights', {})
                    # Nếu không có weights, dùng trọng số đều
                    if not self.model_weights:
                        default_weight = 1.0 / len(self.models)
                        self.model_weights = {name: default_weight for name in self.models}
                else:
                    # Format cũ: chỉ có models
                    self.models = save_data
                    default_weight = 1.0 / len(self.models)
                    self.model_weights = {name: default_weight for name in self.models}
            # print(f"📂 [System] Loaded {len(self.models)} models.")
        else:
            print("⚠️ No saved models found. Please train first.")
            self.build_models()