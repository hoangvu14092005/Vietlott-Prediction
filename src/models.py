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
        print("🔨 [Model Factory] Đang khởi tạo Models...")
        best = self.load_best_params()
        
        # 1. LightGBM
        p_lgb = best.get('lgb', {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 31})
        p_lgb['verbose'] = -1
        # LƯU Ý: Xóa early_stopping_rounds trong param khởi tạo để tránh lỗi với MultiOutput
        if 'early_stopping_rounds' in p_lgb: del p_lgb['early_stopping_rounds']
            
        self.models['lgb'] = MultiOutputClassifier(LGBMClassifier(**p_lgb, random_state=42))
        
        # 2. XGBoost
        p_xgb = best.get('xgb', {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03})
        p_xgb['verbosity'] = 0
        p_xgb['tree_method'] = 'hist'
        # Xóa early_stopping_rounds
        if 'early_stopping_rounds' in p_xgb: del p_xgb['early_stopping_rounds']

        self.models['xgb'] = MultiOutputClassifier(XGBClassifier(**p_xgb, random_state=42))
        
        # 3. CatBoost
        p_cat = best.get('cat', {'iterations': 500, 'depth': 6, 'learning_rate': 0.03})
        p_cat['verbose'] = 0
        p_cat['allow_writing_files'] = False
        # Xóa early_stopping_rounds
        if 'early_stopping_rounds' in p_cat: del p_cat['early_stopping_rounds']

        self.models['cat'] = MultiOutputClassifier(CatBoostClassifier(**p_cat, random_state=42))
        
        # 4. Random Forest (Giữ nguyên)
        self.models['rf'] = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)

        # 5. Logistic Regression (Giữ nguyên)
        self.models['lr'] = MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=42))

        # 6. TabNet (Native Multi-task -> Hỗ trợ tốt Validation)
        self.models['tab'] = TabNetMultiTaskClassifier(
            verbose=0, 
            optimizer_params=dict(lr=0.02),
            seed=42
        )

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        if not self.models:
            self.build_models()
            
        print(f"🚀 [Training] Bắt đầu huấn luyện trên {X_train.shape[0]} mẫu...")
        
        # Chuyển đổi dữ liệu sang float32 cho TabNet (tránh lỗi pytorch)
        X_train_Tab = X_train.astype(np.float32)
        y_train_Tab = y_train.astype(np.float32)
        if X_val is not None:
            X_val_Tab = X_val.astype(np.float32)
            y_val_Tab = y_val.astype(np.float32)

        default_weight = 1.0 / len(self.models)
        
        for name, model in self.models.items():
            print(f"   ⮞ Training {name.upper()}...", end=" ")
            try:
                # --- TABNET (Xử lý riêng vì hỗ trợ native validation) ---
                if name == 'tab':
                    if X_val is not None:
                        model.fit(
                            X_train_Tab, y_train_Tab,
                            eval_set=[(X_val_Tab, y_val_Tab)],
                            patience=15, max_epochs=100, # TabNet dùng patience thay vì early_stopping_rounds
                            batch_size=128, virtual_batch_size=64
                        )
                    else:
                        model.fit(X_train_Tab, y_train_Tab, max_epochs=50)

                # --- CÁC MODEL WRAPPER (LGBM, XGB, CAT, RF, LR) ---
                # Vì dùng MultiOutputClassifier nên ta KHÔNG truyền eval_set vào đây
                # để tránh lỗi dimension mismatch. Ta train full số trees.
                else:
                    model.fit(X_train, y_train)
                
                print("✅ Done.")
                self.model_weights[name] = default_weight
            
            except Exception as e:
                print(f"❌ Failed: {e}")
                import traceback
                traceback.print_exc() # In chi tiết lỗi để debug
                self.model_weights[name] = 0.0

        # Cân bằng trọng số
        total = sum(self.model_weights.values())
        if total > 0:
            for k in self.model_weights: self.model_weights[k] /= total
            
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