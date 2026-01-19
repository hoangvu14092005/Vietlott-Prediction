# main.py
# ================================================================
# VIETLOTT 6/45 - ULTRA PIPELINE V7 (ENHANCED WITH PROBABILITY OUTPUT)
# ================================================================
# Cải tiến:
# - Features: Tăng từ 300 lên 600+ features (Pair/Triplet, Hot/Cold, Zone, Gap)
# - Models: Weighted ensemble với early stopping
# - Output: Xác suất cho tất cả 45 số thay vì chỉ top-K
# ================================================================

import os
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer

# Import các module từ thư mục src
from src.data_loader import LotteryDataLoader
from src.features import UltraFeatureEngine
from src.models import UltraModelManager
from src.tuner import UltraTuner
from src.visualizer import ProbabilityVisualizer
from src.evaluator import LotteryEvaluator

# --- CẤU HÌNH ---
# Quan trọng: Đặt FORCE_RETRAIN = True cho lần chạy đầu tiên này 
# để xóa bỏ model "ngu" (train bằng dummy data) và train lại bằng data thật.
RUN_TUNING = False # chạy 1 lần rồi tắt đi
FORCE_RETRAIN = True 
PAST_WINDOW = 100   # Nhìn lại 100 kỳ quá khứ
TEST_SIZE = 50      # Dùng 50 kỳ cuối để kiểm tra độ chính xác
TOP_K = 8           # Gợi ý top 8 số

warnings.filterwarnings("ignore")

def main():
    print("\n" + "="*50)
    print("   🚀 VIETLOTT ULTRA ENSEMBLE V4 - REAL DATA MODE")
    print("="*50)

    # ------------------------------------------------------
    # BƯỚC 1: DATA LAYER (NẠP FILE NPY)
    # ------------------------------------------------------
    print("\n[1/5] 📥 DATA LAYER")
    loader = LotteryDataLoader()
    
    # --- ĐOẠN QUAN TRỌNG NHẤT ---
    # Đọc file lottery_data.npy của bạn
    # Hãy chắc chắn file này nằm cùng cấp với main.py
    loader.import_from_npy("lottery_data.npy")
    
    df = loader.load_data()
    print(f"   -> Đã tải thành công {len(df)} kỳ quay.")

    # ------------------------------------------------------
    # BƯỚC 2: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG)
    # ------------------------------------------------------
    print("\n[2/5] ⚙️  FEATURE ENGINEERING")
    print("   -> Đang tính toán tần suất, chu kỳ, cặp số...")
    
    feat_engine = UltraFeatureEngine(past_window=PAST_WINDOW)
    X, y = feat_engine.prepare_training_dataset(df)
    
    mlb = MultiLabelBinarizer(classes=range(1, 46))
    y_bin = mlb.fit_transform(y)
    
    X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
    y_train, y_test = y_bin[:-TEST_SIZE], y_bin[-TEST_SIZE:]
    
    print(f"   -> Kích thước Train: {X_train.shape[0]} dòng | Test: {X_test.shape[0]} dòng")
    
    # BƯỚC 2.5: HYPERPARAMETER TUNING (OPTUNA)
    if RUN_TUNING:
        print("\n[2.5] 🧪 HYPERPARAMETER TUNING (OPTUNA)")
        print("   -> Đang tìm kiếm bộ tham số tốt nhất (Sẽ mất vài phút)...")
        
        # Khởi tạo Tuner với toàn bộ dữ liệu (nó sẽ tự chia Validation)
        tuner = UltraTuner(X, y_bin) # Chú ý truyền y_bin (đã mã hóa one-hot)
        
        # Chạy 20 vòng thử nghiệm cho mỗi model (Tăng lên 50 nếu máy mạnh)
        tuner.run_optimization(n_trials=20)
        
        print("   -> Đã Tuning xong! Các model sau đây sẽ dùng tham số mới.")

    # ------------------------------------------------------
    # BƯỚC 3: MODEL TRAINING (HUẤN LUYỆN)
    # ------------------------------------------------------
    print("\n[3/5] 🧠 MODEL FACTORY")
    manager = UltraModelManager(model_path="data/ultra_ensemble_v4.pkl")
    
    model_file_exists = os.path.exists(manager.model_path)
    
    if FORCE_RETRAIN or not model_file_exists:
        print("   ⚠️  Phát hiện yêu cầu Retrain hoặc chưa có Model. Đang huấn luyện lại...")
        manager.train_all(X_train, y_train)
    else:
        print("   ✅ Đã tìm thấy Model cũ. Đang tải lên...")
        manager.load_models()
        
    print(f"   -> Các Model hoạt động: {list(manager.models.keys())}")

    # ------------------------------------------------------
    # BƯỚC 4: ĐÁNH GIÁ HIỆU SUẤT TOÀN DIỆN
    # ------------------------------------------------------
    print(f"\n[4/5] 🏁 KIỂM THỬ TRÊN {TEST_SIZE} KỲ CUỐI")
    
    # Dự báo cho toàn bộ test set
    print("   ⮞ Đang dự báo cho test set...")
    test_predictions = []
    for i in range(len(X_test)):
        probas = manager.predict_ensemble(X_test[i].reshape(1, -1))
        test_predictions.append(probas)
    
    test_predictions = np.array(test_predictions)
    
    # Hiển thị một vài kỳ mẫu
    print("\n   📋 MẪU DỰ ĐOÁN (3 kỳ cuối):")
    for i in range(max(0, len(X_test) - 3), len(X_test)):
        probas = test_predictions[i]
        top_indices = np.argsort(probas)[-TOP_K:][::-1]
        pred_nums = [idx + 1 for idx in top_indices]
        actual_indices = np.where(y_test[i] == 1)[0]
        actual_nums = [idx + 1 for idx in actual_indices]
        hits = len(set(pred_nums) & set(actual_nums))
        real_idx = len(X_train) + i + 1
        print(f"      Kỳ {real_idx}: Dự đoán {[int(x) for x in sorted(pred_nums)]} | KQ {[int(x) for x in sorted(actual_nums)]} | Trúng: {hits}")
    
    # Đánh giá toàn diện
    evaluator = LotteryEvaluator(top_k=TOP_K)
    evaluation_results = evaluator.comprehensive_evaluate(test_predictions, y_test)
    
    # In báo cáo
    evaluator.print_evaluation_report(evaluation_results)
    
    # Hiển thị phân phối xác suất trung bình trên test set
    print(f"\n   📈 PHÂN TÍCH XÁC SUẤT TRUNG BÌNH TRÊN TEST SET:")
    avg_probas = np.mean(test_predictions, axis=0)
    top_5_avg = np.argsort(avg_probas)[-5:][::-1]
    print(f"      Top 5 số có xác suất trung bình cao nhất:")
    for idx in top_5_avg:
        print(f"         Số {idx+1}: {avg_probas[idx]:.4f} ({avg_probas[idx]*100:.2f}%)")

    # ------------------------------------------------------
    # BƯỚC 5: DỰ BÁO TƯƠNG LAI VỚI XÁC SUẤT
    # ------------------------------------------------------
    print("\n" + "="*80)
    print(f"🔥 DỰ BÁO KỲ TIẾP THEO (Index {len(df) + 1}) - XÁC SUẤT")
    print("="*80)
    
    future_feat = feat_engine.create_single_feature(df, len(df))
    future_probas = manager.predict_ensemble(future_feat.reshape(1, -1))
    
    # Sử dụng Visualizer để hiển thị
    visualizer = ProbabilityVisualizer()
    
    # Hiển thị bảng xác suất đầy đủ
    visualizer.print_probability_table(future_probas, top_n=45)
    
    # Hiển thị top K số
    print(f"\n🎯 TOP {TOP_K} SỐ CÓ XÁC SUẤT CAO NHẤT:")
    print("-" * 80)
    top_numbers = visualizer.get_top_numbers(future_probas, k=TOP_K)
    for i, (num, prob) in enumerate(top_numbers, 1):
        level = ProbabilityVisualizer._get_probability_level(prob)
        bar = "█" * int(prob * 40)
        print(f"  {i:2d}. Số {num:2d}: {prob:.6f} ({prob*100:5.2f}%) {level:>12} {bar}")
    
    # Hiển thị tóm tắt
    visualizer.print_probability_summary(future_probas)
    
    # Phân tích theo vùng và chẵn/lẻ
    visualizer.print_zone_analysis(future_probas)
    
    print("="*80)
    
    # Xuất dictionary xác suất (có thể lưu vào file nếu cần)
    probability_dict = visualizer.export_probabilities_to_dict(future_probas)
    print(f"\n💾 Xác suất đã được tính toán cho tất cả 45 số.")
    print(f"   Bạn có thể sử dụng probability_dict để lưu hoặc xử lý thêm.")

if __name__ == "__main__":
    main()
