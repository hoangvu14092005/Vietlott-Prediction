# 📊 Tài liệu Phương pháp Đánh giá Model

## 🎯 Tổng quan

Module `src/evaluator.py` cung cấp hệ thống đánh giá toàn diện với nhiều metrics phù hợp cho bài toán dự đoán xổ số Vietlott 6/45.

## 📈 Các Metrics được sử dụng

### 1. **Hits Metrics** (Số trúng)

#### Mô tả:
Đếm số lượng số thực tế nằm trong Top-K dự đoán.

#### Metrics:
- **Mean Hits**: Trung bình số trúng/kỳ
- **Std Hits**: Độ lệch chuẩn
- **Median Hits**: Median số trúng
- **Min/Max Hits**: Số trúng thấp nhất/cao nhất
- **Hit Rate**: Tỷ lệ trúng = Mean Hits / K
- **Hits Distribution**: Phân phối số trúng (ví dụ: 0 trúng: 10 kỳ, 1 trúng: 15 kỳ...)

#### Ý nghĩa:
- **Kỳ vọng**: Với K=8, trung bình 2-3 số trúng/kỳ (~25-37%)
- **Tốt**: Mean Hits > 2.5 với K=8
- **Xuất sắc**: Mean Hits > 3.0 với K=8

#### Ví dụ:
```
Kỳ 1: Dự đoán [5, 12, 18, 23, 28, 35, 40, 42]
      Kết quả [5, 12, 19, 23, 29, 35]
      → Trúng: 4 số (5, 12, 23, 35)
```

---

### 2. **Precision, Recall, F1-Score**

#### Mô tả:
Đánh giá độ chính xác của dự đoán dựa trên classification metrics.

#### Metrics:
- **Precision**: Tỷ lệ số dự đoán đúng / tổng số dự đoán
  - Precision = TP / (TP + FP)
  - Cao = ít false positive
  
- **Recall**: Tỷ lệ số thực tế được dự đoán đúng
  - Recall = TP / (TP + FN)
  - Cao = ít false negative
  
- **F1-Score**: Trung bình điều hòa của Precision và Recall
  - F1 = 2 * (Precision * Recall) / (Precision + Recall)

#### Ý nghĩa:
- **Precision cao**: Dự đoán ít sai, nhưng có thể bỏ sót số thực tế
- **Recall cao**: Bắt được nhiều số thực tế, nhưng có thể dự đoán sai
- **F1 cao**: Cân bằng giữa Precision và Recall

#### Kỳ vọng:
- Precision: ~0.3-0.4 (vì chỉ dự đoán 8/45 số)
- Recall: ~0.5-0.7 (bắt được 3-4/6 số thực tế)
- F1: ~0.4-0.5

---

### 3. **Rank Metrics** (Vị trí trong Ranking)

#### Mô tả:
Đánh giá vị trí của số thực tế trong ranking xác suất.

#### Metrics:
- **Mean Rank**: Trung bình vị trí của số thực tế trong ranking
- **Median Rank**: Median vị trí
- **Mean Rank in Top-K**: Trung bình vị trí của số thực tế nằm trong Top-K
- **Coverage at K**: Tỷ lệ số thực tế nằm trong Top-K
- **Min/Max Rank**: Vị trí thấp nhất/cao nhất

#### Ý nghĩa:
- **Rank thấp** (1-8): Số thực tế có xác suất cao → Model tốt
- **Rank cao** (>20): Số thực tế có xác suất thấp → Model chưa tốt
- **Coverage cao**: Nhiều số thực tế nằm trong Top-K

#### Kỳ vọng:
- Mean Rank: 15-20 (giữa 45 số)
- Coverage at K: 50-70% (3-4/6 số thực tế trong Top-K)

---

### 4. **Coverage Metrics** (Độ bao phủ)

#### Mô tả:
Đánh giá bao nhiêu số thực tế được dự đoán.

#### Metrics:
- **Mean Coverage**: Trung bình coverage/kỳ
  - Coverage = Số trúng / Tổng số thực tế
  
- **Total Coverage**: Coverage tổng thể
  - Tỷ lệ số thực tế unique được dự đoán trong toàn bộ test set
  
- **Unique Counts**: Số lượng số thực tế/dự đoán unique

#### Ý nghĩa:
- **Coverage cao**: Model bắt được nhiều số thực tế
- **Total Coverage**: Cho biết model có bias về một số số cụ thể không

#### Kỳ vọng:
- Mean Coverage: 50-70% (3-4/6 số thực tế)
- Total Coverage: 60-80% (nhiều số thực tế được dự đoán ít nhất 1 lần)

---

### 5. **Probability Calibration** (Hiệu chuẩn Xác suất)

#### Mô tả:
Đánh giá độ chính xác của xác suất dự đoán.

#### Metrics:
- **Brier Score**: Độ lỗi bình phương trung bình
  - Brier = mean((predicted_prob - actual)^2)
  - Thấp hơn = tốt hơn (0 = hoàn hảo)
  
- **Calibration Error**: Độ lệch giữa xác suất dự đoán và xác suất thực tế
  - So sánh predicted probability vs empirical probability trong các bins

#### Ý nghĩa:
- **Brier Score thấp**: Xác suất dự đoán gần với thực tế
- **Calibration Error thấp**: Xác suất được hiệu chuẩn tốt

#### Kỳ vọng:
- Brier Score: 0.08-0.12 (cho bài toán multi-label)
- Calibration Error: < 0.05

---

### 6. **Baseline Comparison** (So sánh với Baseline)

#### Mô tả:
So sánh model với các phương pháp baseline đơn giản.

#### Baselines:
1. **Random Baseline**: Chọn ngẫu nhiên K số
2. **Frequency Baseline**: Chọn K số có tần suất cao nhất trong training data

#### Metrics:
- **Model Hits**: Số trúng của model
- **Random Hits**: Số trúng của random baseline
- **Frequency Hits**: Số trúng của frequency baseline
- **Improvement**: Cải thiện so với baseline

#### Ý nghĩa:
- **Improvement > 0**: Model tốt hơn baseline
- **Improvement > 50%**: Model tốt hơn đáng kể

#### Kỳ vọng:
- Random Baseline: ~1.07 hits/kỳ (K=8, 6 số thực tế)
- Frequency Baseline: ~1.5-2.0 hits/kỳ
- Model: > 2.0 hits/kỳ (cải thiện 30-50%)

---

## 🔄 Quy trình Đánh giá

### 1. **Train/Test Split**
```
Total Data: N kỳ
├── Train: N - TEST_SIZE kỳ (để train model)
└── Test: TEST_SIZE kỳ cuối (để đánh giá)
```

### 2. **Time-Series Split**
- **Quan trọng**: Không shuffle dữ liệu
- Test set phải là các kỳ **sau** training set
- Tránh data leakage

### 3. **Evaluation Process**
```python
# 1. Dự báo cho test set
predictions = model.predict(test_features)  # (n_samples, 45)

# 2. Đánh giá toàn diện
evaluator = LotteryEvaluator(top_k=8)
results = evaluator.comprehensive_evaluate(predictions, test_labels)

# 3. In báo cáo
evaluator.print_evaluation_report(results)
```

---

## 📊 Cách Đọc Báo cáo

### Ví dụ Output:
```
📊 BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN
================================================================================

🎯 1. HITS METRICS (Số trúng)
--------------------------------------------------------------------------------
   Trung bình số trúng/kỳ:     2.450 ± 0.850
   Median số trúng:            2.00
   Min/Max:                     0 / 5
   Hit Rate (trúng/K):          30.62%
   Phân phối:                  {0: 5, 1: 12, 2: 15, 3: 10, 4: 6, 5: 2}

📈 2. PRECISION/RECALL/F1
--------------------------------------------------------------------------------
   Precision:                   0.3062 ± 0.0850
   Recall:                     0.4083 ± 0.1133
   F1-Score:                   0.3500 ± 0.0950

📊 3. RANK METRICS
--------------------------------------------------------------------------------
   Mean Rank:                   18.50
   Median Rank:                 16.00
   Mean Rank trong Top-8:       4.25
   Coverage tại Top-8:          65.00%

🎯 4. COVERAGE
--------------------------------------------------------------------------------
   Mean Coverage:               40.83%
   Total Coverage:              73.33%
   Số thực tế unique:          45
   Số dự đoán unique:          42
   Overlap:                     33

📉 5. PROBABILITY CALIBRATION
--------------------------------------------------------------------------------
   Mean Brier Score:            0.0950
   Calibration Error:           0.0320

⚖️  6. BASELINE COMPARISON
--------------------------------------------------------------------------------
   Model Hits:                  2.450
   Random Baseline:             1.067
   Frequency Baseline:          1.850
   Cải thiện vs Random:         1.383 (129.62%)
   Cải thiện vs Frequency:      0.600
```

### Giải thích:
- **Hits**: Model trung bình trúng 2.45 số/kỳ (tốt!)
- **Precision/Recall**: Cân bằng, không quá thiên về một phía
- **Rank**: Số thực tế thường nằm ở vị trí 18.5 (trung bình), nhưng 65% nằm trong Top-8
- **Coverage**: Bắt được 40.83% số thực tế mỗi kỳ, tổng thể 73.33%
- **Calibration**: Xác suất khá chính xác (Brier = 0.095)
- **Baseline**: Tốt hơn Random 129%, tốt hơn Frequency 32%

---

## 🎯 Tiêu chí Đánh giá Tổng thể

### Model Tốt:
- ✅ Mean Hits > 2.5 (với K=8)
- ✅ Hit Rate > 30%
- ✅ Coverage > 60%
- ✅ Improvement vs Random > 100%
- ✅ Brier Score < 0.10

### Model Xuất sắc:
- ✅ Mean Hits > 3.0
- ✅ Hit Rate > 37%
- ✅ Coverage > 70%
- ✅ Improvement vs Random > 150%
- ✅ Brier Score < 0.08

### Model Cần Cải thiện:
- ⚠️ Mean Hits < 2.0
- ⚠️ Hit Rate < 25%
- ⚠️ Coverage < 50%
- ⚠️ Improvement vs Random < 50%

---

## 🔧 Tùy chỉnh Đánh giá

### Thay đổi Top-K:
```python
evaluator = LotteryEvaluator(top_k=10)  # Thay vì 8
```

### Chỉ đánh giá một số metrics:
```python
# Chỉ đánh giá hits
hits_results = evaluator.evaluate_hits(predictions, actuals)

# Chỉ đánh giá rank
rank_results = evaluator.evaluate_rank_metrics(predictions, actuals)
```

### So sánh nhiều models:
```python
results_model1 = evaluator.comprehensive_evaluate(pred1, actuals)
results_model2 = evaluator.comprehensive_evaluate(pred2, actuals)

# So sánh
print(f"Model 1 Hits: {results_model1['hits']['mean_hits']:.3f}")
print(f"Model 2 Hits: {results_model2['hits']['mean_hits']:.3f}")
```

---

## ⚠️ Lưu ý Quan trọng

1. **Xổ số là ngẫu nhiên**: Không thể đạt 100% accuracy
2. **Overfitting**: Cần kiểm tra performance trên validation set
3. **Time-series**: Không shuffle dữ liệu, test phải sau train
4. **Baseline**: Luôn so sánh với baseline để biết model có thực sự tốt không
5. **Multiple Metrics**: Dùng nhiều metrics để đánh giá toàn diện

---

## 📚 Tài liệu Tham khảo

- **Brier Score**: https://en.wikipedia.org/wiki/Brier_score
- **Precision/Recall**: https://en.wikipedia.org/wiki/Precision_and_recall
- **Calibration**: https://scikit-learn.org/stable/modules/calibration.html

