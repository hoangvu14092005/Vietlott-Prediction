# src/evaluator.py
# Module đánh giá toàn diện cho bài toán dự đoán Vietlott

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

class LotteryEvaluator:
    """
    Class đánh giá hiệu suất model với nhiều metrics khác nhau
    phù hợp cho bài toán dự đoán xổ số
    """
    
    def __init__(self, top_k: int = 8):
        """
        Args:
            top_k: Số lượng số được đề xuất (mặc định 8)
        """
        self.top_k = top_k
        self.results = {}
    
    def evaluate_hits(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá số lượng số trúng (Hits)
        
        Args:
            predictions: Array (n_samples, 45) - xác suất cho 45 số
            actuals: Array (n_samples, 45) - one-hot encoding của số thực tế
            
        Returns:
            Dict chứa các metrics về hits
        """
        n_samples = len(predictions)
        hits_per_sample = []
        hit_rates = []
        
        for i in range(n_samples):
            # Lấy top-K số có xác suất cao nhất
            top_indices = np.argsort(predictions[i])[-self.top_k:][::-1]
            pred_nums = set(top_indices)
            
            # Số thực tế
            actual_indices = np.where(actuals[i] == 1)[0]
            actual_nums = set(actual_indices)
            
            # Số trúng
            hits = len(pred_nums & actual_nums)
            hits_per_sample.append(hits)
            hit_rates.append(hits / self.top_k)
        
        hits_array = np.array(hits_per_sample)
        
        return {
            'mean_hits': np.mean(hits_array),
            'std_hits': np.std(hits_array),
            'median_hits': np.median(hits_array),
            'max_hits': np.max(hits_array),
            'min_hits': np.min(hits_array),
            'mean_hit_rate': np.mean(hit_rates),
            'hits_distribution': self._get_distribution(hits_array),
            'hits_per_sample': hits_per_sample
        }
    
    def evaluate_precision_recall(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá Precision, Recall, F1-score
        
        Args:
            predictions: Array (n_samples, 45) - xác suất
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict chứa precision, recall, f1
        """
        n_samples = len(predictions)
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(n_samples):
            # Top-K predictions (binary)
            top_indices = np.argsort(predictions[i])[-self.top_k:][::-1]
            pred_binary = np.zeros(45)
            pred_binary[top_indices] = 1
            
            # Actual (binary)
            actual_binary = actuals[i]
            
            # Tính metrics
            precision = precision_score(actual_binary, pred_binary, zero_division=0)
            recall = recall_score(actual_binary, pred_binary, zero_division=0)
            f1 = f1_score(actual_binary, pred_binary, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_f1': np.mean(f1_scores),
            'std_precision': np.std(precisions),
            'std_recall': np.std(recalls),
            'std_f1': np.std(f1_scores)
        }
    
    def evaluate_rank_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá dựa trên ranking (vị trí của số thực tế trong ranking)
        
        Args:
            predictions: Array (n_samples, 45) - xác suất
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict chứa các metrics về ranking
        """
        n_samples = len(predictions)
        ranks = []
        top_k_ranks = []
        
        for i in range(n_samples):
            # Sắp xếp theo xác suất giảm dần
            sorted_indices = np.argsort(predictions[i])[::-1]
            
            # Số thực tế
            actual_indices = np.where(actuals[i] == 1)[0]
            
            # Vị trí của mỗi số thực tế trong ranking
            sample_ranks = []
            for actual_idx in actual_indices:
                rank = np.where(sorted_indices == actual_idx)[0][0] + 1  # 1-indexed
                sample_ranks.append(rank)
                ranks.append(rank)
                
                # Chỉ tính nếu trong top-K
                if rank <= self.top_k:
                    top_k_ranks.append(rank)
        
        if len(ranks) == 0:
            return {
                'mean_rank': 0,
                'median_rank': 0,
                'mean_rank_in_topk': 0,
                'coverage_at_k': 0
            }
        
        ranks_array = np.array(ranks)
        coverage = len(top_k_ranks) / len(ranks) if len(ranks) > 0 else 0
        
        return {
            'mean_rank': np.mean(ranks_array),
            'median_rank': np.median(ranks_array),
            'std_rank': np.std(ranks_array),
            'mean_rank_in_topk': np.mean(top_k_ranks) if len(top_k_ranks) > 0 else 0,
            'coverage_at_k': coverage,  # Tỷ lệ số thực tế nằm trong top-K
            'min_rank': np.min(ranks_array),
            'max_rank': np.max(ranks_array)
        }
    
    def evaluate_probability_calibration(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá độ chính xác của xác suất (calibration)
        
        Args:
            predictions: Array (n_samples, 45) - xác suất
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict chứa các metrics về calibration
        """
        n_samples = len(predictions)
        
        # Chia xác suất thành các bins
        bins = np.linspace(0, 1, 11)  # 10 bins: 0-0.1, 0.1-0.2, ...
        bin_counts = np.zeros(len(bins) - 1)
        bin_actuals = np.zeros(len(bins) - 1)
        
        for i in range(n_samples):
            for j in range(45):
                prob = predictions[i, j]
                actual = actuals[i, j]
                
                # Tìm bin
                bin_idx = np.digitize(prob, bins) - 1
                bin_idx = max(0, min(bin_idx, len(bins) - 2))
                
                bin_counts[bin_idx] += 1
                bin_actuals[bin_idx] += actual
        
        # Tính empirical probability cho mỗi bin
        empirical_probs = []
        predicted_probs = []
        for i in range(len(bins) - 1):
            if bin_counts[i] > 0:
                empirical = bin_actuals[i] / bin_counts[i]
                predicted = (bins[i] + bins[i+1]) / 2
                empirical_probs.append(empirical)
                predicted_probs.append(predicted)
        
        # Tính calibration error (Brier score)
        brier_scores = []
        for i in range(n_samples):
            for j in range(45):
                prob = predictions[i, j]
                actual = actuals[i, j]
                brier = (prob - actual) ** 2
                brier_scores.append(brier)
        
        return {
            'mean_brier_score': np.mean(brier_scores),
            'calibration_error': np.mean(np.abs(np.array(empirical_probs) - np.array(predicted_probs))) if len(empirical_probs) > 0 else 0,
            'empirical_probs': empirical_probs,
            'predicted_probs': predicted_probs
        }
    
    def evaluate_coverage(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá coverage - bao nhiêu số thực tế được dự đoán
        
        Args:
            predictions: Array (n_samples, 45) - xác suất
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict chứa coverage metrics
        """
        n_samples = len(predictions)
        unique_actuals = set()
        unique_predicted = set()
        coverage_per_sample = []
        
        for i in range(n_samples):
            # Số thực tế
            actual_indices = np.where(actuals[i] == 1)[0]
            unique_actuals.update(actual_indices)
            
            # Top-K predicted
            top_indices = np.argsort(predictions[i])[-self.top_k:][::-1]
            unique_predicted.update(top_indices)
            
            # Coverage cho sample này
            coverage = len(set(actual_indices) & set(top_indices)) / len(actual_indices) if len(actual_indices) > 0 else 0
            coverage_per_sample.append(coverage)
        
        total_coverage = len(unique_actuals & unique_predicted) / len(unique_actuals) if len(unique_actuals) > 0 else 0
        
        return {
            'mean_coverage': np.mean(coverage_per_sample),
            'total_coverage': total_coverage,
            'unique_actuals_count': len(unique_actuals),
            'unique_predicted_count': len(unique_predicted),
            'overlap_count': len(unique_actuals & unique_predicted)
        }
    
    def compare_with_baseline(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        So sánh với baseline (random và frequency-based)
        
        Args:
            predictions: Array (n_samples, 45) - xác suất từ model
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict so sánh với baselines
        """
        n_samples = len(predictions)
        
        # Baseline 1: Random
        random_hits = []
        for i in range(n_samples):
            random_pred = np.random.choice(45, size=self.top_k, replace=False)
            actual_indices = np.where(actuals[i] == 1)[0]
            hits = len(set(random_pred) & set(actual_indices))
            random_hits.append(hits)
        
        # Baseline 2: Frequency-based (dựa trên tần suất trong training)
        # Tính tần suất từ actuals
        freq = np.sum(actuals, axis=0)
        freq_normalized = freq / (np.sum(freq) + 1e-10)
        
        freq_hits = []
        for i in range(n_samples):
            # Chọn top-K số có tần suất cao nhất
            top_freq_indices = np.argsort(freq_normalized)[-self.top_k:][::-1]
            actual_indices = np.where(actuals[i] == 1)[0]
            hits = len(set(top_freq_indices) & set(actual_indices))
            freq_hits.append(hits)
        
        # Model hits
        model_hits = []
        for i in range(n_samples):
            top_indices = np.argsort(predictions[i])[-self.top_k:][::-1]
            actual_indices = np.where(actuals[i] == 1)[0]
            hits = len(set(top_indices) & set(actual_indices))
            model_hits.append(hits)
        
        return {
            'model_mean_hits': np.mean(model_hits),
            'random_mean_hits': np.mean(random_hits),
            'frequency_mean_hits': np.mean(freq_hits),
            'improvement_over_random': np.mean(model_hits) - np.mean(random_hits),
            'improvement_over_frequency': np.mean(model_hits) - np.mean(freq_hits),
            'improvement_over_random_pct': ((np.mean(model_hits) - np.mean(random_hits)) / np.mean(random_hits) * 100) if np.mean(random_hits) > 0 else 0
        }
    
    def comprehensive_evaluate(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Đánh giá toàn diện với tất cả metrics
        
        Args:
            predictions: Array (n_samples, 45) - xác suất
            actuals: Array (n_samples, 45) - one-hot encoding
            
        Returns:
            Dict chứa tất cả metrics
        """
        results = {}
        
        print("📊 Đang đánh giá với nhiều metrics...")
        
        # 1. Hits metrics
        print("   ⮞ Đang tính Hits metrics...")
        results['hits'] = self.evaluate_hits(predictions, actuals)
        
        # 2. Precision/Recall/F1
        print("   ⮞ Đang tính Precision/Recall/F1...")
        results['precision_recall'] = self.evaluate_precision_recall(predictions, actuals)
        
        # 3. Rank metrics
        print("   ⮞ Đang tính Rank metrics...")
        results['rank'] = self.evaluate_rank_metrics(predictions, actuals)
        
        # 4. Coverage
        print("   ⮞ Đang tính Coverage...")
        results['coverage'] = self.evaluate_coverage(predictions, actuals)
        
        # 5. Probability calibration
        print("   ⮞ Đang tính Probability Calibration...")
        results['calibration'] = self.evaluate_probability_calibration(predictions, actuals)
        
        # 6. Baseline comparison
        print("   ⮞ Đang so sánh với Baseline...")
        results['baseline'] = self.compare_with_baseline(predictions, actuals)
        
        self.results = results
        return results
    
    def print_evaluation_report(self, results: Optional[Dict] = None):
        """
        In báo cáo đánh giá đẹp
        
        Args:
            results: Dict kết quả (nếu None thì dùng self.results)
        """
        if results is None:
            results = self.results
        
        if not results:
            print("⚠️ Chưa có kết quả đánh giá. Hãy chạy comprehensive_evaluate() trước.")
            return
        
        print("\n" + "="*80)
        print("📊 BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN")
        print("="*80)
        
        # 1. Hits Metrics
        hits = results.get('hits', {})
        print("\n🎯 1. HITS METRICS (Số trúng)")
        print("-" * 80)
        print(f"   Trung bình số trúng/kỳ:     {hits.get('mean_hits', 0):.3f} ± {hits.get('std_hits', 0):.3f}")
        print(f"   Median số trúng:            {hits.get('median_hits', 0):.2f}")
        print(f"   Min/Max:                     {hits.get('min_hits', 0)} / {hits.get('max_hits', 0)}")
        print(f"   Hit Rate (trúng/K):         {hits.get('mean_hit_rate', 0)*100:.2f}%")
        print(f"   Phân phối:                  {hits.get('hits_distribution', {})}")
        
        # 2. Precision/Recall/F1
        pr = results.get('precision_recall', {})
        print("\n📈 2. PRECISION/RECALL/F1")
        print("-" * 80)
        print(f"   Precision:                   {pr.get('mean_precision', 0):.4f} ± {pr.get('std_precision', 0):.4f}")
        print(f"   Recall:                      {pr.get('mean_recall', 0):.4f} ± {pr.get('std_recall', 0):.4f}")
        print(f"   F1-Score:                    {pr.get('mean_f1', 0):.4f} ± {pr.get('std_f1', 0):.4f}")
        
        # 3. Rank Metrics
        rank = results.get('rank', {})
        print("\n📊 3. RANK METRICS")
        print("-" * 80)
        print(f"   Mean Rank:                   {rank.get('mean_rank', 0):.2f}")
        print(f"   Median Rank:                 {rank.get('median_rank', 0):.2f}")
        print(f"   Mean Rank trong Top-{self.top_k}: {rank.get('mean_rank_in_topk', 0):.2f}")
        print(f"   Coverage tại Top-{self.top_k}:   {rank.get('coverage_at_k', 0)*100:.2f}%")
        
        # 4. Coverage
        coverage = results.get('coverage', {})
        print("\n🎯 4. COVERAGE")
        print("-" * 80)
        print(f"   Mean Coverage:               {coverage.get('mean_coverage', 0)*100:.2f}%")
        print(f"   Total Coverage:              {coverage.get('total_coverage', 0)*100:.2f}%")
        print(f"   Số thực tế unique:           {coverage.get('unique_actuals_count', 0)}")
        print(f"   Số dự đoán unique:           {coverage.get('unique_predicted_count', 0)}")
        print(f"   Overlap:                      {coverage.get('overlap_count', 0)}")
        
        # 5. Calibration
        cal = results.get('calibration', {})
        print("\n📉 5. PROBABILITY CALIBRATION")
        print("-" * 80)
        print(f"   Mean Brier Score:            {cal.get('mean_brier_score', 0):.4f}")
        print(f"   Calibration Error:           {cal.get('calibration_error', 0):.4f}")
        
        # 6. Baseline Comparison
        baseline = results.get('baseline', {})
        print("\n⚖️  6. BASELINE COMPARISON")
        print("-" * 80)
        print(f"   Model Hits:                  {baseline.get('model_mean_hits', 0):.3f}")
        print(f"   Random Baseline:             {baseline.get('random_mean_hits', 0):.3f}")
        print(f"   Frequency Baseline:          {baseline.get('frequency_mean_hits', 0):.3f}")
        print(f"   Cải thiện vs Random:         {baseline.get('improvement_over_random', 0):.3f} ({baseline.get('improvement_over_random_pct', 0):.2f}%)")
        print(f"   Cải thiện vs Frequency:       {baseline.get('improvement_over_frequency', 0):.3f}")
        
        print("="*80)
    
    def _get_distribution(self, array: np.ndarray) -> Dict:
        """Tính phân phối của array"""
        unique, counts = np.unique(array, return_counts=True)
        total = len(array)
        return {int(k): int(v) for k, v in zip(unique, counts)}

