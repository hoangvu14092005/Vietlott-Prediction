# src/features.py
# ULTRA FEATURE ENGINE V7 - ENHANCED WITH PAIR/TRIPLET, HOT/COLD, ZONE ANALYSIS

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy
from collections import Counter, defaultdict

class UltraFeatureEngine:
    def __init__(self, past_window=100, feature_size=600):
        self.past_window = past_window
        self.feature_size = feature_size 

    def _get_row_numbers(self, row):
        return row[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values.astype(int)

    def _calculate_fft_magnitude(self, sequence):
        """
        Dùng biến đổi Fourier để tìm chu kỳ ẩn trong chuỗi xuất hiện của số.
        Input: Chuỗi nhị phân (0, 0, 1, 0, 1...) của một số qua 100 kỳ.
        Output: Cường độ tín hiệu mạnh nhất (Chu kỳ lặp).
        """
        # Nếu chuỗi quá ngắn hoặc toàn 0
        if len(sequence) < 10 or np.sum(sequence) == 0:
            return 0.0
        
        # Biến đổi Fourier
        f_transform = fft(sequence)
        # Lấy độ lớn (Magnitude) của các tần số, bỏ qua tần số 0 (DC component)
        magnitudes = np.abs(f_transform)[1:len(sequence)//2]
        
        if len(magnitudes) == 0: return 0.0
        return np.max(magnitudes) # Trả về tín hiệu sóng mạnh nhất

    def _calculate_pair_frequency(self, window_data):
        """Tính tần suất xuất hiện của các cặp số"""
        pair_freq = defaultdict(int)
        for row in window_data:
            row_sorted = sorted(row)
            for i in range(len(row_sorted)):
                for j in range(i+1, len(row_sorted)):
                    pair = (row_sorted[i], row_sorted[j])
                    pair_freq[pair] += 1
        return pair_freq

    def _calculate_triplet_frequency(self, window_data):
        """Tính tần suất xuất hiện của các bộ 3 số"""
        triplet_freq = defaultdict(int)
        for row in window_data:
            row_sorted = sorted(row)
            for i in range(len(row_sorted)):
                for j in range(i+1, len(row_sorted)):
                    for k in range(j+1, len(row_sorted)):
                        triplet = (row_sorted[i], row_sorted[j], row_sorted[k])
                        triplet_freq[triplet] += 1
        return triplet_freq

    def _get_hot_cold_numbers(self, freq, window_size):
        """Phân loại số nóng (hot) và lạnh (cold)"""
        avg_freq = np.mean(freq)
        std_freq = np.std(freq)
        hot_threshold = avg_freq + 0.5 * std_freq
        cold_threshold = avg_freq - 0.5 * std_freq
        
        hot_numbers = np.where(freq >= hot_threshold)[0]
        cold_numbers = np.where(freq <= cold_threshold)[0]
        return hot_numbers, cold_numbers

    def _calculate_zone_distribution(self, numbers):
        """Chia 45 số thành 3 vùng: 1-15, 16-30, 31-45"""
        zone1 = sum(1 for n in numbers if 1 <= n <= 15)
        zone2 = sum(1 for n in numbers if 16 <= n <= 30)
        zone3 = sum(1 for n in numbers if 31 <= n <= 45)
        return [zone1, zone2, zone3]

    def create_single_feature(self, df, current_idx):
        start = max(0, current_idx - self.past_window)
        end = current_idx
        
        if end <= start:
            return np.zeros(self.feature_size, dtype=np.float32)

        window_df = df.iloc[start:end]
        # Chuyển về numpy array để tính toán cho nhanh
        window_data = window_df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
        
        feature = []

        # --- PHẦN 1: TẦN SUẤT & GAN (Cơ bản nhưng quan trọng) ---
        freq = np.zeros(45)
        last_seen = np.full(45, 100.0)
        
        # Ma trận xuất hiện: 100 dòng x 45 cột (1 nếu số xuất hiện, 0 nếu không)
        # Dùng để tính toán FFT và Correlation
        appearance_matrix = np.zeros((len(window_data), 45))

        for i, row in enumerate(window_data):
            # Tính Tần suất
            for num in row:
                freq[num-1] += 1
                appearance_matrix[i, num-1] = 1
                last_seen[num-1] = 0 # Reset gan khi thấy số
            
            # Tăng gan cho các số không xuất hiện
            last_seen += 1 # Cộng 1 cho tất cả, dòng trên đã reset về 0 những số xuất hiện
        
        # Chuẩn hóa
        feature.extend(freq / len(window_data))      # 45 features
        feature.extend(last_seen / 100.0)            # 45 features

        # --- PHẦN 2: FFT SIGNALS (SÓNG NGẦM - NÂNG CAO) ---
        # Phân tích chu kỳ sóng của từng số
        fft_signals = []
        for num_idx in range(45):
            # Lấy chuỗi lịch sử của số (ví dụ số 1: [0, 1, 0, 0, 1...])
            seq = appearance_matrix[:, num_idx]
            signal_strength = self._calculate_fft_magnitude(seq)
            fft_signals.append(signal_strength)
        
        # Chuẩn hóa FFT vector
        fft_signals = np.array(fft_signals)
        if fft_signals.max() > 0:
            fft_signals = fft_signals / fft_signals.max()
        feature.extend(fft_signals) # 45 features

        # --- PHẦN 3: DELTA & SKEWNESS (Cấu trúc bộ số) ---
        # Phân tích kỳ quay gần nhất có cấu trúc thế nào
        if len(window_data) > 0:
            last_draw = np.sort(window_data[-1])
            # Delta: Khoảng cách giữa các số (Ví dụ: 05, 12, 18 -> Delta: 7, 6...)
            deltas = np.diff(last_draw) 
            # Thêm trung bình delta và độ lệch chuẩn delta
            feature.append(np.mean(deltas) / 45.0) 
            feature.append(np.std(deltas) / 20.0)
            
            # Skewness: Độ lệch của phân phối số trong kỳ quay
            feature.append(skew(last_draw))
        else:
            feature.extend([0, 0, 0])

        # --- PHẦN 4: POISSON PROBABILITY (Xác suất kỳ vọng) ---
        # Theo lý thuyết, xác suất ra 1 số là 6/45 = 0.133
        # Nếu 1 số ra quá nhiều (>0.2) hoặc quá ít (<0.05), nó mất cân bằng
        poisson_vec = []
        expected_prob = 6/45
        for f in freq:
            actual_prob = f / len(window_data)
            # Độ lệch so với lý thuyết
            diff = actual_prob - expected_prob 
            poisson_vec.append(diff)
        
        feature.extend(poisson_vec) # 45 features

        # --- PHẦN 5: HOT/COLD NUMBERS (Số nóng/lạnh) ---
        hot_nums, cold_nums = self._get_hot_cold_numbers(freq, len(window_data))
        hot_vector = np.zeros(45)
        cold_vector = np.zeros(45)
        hot_vector[hot_nums] = 1.0
        cold_vector[cold_nums] = 1.0
        feature.extend(hot_vector)  # 45 features
        feature.extend(cold_vector) # 45 features

        # --- PHẦN 6: PAIR FREQUENCY (Tần suất cặp số) ---
        # Tính tần suất các cặp số phổ biến nhất
        pair_freq = self._calculate_pair_frequency(window_data)
        if pair_freq:
            top_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])[:20]
            pair_features = np.zeros(45)
            for (a, b), count in top_pairs:
                pair_features[a-1] += count / len(window_data)
                pair_features[b-1] += count / len(window_data)
            if pair_features.max() > 0:
                pair_features = pair_features / pair_features.max()
        else:
            pair_features = np.zeros(45)
        feature.extend(pair_features) # 45 features

        # --- PHẦN 7: GAP ANALYSIS (Phân tích khoảng cách) ---
        # Tính khoảng cách trung bình giữa các số trong các kỳ quay gần đây
        gap_features = np.zeros(45)
        recent_window = window_data[-min(10, len(window_data)):]
        for row in recent_window:
            row_sorted = np.sort(row)
            gaps = np.diff(row_sorted)
            avg_gap = np.mean(gaps) if len(gaps) > 0 else 0
            for num in row:
                # Số có khoảng cách nhỏ với số khác có thể có pattern
                gap_features[num-1] += avg_gap / 45.0
        if gap_features.max() > 0:
            gap_features = gap_features / gap_features.max()
        feature.extend(gap_features) # 45 features

        # --- PHẦN 8: SUM & STATISTICS (Tổng và thống kê) ---
        if len(window_data) > 0:
            recent_sums = [np.sum(row) for row in window_data[-10:]]
            avg_sum = np.mean(recent_sums) / (6 * 45)  # Chuẩn hóa
            std_sum = np.std(recent_sums) / (6 * 45)
            feature.append(avg_sum)
            feature.append(std_sum)
            
            # Tỷ lệ số chẵn/lẻ trong kỳ gần nhất
            last_draw = window_data[-1]
            even_count = sum(1 for n in last_draw if n % 2 == 0)
            odd_count = 6 - even_count
            feature.append(even_count / 6.0)
            feature.append(odd_count / 6.0)
            
            # Số liên tiếp trong kỳ gần nhất
            last_sorted = np.sort(last_draw)
            consecutive = 0
            for i in range(len(last_sorted)-1):
                if last_sorted[i+1] - last_sorted[i] == 1:
                    consecutive += 1
            feature.append(consecutive / 6.0)
        else:
            feature.extend([0, 0, 0, 0, 0])

        # --- PHẦN 9: ZONE DISTRIBUTION (Phân bố vùng) ---
        if len(window_data) > 0:
            recent_zones = []
            for row in window_data[-10:]:
                zones = self._calculate_zone_distribution(row)
                recent_zones.append(zones)
            avg_zones = np.mean(recent_zones, axis=0) / 6.0
            feature.extend(avg_zones.tolist()) # 3 features
        else:
            feature.extend([0, 0, 0])

        # --- PHẦN 10: TREND ANALYSIS (Phân tích xu hướng) ---
        # Xu hướng tần suất của mỗi số trong 20 kỳ gần nhất
        trend_window = min(20, len(window_data))
        if trend_window > 5:
            trend_features = np.zeros(45)
            for num_idx in range(45):
                recent_freq = []
                for i in range(max(0, len(window_data)-trend_window), len(window_data)):
                    if num_idx+1 in window_data[i]:
                        recent_freq.append(1)
                    else:
                        recent_freq.append(0)
                if len(recent_freq) > 1:
                    # Tính xu hướng (tăng/giảm)
                    trend = (recent_freq[-1] - recent_freq[0]) / len(recent_freq)
                    trend_features[num_idx] = trend
            feature.extend(trend_features) # 45 features
        else:
            feature.extend([0] * 45)

        # --- PHẦN 11: CORRELATION MATRIX (Ma trận tương quan) ---
        # Tính tương quan giữa các số (số nào thường xuất hiện cùng nhau)
        if len(window_data) > 10:
            corr_features = np.zeros(45)
            for num_idx in range(45):
                num = num_idx + 1
                cooccurrence = 0
                for row in window_data[-20:]:
                    if num in row:
                        # Đếm số lần các số khác xuất hiện cùng số này
                        cooccurrence += len([x for x in row if x != num])
                corr_features[num_idx] = cooccurrence / (20 * 5)  # Chuẩn hóa
            if corr_features.max() > 0:
                corr_features = corr_features / corr_features.max()
            feature.extend(corr_features) # 45 features
        else:
            feature.extend([0] * 45)

        # --- PHẦN 12: ENTROPY & VARIANCE (Entropy và phương sai) ---
        # Entropy của phân phối tần suất
        freq_normalized = freq / (np.sum(freq) + 1e-10)
        freq_entropy = entropy(freq_normalized + 1e-10)
        feature.append(freq_entropy / np.log(45))  # Chuẩn hóa
        
        # Variance của tần suất
        freq_variance = np.var(freq) / (len(window_data) ** 2)
        feature.append(freq_variance)

        # --- PADDING ---
        current_len = len(feature)
        if current_len < self.feature_size:
            feature.extend([0] * (self.feature_size - current_len))
        
        return np.array(feature[:self.feature_size], dtype=np.float32)

    def prepare_training_dataset(self, df):
        X = []
        y = []
        n_rows = len(df)
        print(f"⚙️ [FeatureEngine V7] Đang tính toán {self.feature_size} features nâng cao...")
        
        for i in range(self.past_window + 1, n_rows):
            feat = self.create_single_feature(df, i)
            target = self._get_row_numbers(df.iloc[i])
            X.append(feat)
            y.append(target)
            
        return np.array(X), np.array(y)