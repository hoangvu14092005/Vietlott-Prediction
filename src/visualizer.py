# src/visualizer.py
# Module hiển thị xác suất dự đoán một cách trực quan

import numpy as np
from typing import Dict, List, Tuple

class ProbabilityVisualizer:
    """Class để hiển thị và phân tích xác suất dự đoán"""
    
    @staticmethod
    def print_probability_table(probabilities: np.ndarray, top_n: int = None):
        """
        In bảng xác suất cho tất cả 45 số
        
        Args:
            probabilities: Array 45 phần tử chứa xác suất cho số 1-45
            top_n: Nếu không None, chỉ hiển thị top N số
        """
        sorted_indices = np.argsort(probabilities)[::-1]
        
        if top_n:
            sorted_indices = sorted_indices[:top_n]
        
        print("\n" + "="*80)
        print("📊 BẢNG XÁC SUẤT DỰ ĐOÁN")
        print("="*80)
        print(f"{'STT':<6} {'Số':<6} {'Xác suất':<15} {'%':<10} {'Mức độ':<15} {'Biểu đồ':<20}")
        print("-"*80)
        
        for rank, idx in enumerate(sorted_indices, 1):
            num = idx + 1
            prob = probabilities[idx]
            percentage = prob * 100
            level = ProbabilityVisualizer._get_probability_level(prob)
            bar = "█" * int(prob * 30)  # Bar chart
            
            print(f"{rank:<6} {num:<6} {prob:<15.6f} {percentage:<10.2f} {level:<15} {bar}")
        
        print("="*80)
    
    @staticmethod
    def print_probability_summary(probabilities: np.ndarray):
        """
        In tóm tắt phân phối xác suất
        
        Args:
            probabilities: Array 45 phần tử chứa xác suất
        """
        sorted_probs = np.sort(probabilities)[::-1]
        
        print("\n" + "="*60)
        print("📈 TÓM TẮT PHÂN PHỐI XÁC SUẤT")
        print("="*60)
        print(f"   Tổng xác suất: {np.sum(probabilities):.4f}")
        print(f"   Xác suất trung bình: {np.mean(probabilities):.4f}")
        print(f"   Xác suất cao nhất: {sorted_probs[0]:.4f} ({sorted_probs[0]*100:.2f}%)")
        print(f"   Xác suất thấp nhất: {sorted_probs[-1]:.4f} ({sorted_probs[-1]*100:.2f}%)")
        print(f"   Độ lệch chuẩn: {np.std(probabilities):.4f}")
        print(f"   Số có xác suất > 0.10: {np.sum(probabilities > 0.10)}")
        print(f"   Số có xác suất > 0.12: {np.sum(probabilities > 0.12)}")
        print(f"   Số có xác suất > 0.15: {np.sum(probabilities > 0.15)}")
        print("="*60)
    
    @staticmethod
    def get_top_numbers(probabilities: np.ndarray, k: int = 8) -> List[Tuple[int, float]]:
        """
        Lấy top K số có xác suất cao nhất
        
        Args:
            probabilities: Array 45 phần tử
            k: Số lượng số cần lấy
            
        Returns:
            List of tuples (số, xác_suất) sắp xếp giảm dần
        """
        sorted_indices = np.argsort(probabilities)[::-1][:k]
        return [(idx + 1, float(probabilities[idx])) for idx in sorted_indices]
    
    @staticmethod
    def get_probability_by_zones(probabilities: np.ndarray) -> Dict[str, float]:
        """
        Tính tổng xác suất theo các vùng số
        
        Args:
            probabilities: Array 45 phần tử
            
        Returns:
            Dict với key là tên vùng và value là tổng xác suất
        """
        zone1 = np.sum(probabilities[0:15])   # Số 1-15
        zone2 = np.sum(probabilities[15:30])  # Số 16-30
        zone3 = np.sum(probabilities[30:45])  # Số 31-45
        
        return {
            "Vùng 1 (1-15)": zone1,
            "Vùng 2 (16-30)": zone2,
            "Vùng 3 (31-45)": zone3
        }
    
    @staticmethod
    def get_probability_by_parity(probabilities: np.ndarray) -> Dict[str, float]:
        """
        Tính tổng xác suất theo số chẵn/lẻ
        
        Args:
            probabilities: Array 45 phần tử
            
        Returns:
            Dict với tổng xác suất số chẵn và lẻ
        """
        even_probs = []
        odd_probs = []
        
        for i, prob in enumerate(probabilities):
            num = i + 1
            if num % 2 == 0:
                even_probs.append(prob)
            else:
                odd_probs.append(prob)
        
        return {
            "Số chẵn": np.sum(even_probs),
            "Số lẻ": np.sum(odd_probs)
        }
    
    @staticmethod
    def print_zone_analysis(probabilities: np.ndarray):
        """In phân tích theo vùng và chẵn/lẻ"""
        zones = ProbabilityVisualizer.get_probability_by_zones(probabilities)
        parity = ProbabilityVisualizer.get_probability_by_parity(probabilities)
        
        print("\n" + "="*60)
        print("🗺️  PHÂN TÍCH THEO VÙNG VÀ CHẴN/LẺ")
        print("="*60)
        print("\n📍 Theo vùng:")
        for zone_name, total_prob in zones.items():
            print(f"   {zone_name}: {total_prob:.4f} ({total_prob*100:.2f}%)")
        
        print("\n🔢 Theo chẵn/lẻ:")
        for parity_name, total_prob in parity.items():
            print(f"   {parity_name}: {total_prob:.4f} ({total_prob*100:.2f}%)")
        print("="*60)
    
    @staticmethod
    def export_probabilities_to_dict(probabilities: np.ndarray) -> Dict[int, float]:
        """
        Xuất xác suất ra dictionary để dễ sử dụng
        
        Args:
            probabilities: Array 45 phần tử
            
        Returns:
            Dict {số: xác_suất}
        """
        return {num: float(probabilities[num-1]) for num in range(1, 46)}
    
    @staticmethod
    def _get_probability_level(prob: float) -> str:
        """Phân loại mức độ xác suất"""
        if prob >= 0.15:
            return "🔥 Rất cao"
        elif prob >= 0.12:
            return "⭐ Cao"
        elif prob >= 0.10:
            return "✓ Trung bình"
        elif prob >= 0.08:
            return "○ Thấp"
        else:
            return "✗ Rất thấp"

