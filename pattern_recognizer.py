import numpy as np
from typing import List, Dict, Tuple

class PatternRecognizer:
    @staticmethod
    def find_local_extrema(prices: List[float], window: int = 3) -> Tuple[List[int], List[int]]:
        peaks = []
        valleys = []
        
        for i in range(window, len(prices) - window):
            is_peak = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1) if j != i)
            is_valley = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1) if j != i)
            
            if is_peak:
                peaks.append(i)
            if is_valley:
                valleys.append(i)
        
        return peaks, valleys
    
    @staticmethod
    def detect_head_shoulders(prices: List[float]) -> float:
        if len(prices) < 7:
            return 0.0
        
        peaks, valleys = PatternRecognizer.find_local_extrema(prices, window=2)
        
        if len(peaks) < 3:
            return 0.0
        
        confidence = 0.0
        
        for i in range(len(peaks) - 2):
            left_shoulder = prices[peaks[i]]
            head = prices[peaks[i+1]]
            right_shoulder = prices[peaks[i+2]]
            
            if head > left_shoulder and head > right_shoulder:
                shoulder_similarity = 1.0 - abs(left_shoulder - right_shoulder) / (head + 1e-10)
                if shoulder_similarity > 0.7:
                    confidence = max(confidence, shoulder_similarity)
        
        return confidence
    
    @staticmethod
    def detect_double_top_bottom(prices: List[float]) -> Tuple[float, str]:
        if len(prices) < 6:
            return 0.0, 'none'
        
        peaks, valleys = PatternRecognizer.find_local_extrema(prices, window=2)
        
        double_top_conf = 0.0
        double_bottom_conf = 0.0
        
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = prices[peaks[i]]
                peak2 = prices[peaks[i+1]]
                
                similarity = 1.0 - abs(peak1 - peak2) / (max(peak1, peak2) + 1e-10)
                if similarity > 0.85:
                    double_top_conf = max(double_top_conf, similarity)
        
        if len(valleys) >= 2:
            for i in range(len(valleys) - 1):
                valley1 = prices[valleys[i]]
                valley2 = prices[valleys[i+1]]
                
                similarity = 1.0 - abs(valley1 - valley2) / (max(abs(valley1), abs(valley2)) + 1e-10)
                if similarity > 0.85:
                    double_bottom_conf = max(double_bottom_conf, similarity)
        
        if double_top_conf > double_bottom_conf and double_top_conf > 0.7:
            return double_top_conf, 'double_top'
        elif double_bottom_conf > 0.7:
            return double_bottom_conf, 'double_bottom'
        
        return 0.0, 'none'
    
    @staticmethod
    def detect_triangle_pattern(prices: List[float]) -> Tuple[float, str]:
        if len(prices) < 8:
            return 0.0, 'none'
        
        peaks, valleys = PatternRecognizer.find_local_extrema(prices)
        
        if len(peaks) < 2 or len(valleys) < 2:
            return 0.0, 'none'
        
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
        recent_valleys = valleys[-3:] if len(valleys) >= 3 else valleys
        
        if len(recent_peaks) < 2 or len(recent_valleys) < 2:
            return 0.0, 'none'
        
        peak_prices = [prices[p] for p in recent_peaks]
        valley_prices = [prices[v] for v in recent_valleys]
        
        peak_trend = np.polyfit(range(len(peak_prices)), peak_prices, 1)[0]
        valley_trend = np.polyfit(range(len(valley_prices)), valley_prices, 1)[0]
        
        if peak_trend < 0 and valley_trend > 0:
            confidence = min(abs(peak_trend) / 0.001, abs(valley_trend) / 0.001) / 100.0
            return min(1.0, confidence), 'ascending_triangle'
        elif peak_trend > 0 and valley_trend < 0:
            confidence = min(abs(peak_trend) / 0.001, abs(valley_trend) / 0.001) / 100.0
            return min(1.0, confidence), 'descending_triangle'
        elif abs(peak_trend) < 0.0001 and abs(valley_trend) < 0.0001:
            return 0.6, 'symmetrical_triangle'
        
        return 0.0, 'none'
    
    @staticmethod
    def detect_candlestick_patterns(prices: List[float]) -> Dict:
        if len(prices) < 3:
            return {}
        
        patterns = {}
        recent = prices[-3:]
        
        if recent[1] > recent[0] and recent[1] > recent[2]:
            patterns['hanging_man'] = 0.6
        
        if recent[1] < recent[0] and recent[1] < recent[2]:
            patterns['hammer'] = 0.6
        
        if recent[0] < recent[1] < recent[2]:
            patterns['three_white_soldiers'] = 0.7
        
        if recent[0] > recent[1] > recent[2]:
            patterns['three_black_crows'] = 0.7
        
        return patterns
    
    @staticmethod
    def extract_all_patterns(prices: List[float]) -> Dict:
        patterns = {
            'head_shoulders': PatternRecognizer.detect_head_shoulders(prices),
            'double_pattern': PatternRecognizer.detect_double_top_bottom(prices),
            'triangle': PatternRecognizer.detect_triangle_pattern(prices),
            'candlestick': PatternRecognizer.detect_candlestick_patterns(prices)
        }
        
        return patterns

