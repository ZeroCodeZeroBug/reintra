import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Dict
from chart_storage import Chart
from technical_indicators import TechnicalIndicators

class SimilarityMatcher:
    def __init__(self):
        self.cache = {}
        self.indicators = TechnicalIndicators()
        self.cache_size_limit = 5000
    
    def calculate_trend(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        price_array = np.array(prices)
        x = np.arange(len(price_array))
        slope = np.polyfit(x, price_array, 1)[0]
        return slope / price_array[0] if price_array[0] != 0 else 0.0
    
    def calculate_momentum(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
    
    def calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns)) if len(returns) > 0 else 0.0
    
    def extract_shape_features(self, prices: List[float]) -> Dict:
        if len(prices) < 5:
            return {}
        
        prices_arr = np.array(prices)
        normalized = (prices_arr - prices_arr.min()) / (prices_arr.max() - prices_arr.min() + 1e-10)
        
        peaks, _ = signal.find_peaks(normalized, prominence=0.1)
        valleys, _ = signal.find_peaks(-normalized, prominence=0.1)
        
        peak_count = len(peaks)
        valley_count = len(valleys)
        
        if len(normalized) > 1:
            first_derivative = np.diff(normalized)
            second_derivative = np.diff(first_derivative) if len(first_derivative) > 1 else np.array([0])
            
            curvature = np.abs(second_derivative) if len(second_derivative) > 0 else np.array([0])
            avg_curvature = float(np.mean(curvature)) if len(curvature) > 0 else 0.0
        else:
            avg_curvature = 0.0
        
        price_range = prices_arr.max() - prices_arr.min()
        if price_range > 0:
            relative_volatility = np.std(prices_arr) / price_range
        else:
            relative_volatility = 0.0
        
        return {
            'peak_count': peak_count,
            'valley_count': valley_count,
            'curvature': avg_curvature,
            'relative_volatility': relative_volatility,
            'price_range': float(price_range),
            'normalized_shape': normalized.tolist()[:20]
        }
    
    def calculate_fourier_similarity(self, chart1: List[float], chart2: List[float]) -> float:
        if len(chart1) < 4 or len(chart2) < 4:
            return 0.5
        
        min_len = min(len(chart1), len(chart2))
        chart1_arr = np.array(chart1[:min_len])
        chart2_arr = np.array(chart2[:min_len])
        
        chart1_normalized = (chart1_arr - chart1_arr.min()) / (chart1_arr.max() - chart1_arr.min() + 1e-10)
        chart2_normalized = (chart2_arr - chart2_arr.min()) / (chart2_arr.max() - chart2_arr.min() + 1e-10)
        
        fft1 = np.abs(np.fft.fft(chart1_normalized)[:min_len//2])
        fft2 = np.abs(np.fft.fft(chart2_normalized)[:min_len//2])
        
        if len(fft1) == 0 or len(fft2) == 0:
            return 0.5
        
        min_fft_len = min(len(fft1), len(fft2))
        fft1 = fft1[:min_fft_len]
        fft2 = fft2[:min_fft_len]
        
        if np.sum(fft1) == 0 or np.sum(fft2) == 0:
            return 0.5
        
        fft1_normalized = fft1 / (np.sum(fft1) + 1e-10)
        fft2_normalized = fft2 / (np.sum(fft2) + 1e-10)
        
        norm1 = np.linalg.norm(fft1_normalized)
        norm2 = np.linalg.norm(fft2_normalized)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            cosine_sim = 1.0 if np.allclose(fft1_normalized, fft2_normalized) else 0.0
        else:
            cosine_sim = np.dot(fft1_normalized, fft2_normalized) / (norm1 * norm2)
            if np.isnan(cosine_sim) or np.isinf(cosine_sim):
                cosine_sim = 0.0
        
        return float(np.clip(cosine_sim, 0.0, 1.0))
    
    def calculate_wavelet_similarity(self, chart1: List[float], chart2: List[float]) -> float:
        if len(chart1) < 8 or len(chart2) < 8:
            return 0.5
        
        min_len = min(len(chart1), len(chart2))
        chart1_arr = np.array(chart1[:min_len])
        chart2_arr = np.array(chart2[:min_len])
        
        chart1_normalized = (chart1_arr - chart1_arr.mean()) / (chart1_arr.std() + 1e-10)
        chart2_normalized = (chart2_arr - chart2_arr.mean()) / (chart2_arr.std() + 1e-10)
        
        try:
            wavelet1 = signal.cwt(chart1_normalized, signal.ricker, [1, 2, 4]) if len(chart1_normalized) >= 8 else None
            wavelet2 = signal.cwt(chart2_normalized, signal.ricker, [1, 2, 4]) if len(chart2_normalized) >= 8 else None
            
            if wavelet1 is None or wavelet2 is None:
                return 0.5
            
            if wavelet1.shape != wavelet2.shape:
                return 0.5
            
            w1_flat = wavelet1.flatten()
            w2_flat = wavelet2.flatten()
            
            std1 = np.std(w1_flat)
            std2 = np.std(w2_flat)
            
            if std1 < 1e-10 or std2 < 1e-10:
                correlation = 1.0 if np.allclose(w1_flat, w2_flat) else 0.0
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    correlation = np.corrcoef(w1_flat, w2_flat)[0, 1]
                    if np.isnan(correlation) or np.isinf(correlation):
                        correlation = 0.0
            
            return float(np.clip(correlation, 0.0, 1.0))
        except:
            return 0.5
    
    def extract_features(self, chart: List[float]) -> Dict:
        base_features = {
            'trend_value': self.calculate_trend(chart),
            'momentum': self.calculate_momentum(chart),
            'volatility': self.calculate_volatility(chart),
            'mean': np.mean(chart),
            'std': np.std(chart)
        }
        
        technical_features = self.indicators.extract_all_indicators(chart)
        base_features.update(technical_features)
        base_features['trend_direction'] = technical_features.get('trend', 'neutral')
        
        shape_features = self.extract_shape_features(chart)
        base_features.update(shape_features)
        
        return base_features
    
    def calculate_similarity(self, chart1: List[float], chart2: List[float]) -> float:
        if not chart1 or not chart2:
            return 0.0
        
        cache_key = (id(chart1), id(chart2))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if len(self.cache) > self.cache_size_limit:
            keys_to_remove = list(self.cache.keys())[:self.cache_size_limit // 2]
            for key in keys_to_remove:
                del self.cache[key]
        
        if len(chart1) != len(chart2):
            min_len = min(len(chart1), len(chart2))
            chart1 = chart1[:min_len]
            chart2 = chart2[:min_len]
        
        chart1_arr = np.array(chart1)
        chart2_arr = np.array(chart2)
        
        if len(chart1_arr) == 0:
            return 0.0
        
        min1, max1 = chart1_arr.min(), chart1_arr.max()
        min2, max2 = chart2_arr.min(), chart2_arr.max()
        
        if max1 - min1 == 0 or max2 - min2 == 0:
            similarity = 1.0 if np.allclose(chart1_arr, chart2_arr) else 0.0
            self.cache[cache_key] = similarity
            return similarity
        
        chart1_normalized = (chart1_arr - min1) / (max1 - min1 + 1e-10)
        chart2_normalized = (chart2_arr - min2) / (max2 - min2 + 1e-10)
        
        std1 = np.std(chart1_normalized)
        std2 = np.std(chart2_normalized)
        
        if std1 < 1e-10 or std2 < 1e-10:
            correlation_sim = 1.0 if np.allclose(chart1_normalized, chart2_normalized) else 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(chart1_normalized, chart2_normalized)[0, 1]
                if np.isnan(correlation) or np.isinf(correlation):
                    correlation = 0.0
                correlation_sim = max(0.0, correlation)
        
        euclidean_diff = np.sqrt(np.mean((chart1_normalized - chart2_normalized) ** 2))
        euclidean_similarity = 1.0 / (1.0 + euclidean_diff)
        
        dtw_distance = self._dtw_distance(chart1_normalized, chart2_normalized)
        dtw_similarity = 1.0 / (1.0 + dtw_distance)
        
        fourier_sim = self.calculate_fourier_similarity(chart1, chart2)
        wavelet_sim = self.calculate_wavelet_similarity(chart1, chart2)
        
        features1 = self.extract_features(chart1)
        features2 = self.extract_features(chart2)
        
        trend_val1 = features1.get('trend_value', 0.0)
        trend_val2 = features2.get('trend_value', 0.0)
        if isinstance(trend_val1, (int, float)) and isinstance(trend_val2, (int, float)):
            trend_sim = 1.0 / (1.0 + abs(float(trend_val1) - float(trend_val2)))
        else:
            trend_sim = 0.5
        
        momentum1 = features1.get('momentum', 0.0)
        momentum2 = features2.get('momentum', 0.0)
        if isinstance(momentum1, (int, float)) and isinstance(momentum2, (int, float)):
            momentum_sim = 1.0 / (1.0 + abs(float(momentum1) - float(momentum2)))
        else:
            momentum_sim = 0.5
        
        volatility1 = features1.get('volatility', 0.0)
        volatility2 = features2.get('volatility', 0.0)
        if isinstance(volatility1, (int, float)) and isinstance(volatility2, (int, float)):
            volatility_sim = 1.0 / (1.0 + abs(float(volatility1) - float(volatility2)) * 10)
        else:
            volatility_sim = 0.5
        
        peak_count1 = features1.get('peak_count', 0)
        peak_count2 = features2.get('peak_count', 0)
        valley_count1 = features1.get('valley_count', 0)
        valley_count2 = features2.get('valley_count', 0)
        
        shape_sim = 0.0
        if isinstance(peak_count1, (int, float)) and isinstance(peak_count2, (int, float)):
            peak_diff = abs(float(peak_count1) - float(peak_count2))
            valley_diff = abs(float(valley_count1) - float(valley_count2))
            shape_sim = 1.0 / (1.0 + (peak_diff + valley_diff) * 0.3)
        
        curvature1 = features1.get('curvature', 0.0)
        curvature2 = features2.get('curvature', 0.0)
        if isinstance(curvature1, (int, float)) and isinstance(curvature2, (int, float)):
            curvature_sim = 1.0 / (1.0 + abs(float(curvature1) - float(curvature2)) * 10)
        else:
            curvature_sim = 0.5
        
        rsi1 = features1.get('rsi', 50.0)
        rsi2 = features2.get('rsi', 50.0)
        if isinstance(rsi1, (int, float)) and isinstance(rsi2, (int, float)):
            rsi_sim = 1.0 / (1.0 + abs(float(rsi1) - float(rsi2)) / 50.0)
        else:
            rsi_sim = 0.5
        
        trend_dir1 = features1.get('trend_direction', 'neutral')
        trend_dir2 = features2.get('trend_direction', 'neutral')
        trend_match = 1.0 if trend_dir1 == trend_dir2 else 0.5
        
        price_pos1 = features1.get('price_position', 0.5)
        price_pos2 = features2.get('price_position', 0.5)
        if isinstance(price_pos1, (int, float)) and isinstance(price_pos2, (int, float)):
            price_pos_sim = 1.0 / (1.0 + abs(float(price_pos1) - float(price_pos2)) * 2)
        else:
            price_pos_sim = 0.5
        
        macd_sim = 1.0
        if 'macd' in features1 and 'macd' in features2:
            macd1 = features1.get('macd', {})
            macd2 = features2.get('macd', {})
            if isinstance(macd1, dict) and isinstance(macd2, dict):
                histogram1 = macd1.get('histogram', 0)
                histogram2 = macd2.get('histogram', 0)
                if isinstance(histogram1, (int, float)) and isinstance(histogram2, (int, float)):
                    macd_diff = abs(float(histogram1) - float(histogram2))
                    macd_sim = 1.0 / (1.0 + macd_diff * 100)
                else:
                    macd_sim = 0.5
        
        combined_similarity = (
            correlation_sim * 0.20 +
            euclidean_similarity * 0.15 +
            dtw_similarity * 0.12 +
            fourier_sim * 0.10 +
            wavelet_sim * 0.08 +
            shape_sim * 0.08 +
            trend_sim * 0.07 +
            momentum_sim * 0.05 +
            volatility_sim * 0.04 +
            rsi_sim * 0.04 +
            trend_match * 0.03 +
            price_pos_sim * 0.02 +
            curvature_sim * 0.01 +
            macd_sim * 0.01
        )
        
        combined_similarity = max(0.0, min(1.0, combined_similarity))
        self.cache[cache_key] = combined_similarity
        return float(combined_similarity)
    
    def _dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return float(dtw_matrix[n, m] / max(n, m))
    
    def find_most_similar_charts(self, current_chart: List[float], charts: List[Chart], top_n: int = 5) -> List[Tuple[Chart, float, float]]:
        if not charts:
            return []
        
        scored_charts = []
        for chart in charts:
            similarity = self.calculate_similarity(current_chart, chart.chart_data)
            if similarity > 0.1:
                weighted_score = similarity * (1.0 + chart.rating * 0.5)
                scored_charts.append((chart, similarity, weighted_score))
        
        scored_charts.sort(key=lambda x: x[2], reverse=True)
        return scored_charts[:top_n]
    
    def find_most_similar_chart(self, current_chart: List[float], charts: List[Chart]) -> Optional[Tuple[Chart, float]]:
        top_matches = self.find_most_similar_charts(current_chart, charts, top_n=1)
        
        if top_matches:
            best_match, raw_similarity, _ = top_matches[0]
            return (best_match, raw_similarity)
        
        return None
    
    def calculate_ensemble_confidence(self, top_matches: List[Tuple[Chart, float, float]]) -> Tuple[bool, float]:
        if not top_matches:
            return True, 50.0
        
        profit_votes = 0.0
        total_weight = 0.0
        avg_similarity = 0.0
        
        for chart, similarity, weighted_score in top_matches:
            weight = similarity * chart.rating
            if chart.category == "profit":
                profit_votes += weight
            total_weight += weight
            avg_similarity += similarity
        
        avg_similarity /= len(top_matches)
        
        if total_weight == 0:
            return True, 50.0
        
        profit_probability = (profit_votes / total_weight) * 100.0
        confidence = avg_similarity * 100.0 * (1.0 + min(1.0, len(top_matches) / 5.0) * 0.2)
        
        predicted_profitable = profit_probability > 50.0
        confidence = max(30.0, min(95.0, confidence))
        
        return predicted_profitable, confidence

