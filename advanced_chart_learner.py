import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from chart_storage import Chart

class AdvancedChartLearner:
    def __init__(self):
        self.chart_pattern_memory = {}
        self.successful_patterns = deque(maxlen=1000)
        self.failed_patterns = deque(maxlen=1000)
        self.pattern_confidence_history = {}
        self.chart_shape_memory = {}
        self.regime_aware_patterns = {}
        
    def learn_from_chart_outcome(self, chart_data: List[float], 
                                 predicted_profitable: bool,
                                 was_profitable: bool,
                                 confidence: float,
                                 market_regime: str,
                                 indicators: Dict,
                                 patterns: Dict):
        chart_signature = self._generate_chart_signature(chart_data)
        
        if chart_signature not in self.chart_pattern_memory:
            self.chart_pattern_memory[chart_signature] = {
                'total_occurrences': 0,
                'successful': 0,
                'failed': 0,
                'avg_confidence': 0.0,
                'regimes': {},
                'pattern_indicators': {}
            }
        
        memory = self.chart_pattern_memory[chart_signature]
        memory['total_occurrences'] += 1
        
        if predicted_profitable == was_profitable:
            memory['successful'] += 1
            self.successful_patterns.append({
                'chart': chart_data,
                'confidence': confidence,
                'regime': market_regime,
                'indicators': indicators,
                'patterns': patterns
            })
        else:
            memory['failed'] += 1
            self.failed_patterns.append({
                'chart': chart_data,
                'confidence': confidence,
                'regime': market_regime,
                'indicators': indicators,
                'patterns': patterns
            })
        
        memory['avg_confidence'] = (
            (memory['avg_confidence'] * (memory['total_occurrences'] - 1) + confidence) /
            memory['total_occurrences']
        )
        
        if market_regime not in memory['regimes']:
            memory['regimes'][market_regime] = {'total': 0, 'successful': 0}
        memory['regimes'][market_regime]['total'] += 1
        if predicted_profitable == was_profitable:
            memory['regimes'][market_regime]['successful'] += 1
        
        key_patterns = self._extract_key_patterns(patterns, indicators)
        for pattern_key, pattern_value in key_patterns.items():
            if pattern_key not in memory['pattern_indicators']:
                memory['pattern_indicators'][pattern_key] = {'total': 0, 'successful': 0}
            memory['pattern_indicators'][pattern_key]['total'] += 1
            if predicted_profitable == was_profitable:
                memory['pattern_indicators'][pattern_key]['successful'] += 1
    
    def get_pattern_confidence_boost(self, chart_data: List[float], 
                                    market_regime: str,
                                    patterns: Dict,
                                    indicators: Dict) -> float:
        chart_signature = self._generate_chart_signature(chart_data)
        
        if chart_signature not in self.chart_pattern_memory:
            return 0.0
        
        memory = self.chart_pattern_memory[chart_signature]
        
        if memory['total_occurrences'] < 3:
            return 0.0
        
        success_rate = memory['successful'] / memory['total_occurrences']
        
        if success_rate > 0.7:
            boost = (success_rate - 0.5) * 20.0
        elif success_rate < 0.3:
            boost = (success_rate - 0.5) * 15.0
        else:
            boost = 0.0
        
        regime_key = f"{chart_signature}_{market_regime}"
        if regime_key in self.regime_aware_patterns:
            regime_memory = self.regime_aware_patterns[regime_key]
            if regime_memory['total'] > 2:
                regime_success_rate = regime_memory['successful'] / regime_memory['total']
                if regime_success_rate > 0.75:
                    boost += 5.0
                elif regime_success_rate < 0.25:
                    boost -= 5.0
        
        key_patterns = self._extract_key_patterns(patterns, indicators)
        for pattern_key, pattern_value in key_patterns.items():
            if pattern_key in memory['pattern_indicators']:
                pattern_stats = memory['pattern_indicators'][pattern_key]
                if pattern_stats['total'] > 2:
                    pattern_success_rate = pattern_stats['successful'] / pattern_stats['total']
                    if pattern_success_rate > 0.7:
                        boost += 2.0
                    elif pattern_success_rate < 0.3:
                        boost -= 2.0
        
        return boost
    
    def find_similar_successful_charts(self, chart_data: List[float], 
                                      top_n: int = 5) -> List[Dict]:
        chart_signature = self._generate_chart_signature(chart_data)
        
        similar_charts = []
        for pattern in self.successful_patterns:
            similarity = self._calculate_shape_similarity(chart_data, pattern['chart'])
            if similarity > 0.7:
                similar_charts.append({
                    'chart': pattern['chart'],
                    'similarity': similarity,
                    'confidence': pattern['confidence'],
                    'regime': pattern['regime'],
                    'indicators': pattern['indicators'],
                    'patterns': pattern['patterns']
                })
        
        similar_charts.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_charts[:top_n]
    
    def _generate_chart_signature(self, chart_data: List[float]) -> str:
        if len(chart_data) < 5:
            return "short"
        
        prices = np.array(chart_data)
        normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        
        features = [
            np.mean(normalized),
            np.std(normalized),
            (normalized[-1] - normalized[0]),
            len([i for i in range(1, len(normalized)) if normalized[i] > normalized[i-1]]),
            len([i for i in range(1, len(normalized)) if normalized[i] < normalized[i-1]])
        ]
        
        signature = "_".join([f"{f:.3f}" for f in features])
        return signature
    
    def _calculate_shape_similarity(self, chart1: List[float], chart2: List[float]) -> float:
        if len(chart1) < 3 or len(chart2) < 3:
            return 0.0
        
        min_len = min(len(chart1), len(chart2))
        arr1 = np.array(chart1[:min_len])
        arr2 = np.array(chart2[:min_len])
        
        arr1_norm = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-10)
        arr2_norm = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-10)
        
        std1 = np.std(arr1_norm)
        std2 = np.std(arr2_norm)
        
        if std1 < 1e-10 or std2 < 1e-10:
            correlation = 1.0 if np.allclose(arr1_norm, arr2_norm) else 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(arr1_norm, arr2_norm)[0, 1]
                if np.isnan(correlation) or np.isinf(correlation):
                    correlation = 0.0
        
        euclidean = np.sqrt(np.mean((arr1_norm - arr2_norm) ** 2))
        similarity = (max(0.0, correlation) + (1.0 / (1.0 + euclidean))) / 2.0
        
        return float(similarity)
    
    def _extract_key_patterns(self, patterns: Dict, indicators: Dict) -> Dict:
        key_patterns = {}
        
        double_pattern = patterns.get('double_pattern', (0.0, 'none'))
        if double_pattern[1] != 'none':
            key_patterns[f"double_{double_pattern[1]}"] = double_pattern[0]
        
        triangle = patterns.get('triangle', (0.0, 'none'))
        if triangle[1] != 'none':
            key_patterns[f"triangle_{triangle[1]}"] = triangle[0]
        
        trend = indicators.get('trend', 'neutral')
        if trend != 'neutral':
            key_patterns[f"trend_{trend}"] = 1.0
        
        rsi = indicators.get('rsi', 50.0)
        if rsi < 30:
            key_patterns['rsi_oversold'] = 1.0
        elif rsi > 70:
            key_patterns['rsi_overbought'] = 1.0
        
        macd = indicators.get('macd', {})
        if isinstance(macd, dict):
            histogram = macd.get('histogram', 0)
            if histogram > 0:
                key_patterns['macd_bullish'] = abs(histogram)
            elif histogram < 0:
                key_patterns['macd_bearish'] = abs(histogram)
        
        return key_patterns

