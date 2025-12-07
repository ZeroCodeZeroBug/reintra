import numpy as np
from typing import List, Dict, Tuple

class TechnicalIndicators:
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return []
        return np.convolve(prices, np.ones(period)/period, mode='valid').tolist()
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return []
        prices_arr = np.array(prices)
        ema_values = []
        multiplier = 2.0 / (period + 1)
        ema = prices_arr[0]
        
        for price in prices_arr:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values[period-1:]
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        fast_ema = TechnicalIndicators.ema(prices, fast)
        slow_ema = TechnicalIndicators.ema(prices, slow)
        
        if len(fast_ema) == 0 or len(slow_ema) == 0:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        min_len = min(len(fast_ema), len(slow_ema))
        macd_line = np.array(fast_ema[-min_len:]) - np.array(slow_ema[-min_len:])
        
        if len(macd_line) < signal:
            return {'macd': float(macd_line[-1]) if len(macd_line) > 0 else 0, 'signal': 0, 'histogram': 0}
        
        signal_line = TechnicalIndicators.ema(macd_line.tolist(), signal)
        
        if len(signal_line) == 0:
            return {'macd': float(macd_line[-1]), 'signal': 0, 'histogram': 0}
        
        macd_val = macd_line[-1]
        signal_val = signal_line[-1]
        histogram = macd_val - signal_val
        
        return {
            'macd': float(macd_val),
            'signal': float(signal_val),
            'histogram': float(histogram)
        }
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Dict:
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}
        
        prices_arr = np.array(prices[-period:])
        middle = np.mean(prices_arr)
        std = np.std(prices_arr)
        
        return {
            'upper': float(middle + (std * std_dev)),
            'middle': float(middle),
            'lower': float(middle - (std * std_dev))
        }
    
    @staticmethod
    def support_resistance_levels(prices: List[float]) -> Dict:
        if len(prices) < 5:
            return {'support': prices[0] if prices else 0, 'resistance': prices[0] if prices else 0}
        
        prices_arr = np.array(prices)
        window = max(3, len(prices) // 4)
        
        local_minima = []
        local_maxima = []
        
        for i in range(window, len(prices_arr) - window):
            window_prices = prices_arr[i-window:i+window+1]
            if prices_arr[i] == window_prices.min():
                local_minima.append(prices_arr[i])
            if prices_arr[i] == window_prices.max():
                local_maxima.append(prices_arr[i])
        
        support = np.mean(local_minima) if local_minima else prices_arr.min()
        resistance = np.mean(local_maxima) if local_maxima else prices_arr.max()
        
        return {'support': float(support), 'resistance': float(resistance)}
    
    @staticmethod
    def detect_trend(prices: List[float]) -> str:
        if len(prices) < 3:
            return 'neutral'
        
        prices_arr = np.array(prices)
        
        if len(prices_arr) >= 10:
            short_avg = np.mean(prices_arr[-5:])
            long_avg = np.mean(prices_arr[-10:])
            
            if short_avg > long_avg * 1.01:
                return 'uptrend'
            elif short_avg < long_avg * 0.99:
                return 'downtrend'
        
        slope = np.polyfit(range(len(prices_arr)), prices_arr, 1)[0]
        if slope > 0:
            return 'uptrend'
        elif slope < 0:
            return 'downtrend'
        
        return 'neutral'
    
    @staticmethod
    def calculate_volume_profile(prices: List[float]) -> Dict:
        if len(prices) < 2:
            return {'high_volume_zones': [], 'low_volume_zones': []}
        
        prices_arr = np.array(prices)
        price_range = prices_arr.max() - prices_arr.min()
        
        if price_range == 0:
            return {'high_volume_zones': [], 'low_volume_zones': []}
        
        bins = 10
        hist, bin_edges = np.histogram(prices_arr, bins=bins)
        
        threshold_high = np.percentile(hist, 75)
        threshold_low = np.percentile(hist, 25)
        
        high_zones = [(bin_edges[i], bin_edges[i+1]) for i in range(len(hist)) if hist[i] > threshold_high]
        low_zones = [(bin_edges[i], bin_edges[i+1]) for i in range(len(hist)) if hist[i] < threshold_low]
        
        return {
            'high_volume_zones': [(float(low), float(high)) for low, high in high_zones],
            'low_volume_zones': [(float(low), float(high)) for low, high in low_zones]
        }
    
    @staticmethod
    def extract_all_indicators(prices: List[float]) -> Dict:
        if len(prices) < 5:
            return {}
        
        rsi_val = TechnicalIndicators.rsi(prices)
        macd_data = TechnicalIndicators.macd(prices)
        bb_data = TechnicalIndicators.bollinger_bands(prices)
        sr_data = TechnicalIndicators.support_resistance_levels(prices)
        trend = TechnicalIndicators.detect_trend(prices)
        volume_profile = TechnicalIndicators.calculate_volume_profile(prices)
        
        prices_arr = np.array(prices)
        current_price = prices_arr[-1]
        
        sma_5 = TechnicalIndicators.sma(prices, 5)
        sma_10 = TechnicalIndicators.sma(prices, 10) if len(prices) >= 10 else []
        
        return {
            'rsi': rsi_val,
            'macd': macd_data,
            'bollinger': bb_data,
            'support_resistance': sr_data,
            'trend': trend,
            'volume_profile': volume_profile,
            'current_price': float(current_price),
            'price_position': float((current_price - sr_data['support']) / (sr_data['resistance'] - sr_data['support'] + 1e-10)),
            'sma_5': float(sma_5[-1]) if sma_5 else current_price,
            'sma_10': float(sma_10[-1]) if sma_10 else current_price,
            'volatility': float(np.std(np.diff(prices_arr)) / (np.mean(prices_arr) + 1e-10)),
            'momentum': float((prices_arr[-1] - prices_arr[0]) / (prices_arr[0] + 1e-10)) if len(prices_arr) > 0 else 0.0
        }

