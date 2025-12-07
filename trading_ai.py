import time
import warnings
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List
from api_client import StockAPIClient
from chart_storage import ChartStorage, Chart
from similarity_matcher import SimilarityMatcher
from technical_indicators import TechnicalIndicators
from pattern_recognizer import PatternRecognizer
from neural_model import TradingNeuralModel
from advanced_chart_learner import AdvancedChartLearner
from performance_cache import PerformanceCache
from config import Config
from logger_config import get_logger

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')

class TradingAI:
    def __init__(self, api_key: Optional[str] = None, db_file: str = None):
        self.logger = get_logger()
        self.api_client = StockAPIClient(api_key=api_key)
        self.storage = ChartStorage(db_file=db_file or Config.DB_FILE)
        self.similarity_matcher = SimilarityMatcher()
        self.indicators = TechnicalIndicators()
        self.pattern_recognizer = PatternRecognizer()
        self.neural_model = TradingNeuralModel()
        self.chart_learner = AdvancedChartLearner()
        self.performance_cache = PerformanceCache(max_size=Config.SIMILARITY_CACHE_SIZE)
        
        self.prediction_window = Config.PREDICTION_WINDOW
        self.base_rating_increase = Config.BASE_RATING_INCREASE
        self.base_rating_decrease = Config.BASE_RATING_DECREASE
        self.min_confidence_threshold = Config.MIN_CONFIDENCE_THRESHOLD
        
        self.recent_memory = deque(maxlen=Config.RECENT_MEMORY_SIZE)
        self.experience_replay = deque(maxlen=Config.EXPERIENCE_REPLAY_SIZE)
        self.memory_size = Config.MEMORY_SIZE
        
        self.feature_weights = {
            'similarity': 0.35,
            'technical': 0.25,
            'pattern': 0.20,
            'context': 0.15,
            'momentum': 0.05
        }
        
        self.state_values = {}
        self.exploration_rate = Config.EXPLORATION_RATE
        self.min_exploration_rate = Config.MIN_EXPLORATION_RATE
        
        self.recent_performance = deque(maxlen=20)
        self.adaptive_learning = True
        
        self.current_prediction = None
        self.current_confidence = None
        self.current_price = None
        
        self.timeframe_predictions = {
            '10s': None,
            '1h': None,
            '12h': None,
            '1d': None,
            '3d': None
        }
        
        self.timeframe_accuracy = {
            '10s': {'total': 0, 'correct': 0},
            '1h': {'total': 0, 'correct': 0},
            '12h': {'total': 0, 'correct': 0},
            '1d': {'total': 0, 'correct': 0},
            '3d': {'total': 0, 'correct': 0}
        }
        
        self.market_regime_history = deque(maxlen=100)
        self.pattern_persistence = {}
        self.cross_timeframe_correlation = {}
        self.market_microstructure = {
            'volume_relationships': {},
            'volatility_patterns': {},
            'momentum_decay': {}
        }
        
        self.regime_pattern_performance = {}
        self.pattern_regime_mapping = {}
        
        total_pred, correct_pred = self.storage.get_statistics()
        self.total_predictions = total_pred
        self.correct_predictions = correct_pred
        
        self.learning_rate = max(0.1, 1.0 - (total_pred / 1000.0))
        
        self.timeframe_learning_rates = {
            '10s': 0.3,
            '1h': 0.25,
            '12h': 0.2,
            '1d': 0.15,
            '3d': 0.1
        }
        
        self.training_data = deque(maxlen=Config.EXPERIENCE_REPLAY_SIZE)
        self.last_training_prediction = 0
        
        self.logger.info(f"TradingAI initialized with {self.total_predictions} previous predictions")
    
    def get_profit_probability(self) -> float:
        if self.total_predictions == 0:
            return 50.0
        return (self.correct_predictions / self.total_predictions) * 100.0
    
    def calculate_fractal_dimension(self, prices: List[float]) -> float:
        if len(prices) < 5:
            return 1.0
        
        prices_arr = np.array(prices)
        n = len(prices_arr)
        
        if n < 10:
            ranges = np.array([np.max(prices_arr) - np.min(prices_arr)])
            stds = np.array([np.std(prices_arr)])
        else:
            window = max(2, n // 5)
            ranges = []
            stds = []
            for i in range(0, n - window, window):
                window_data = prices_arr[i:i+window]
                ranges.append(np.max(window_data) - np.min(window_data))
                stds.append(np.std(window_data))
            ranges = np.array(ranges)
            stds = np.array(stds)
        
        if np.mean(ranges) == 0 or np.mean(stds) == 0:
            return 1.0
        
        hurst = np.log(np.mean(ranges) / np.mean(stds) + 1e-10) / np.log(n + 1e-10)
        fractal_dim = 2.0 - hurst
        return float(np.clip(fractal_dim, 1.0, 2.0))
    
    def detect_market_regime(self, prices: List[float], indicators: Dict) -> str:
        if len(prices) < 10:
            return 'neutral'
        
        prices_arr = np.array(prices)
        volatility = indicators.get('volatility', 0.01)
        
        trend = indicators.get('trend', 'neutral')
        rsi = indicators.get('rsi', 50)
        
        adx_like = volatility * 100
        
        if adx_like > 2.0 and trend != 'neutral':
            return 'trending'
        elif adx_like < 1.0:
            return 'ranging'
        else:
            return 'transitional'
    
    def calculate_momentum_divergence(self, prices: List[float], indicators: Dict) -> float:
        if len(prices) < 10:
            return 0.0
        
        prices_arr = np.array(prices)
        price_momentum = (prices_arr[-1] - prices_arr[-5]) / (prices_arr[-5] + 1e-10)
        indicator_momentum = indicators.get('momentum', 0.0)
        
        if abs(price_momentum) < 1e-6:
            return 0.0
        
        divergence = abs(price_momentum - indicator_momentum) / (abs(price_momentum) + 1e-6)
        return float(divergence)
    
    def analyze_chart_context(self, current_chart: List[float]) -> Dict:
        indicators = self.indicators.extract_all_indicators(current_chart)
        patterns = self.pattern_recognizer.extract_all_patterns(current_chart)
        
        context_score = 0.0
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0.0
        
        fractal_dim = self.calculate_fractal_dimension(current_chart)
        market_regime = self.detect_market_regime(current_chart, indicators)
        momentum_div = self.calculate_momentum_divergence(current_chart, indicators)
        
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            bullish_signals += 2
            context_score += 0.15
            signal_strength += 0.2
        elif rsi < 40:
            bullish_signals += 1
            context_score += 0.08
            signal_strength += 0.1
        elif rsi > 70:
            bearish_signals += 2
            context_score -= 0.15
            signal_strength += 0.2
        elif rsi > 60:
            bearish_signals += 1
            context_score -= 0.08
            signal_strength += 0.1
        
        macd_hist = indicators.get('macd', {}).get('histogram', 0)
        if macd_hist > 0:
            bullish_signals += 2
            context_score += 0.2
            signal_strength += 0.15
        elif macd_hist < 0:
            bearish_signals += 2
            context_score -= 0.2
            signal_strength += 0.15
        
        trend = indicators.get('trend', 'neutral')
        if trend == 'uptrend':
            bullish_signals += 3
            context_score += 0.25
            signal_strength += 0.25
        elif trend == 'downtrend':
            bearish_signals += 3
            context_score -= 0.25
            signal_strength += 0.25
        
        price_arr = np.array(current_chart)
        current_price = price_arr[-1]
        sma_5 = indicators.get('sma_5', current_price)
        sma_10 = indicators.get('sma_10', current_price)
        
        if current_price > sma_5 > sma_10:
            bullish_signals += 2
            context_score += 0.12
            signal_strength += 0.15
        elif current_price < sma_5 < sma_10:
            bearish_signals += 2
            context_score -= 0.12
            signal_strength += 0.15
        
        bb = indicators.get('bollinger', {})
        if bb:
            bb_upper = bb.get('upper', current_price)
            bb_lower = bb.get('lower', current_price)
            bb_middle = bb.get('middle', current_price)
            
            if current_price < bb_lower:
                bullish_signals += 1
                context_score += 0.08
            elif current_price > bb_upper:
                bearish_signals += 1
                context_score -= 0.08
        
        candlestick_patterns = patterns.get('candlestick', {})
        if 'hammer' in candlestick_patterns:
            bullish_signals += 2
            context_score += 0.12
            signal_strength += 0.15
        if 'three_white_soldiers' in candlestick_patterns:
            bullish_signals += 3
            context_score += 0.18
            signal_strength += 0.2
        if 'hanging_man' in candlestick_patterns:
            bearish_signals += 2
            context_score -= 0.12
            signal_strength += 0.15
        if 'three_black_crows' in candlestick_patterns:
            bearish_signals += 3
            context_score -= 0.18
            signal_strength += 0.2
        
        double_pattern = patterns.get('double_pattern', (0.0, 'none'))
        if double_pattern[1] == 'double_bottom':
            bullish_signals += 2
            context_score += 0.15
            signal_strength += 0.18
        elif double_pattern[1] == 'double_top':
            bearish_signals += 2
            context_score -= 0.15
            signal_strength += 0.18
        
        triangle_pattern = patterns.get('triangle', (0.0, 'none'))
        if triangle_pattern[1] == 'ascending_triangle':
            bullish_signals += 2
            context_score += 0.12
        elif triangle_pattern[1] == 'descending_triangle':
            bearish_signals += 2
            context_score -= 0.12
        
        support_resistance = indicators.get('support_resistance', {})
        price_position = indicators.get('price_position', 0.5)
        
        if price_position < 0.2:
            bullish_signals += 1
            context_score += 0.08
        elif price_position > 0.8:
            bearish_signals += 1
            context_score -= 0.08
        
        if market_regime == 'trending':
            signal_strength *= 1.2
        elif market_regime == 'ranging':
            signal_strength *= 0.8
        
        if momentum_div > 0.3:
            signal_strength *= 0.9
        
        context_score *= (1.0 + signal_strength * 0.3)
        context_score = np.clip(context_score, -1.0, 1.0)
        
        return {
            'score': context_score,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'signal_strength': signal_strength,
            'indicators': indicators,
            'patterns': patterns,
            'fractal_dimension': fractal_dim,
            'market_regime': market_regime,
            'momentum_divergence': momentum_div
        }
    
    def estimate_state_value(self, chart_features: Dict, market_regime: str) -> float:
        state_key = f"{market_regime}_{hash(str(sorted(chart_features.items())))}"
        
        if state_key in self.state_values:
            return self.state_values[state_key]
        
        return 0.5
    
    def update_state_value(self, chart_features: Dict, market_regime: str, reward: float, alpha: float = 0.1):
        state_key = f"{market_regime}_{hash(str(sorted(chart_features.items())))}"
        
        if state_key in self.state_values:
            self.state_values[state_key] = (1 - alpha) * self.state_values[state_key] + alpha * reward
        else:
            self.state_values[state_key] = reward
        
        if len(self.state_values) > 1000:
            oldest_key = min(self.state_values.keys(), key=lambda k: hash(k))
            del self.state_values[oldest_key]
    
    def calculate_multi_strategy_ensemble(self, current_chart: List[float], context: Dict, top_matches: List) -> Tuple[bool, float]:
        chart_ratings = [chart.rating for chart, _, _ in top_matches] if top_matches else []
        
        features = self.neural_model.extract_features_for_model(
            current_chart, context['indicators'], 
            context.get('patterns', {}), context, 
            top_matches, chart_ratings
        )
        
        neural_pred, neural_conf = self.neural_model.predict(features)
        
        strategies = []
        
        if top_matches:
            similarity_strategy = self.similarity_matcher.calculate_ensemble_confidence(top_matches)
            strategies.append(('similarity', similarity_strategy[0], similarity_strategy[1]))
        
        technical_score = context['score']
        technical_confidence = abs(technical_score) * 70 + 30
        technical_prediction = technical_score > 0
        strategies.append(('technical', technical_prediction, technical_confidence))
        
        pattern_bullish = context['bullish_signals'] > context['bearish_signals']
        signal_ratio = max(context['bullish_signals'], context['bearish_signals']) / max(1, context['bullish_signals'] + context['bearish_signals'])
        pattern_confidence = 30.0 + (signal_ratio * 55.0)
        strategies.append(('pattern', pattern_bullish, pattern_confidence))
        
        momentum = context['indicators'].get('momentum', 0.0)
        momentum_prediction = momentum > 0
        momentum_confidence = min(75.0, max(30.0, abs(momentum) * 500 + 30))
        strategies.append(('momentum', momentum_prediction, momentum_confidence))
        
        if top_matches and len(top_matches) > 0:
            avg_rating = np.mean([chart.rating for chart, _, _ in top_matches[:5]])
            rating_boost = (avg_rating - 2.5) / 2.5
            rating_confidence = 50.0 + (rating_boost * 30.0)
            rating_prediction = avg_rating > 2.5
            strategies.append(('rating', rating_prediction, rating_confidence))
        
        weighted_votes_profit = 0.0
        weighted_votes_loss = 0.0
        total_confidence_weight = 0.0
        strategy_confidences = []
        
        neural_weight = 0.4 if self.neural_model.is_trained else 0.0
        adjusted_weights = {k: v * (1.0 - neural_weight) for k, v in self.feature_weights.items()}
        
        if self.neural_model.is_trained:
            if neural_pred:
                weighted_votes_profit += neural_conf * neural_weight
            else:
                weighted_votes_loss += neural_conf * neural_weight
            total_confidence_weight += neural_conf * neural_weight
            strategy_confidences.append(neural_conf)
        
        for strategy_name, prediction, confidence in strategies:
            strategy_weight = adjusted_weights.get(strategy_name, 0.2)
            weighted_confidence = confidence * strategy_weight
            
            if prediction:
                weighted_votes_profit += weighted_confidence
            else:
                weighted_votes_loss += weighted_confidence
            
            total_confidence_weight += weighted_confidence
            strategy_confidences.append(confidence)
        
        if total_confidence_weight == 0:
            return True, 50.0
        
        profit_probability = (weighted_votes_profit / total_confidence_weight) * 100.0
        
        avg_confidence = np.mean(strategy_confidences) if strategy_confidences else 50.0
        agreement_ratio = max(weighted_votes_profit, weighted_votes_loss) / max(0.001, total_confidence_weight)
        
        ensemble_confidence = avg_confidence * (0.5 + agreement_ratio * 0.5)
        
        profit_strength = abs(profit_probability - 50.0) / 50.0
        ensemble_confidence *= (1.0 + profit_strength * 0.4)
        
        if top_matches and len(top_matches) > 0:
            high_rated_count = sum(1 for chart, _, _ in top_matches if chart.rating > 3.0)
            if high_rated_count >= 3:
                ensemble_confidence *= 1.15
            elif high_rated_count == 0:
                ensemble_confidence *= 0.9
        
        chart_learner_boost = self.chart_learner.get_pattern_confidence_boost(
            current_chart,
            context.get('market_regime', 'neutral'),
            context.get('patterns', {}),
            context['indicators']
        )
        ensemble_confidence += chart_learner_boost
        
        similar_successful = self.chart_learner.find_similar_successful_charts(current_chart, top_n=3)
        if similar_successful:
            avg_success_conf = np.mean([c['confidence'] for c in similar_successful])
            if avg_success_conf > 60.0:
                ensemble_confidence += 3.0
        
        final_prediction = profit_probability > 50.0
        final_confidence = min(Config.CONFIDENCE_MAX, max(Config.CONFIDENCE_MIN, ensemble_confidence))
        
        return final_prediction, final_confidence
    
    def make_prediction(self, current_chart: list, symbol: str) -> Tuple[bool, Optional[Chart], float]:
        context = self.analyze_chart_context(current_chart)
        
        all_charts = self.storage.get_all_charts_sorted_by_rating()
        
        if not all_charts:
            profit_prob = self.get_profit_probability()
            context_score = context['score']
            signal_strength = context.get('signal_strength', 0.0)
            
            base_confidence = 40.0 + (profit_prob - 50.0) * 0.4
            context_modifier = abs(context_score) * 40
            signal_modifier = signal_strength * 20
            
            confidence = base_confidence + context_modifier + signal_modifier
            confidence = max(35.0, min(75.0, confidence))
            
            predicted_profitable = context_score > 0 or profit_prob > 52.0
            
            return predicted_profitable, None, confidence
        
        recent_charts = self.storage.get_all_charts_sorted_by_rating(limit=150)
        top_matches = self.similarity_matcher.find_most_similar_charts(current_chart, recent_charts, top_n=10)
        
        if top_matches:
            best_match = top_matches[0][0]
            
            predicted_profitable, base_confidence = self.calculate_multi_strategy_ensemble(current_chart, context, top_matches)
            
            market_regime = context.get('market_regime', 'neutral')
            for chart, _, _ in top_matches:
                pattern_key = f"{chart.chart_id}_{market_regime}"
                if pattern_key in self.regime_pattern_performance:
                    perf = self.regime_pattern_performance[pattern_key]
                    if perf['total'] > 0:
                        pattern_accuracy = perf['correct'] / perf['total']
                        if pattern_accuracy > 0.7:
                            base_confidence += 5.0
                        elif pattern_accuracy < 0.3:
                            base_confidence -= 5.0
            
            if len(top_matches) > 1:
                first_category = top_matches[0][0].category == "profit"
                agreement = sum(1 for chart, _, _ in top_matches if (chart.category == "profit") == first_category)
                agreement_bonus = (agreement / len(top_matches)) * 0.2
                base_confidence += agreement_bonus * 100
            
            chart_features = {
                'fractal_dim': context['fractal_dimension'],
                'regime': context['market_regime'],
                'signal_strength': context['signal_strength']
            }
            state_value = self.estimate_state_value(chart_features, context['market_regime'])
            state_value_adjustment = (state_value - 0.5) * 15
            base_confidence += state_value_adjustment
            
            microstructure_adjustment = self.get_microstructure_adjustment(current_chart)
            base_confidence += microstructure_adjustment * 10
            
            context_modifier = context['score'] * 20 * (1.0 + context['signal_strength'] * 0.2)
            confidence = base_confidence + context_modifier
            
            similarity_bonus = 0.0
            if top_matches:
                avg_similarity = np.mean([sim for _, sim, _ in top_matches[:3]])
                similarity_bonus = (avg_similarity - 0.5) * 30
                confidence += similarity_bonus
            
            if np.random.random() < self.exploration_rate:
                predicted_profitable = not predicted_profitable
                confidence = max(30.0, confidence * 0.75)
            
            confidence = min(95.0, max(30.0, confidence))
            
            return predicted_profitable, best_match, confidence
        
        profit_prob = self.get_profit_probability()
        context_prediction = context['score'] > 0
        
        predicted_profitable, ensemble_confidence = self.calculate_multi_strategy_ensemble(current_chart, context, [])
        
        if profit_prob > 60.0 or (context['bullish_signals'] > context['bearish_signals']):
            return True, None, max(45.0, profit_prob * 0.6 + ensemble_confidence * 0.4)
        elif profit_prob < 40.0 or (context['bearish_signals'] > context['bullish_signals']):
            return False, None, max(45.0, (100 - profit_prob) * 0.6 + ensemble_confidence * 0.4)
        
        return context_prediction, None, max(45.0, ensemble_confidence)
    
    def calculate_profit(self, initial_price: float, final_price: float) -> bool:
        if initial_price == 0:
            return False
        profit_percent = ((final_price - initial_price) / initial_price) * 100.0
        return profit_percent > 0
    
    def calculate_reward(self, was_correct: bool, confidence: float, price_change: float, 
                         predicted_profitable: bool, was_profitable: bool, timeframe: str = '10s') -> float:
        base_reward = 1.0 if was_correct else -0.5
        
        confidence_weight = confidence / 100.0
        magnitude_weight = min(3.0, abs(price_change) * 15)
        
        direction_consistency = 1.0 if (predicted_profitable == was_profitable) else -0.5
        
        timeframe_multiplier = self.timeframe_learning_rates.get(timeframe, 0.2)
        
        if was_correct:
            reward = base_reward * (1.0 + confidence_weight * 0.6) * (1.0 + magnitude_weight * 0.4) * (1.0 + direction_consistency * 0.3)
            reward *= (1.0 + timeframe_multiplier * 0.5)
        else:
            penalty_multiplier = 1.0 + abs(confidence_weight - 0.5) * 0.5
            reward = base_reward * penalty_multiplier * (1.0 + magnitude_weight * 0.25)
            reward *= (1.0 - timeframe_multiplier * 0.3)
        
        if abs(price_change) > 1.0:
            reward *= (1.0 + abs(price_change) * 0.1)
        
        if was_correct and abs(price_change) > 0.5:
            reward *= 1.2
        
        if not was_correct and abs(price_change) > 0.5:
            reward *= 1.3
        
        return reward
    
    def update_feature_weights(self, was_correct: bool, strategies_used: List[str], reward: float):
        if not self.adaptive_learning:
            return
        
        update_rate = 0.05
        total_weight = sum(self.feature_weights.values())
        
        for strategy in strategies_used:
            if strategy in self.feature_weights:
                if was_correct:
                    adjustment = update_rate * reward
                else:
                    adjustment = -update_rate * abs(reward) * 0.5
                
                self.feature_weights[strategy] = max(0.05, min(0.5, self.feature_weights[strategy] + adjustment))
        
        normalization_factor = total_weight / sum(self.feature_weights.values())
        for key in self.feature_weights:
            self.feature_weights[key] *= normalization_factor
    
    def update_chart_rating(self, chart: Chart, was_correct: bool, confidence: float = 1.0, 
                           price_change: float = 0.0, predicted_profitable: bool = False, 
                           was_profitable: bool = False, timeframe: str = '10s'):
        if chart.chart_id is not None:
            reward = self.calculate_reward(was_correct, confidence, price_change, predicted_profitable, was_profitable, timeframe)
            
            recent_accuracy = np.mean([p['correct'] for p in self.recent_performance]) if self.recent_performance else 0.5
            adaptive_rate = self.learning_rate * (1.0 + recent_accuracy * 0.2)
            
            timeframe_adjustment = 1.0
            if timeframe in self.timeframe_learning_rates:
                tf_stats = self.timeframe_accuracy.get(timeframe, {'total': 0, 'correct': 0})
                if tf_stats['total'] > 0:
                    tf_accuracy = tf_stats['correct'] / tf_stats['total']
                    timeframe_adjustment = 1.0 + (tf_accuracy - 0.5) * 0.3
            
            if was_correct:
                rating_change = self.base_rating_increase * adaptive_rate * (1.0 + reward * 0.6) * timeframe_adjustment
                self.storage.update_chart_rating(chart, rating_change)
            else:
                rating_change = -self.base_rating_decrease * adaptive_rate * (1.0 + abs(reward) * 0.4) * timeframe_adjustment
                self.storage.update_chart_rating(chart, rating_change)
            
            if chart.rating > 5.0:
                chart.rating = 5.0
            elif chart.rating < 0.1:
                chart.rating = 0.1
            
            experience = {
                'chart_id': chart.chart_id,
                'was_correct': was_correct,
                'reward': reward,
                'confidence': confidence,
                'price_change': price_change
            }
            self.experience_replay.append(experience)
            
            if len(self.experience_replay) > 50 and np.random.random() < 0.3:
                replay_list = list(self.experience_replay)
                sample_size = min(10, len(replay_list))
                sample_indices = np.random.choice(len(replay_list), size=sample_size, replace=False)
                sample = [replay_list[i] for i in sample_indices]
                avg_reward = np.mean([exp['reward'] for exp in sample if exp['was_correct']])
                if avg_reward > 0:
                    for exp in sample:
                        if exp['was_correct']:
                            matching_chart = self.storage.find_chart_by_id(exp['chart_id'])
                            if matching_chart:
                                bonus = self.base_rating_increase * 0.1 * (exp['reward'] / avg_reward)
                                self.storage.update_chart_rating(matching_chart, bonus)
    
    def trade(self, symbol: str):
        print(f"\nCHECKING PRICE FOR ${symbol} ")
        
        initial_price = self.api_client.get_current_price(symbol)
        if initial_price is None:
            print(f"Could not fetch price for {symbol}")
            return
        
        print(f"Initial price: ${initial_price:.2f}")
        
        current_chart = self.api_client.get_price_history(symbol, duration_seconds=5)
        if len(current_chart) < 2:
            print("Not enough data collected")
            return
        
        predicted_profitable, matched_chart, confidence = self.make_prediction(current_chart, symbol)
        
        self.current_prediction = predicted_profitable
        self.current_confidence = confidence
        self.current_price = initial_price
        
        profit_probability = self.get_profit_probability()
        print(f"Profit probability: {profit_probability:.2f}%")
        print(f"Prediction: {'PROFIT' if predicted_profitable else 'NON-PROFIT'}")
        if matched_chart:
            print(f"Matched chart (rating: {matched_chart.rating:.2f}, similarity: {confidence:.2f}%)")
        
        print(f"\nWaiting {self.prediction_window} seconds...")
        time.sleep(self.prediction_window)
        
        final_price = self.api_client.get_current_price(symbol)
        if final_price is None:
            print("Could not fetch final price")
            return
        
        print(f"Final price: ${final_price:.2f}")
        
        was_profitable = self.calculate_profit(initial_price, final_price)
        price_change = ((final_price - initial_price) / initial_price) * 100.0
        print(f"Price change: {price_change:+.2f}%")
        print(f"Result: {'PROFIT' if was_profitable else 'NON-PROFIT'}")
        
        category = "profit" if was_profitable else "non_profit"
        saved_chart = self.storage.add_chart(current_chart, category, symbol, initial_rating=1.0)
        
        matched_chart_id = None
        similarity_value = 0.0
        strategies_used = []
        
        if matched_chart:
            matched_chart_id = matched_chart.chart_id
            similarity_value = confidence / 100.0
            was_correct = (predicted_profitable == was_profitable)
            strategies_used = ['similarity', 'technical', 'pattern', 'momentum']
            self.update_chart_rating(matched_chart, was_correct, confidence, price_change, 
                                    predicted_profitable, was_profitable, timeframe='10s')
            
            context = self.analyze_chart_context(current_chart)
            chart_features = {
                'fractal_dim': context['fractal_dimension'],
                'regime': context['market_regime'],
                'signal_strength': context['signal_strength']
            }
            reward = self.calculate_reward(was_correct, confidence, price_change, predicted_profitable, was_profitable, '10s')
            self.update_state_value(chart_features, context['market_regime'], reward)
            
            if was_correct:
                self.correct_predictions += 1
                print(f"✓ Prediction correct (confidence: {confidence:.1f}%) - rating increased")
            else:
                print(f"✗ Prediction incorrect (confidence: {confidence:.1f}%) - rating decreased")
        else:
            if predicted_profitable == was_profitable:
                self.correct_predictions += 1
                print(f"✓ Baseline prediction correct (confidence: {confidence:.1f}%)")
            strategies_used = ['technical', 'pattern', 'momentum']
        
        if strategies_used:
            reward = self.calculate_reward(predicted_profitable == was_profitable, confidence, price_change, 
                                         predicted_profitable, was_profitable, '10s')
            self.update_feature_weights(predicted_profitable == was_profitable, strategies_used, reward)
        
        self.recent_memory.append({
            'chart': current_chart,
            'profit': was_profitable,
            'price_change': price_change,
            'timestamp': time.time()
        })
        
        self.recent_performance.append({
            'correct': predicted_profitable == was_profitable,
            'confidence': confidence,
            'price_change': price_change
        })
        
        self.total_predictions += 1
        
        recent_accuracy = np.mean([p['correct'] for p in self.recent_performance]) if self.recent_performance else 0.5
        
        if recent_accuracy < 0.5:
            self.learning_rate = min(0.3, max(0.1, 0.2 + (0.5 - recent_accuracy) * 0.2))
        else:
            self.learning_rate = max(0.05, 0.15 - (self.total_predictions / 3000.0))
        
        context = self.analyze_chart_context(current_chart)
        recent_charts = self.storage.get_all_charts_sorted_by_rating(limit=150)
        top_matches = self.similarity_matcher.find_most_similar_charts(current_chart, recent_charts, top_n=10) if recent_charts else []
        
        if matched_chart and top_matches:
            chart_ratings = [chart.rating for chart, _, _ in top_matches[:5]]
        else:
            chart_ratings = []
        
        training_features = self.neural_model.extract_features_for_model(
            current_chart, context['indicators'],
            context.get('patterns', {}), context,
            top_matches if top_matches else [], chart_ratings
        )
        
        self.training_data.append({
            'features': training_features[0],
            'prediction': predicted_profitable,
            'actual': was_profitable,
            'confidence': confidence
        })
        
        if matched_chart and self.neural_model.is_trained:
            actual_confidence = 100.0 if was_profitable == predicted_profitable else 30.0
            self.neural_model.update_online(training_features, was_profitable == predicted_profitable, actual_confidence, self.learning_rate * 0.01)
        
        if (self.total_predictions - self.last_training_prediction >= Config.TRAINING_INTERVAL and 
            len(self.training_data) >= Config.MIN_TRAINING_SAMPLES):
            self.logger.info(f"Training neural model with {len(self.training_data)} samples")
            self._train_neural_model()
            self.last_training_prediction = self.total_predictions
        
        recent_accuracy = np.mean([p['correct'] for p in self.recent_performance])
        if recent_accuracy > 0.6:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * 0.98)
        elif recent_accuracy < 0.4:
            self.exploration_rate = min(0.25, self.exploration_rate * 1.02)
        
        market_regime = context.get('market_regime', 'neutral')
        
        self.market_regime_history.append({
            'regime': market_regime,
            'correct': was_profitable == predicted_profitable,
            'price_change': price_change,
            'timestamp': time.time()
        })
        
        self.learn_market_microstructure(current_chart, price_change, was_profitable)
        
        context = self.analyze_chart_context(current_chart)
        market_regime = context.get('market_regime', 'neutral')
        
        self.chart_learner.learn_from_chart_outcome(
            current_chart,
            predicted_profitable,
            was_profitable,
            confidence,
            market_regime,
            context['indicators'],
            context.get('patterns', {})
        )
        
        if matched_chart:
            self.learn_pattern_persistence(matched_chart, '10s', was_profitable == predicted_profitable, price_change)
            
            pattern_key = f"{matched_chart.chart_id}_{market_regime}"
            if pattern_key not in self.regime_pattern_performance:
                self.regime_pattern_performance[pattern_key] = {
                    'total': 0,
                    'correct': 0,
                    'avg_reward': 0.0
                }
            
            perf = self.regime_pattern_performance[pattern_key]
            perf['total'] += 1
            if was_profitable == predicted_profitable:
                perf['correct'] += 1
                reward = self.calculate_reward(True, confidence, price_change, predicted_profitable, was_profitable, '10s')
                perf['avg_reward'] = (perf['avg_reward'] * (perf['total'] - 1) + reward) / perf['total']
            
            if len(self.regime_pattern_performance) > 2000:
                oldest_key = min(self.regime_pattern_performance.keys(),
                               key=lambda k: self.regime_pattern_performance[k].get('total', 0))
                del self.regime_pattern_performance[oldest_key]
        
        self.timeframe_accuracy['10s']['total'] += 1
        if was_profitable == predicted_profitable:
            self.timeframe_accuracy['10s']['correct'] += 1
        
        short_term_result = was_profitable
        self.update_timeframe_predictions(symbol)
        
        period_map = {
            '1h': ('1d', '5m', 12),
            '12h': ('5d', '1h', 12),
            '1d': ('5d', '1h', 24),
            '3d': ('10d', '4h', 18)
        }
        
        for timeframe in ['1h', '12h', '1d', '3d']:
            tf_prediction = self.timeframe_predictions.get(timeframe)
            if tf_prediction and tf_prediction.get('prediction') is not None:
                period, interval, max_points = period_map.get(timeframe, ('5d', '1h', 24))
                long_term_chart = self.api_client.get_historical_data(symbol, period, interval, max_points)
                if len(long_term_chart) >= 5:
                    long_term_result = None
                    self.learn_cross_timeframe_correlation(
                        current_chart,
                        long_term_chart,
                        short_term_result,
                        long_term_result
                    )
        
        self.verify_timeframe_predictions(symbol, initial_price)
        
        accuracy = (self.correct_predictions/self.total_predictions)*100 if self.total_predictions > 0 else 0.0
        
        self.storage.add_trade(
            symbol=symbol,
            initial_price=initial_price,
            final_price=final_price,
            predicted_profitable=predicted_profitable,
            was_profitable=was_profitable,
            price_change_percent=price_change,
            confidence=confidence,
            matched_chart_id=matched_chart_id,
            similarity=similarity_value,
            chart_data=current_chart,
            accuracy_at_time=accuracy
        )
        
        self.storage.update_statistics(self.total_predictions, self.correct_predictions)
        self.storage.commit()
        
        self.current_prediction = None
        self.current_confidence = None
        self.current_price = None
        
        accuracy = (self.correct_predictions/self.total_predictions)*100 if self.total_predictions > 0 else 0.0
        print(f"Overall accuracy: {accuracy:.2f}%")
        
        if self.total_predictions % 10 == 0 and self.total_predictions > 0:
            print("\n--- Learning Progress Report ---")
            for tf in ['10s', '1h', '12h', '1d', '3d']:
                tf_stats = self.timeframe_accuracy.get(tf, {'total': 0, 'correct': 0})
                if tf_stats['total'] > 0:
                    tf_acc = (tf_stats['correct'] / tf_stats['total']) * 100.0
                    print(f"  {tf:>4} accuracy: {tf_acc:.1f}% ({tf_stats['correct']}/{tf_stats['total']})")
            
            print(f"  Market regimes learned: {len(set(r.get('regime', 'unknown') for r in self.market_regime_history))}")
            print(f"  Patterns tracked: {len(self.pattern_persistence)}")
            print(f"  Microstructure patterns: {len(self.market_microstructure['volatility_patterns'])}")
            print(f"  Cross-timeframe correlations: {len(self.cross_timeframe_correlation)}")
            print("---")
    
    def get_microstructure_adjustment(self, chart_data: List[float]) -> float:
        """Use learned market microstructure patterns to adjust predictions"""
        if len(chart_data) < 10:
            return 0.0
        
        prices_arr = np.array(chart_data)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        volatility = np.std(returns)
        momentum = (prices_arr[-1] - prices_arr[0]) / prices_arr[0]
        
        vol_key = f"{volatility:.4f}"
        momentum_key = f"{momentum:.4f}"
        
        adjustment = 0.0
        
        if vol_key in self.market_microstructure['volatility_patterns']:
            vol_stats = self.market_microstructure['volatility_patterns'][vol_key]
            if vol_stats['total'] > 0:
                profit_rate = vol_stats['profit'] / vol_stats['total']
                adjustment += (profit_rate - 0.5) * 0.3
        
        if momentum_key in self.market_microstructure['momentum_decay']:
            mom_stats = self.market_microstructure['momentum_decay'][momentum_key]
            if mom_stats['total'] > 0:
                profit_rate = mom_stats['profit'] / mom_stats['total']
                adjustment += (profit_rate - 0.5) * 0.2
        
        return adjustment
    
    def learn_market_microstructure(self, chart_data: List[float], price_change: float, was_profitable: bool):
        """Learn relationships between market microstructure and outcomes"""
        if len(chart_data) < 10:
            return
        
        prices_arr = np.array(chart_data)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        volatility = np.std(returns)
        volume_like = len([x for x in returns if abs(x) > volatility])
        momentum = (prices_arr[-1] - prices_arr[0]) / prices_arr[0]
        
        vol_key = f"{volatility:.4f}"
        if vol_key not in self.market_microstructure['volatility_patterns']:
            self.market_microstructure['volatility_patterns'][vol_key] = {'total': 0, 'profit': 0}
        self.market_microstructure['volatility_patterns'][vol_key]['total'] += 1
        if was_profitable:
            self.market_microstructure['volatility_patterns'][vol_key]['profit'] += 1
        
        momentum_key = f"{momentum:.4f}"
        if momentum_key not in self.market_microstructure['momentum_decay']:
            self.market_microstructure['momentum_decay'][momentum_key] = {'total': 0, 'profit': 0}
        self.market_microstructure['momentum_decay'][momentum_key]['total'] += 1
        if was_profitable:
            self.market_microstructure['momentum_decay'][momentum_key]['profit'] += 1
        
        if len(self.market_microstructure['volatility_patterns']) > 1000:
            oldest_key = min(self.market_microstructure['volatility_patterns'].keys())
            del self.market_microstructure['volatility_patterns'][oldest_key]
    
    def learn_cross_timeframe_correlation(self, short_term_chart: List[float], long_term_chart: List[float], 
                                         short_result: bool, long_result: Optional[bool]):
        """Learn how short-term patterns correlate with long-term outcomes"""
        if len(short_term_chart) < 5:
            return
        
        short_features = self.similarity_matcher.extract_features(short_term_chart)
        short_trend = short_features.get('trend_value', 0.0)
        short_momentum = short_features.get('momentum', 0.0)
        
        if len(long_term_chart) >= 5:
            long_features = self.similarity_matcher.extract_features(long_term_chart)
            long_trend = long_features.get('trend_value', 0.0)
        else:
            long_trend = short_trend
        
        correlation_key = f"{short_trend:.3f}_{short_momentum:.3f}_{long_trend:.3f}"
        if correlation_key not in self.cross_timeframe_correlation:
            self.cross_timeframe_correlation[correlation_key] = {
                'total': 0,
                'both_profit': 0,
                'both_loss': 0,
                'divergent': 0
            }
        
        self.cross_timeframe_correlation[correlation_key]['total'] += 1
        if long_result is not None:
            if short_result and long_result:
                self.cross_timeframe_correlation[correlation_key]['both_profit'] += 1
            elif not short_result and not long_result:
                self.cross_timeframe_correlation[correlation_key]['both_loss'] += 1
            else:
                self.cross_timeframe_correlation[correlation_key]['divergent'] += 1
        
        if len(self.cross_timeframe_correlation) > 500:
            oldest_key = min(self.cross_timeframe_correlation.keys(),
                           key=lambda k: self.cross_timeframe_correlation[k].get('total', 0))
            del self.cross_timeframe_correlation[oldest_key]
    
    def get_cross_timeframe_adjustment(self, short_chart: List[float], target_timeframe: str) -> float:
        """Use cross-timeframe correlations to adjust predictions"""
        if len(short_chart) < 5:
            return 0.0
        
        short_features = self.similarity_matcher.extract_features(short_chart)
        short_trend = short_features.get('trend_value', 0.0)
        short_momentum = short_features.get('momentum', 0.0)
        
        adjustment = 0.0
        matches = 0
        
        for key, corr_data in self.cross_timeframe_correlation.items():
            parts = key.split('_')
            if len(parts) >= 2:
                try:
                    key_trend = float(parts[0])
                    key_momentum = float(parts[1])
                    
                    if abs(key_trend - short_trend) < 0.1 and abs(key_momentum - short_momentum) < 0.1:
                        total = corr_data.get('total', 0)
                        if total > 5:
                            both_profit_rate = corr_data.get('both_profit', 0) / total
                            both_loss_rate = corr_data.get('both_loss', 0) / total
                            divergence_rate = corr_data.get('divergent', 0) / total
                            
                            if both_profit_rate > 0.6:
                                adjustment += 0.15
                            elif both_loss_rate > 0.6:
                                adjustment -= 0.15
                            
                            matches += 1
                            if matches >= 3:
                                break
                except ValueError:
                    continue
        
        return adjustment / max(1, matches) if matches > 0 else 0.0
    
    def learn_pattern_persistence(self, chart: Chart, timeframe: str, was_correct: bool, price_change: float):
        """Learn how long patterns persist and remain valid"""
        chart_key = f"{chart.chart_id}_{timeframe}"
        
        if chart_key not in self.pattern_persistence:
            self.pattern_persistence[chart_key] = {
                'first_seen': time.time(),
                'last_seen': time.time(),
                'uses': 0,
                'correct': 0,
                'avg_price_change': 0.0
            }
        
        pattern = self.pattern_persistence[chart_key]
        pattern['last_seen'] = time.time()
        pattern['uses'] += 1
        if was_correct:
            pattern['correct'] += 1
        
        pattern['avg_price_change'] = (pattern['avg_price_change'] * (pattern['uses'] - 1) + abs(price_change)) / pattern['uses']
        
        if len(self.pattern_persistence) > 2000:
            oldest_key = min(self.pattern_persistence.keys(), 
                           key=lambda k: self.pattern_persistence[k]['last_seen'])
            del self.pattern_persistence[oldest_key]
    
    def get_timeframe_adjusted_confidence(self, base_confidence: float, timeframe: str) -> float:
        """Adjust confidence based on timeframe-specific accuracy"""
        tf_stats = self.timeframe_accuracy.get(timeframe, {'total': 0, 'correct': 0})
        
        if tf_stats['total'] == 0:
            return base_confidence
        
        tf_accuracy = (tf_stats['correct'] / tf_stats['total']) * 100.0
        accuracy_modifier = (tf_accuracy - 50.0) / 50.0
        
        adjusted_confidence = base_confidence * (1.0 + accuracy_modifier * 0.2)
        return min(95.0, max(25.0, adjusted_confidence))
    
    def calculate_timeframe_prediction(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Calculate prediction for a specific timeframe with enhanced learning"""
        try:
            period_map = {
                '1h': ('1d', '5m', 12),
                '12h': ('5d', '1h', 12),
                '1d': ('5d', '1h', 24),
                '3d': ('10d', '4h', 18)
            }
            
            if timeframe not in period_map:
                return None
            
            period, interval, max_points = period_map[timeframe]
            chart_data = self.api_client.get_historical_data(symbol, period, interval, max_points)
            
            if len(chart_data) < 5:
                return None
            
            predicted_profitable, matched_chart, confidence = self.make_prediction(chart_data, symbol)
            
            confidence = self.get_timeframe_adjusted_confidence(confidence, timeframe)
            
            short_term_chart = self.api_client.get_price_history(symbol, duration_seconds=5)
            if len(short_term_chart) >= 5:
                cross_tf_adjustment = self.get_cross_timeframe_adjustment(short_term_chart, timeframe)
                confidence = min(95.0, max(25.0, confidence + cross_tf_adjustment * 100))
            
            if matched_chart:
                chart_key = f"{matched_chart.chart_id}_{timeframe}"
                if chart_key in self.pattern_persistence:
                    persistence = self.pattern_persistence[chart_key]
                    age_factor = min(1.0, (time.time() - persistence['first_seen']) / 3600.0)
                    persistence_bonus = (persistence['correct'] / max(1, persistence['uses'])) * age_factor * 0.1
                    confidence = min(95.0, confidence + persistence_bonus * 100)
            
            context = self.analyze_chart_context(chart_data)
            market_regime = context.get('market_regime', 'neutral')
            
            regime_history = [r for r in self.market_regime_history if r.get('regime') == market_regime]
            if regime_history:
                regime_correct = sum(1 for r in regime_history if r.get('correct', False))
                regime_accuracy = regime_correct / max(1, len(regime_history))
                if regime_accuracy > 0.6:
                    confidence *= 1.05
                elif regime_accuracy < 0.4:
                    confidence *= 0.95
            
            microstructure_adjustment = self.get_microstructure_adjustment(chart_data)
            confidence = min(95.0, max(25.0, confidence + microstructure_adjustment * 100))
            
            return {
                'prediction': predicted_profitable,
                'confidence': confidence,
                'timeframe': timeframe,
                'market_regime': market_regime
            }
        except Exception as e:
            print(f"Error calculating {timeframe} prediction: {e}")
            return None
    
    def verify_timeframe_predictions(self, symbol: str, reference_price: float):
        """Verify and learn from previous timeframe predictions"""
        current_price = self.api_client.get_current_price(symbol)
        if current_price is None:
            return
        
        for timeframe in ['1h', '12h', '1d', '3d']:
            prev_prediction = self.timeframe_predictions.get(timeframe)
            if prev_prediction is None:
                continue
            
            prediction_timestamp = prev_prediction.get('timestamp', 0)
            time_elapsed = time.time() - prediction_timestamp
            
            timeframe_seconds = {
                '1h': 3600,
                '12h': 43200,
                '1d': 86400,
                '3d': 259200
            }.get(timeframe, 0)
            
            if timeframe_seconds > 0 and time_elapsed >= timeframe_seconds * 0.9:
                predicted_price_change = prev_prediction.get('expected_change', 0)
                actual_price_change = ((current_price - reference_price) / reference_price) * 100.0
                
                predicted_profitable = prev_prediction.get('prediction', False)
                was_profitable = actual_price_change > 0
                
                was_correct = predicted_profitable == was_profitable
                
                self.timeframe_accuracy[timeframe]['total'] += 1
                if was_correct:
                    self.timeframe_accuracy[timeframe]['correct'] += 1
                
                accuracy = (self.timeframe_accuracy[timeframe]['correct'] / 
                          max(1, self.timeframe_accuracy[timeframe]['total'])) * 100.0
                
                if was_correct:
                    self.timeframe_learning_rates[timeframe] = min(0.5, 
                        self.timeframe_learning_rates[timeframe] * 1.02)
                else:
                    self.timeframe_learning_rates[timeframe] = max(0.05,
                        self.timeframe_learning_rates[timeframe] * 0.98)
    
    def update_timeframe_predictions(self, symbol: str):
        """Update predictions for all timeframes with enhanced learning"""
        timeframes = ['1h', '12h', '1d', '3d']
        
        reference_price = self.api_client.get_current_price(symbol)
        
        for timeframe in timeframes:
            try:
                prediction = self.calculate_timeframe_prediction(symbol, timeframe)
                if prediction:
                    prediction['timestamp'] = time.time()
                    prediction['reference_price'] = reference_price
                    self.timeframe_predictions[timeframe] = prediction
            except Exception as e:
                print(f"Error updating {timeframe} prediction: {e}")
    
    def _train_neural_model(self):
        if len(self.training_data) < Config.MIN_TRAINING_SAMPLES:
            return
        
        try:
            X = np.array([item['features'] for item in self.training_data])
            y_pred = np.array([[1.0 if item['prediction'] == item['actual'] else 0.0] for item in self.training_data])
            
            actual_confidences = []
            for item in self.training_data:
                if item['prediction'] == item['actual']:
                    actual_confidences.append([min(Config.CONFIDENCE_MAX, item['confidence'] + 5.0)])
                else:
                    actual_confidences.append([max(Config.CONFIDENCE_MIN, item['confidence'] - 10.0)])
            
            y_conf = np.array(actual_confidences)
            
            self.neural_model.train(X, y_pred, y_conf, validation_split=0.2, epochs=15)
            self.logger.info("Neural model training completed successfully")
        except Exception as e:
            self.logger.error(f"Neural model training error: {e}")
    
    def get_current_prediction(self):
        current_price = self.api_client.get_current_price("TSLA") if self.current_price is None else self.current_price
        if current_price and self.current_price is None:
            self.current_price = current_price
        
        return {
            'prediction': self.current_prediction,
            'confidence': self.current_confidence,
            'current_price': self.current_price if self.current_price else current_price,
            'timeframes': {
                '1h': self.timeframe_predictions.get('1h'),
                '12h': self.timeframe_predictions.get('12h'),
                '1d': self.timeframe_predictions.get('1d'),
                '3d': self.timeframe_predictions.get('3d')
            }
        }

