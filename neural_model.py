import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import pickle
from config import Config

class TradingNeuralModel:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.weights = None
        self.biases = None
        self.feature_scaler_mean = None
        self.feature_scaler_std = None
        self.training_history = []
        self.is_trained = False
        self.dropout_rate = Config.NEURAL_NETWORK_DROPOUT_RATE
        self.training_mode = False
        
        self.input_dim = 45
        self.hidden_dims = Config.NEURAL_NETWORK_HIDDEN_LAYERS.copy()
        
        self._initialize_weights()
        
        if os.path.exists(self.model_path):
            try:
                self._load_model()
                self.is_trained = True
            except Exception as e:
                pass
    
    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        
        dims = [self.input_dim] + self.hidden_dims + [2]
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _forward(self, X, apply_dropout: bool = False):
        activations = [X]
        z_values = []
        dropout_masks = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._relu(z)
            
            if apply_dropout and self.training_mode:
                dropout_mask = (np.random.random(a.shape) > self.dropout_rate).astype(float)
                dropout_mask /= (1 - self.dropout_rate)
                a *= dropout_mask
                dropout_masks.append(dropout_mask)
            else:
                dropout_masks.append(None)
            
            activations.append(a)
        
        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z_final)
        output = self._sigmoid(z_final)
        activations.append(output)
        
        return activations, z_values, dropout_masks
    
    def _predict_forward(self, X):
        a = X
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._relu(z)
        z_final = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self._sigmoid(z_final)
        return output
    
    def extract_features_for_model(self, chart_data: List[float], indicators: Dict, 
                                   patterns: Dict, context: Dict, 
                                   top_matches: List, chart_ratings: List[float] = None) -> np.ndarray:
        features = []
        
        if len(chart_data) < 5:
            chart_data = chart_data + [chart_data[-1]] * (5 - len(chart_data)) if chart_data else [0.0] * 5
        
        prices_arr = np.array(chart_data[-20:] if len(chart_data) > 20 else chart_data)
        if len(prices_arr) > 0 and prices_arr.max() != prices_arr.min():
            prices_normalized = (prices_arr - prices_arr.min()) / (prices_arr.max() - prices_arr.min() + 1e-10)
        else:
            prices_normalized = np.zeros(min(20, len(chart_data)))
        
        features.extend(prices_normalized.tolist()[:15])
        
        while len(features) < 15:
            features.append(0.0)
        
        features.append(indicators.get('rsi', 50.0) / 100.0)
        features.append(indicators.get('volatility', 0.01) * 100)
        
        macd = indicators.get('macd', {})
        if isinstance(macd, dict):
            features.append(macd.get('histogram', 0.0) / 10.0)
        else:
            features.append(0.0)
        
        bb = indicators.get('bollinger', {})
        if isinstance(bb, dict) and len(chart_data) > 0:
            current_price = chart_data[-1]
            bb_middle = bb.get('middle', current_price)
            if bb_middle > 0:
                features.append((current_price - bb_middle) / bb_middle)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        trend = indicators.get('trend', 'neutral')
        features.append(1.0 if trend == 'uptrend' else 0.0)
        features.append(1.0 if trend == 'downtrend' else 0.0)
        
        features.append(context.get('fractal_dimension', 1.0) / 2.0)
        
        market_regime = context.get('market_regime', 'neutral')
        features.append(1.0 if market_regime == 'trending' else 0.0)
        features.append(1.0 if market_regime == 'ranging' else 0.0)
        
        features.append(context.get('signal_strength', 0.0))
        features.append(context.get('momentum_divergence', 0.0))
        
        bullish_signals = context.get('bullish_signals', 0)
        bearish_signals = context.get('bearish_signals', 0)
        total_signals = max(1, bullish_signals + bearish_signals)
        features.append(bullish_signals / total_signals)
        features.append(bearish_signals / total_signals)
        
        if top_matches and len(top_matches) > 0:
            avg_similarity = np.mean([sim for _, sim, _ in top_matches[:5]])
            features.append(avg_similarity)
            
            profit_count = sum(1 for chart, _, _ in top_matches[:5] if chart.category == "profit")
            features.append(profit_count / min(5, len(top_matches)))
            
            if chart_ratings and len(chart_ratings) > 0:
                avg_rating = np.mean(chart_ratings[:5])
                features.append(avg_rating / 5.0)
            else:
                avg_rating = np.mean([chart.rating for chart, _, _ in top_matches[:5]])
                features.append(avg_rating / 5.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        pattern_confidences = []
        double_pattern = patterns.get('double_pattern', (0.0, 'none'))
        pattern_confidences.append(double_pattern[0] if double_pattern[1] != 'none' else 0.0)
        
        triangle = patterns.get('triangle', (0.0, 'none'))
        pattern_confidences.append(triangle[0] if triangle[1] != 'none' else 0.0)
        
        candlestick = patterns.get('candlestick', {})
        if isinstance(candlestick, dict):
            pattern_confidences.append(1.0 if 'hammer' in candlestick or 'three_white_soldiers' in candlestick else 0.0)
            pattern_confidences.append(1.0 if 'hanging_man' in candlestick or 'three_black_crows' in candlestick else 0.0)
        else:
            pattern_confidences.extend([0.0, 0.0])
        
        features.extend(pattern_confidences[:4])
        
        while len(features) < self.input_dim:
            features.append(0.0)
        
        features = features[:self.input_dim]
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        if not self.is_trained:
            return True, 50.0
        
        try:
            if self.feature_scaler_mean is not None:
                features_scaled = (features - self.feature_scaler_mean) / (self.feature_scaler_std + 1e-10)
            else:
                features_scaled = features
            
            output = self._predict_forward(features_scaled)
            pred_bool = bool(output[0, 0] > 0.5)
            conf_value = float(output[0, 1] * 100.0)
            conf_value = max(30.0, min(95.0, conf_value))
            return pred_bool, conf_value
        except Exception as e:
            print(f"Model prediction error: {e}")
            return True, 50.0
    
    def train(self, X: np.ndarray, y_prediction: np.ndarray, y_confidence: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 10):
        if len(X) < 10:
            return
        
        if self.feature_scaler_mean is None:
            self.feature_scaler_mean = np.mean(X, axis=0)
            self.feature_scaler_std = np.std(X, axis=0) + 1e-10
        
        X_scaled = (X - self.feature_scaler_mean) / (self.feature_scaler_std + 1e-10)
        
        y_combined = np.hstack([y_prediction, y_confidence / 100.0])
        
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]
        
        learning_rate = 0.001
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(X_train, y_train, learning_rate)
            
            if len(X_val) > 0:
                val_pred = self._predict_forward(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        self.is_trained = True
        self._save_model()
    
    def _train_epoch(self, X, y, learning_rate):
        batch_size = min(Config.NEURAL_NETWORK_BATCH_SIZE, len(X))
        total_loss = 0.0
        
        self.training_mode = True
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            activations, z_values, dropout_masks = self._forward(X_batch, apply_dropout=True)
            
            output = activations[-1]
            loss = np.mean((output - y_batch) ** 2)
            total_loss += loss
            
            error = output - y_batch
            delta = error * output * (1 - output)
            
            for layer_idx in range(len(self.weights) - 1, -1, -1):
                if layer_idx > 0:
                    prev_activation = activations[layer_idx]
                    if dropout_masks[layer_idx - 1] is not None:
                        prev_activation = prev_activation * dropout_masks[layer_idx - 1]
                    
                    grad_weights = np.dot(prev_activation.T, delta)
                    grad_biases = np.mean(delta, axis=0, keepdims=True)
                    
                    self.weights[layer_idx] -= learning_rate * grad_weights
                    self.biases[layer_idx] -= learning_rate * grad_biases
                    
                    if layer_idx > 0:
                        delta = np.dot(delta, self.weights[layer_idx].T) * self._relu_derivative(z_values[layer_idx-1])
                        if dropout_masks[layer_idx - 1] is not None:
                            delta *= dropout_masks[layer_idx - 1]
                else:
                    grad_weights = np.dot(X_batch.T, delta)
                    grad_biases = np.mean(delta, axis=0, keepdims=True)
                    
                    self.weights[layer_idx] -= learning_rate * grad_weights
                    self.biases[layer_idx] -= learning_rate * grad_biases
        
        self.training_mode = False
        return total_loss / (len(X) // batch_size + 1)
    
    def update_online(self, features: np.ndarray, was_correct: bool, 
                     actual_confidence: float, learning_rate: float = 0.01):
        if not self.is_trained:
            return
        
        try:
            if self.feature_scaler_mean is not None:
                features_scaled = (features - self.feature_scaler_mean) / (self.feature_scaler_std + 1e-10)
            else:
                features_scaled = features
            
            target = np.array([[1.0 if was_correct else 0.0, actual_confidence / 100.0]])
            
            activations, z_values = self._forward(features_scaled)
            output = activations[-1]
            
            error = output - target
            delta = error * output * (1 - output)
            
            online_lr = learning_rate * 0.1
            
            for i in range(len(self.weights) - 1, -1, -1):
                if i > 0:
                    prev_activation = activations[i]
                    grad_weights = np.dot(prev_activation.T, delta)
                    grad_biases = delta
                    
                    self.weights[i] -= online_lr * grad_weights
                    self.biases[i] -= online_lr * grad_biases
                    
                    if i > 0:
                        delta = np.dot(delta, self.weights[i].T) * self._relu_derivative(z_values[i-1])
                else:
                    grad_weights = np.dot(features_scaled.T, delta)
                    grad_biases = delta
                    
                    self.weights[i] -= online_lr * grad_weights
                    self.biases[i] -= online_lr * grad_biases
        except Exception as e:
            pass
    
    def _save_model(self):
        try:
            model_data = {
                'weights': self.weights,
                'biases': self.biases,
                'feature_scaler_mean': self.feature_scaler_mean,
                'feature_scaler_std': self.feature_scaler_std
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            pass
    
    def _load_model(self):
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.weights = model_data['weights']
            self.biases = model_data['biases']
            self.feature_scaler_mean = model_data.get('feature_scaler_mean')
            self.feature_scaler_std = model_data.get('feature_scaler_std')
