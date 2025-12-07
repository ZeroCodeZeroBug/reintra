import os
from typing import Dict, Any

class Config:
    PREDICTION_WINDOW = int(os.getenv('PREDICTION_WINDOW', '10'))
    BASE_RATING_INCREASE = float(os.getenv('BASE_RATING_INCREASE', '0.3'))
    BASE_RATING_DECREASE = float(os.getenv('BASE_RATING_DECREASE', '0.15'))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.3'))
    
    NEURAL_NETWORK_HIDDEN_LAYERS = [128, 64, 32]
    NEURAL_NETWORK_LEARNING_RATE = 0.001
    NEURAL_NETWORK_BATCH_SIZE = 32
    NEURAL_NETWORK_DROPOUT_RATE = 0.3
    
    TRAINING_INTERVAL = 50
    MIN_TRAINING_SAMPLES = 30
    ONLINE_LEARNING_RATE = 0.01
    
    SIMILARITY_CACHE_SIZE = 5000
    CHART_MATCH_LIMIT = 150
    TOP_MATCHES_COUNT = 10
    
    MEMORY_SIZE = 50
    EXPERIENCE_REPLAY_SIZE = 500
    RECENT_MEMORY_SIZE = 50
    
    ENABLE_ENSEMBLE = True
    ENSEMBLE_MODELS_COUNT = 3
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_METRICS = True
    
    DB_FILE = os.getenv('DB_FILE', 'trading_ai.db')
    MODEL_PATH = os.getenv('MODEL_PATH', 'trading_model.pkl')
    
    EXPLORATION_RATE = 0.15
    MIN_EXPLORATION_RATE = 0.05
    
    CHART_RATING_MIN = 0.1
    CHART_RATING_MAX = 5.0
    
    CONFIDENCE_MIN = 30.0
    CONFIDENCE_MAX = 95.0
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        return {
            'prediction_window': cls.PREDICTION_WINDOW,
            'neural_network_layers': cls.NEURAL_NETWORK_HIDDEN_LAYERS,
            'training_interval': cls.TRAINING_INTERVAL,
            'enable_ensemble': cls.ENABLE_ENSEMBLE
        }

