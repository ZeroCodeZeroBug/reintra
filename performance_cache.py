from functools import lru_cache
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import OrderedDict
import hashlib

class PerformanceCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.similarity_cache = OrderedDict()
        self.feature_cache = OrderedDict()
        self.prediction_cache = OrderedDict()
        
    def _hash_chart(self, chart_data: List[float]) -> str:
        chart_str = ",".join([f"{x:.6f}" for x in chart_data[-10:]])
        return hashlib.md5(chart_str.encode()).hexdigest()
    
    def get_similarity(self, chart1_hash: str, chart2_hash: str) -> Optional[float]:
        key = f"{chart1_hash}_{chart2_hash}"
        if key in self.similarity_cache:
            self.similarity_cache.move_to_end(key)
            return self.similarity_cache[key]
        return None
    
    def set_similarity(self, chart1_hash: str, chart2_hash: str, similarity: float):
        key = f"{chart1_hash}_{chart2_hash}"
        self.similarity_cache[key] = similarity
        self.similarity_cache.move_to_end(key)
        
        if len(self.similarity_cache) > self.max_size:
            self.similarity_cache.popitem(last=False)
    
    def get_features(self, chart_hash: str) -> Optional[np.ndarray]:
        if chart_hash in self.feature_cache:
            self.feature_cache.move_to_end(chart_hash)
            return self.feature_cache[chart_hash]
        return None
    
    def set_features(self, chart_hash: str, features: np.ndarray):
        self.feature_cache[chart_hash] = features
        self.feature_cache.move_to_end(chart_hash)
        
        if len(self.feature_cache) > self.max_size:
            self.feature_cache.popitem(last=False)
    
    def get_prediction(self, chart_hash: str, context_hash: str) -> Optional[Tuple[bool, float]]:
        key = f"{chart_hash}_{context_hash}"
        if key in self.prediction_cache:
            self.prediction_cache.move_to_end(key)
            return self.prediction_cache[key]
        return None
    
    def set_prediction(self, chart_hash: str, context_hash: str, prediction: bool, confidence: float):
        key = f"{chart_hash}_{context_hash}"
        self.prediction_cache[key] = (prediction, confidence)
        self.prediction_cache.move_to_end(key)
        
        if len(self.prediction_cache) > self.max_size:
            self.prediction_cache.popitem(last=False)
    
    def clear(self):
        self.similarity_cache.clear()
        self.feature_cache.clear()
        self.prediction_cache.clear()

