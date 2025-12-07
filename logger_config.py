import logging
import sys
from datetime import datetime
from config import Config

class TradingLogger:
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingLogger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self):
        self._logger = logging.getLogger('TradingAI')
        self._logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, Config.LOG_LEVEL))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def get_logger(self):
        return self._logger

def get_logger():
    return TradingLogger().get_logger()

