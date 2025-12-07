import yfinance as yf
import time
from typing import List, Optional

class StockAPIClient:
    def __init__(self, api_key: Optional[str] = None):
        pass
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            info = ticker.info
            if 'currentPrice' in info:
                return float(info['currentPrice'])
            elif 'regularMarketPrice' in info:
                return float(info['regularMarketPrice'])
            elif 'previousClose' in info:
                return float(info['previousClose'])
            
            return None
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_price_history(self, symbol: str, duration_seconds: int = 20) -> List[float]:
        prices = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            price = self.get_current_price(symbol)
            if price is not None:
                prices.append(price)
            time.sleep(1)
        
        if not prices:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                prices = hist['Close'].tail(duration_seconds).tolist()
                prices = [float(p) for p in prices]
        
        return prices
    
    def get_historical_data(self, symbol: str, period: str = "1d", interval: str = "1m", max_points: int = 100) -> List[float]:
        """Get historical price data for different timeframes"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if not hist.empty:
                prices = hist['Close'].tail(max_points).tolist()
                return [float(p) for p in prices]
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
        
        return []

