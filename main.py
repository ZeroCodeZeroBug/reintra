from trading_ai import TradingAI
import time
import threading
from web_server import app, set_trading_ai

def run_web_server():
    app.run(debug=False, port=5000, use_reloader=False)

def main():
    api_key = None
    symbol = "TSLA"
    
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()
    
    print("Reinforcement Learning Trading AI")
    print("*" * 40)
    print(f"Trading symbol: {symbol}")
    print(f"Prediction window: 10 seconds")
    print(f"Web Dashboard: http://localhost:5000")
    print("*" * 40)
    
    ai = TradingAI(api_key=api_key)
    set_trading_ai(ai)
    
    print("Initializing timeframe predictions...")
    ai.update_timeframe_predictions(symbol)
    
    try:
        while True:
            ai.trade(symbol)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user")
        print(f"Final statistics:")
        print(f"  Total predictions: {ai.total_predictions}")
        print(f"  Correct predictions: {ai.correct_predictions}")
        if ai.total_predictions > 0:
            print(f"  Accuracy-> {(ai.correct_predictions/ai.total_predictions)*100:.2f}%")
        
        all_charts = ai.storage.get_all_charts_sorted_by_rating()
        profit_charts = ai.storage.get_charts_by_category('profit')
        non_profit_charts = ai.storage.get_charts_by_category('non_profit')
        
        print(f"  Stored charts: {len(all_charts)}")
        print(f"  Profit charts: {len(profit_charts)}")
        print(f"  Non-profit charts: {len(non_profit_charts)}")
        
        ai.storage.close()

if __name__ == "__main__":
    main()

