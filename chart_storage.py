import sqlite3
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Chart:
    chart_data: List[float]
    category: str
    rating: float
    timestamp: float
    symbol: str
    chart_id: Optional[int] = None

class ChartStorage:
    def __init__(self, db_file: str = "trading_ai.db"):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA cache_size=-64000')
        self.conn.execute('PRAGMA temp_store=MEMORY')
        self.conn.execute('PRAGMA mmap_size=268435456')
        self.conn.execute('PRAGMA optimize')
        
        self._prepare_statements()
        self.init_database()
    
    def _prepare_statements(self):
        pass
    
    def init_database(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS charts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chart_data TEXT NOT NULL,
                category TEXT NOT NULL,
                rating REAL NOT NULL DEFAULT 1.0,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                initial_price REAL NOT NULL,
                final_price REAL NOT NULL,
                predicted_profitable INTEGER NOT NULL,
                was_profitable INTEGER NOT NULL,
                price_change_percent REAL NOT NULL,
                confidence REAL,
                matched_chart_id INTEGER,
                similarity REAL,
                timestamp REAL NOT NULL,
                chart_data TEXT NOT NULL,
                accuracy_at_time REAL
            )
        ''')
        
        try:
            cursor.execute('ALTER TABLE trades ADD COLUMN accuracy_at_time REAL')
        except sqlite3.OperationalError:
            pass
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_charts_rating ON charts(rating DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_charts_category ON charts(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_charts_timestamp ON charts(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_profitable ON trades(was_profitable, predicted_profitable)')
        
        cursor.execute('SELECT COUNT(*) FROM statistics')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO statistics (id, total_predictions, correct_predictions) VALUES (1, 0, 0)')
        
        self.conn.commit()
        
        self.pending_operations = []
        self.batch_size = 5
    
    def add_chart(self, chart_data: List[float], category: str, symbol: str, initial_rating: float = 1.0):
        chart_json = json.dumps(chart_data)
        timestamp = datetime.now().timestamp()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO charts (chart_data, category, rating, timestamp, symbol)
            VALUES (?, ?, ?, ?, ?)
        ''', (chart_json, category, initial_rating, timestamp, symbol))
        
        chart_id = cursor.lastrowid
        self.pending_operations.append('chart')
        
        if len(self.pending_operations) >= self.batch_size:
            self.conn.commit()
            self.pending_operations.clear()
        
        chart = Chart(
            chart_data=chart_data,
            category=category,
            rating=initial_rating,
            timestamp=timestamp,
            symbol=symbol,
            chart_id=chart_id
        )
        return chart
    
    def get_charts_by_category(self, category: str) -> List[Chart]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM charts WHERE category = ?', (category,))
        rows = cursor.fetchall()
        
        charts = []
        for row in rows:
            chart_data = json.loads(row['chart_data'])
            chart = Chart(
                chart_id=row['id'],
                chart_data=chart_data,
                category=row['category'],
                rating=row['rating'],
                timestamp=row['timestamp'],
                symbol=row['symbol']
            )
            charts.append(chart)
        
        return charts
    
    def get_all_charts_sorted_by_rating(self, limit: Optional[int] = None) -> List[Chart]:
        cursor = self.conn.cursor()
        if limit:
            cursor.execute('SELECT * FROM charts ORDER BY rating DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT * FROM charts ORDER BY rating DESC')
        rows = cursor.fetchall()
        
        charts = []
        for row in rows:
            chart_data = json.loads(row['chart_data'])
            chart = Chart(
                chart_id=row['id'],
                chart_data=chart_data,
                category=row['category'],
                rating=row['rating'],
                timestamp=row['timestamp'],
                symbol=row['symbol']
            )
            charts.append(chart)
        
        return charts
    
    def update_chart_rating(self, chart: Chart, rating_change: float):
        if chart.chart_id is None:
            return
        
        new_rating = max(0.1, min(5.0, chart.rating + rating_change))
        chart.rating = new_rating
        
        cursor = self.conn.cursor()
        cursor.execute('UPDATE charts SET rating = ? WHERE id = ?', (new_rating, chart.chart_id))
        self.pending_operations.append('rating')
        
        if len(self.pending_operations) >= self.batch_size:
            self.conn.commit()
            self.pending_operations.clear()
    
    def find_chart_by_id(self, chart_id: int) -> Optional[Chart]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM charts WHERE id = ?', (chart_id,))
        row = cursor.fetchone()
        
        if row:
            chart_data = json.loads(row['chart_data'])
            return Chart(
                chart_id=row['id'],
                chart_data=chart_data,
                category=row['category'],
                rating=row['rating'],
                timestamp=row['timestamp'],
                symbol=row['symbol']
            )
        return None
    
    def get_statistics(self) -> tuple:
        cursor = self.conn.cursor()
        cursor.execute('SELECT total_predictions, correct_predictions FROM statistics WHERE id = 1')
        row = cursor.fetchone()
        if row:
            return row['total_predictions'], row['correct_predictions']
        return 0, 0
    
    def update_statistics(self, total_predictions: int, correct_predictions: int):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE statistics 
            SET total_predictions = ?, correct_predictions = ?
            WHERE id = 1
        ''', (total_predictions, correct_predictions))
        self.conn.commit()
    
    def add_trade(self, symbol: str, initial_price: float, final_price: float,
                  predicted_profitable: bool, was_profitable: bool,
                  price_change_percent: float, confidence: float,
                  matched_chart_id: Optional[int], similarity: float,
                  chart_data: List[float], accuracy_at_time: Optional[float] = None):
        chart_json = json.dumps(chart_data)
        timestamp = datetime.now().timestamp()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, initial_price, final_price, predicted_profitable,
                               was_profitable, price_change_percent, confidence,
                               matched_chart_id, similarity, timestamp, chart_data, accuracy_at_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, initial_price, final_price, 1 if predicted_profitable else 0,
              1 if was_profitable else 0, price_change_percent, confidence,
              matched_chart_id, similarity, timestamp, chart_json, accuracy_at_time))
        
        trade_id = cursor.lastrowid
        self.pending_operations.append('trade')
        
        if len(self.pending_operations) >= self.batch_size:
            self.conn.commit()
            self.pending_operations.clear()
        
        return trade_id
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                try:
                    chart_data_str = row['chart_data'] if 'chart_data' in row.keys() else None
                    chart_data = json.loads(chart_data_str) if chart_data_str else []
                except:
                    chart_data = []
                
                confidence = float(row['confidence']) if 'confidence' in row.keys() and row['confidence'] is not None else 0.0
                matched_chart_id = row['matched_chart_id'] if 'matched_chart_id' in row.keys() else None
                similarity = float(row['similarity']) if 'similarity' in row.keys() and row['similarity'] is not None else 0.0
                accuracy_at_time = float(row['accuracy_at_time']) if 'accuracy_at_time' in row.keys() and row['accuracy_at_time'] is not None else None
                
                trades.append({
                    'id': row['id'],
                    'symbol': row['symbol'],
                    'initial_price': float(row['initial_price']),
                    'final_price': float(row['final_price']),
                    'predicted_profitable': bool(row['predicted_profitable']),
                    'was_profitable': bool(row['was_profitable']),
                    'price_change_percent': float(row['price_change_percent']),
                    'confidence': confidence,
                    'matched_chart_id': matched_chart_id,
                    'similarity': similarity,
                    'timestamp': float(row['timestamp']),
                    'chart_data': chart_data,
                    'accuracy_at_time': accuracy_at_time
                })
            return trades
        except Exception as e:
            print(f"Error in get_recent_trades: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_trade_statistics(self) -> Dict:
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN predicted_profitable = was_profitable THEN 1 ELSE 0 END) as correct_predictions,
                SUM(CASE WHEN was_profitable = 1 THEN 1 ELSE 0 END) as profitable_trades,
                AVG(CASE WHEN was_profitable = 1 THEN price_change_percent ELSE NULL END) as avg_profit,
                AVG(CASE WHEN was_profitable = 0 THEN ABS(price_change_percent) ELSE NULL END) as avg_loss,
                SUM(CASE WHEN predicted_profitable = 1 THEN 1 ELSE 0 END) as predicted_profit_count
            FROM trades
        ''')
        
        row = cursor.fetchone()
        
        total_trades = row['total_trades'] or 0
        correct_trades = row['correct_predictions'] or 0
        profitable_trades = row['profitable_trades'] or 0
        avg_profit = row['avg_profit'] or 0
        avg_loss = row['avg_loss'] or 0
        predicted_profit = row['predicted_profit_count'] or 0
        
        return {
            'total_trades': total_trades,
            'correct_predictions': correct_trades,
            'accuracy': (correct_trades / total_trades * 100) if total_trades > 0 else 0,
            'profitable_trades': profitable_trades,
            'profit_rate': (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_profit_percent': avg_profit,
            'avg_loss_percent': avg_loss,
            'predicted_profit_count': predicted_profit
        }
    
    def get_chart_analytics(self) -> Dict:
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_charts,
                SUM(CASE WHEN category = "profit" THEN 1 ELSE 0 END) as profit_charts,
                SUM(CASE WHEN category = "non_profit" THEN 1 ELSE 0 END) as non_profit_charts,
                AVG(rating) as avg_rating,
                MAX(rating) as max_rating,
                AVG(CASE WHEN category = "profit" THEN rating ELSE NULL END) as avg_profit_rating,
                AVG(CASE WHEN category = "non_profit" THEN rating ELSE NULL END) as avg_non_profit_rating
            FROM charts
        ''')
        
        row = cursor.fetchone()
        
        return {
            'total_charts': row['total_charts'] or 0,
            'profit_charts': row['profit_charts'] or 0,
            'non_profit_charts': row['non_profit_charts'] or 0,
            'avg_rating': row['avg_rating'] or 0,
            'max_rating': row['max_rating'] or 0,
            'avg_profit_rating': row['avg_profit_rating'] or 0,
            'avg_non_profit_rating': row['avg_non_profit_rating'] or 0
        }
    
    def commit(self):
        self.conn.commit()
    
    def close(self):
        self.conn.commit()
        self.conn.close()

