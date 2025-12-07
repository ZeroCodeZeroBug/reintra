from flask import Flask, jsonify, render_template_string
from chart_storage import ChartStorage
from api_client import StockAPIClient
import json
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.logger.disabled = True

storage = ChartStorage()
api_client = StockAPIClient()
trading_ai_instance = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body {
            font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
            background: #000;
            color: #fff;
            padding: 16px;
            min-height: 100vh;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            border-bottom: 1px solid #333;
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: normal;
            letter-spacing: 2px;
        }
        
        .header-info {
            display: flex;
            gap: 24px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .stock-display {
            font-size: 22px;
            font-weight: bold;
            color: #4ade80;
        }
        
        .refresh-btn {
            background: transparent;
            color: #fff;
            border: 1px solid #333;
            padding: 8px 16px;
            cursor: pointer;
            font-family: inherit;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.2s;
        }
        
        .refresh-btn:hover {
            background: #fff;
            color: #000;
            border-color: #fff;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1px;
            background: #333;
            margin-bottom: 24px;
            border: 1px solid #333;
        }
        
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 480px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .stat-card {
            background: #000;
            padding: 16px;
            border: 1px solid #333;
            display: flex;
            flex-direction: column;
        }
        
        .stat-label {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        
        .stat-card:nth-child(1) .stat-value { color: #4a9eff; }
        .stat-card:nth-child(2) .stat-value { color: #4ade80; }
        .stat-card:nth-child(3) .stat-value { color: #fbbf24; }
        .stat-card:nth-child(4) .stat-value { color: #fb923c; }
        .stat-card:nth-child(5) .stat-value { color: #a78bfa; }
        .stat-card:nth-child(6) .stat-value { color: #f472b6; }
        
        .section {
            border: 1px solid #333;
            margin-bottom: 24px;
            background: #000;
        }
        
        .section-header {
            border-bottom: 1px solid #333;
            padding: 16px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
        }
        
        .section-content {
            padding: 16px;
        }
        
        .chart-wrapper {
            position: relative;
            height: 400px;
            width: 100%;
            overflow: hidden;
        }
        
        @media (max-width: 768px) {
            .chart-wrapper {
                height: 300px;
            }
        }
        
        .trades-table {
            overflow-x: auto;
        }
        
        .trades-container {
            max-height: 600px;
            overflow-y: auto;
            overflow-x: auto;
        }
        
        .trades-container::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .trades-container::-webkit-scrollbar-track {
            background: #111;
        }
        
        .trades-container::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        
        .trades-container::-webkit-scrollbar-thumb:hover {
            background: #444;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        th {
            background: #111;
            padding: 14px 10px;
            text-align: left;
            font-weight: normal;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 12px;
            color: #888;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
        }
        
        td {
            padding: 14px 10px;
            border-bottom: 1px solid #222;
        }
        
        tr:hover {
            background: #111;
        }
        
        .status-correct {
            color: #4ade80;
            font-weight: bold;
        }
        
        .status-incorrect {
            color: #ef4444;
            font-weight: bold;
        }
        
        .positive {
            color: #4ade80;
            font-weight: bold;
        }
        
        .negative {
            color: #ef4444;
            font-weight: bold;
        }
        
        .prediction {
            font-weight: bold;
            color: #fbbf24;
        }
        
        .confidence {
            font-weight: bold;
            color: #4a9eff;
        }
        
        .result-profit {
            color: #4ade80;
            font-weight: bold;
        }
        
        .result-loss {
            color: #ef4444;
            font-weight: bold;
        }
        
        .next-prediction {
            border: 1px solid #333;
            padding: 16px;
            margin-bottom: 24px;
            background: #000;
        }
        
        .next-prediction-header {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 12px;
        }
        
        .next-prediction-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .prediction-main-row {
            display: flex;
            gap: 24px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .prediction-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .prediction-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
        }
        
        .prediction-value {
            font-size: 18px;
            font-weight: bold;
            color: #fff;
        }
        
        .timeframes-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-top: 8px;
        }
        
        .timeframe-card {
            border: 1px solid #333;
            padding: 12px;
            background: #0a0a0a;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .timeframe-label {
            font-size: 10px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .timeframe-prediction {
            font-size: 16px;
            font-weight: bold;
        }
        
        .timeframe-confidence {
            font-size: 13px;
            color: #888;
        }
        
        .status-icon {
            display: inline-block;
            width: 12px;
            text-align: center;
        }
        
        .highlight-row {
            background: rgba(255, 255, 255, 0.03);
        }
        
        @media (max-width: 1024px) {
            table {
                font-size: 11px;
            }
            
            th, td {
                padding: 8px 6px;
            }
        }
        
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .header-info {
                width: 100%;
                justify-content: space-between;
            }
            
            table {
                font-size: 10px;
            }
            
            th, td {
                padding: 6px 4px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-info">
                <div class="stock-display" id="stockPrice">LOADING...</div>
                <button class="refresh-btn" onclick="location.reload()">REFRESH</button>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Next Prediction</div>
            <div class="section-content">
                <div class="next-prediction-content">
                    <div class="prediction-main-row">
                        <div class="prediction-item">
                            <div class="prediction-label">Direction (10s)</div>
                            <div class="prediction-value" id="nextPrediction">LOADING...</div>
                        </div>
                        <div class="prediction-item">
                            <div class="prediction-label">Confidence</div>
                            <div class="prediction-value" id="nextConfidence">--</div>
                        </div>
                        <div class="prediction-item">
                            <div class="prediction-label">Current Price</div>
                            <div class="prediction-value" id="currentPrice">--</div>
                        </div>
                    </div>
                    <div class="timeframes-container">
                        <div class="timeframe-card">
                            <div class="timeframe-label">1 Hour</div>
                            <div class="timeframe-prediction" id="tf1h">--</div>
                            <div class="timeframe-confidence" id="tf1hConf">--</div>
                        </div>
                        <div class="timeframe-card">
                            <div class="timeframe-label">12 Hours</div>
                            <div class="timeframe-prediction" id="tf12h">--</div>
                            <div class="timeframe-confidence" id="tf12hConf">--</div>
                        </div>
                        <div class="timeframe-card">
                            <div class="timeframe-label">1 Day</div>
                            <div class="timeframe-prediction" id="tf1d">--</div>
                            <div class="timeframe-confidence" id="tf1dConf">--</div>
                        </div>
                        <div class="timeframe-card">
                            <div class="timeframe-label">3 Days</div>
                            <div class="timeframe-prediction" id="tf3d">--</div>
                            <div class="timeframe-confidence" id="tf3dConf">--</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value" id="totalPredictions">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value" id="accuracy">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Rate</div>
                <div class="stat-value" id="profitRate">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Profit</div>
                <div class="stat-value" id="avgProfit">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Charts</div>
                <div class="stat-value" id="totalCharts">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Rating</div>
                <div class="stat-value" id="avgRating">0.00</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Profit / Loss Analysis</div>
            <div class="section-content">
                <div class="chart-wrapper">
                    <canvas id="profitLossChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Accuracy Over Time</div>
            <div class="section-content">
                <div class="chart-wrapper">
                    <canvas id="accuracyChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Recent Trades Simulation (Showing 15, scroll for more)</div>
            <div class="section-content">
                <div class="trades-container">
                    <table>
                        <thead style="position: sticky; top: 0; z-index: 10;">
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Initial</th>
                                <th>Final</th>
                                <th>Change</th>
                                <th>Prediction</th>
                                <th>Result</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="tradesBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        Chart.defaults.color = '#fff';
        Chart.defaults.borderColor = '#333';
        Chart.defaults.backgroundColor = '#000';
        
        let profitLossChart;
        let accuracyChart;
        
        // Test if JavaScript is working
        console.log('Dashboard script loaded');
        
        function safeSetText(id, text) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = text;
            } else {
                console.error(`Element not found: ${id}`);
            }
        }
        
        async function loadData() {
            try {
                console.log('Loading data...');
                
                fetch('/api/recent-trades')
                    .then(r => r.ok ? r.json() : [])
                    .then(trades => {
                        updateTradesTable(trades || []);
                    })
                    .catch(e => {
                        console.error('Recent trades fetch error:', e);
                        updateTradesTable([]);
                    });
                
                const responses = await Promise.all([
                    fetch('/api/statistics').catch(e => { console.error('Stats fetch error:', e); return null; }),
                    fetch('/api/trade-statistics').catch(e => { console.error('Trade stats fetch error:', e); return null; }),
                    fetch('/api/chart-analytics').catch(e => { console.error('Chart analytics fetch error:', e); return null; }),
                    fetch('/api/stock-price').catch(e => { console.error('Stock price fetch error:', e); return null; }),
                    fetch('/api/accuracy-history').catch(e => { console.error('Accuracy history fetch error:', e); return null; })
                ]);
                
                console.log('Fetch responses received:', responses.map((r, i) => ({ index: i, ok: r?.ok, status: r?.status })));
                
                const [stats, tradeStats, chartStats, stock, accuracyHistory] = await Promise.all(
                    responses.map(async (r, i) => {
                        if (!r || !r.ok) {
                            console.error(`Response ${i} failed:`, r?.status, r?.statusText);
                            return null;
                        }
                        try {
                            const data = await r.json();
                            console.log(`Response ${i} success:`, data);
                            return data;
                        } catch (e) {
                            console.error(`JSON parse error for response ${i}:`, e);
                            return null;
                        }
                    })
                );
                
                console.log('Parsed data:', { stats, tradeStats, chartStats, stock });
                
                safeSetText('totalPredictions', (stats?.total_predictions ?? 0) || 0);
                safeSetText('accuracy', ((stats?.accuracy ?? 0) || 0).toFixed(1) + '%');
                safeSetText('profitRate', ((tradeStats?.profit_rate ?? 0) || 0).toFixed(1) + '%');
                safeSetText('avgProfit', ((tradeStats?.avg_profit_percent ?? 0) || 0).toFixed(2) + '%');
                safeSetText('totalCharts', (chartStats?.total_charts ?? 0) || 0);
                safeSetText('avgRating', ((chartStats?.avg_rating ?? 0) || 0).toFixed(2));
                
                if (stock?.price) {
                    safeSetText('stockPrice', `${stock.symbol || 'TSLA'} $${stock.price.toFixed(2)}`);
                } else {
                    safeSetText('stockPrice', 'TSLA --');
                }
                
                fetch('/api/next-prediction').then(r => {
                    console.log('Next prediction response status:', r.status, r.statusText);
                    if (!r.ok) {
                        console.error('Next prediction fetch failed:', r.status, r.statusText);
                        return r.json().then(data => {
                            console.error('Error response data:', data);
                            throw new Error(`Fetch failed: ${r.status}`);
                        }).catch(e => {
                            throw new Error(`Fetch failed: ${r.status}`);
                        });
                    }
                    return r.json();
                }).then(prediction => {
                    console.log('Prediction data:', prediction);
                    if (prediction && (prediction.prediction !== null && prediction.prediction !== undefined)) {
                        const predictionText = prediction.prediction ? 'PROFIT' : 'LOSS';
                        const predictionColor = prediction.prediction ? '#4ade80' : '#ef4444';
                        safeSetText('nextPrediction', predictionText);
                        const predEl = document.getElementById('nextPrediction');
                        if (predEl) predEl.style.color = predictionColor;
                        safeSetText('nextConfidence', (prediction.confidence || 0).toFixed(1) + '%');
                        safeSetText('currentPrice', '$' + (prediction.current_price || 0).toFixed(2));
                    } else {
                        safeSetText('nextPrediction', 'WAITING...');
                        const predEl = document.getElementById('nextPrediction');
                        if (predEl) predEl.style.color = '#888';
                        safeSetText('nextConfidence', '--');
                        safeSetText('currentPrice', '--');
                    }
                    
                    const timeframes = ['1h', '12h', '1d', '3d'];
                    timeframes.forEach(tf => {
                        const tfData = prediction?.timeframes && prediction.timeframes[tf];
                        const predEl = document.getElementById(`tf${tf}`);
                        const confEl = document.getElementById(`tf${tf}Conf`);
                        
                        if (tfData && tfData.prediction !== null && tfData.prediction !== undefined) {
                            const tfText = tfData.prediction ? 'PROFIT' : 'LOSS';
                            const tfColor = tfData.prediction ? '#4ade80' : '#ef4444';
                            if (predEl) {
                                predEl.textContent = tfText;
                                predEl.style.color = tfColor;
                            }
                            if (confEl) {
                                confEl.textContent = (tfData.confidence || 0).toFixed(1) + '%';
                                confEl.style.color = '#888';
                            }
                        } else {
                            if (predEl) {
                                predEl.textContent = '--';
                                predEl.style.color = '#888';
                            }
                            if (confEl) {
                                confEl.textContent = '--';
                            }
                        }
                    });
                }).catch(err => {
                    console.error('Prediction fetch error:', err);
                    console.error('Error details:', err.message, err.stack);
                    safeSetText('nextPrediction', 'WAITING...');
                    const predEl = document.getElementById('nextPrediction');
                    if (predEl) predEl.style.color = '#888';
                    safeSetText('nextConfidence', '--');
                    safeSetText('currentPrice', '--');
                });
                
                fetch('/api/recent-trades')
                    .then(r => {
                        console.log('Chart trades fetch status:', r.status);
                        return r.ok ? r.json() : [];
                    })
                    .then(trades => {
                        console.log('Chart trades received:', trades?.length || 0);
                        updateCharts(trades || [], tradeStats || {});
                        return trades;
                    })
                    .catch(e => {
                        console.error('Recent trades fetch error for charts:', e);
                        updateCharts([], tradeStats || {});
                    });
                
                updateAccuracyChart(accuracyHistory || []);
            } catch (error) {
                console.error('Error loading data:', error);
                console.error('Error stack:', error.stack);
                safeSetText('totalPredictions', 'ERROR');
                safeSetText('accuracy', 'ERROR');
            }
        }
        
        function updateCharts(trades, tradeStats) {
            try {
                console.log('updateCharts called with', trades?.length || 0, 'trades');
                
                if (!trades || !Array.isArray(trades) || trades.length === 0) {
                    console.log('No trades data, clearing chart if exists');
                    if (profitLossChart) {
                        profitLossChart.data.labels = [];
                        profitLossChart.data.datasets[0].data = [];
                        profitLossChart.data.datasets[1].data = [];
                        profitLossChart.update('none');
                    }
                    return;
                }
                
                const maxPoints = 50;
                const sortedTrades = trades.slice(0, maxPoints).reverse();
                console.log('Processing', sortedTrades.length, 'trades for chart');
                
                const profitLineData = [];
                const lossBarData = [];
                const labels = [];
                
                sortedTrades.forEach((trade, idx) => {
                    if (!trade || typeof trade.was_profitable === 'undefined') {
                        return;
                    }
                    labels.push(`#${idx + 1}`);
                    if (trade.was_profitable) {
                        profitLineData.push({ x: idx, y: trade.price_change_percent || 0 });
                        lossBarData.push({ x: idx, y: 0 });
                    } else {
                        profitLineData.push({ x: idx, y: 0 });
                        lossBarData.push({ x: idx, y: Math.abs(trade.price_change_percent || 0) });
                    }
                });
                
                console.log('Generated labels:', labels.length, 'labels');
                
                if (labels.length === 0) {
                    console.log('No valid trades to display, skipping chart update');
                    return;
                }
            
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#111',
                        borderColor: '#333',
                        borderWidth: 1,
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        padding: 12,
                        titleFont: { family: 'monospace', size: 12 },
                        bodyFont: { family: 'monospace', size: 12 },
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null && context.parsed.y !== undefined) {
                                    label += context.parsed.y.toFixed(2) + '%';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#222' },
                        ticks: { 
                            color: '#888', 
                            font: { family: 'monospace', size: 11 },
                            maxTicksLimit: 50
                        },
                        title: {
                            display: true,
                            text: 'Iteration',
                            color: '#888',
                            font: { family: 'monospace', size: 12 }
                        }
                    },
                    y: {
                        grid: { color: '#222' },
                        ticks: { 
                            color: '#888', 
                            font: { family: 'monospace', size: 11 },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        },
                        title: {
                            display: true,
                            text: 'Percentage (%)',
                            color: '#888',
                            font: { family: 'monospace', size: 12 }
                        }
                    }
                },
                onHover: (event, activeElements) => {
                    event.native.target.style.cursor = activeElements.length > 0 ? 'pointer' : 'default';
                }
            };
            
                const profitData = sortedTrades.map(t => {
                    if (!t || typeof t.was_profitable === 'undefined') return null;
                    return t.was_profitable ? (t.price_change_percent || 0) : null;
                });
                
                const lossData = sortedTrades.map(t => {
                    if (!t || typeof t.was_profitable === 'undefined') return null;
                    return !t.was_profitable ? Math.abs(t.price_change_percent || 0) : null;
                });
                
                console.log('Profit data points:', profitData.filter(d => d !== null).length);
                console.log('Loss data points:', lossData.filter(d => d !== null).length);
                
                if (!profitLossChart) {
                    console.log('Creating new profit/loss chart');
                    const canvas = document.getElementById('profitLossChart');
                    if (!canvas) {
                        console.error('Profit/Loss chart canvas not found');
                        return;
                    }
                    const ctx = canvas.getContext('2d');
                    if (!ctx) {
                        console.error('Could not get 2d context for chart');
                        return;
                    }
                    profitLossChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: 'Loss %',
                                    data: lossData,
                                    backgroundColor: '#ef4444',
                                    borderColor: '#ef4444',
                                    borderWidth: 1,
                                    yAxisID: 'y',
                                    order: 1
                                },
                                {
                                    label: 'Profit %',
                                    data: profitData,
                                    type: 'line',
                                    borderColor: '#4ade80',
                                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                                    borderWidth: 2,
                                    pointRadius: 4,
                                    pointBackgroundColor: '#4ade80',
                                    pointBorderColor: '#fff',
                                    pointBorderWidth: 2,
                                    pointHoverRadius: 6,
                                    borderDash: [2, 2],
                                    fill: false,
                                    tension: 0.3,
                                    spanGaps: false,
                                    yAxisID: 'y',
                                    order: 2
                                }
                            ]
                        },
                        options: chartOptions
                    });
                    console.log('Chart created successfully with', labels.length, 'data points');
                } else {
                    try {
                        console.log('Updating existing chart with', labels.length, 'data points');
                        if (profitLossChart.data && profitLossChart.data.datasets && profitLossChart.data.datasets.length >= 2) {
                            profitLossChart.data.labels = labels;
                            profitLossChart.data.datasets[0].data = lossData;
                            profitLossChart.data.datasets[1].data = profitData;
                            profitLossChart.update('none');
                            console.log('Chart updated successfully');
                        } else {
                            console.error('Chart data structure invalid, recreating chart');
                            profitLossChart.destroy();
                            profitLossChart = null;
                            updateCharts(trades, tradeStats);
                        }
                    } catch (updateError) {
                        console.error('Error updating profit/loss chart:', updateError);
                        console.error('Error stack:', updateError.stack);
                        if (profitLossChart) {
                            try {
                                profitLossChart.destroy();
                            } catch (e) {
                                console.error('Error destroying chart:', e);
                            }
                        }
                        profitLossChart = null;
                        setTimeout(() => updateCharts(trades, tradeStats), 100);
                    }
                }
            } catch (error) {
                console.error('Error in updateCharts:', error);
                if (profitLossChart) {
                    try {
                        profitLossChart.destroy();
                    } catch (e) {
                        console.error('Error destroying chart:', e);
                    }
                    profitLossChart = null;
                }
            }
        }
        
        function updateAccuracyChart(accuracyData) {
            if (!accuracyData || accuracyData.length === 0) {
                return;
            }
            
            const sortedData = accuracyData.slice().sort((a, b) => a.timestamp - b.timestamp);
            const labels = [];
            const accuracyValues = [];
            
            sortedData.forEach((point, idx) => {
                labels.push(`#${idx + 1}`);
                accuracyValues.push(point.accuracy);
            });
            
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#111',
                        borderColor: '#333',
                        borderWidth: 1,
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        padding: 12,
                        titleFont: { family: 'monospace', size: 12 },
                        bodyFont: { family: 'monospace', size: 12 },
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(2) + '%';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#222' },
                        ticks: { 
                            color: '#888', 
                            font: { family: 'monospace', size: 11 },
                            maxTicksLimit: 50
                        },
                        title: {
                            display: true,
                            text: 'Iteration',
                            color: '#888',
                            font: { family: 'monospace', size: 12 }
                        }
                    },
                    y: {
                        grid: { color: '#222' },
                        ticks: { 
                            color: '#888', 
                            font: { family: 'monospace', size: 11 },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        },
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            color: '#888',
                            font: { family: 'monospace', size: 12 }
                        }
                    }
                },
                onHover: (event, activeElements) => {
                    event.native.target.style.cursor = activeElements.length > 0 ? 'pointer' : 'default';
                }
            };
            
            if (!accuracyChart) {
                const ctx = document.getElementById('accuracyChart').getContext('2d');
                accuracyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Accuracy',
                                data: accuracyValues,
                                borderColor: '#4a9eff',
                                backgroundColor: 'rgba(74, 158, 255, 0.1)',
                                borderWidth: 2,
                                pointRadius: 4,
                                pointBackgroundColor: '#4a9eff',
                                pointBorderColor: '#fff',
                                pointBorderWidth: 2,
                                pointHoverRadius: 6,
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y'
                            }
                        ]
                    },
                    options: chartOptions
                });
            } else {
                accuracyChart.data.labels = labels;
                accuracyChart.data.datasets[0].data = accuracyValues;
                accuracyChart.update('none');
            }
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('tradesBody');
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #888; padding: 24px;">NO TRADES YET</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.map((trade, index) => {
                const displayClass = index < 15 ? 'highlight-row' : '';
                const date = new Date(trade.timestamp * 1000);
                const changeClass = trade.price_change_percent >= 0 ? 'positive' : 'negative';
                const resultClass = trade.was_profitable ? 'result-profit' : 'result-loss';
                
                return `
                    <tr class="${displayClass}">
                        <td>${date.toLocaleTimeString()}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.initial_price.toFixed(2)}</td>
                        <td>${trade.final_price.toFixed(2)}</td>
                        <td class="${changeClass}">${trade.price_change_percent >= 0 ? '+' : ''}${trade.price_change_percent.toFixed(2)}%</td>
                        <td class="prediction">${trade.predicted_profitable ? 'PROFIT' : 'LOSS'}</td>
                        <td class="${resultClass}">${trade.was_profitable ? 'PROFIT' : 'LOSS'}</td>
                        <td class="confidence">${(trade.confidence || 0).toFixed(1)}%</td>
                    </tr>
                `;
            }).join('');
        }
        
        function loadTradesImmediately() {
            fetch('/api/recent-trades')
                .then(r => {
                    if (!r.ok) {
                        console.error('Recent trades API error:', r.status, r.statusText);
                        return [];
                    }
                    return r.json();
                })
                .then(trades => {
                    console.log('Trades received:', trades?.length || 0, trades);
                    if (Array.isArray(trades)) {
                        updateTradesTable(trades);
                        fetch('/api/trade-statistics')
                            .then(r => r.ok ? r.json() : {})
                            .then(tradeStats => {
                                updateCharts(trades, tradeStats);
                            })
                            .catch(e => {
                                console.error('Trade stats error:', e);
                                updateCharts(trades, {});
                            });
                    } else {
                        console.error('Trades is not an array:', trades);
                        updateTradesTable([]);
                    }
                })
                .catch(e => {
                    console.error('Recent trades fetch error:', e);
                    updateTradesTable([]);
                });
        }
        
        // Start loading when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM ready, starting interval');
                loadTradesImmediately();
                loadData();
                setInterval(loadTradesImmediately, 1000);
                setInterval(loadData, 1000);
            });
        } else {
            console.log('DOM already ready, starting immediately');
            loadTradesImmediately();
            loadData();
            setInterval(loadTradesImmediately, 1000);
            setInterval(loadData, 1000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    print("[API] Serving dashboard page")
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/test')
def test():
    return jsonify({'status': 'ok', 'trading_ai_set': trading_ai_instance is not None})

@app.route('/api/statistics')
def get_statistics():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            storage_instance = trading_ai_instance.storage
        else:
            storage_instance = storage
        
        total, correct = storage_instance.get_statistics()
        accuracy = (correct / total * 100) if total > 0 else 0
        result = {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': accuracy
        }
        #print(f"[API] /api/statistics: {result}")
        return jsonify(result)
    except Exception as e:
        #print(f"[API ERROR] /api/statistics: {e}")
        return jsonify({'total_predictions': 0, 'correct_predictions': 0, 'accuracy': 0}), 500

@app.route('/api/trade-statistics')
def get_trade_statistics():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            storage_instance = trading_ai_instance.storage
        else:
            storage_instance = storage
        result = storage_instance.get_trade_statistics()
        #print(f"[API] /api/trade-statistics: {result}")
        return jsonify(result)
    except Exception as e:
        #print(f"[API ERROR] /api/trade-statistics: {e}")
        return jsonify({'profit_rate': 0, 'avg_profit_percent': 0, 'avg_loss_percent': 0}), 500

@app.route('/api/chart-analytics')
def get_chart_analytics():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            storage_instance = trading_ai_instance.storage
        else:
            storage_instance = storage
        return jsonify(storage_instance.get_chart_analytics())
    except Exception as e:
        return jsonify({'total_charts': 0, 'avg_rating': 0, 'profit_charts': 0, 'non_profit_charts': 0}), 500

@app.route('/api/recent-trades')
def get_recent_trades():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            storage_instance = trading_ai_instance.storage
        else:
            storage_instance = storage
        
        trades = storage_instance.get_recent_trades(50)
        #print(f"[API] /api/recent-trades: Returning {len(trades)} trades")
        return jsonify(trades)
    except Exception as e:
        print(f"[API ERROR] /api/recent-trades: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 500

@app.route('/api/stock-price')
def get_stock_price():
    try:
        symbol = "TSLA"
        global trading_ai_instance
        if trading_ai_instance:
            price = trading_ai_instance.api_client.get_current_price(symbol)
        else:
            price = api_client.get_current_price(symbol)
        result = {'symbol': symbol, 'price': price}
        #print(f"[API] /api/stock-price: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"[API ERROR] /api/stock-price: {e}")
        return jsonify({'symbol': symbol, 'price': None}), 500

@app.route('/api/accuracy-history')
def get_accuracy_history():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            storage_instance = trading_ai_instance.storage
        else:
            storage_instance = storage
        
        cursor = storage_instance.conn.cursor()
        cursor.execute('''
            SELECT timestamp, accuracy_at_time 
            FROM trades 
            WHERE accuracy_at_time IS NOT NULL
            ORDER BY timestamp ASC
        ''')
        
        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                'timestamp': row['timestamp'],
                'accuracy': row['accuracy_at_time']
            })
        
        return jsonify(history)
    except Exception as e:
        print(f"[API ERROR] /api/accuracy-history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 500

@app.route('/api/next-prediction')
def get_next_prediction():
    try:
        global trading_ai_instance
        if trading_ai_instance:
            try:
                prediction_data = trading_ai_instance.get_current_prediction()
                
                json_data = {
                    'prediction': bool(prediction_data.get('prediction')) if prediction_data.get('prediction') is not None else None,
                    'confidence': float(prediction_data.get('confidence')) if prediction_data.get('confidence') is not None else None,
                    'current_price': float(prediction_data.get('current_price')) if prediction_data.get('current_price') is not None else None,
                    'timeframes': {}
                }
                
                timeframes = prediction_data.get('timeframes', {})
                for tf in ['1h', '12h', '1d', '3d']:
                    tf_data = timeframes.get(tf)
                    if tf_data:
                        json_data['timeframes'][tf] = {
                            'prediction': bool(tf_data.get('prediction')) if tf_data.get('prediction') is not None else None,
                            'confidence': float(tf_data.get('confidence')) if tf_data.get('confidence') is not None else None,
                            'timeframe': str(tf_data.get('timeframe', tf))
                        }
                    else:
                        json_data['timeframes'][tf] = None
                
                return jsonify(json_data)
            except Exception as e:
                print(f"[API ERROR] /api/next-prediction - get_current_prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        default_response = {
            'prediction': None,
            'confidence': None,
            'current_price': None,
            'timeframes': {
                '1h': None,
                '12h': None,
                '1d': None,
                '3d': None
            }
        }
        return jsonify(default_response)
    except Exception as e:
        print(f"[API ERROR] /api/next-prediction exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'prediction': None,
            'confidence': None,
            'current_price': None,
            'timeframes': {
                '1h': None,
                '12h': None,
                '1d': None,
                '3d': None
            }
        }), 500

def set_trading_ai(ai_instance):
    global trading_ai_instance
    trading_ai_instance = ai_instance

if __name__ == '__main__':
    app.run(debug=False, port=5000)
