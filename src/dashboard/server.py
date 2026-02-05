"""
Web Dashboard for Live Trading Monitoring.
FastAPI + WebSocket for real-time updates.
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

app = FastAPI(title="Trading Dashboard", version="1.0.0")


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# Store for real-time data
class TradingState:
    def __init__(self):
        self.candles: List[dict] = []
        self.signals: List[dict] = []
        self.trades: List[dict] = []
        self.equity: List[dict] = []
        self.current_position: Optional[dict] = None
        self.balance: float = 10000.0
        self.pnl: float = 0.0
        self.last_price: float = 0.0
        
    def to_dict(self) -> dict:
        return {
            "candles": self.candles[-100:],  # Last 100
            "signals": self.signals[-50:],
            "trades": self.trades[-20:],
            "equity": self.equity[-100:],
            "current_position": self.current_position,
            "balance": self.balance,
            "pnl": self.pnl,
            "last_price": self.last_price,
            "timestamp": datetime.utcnow().isoformat()
        }


state = TradingState()


# Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        h1 { 
            font-size: 1.8em;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status { 
            display: flex; 
            gap: 10px; 
            align-items: center;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h3 { color: #888; font-size: 0.9em; margin-bottom: 8px; }
        .card .value { font-size: 1.8em; font-weight: bold; }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        .chart-container {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
        }
        .trades-table {
            width: 100%;
            border-collapse: collapse;
        }
        .trades-table th, .trades-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .trades-table th { color: #888; font-weight: normal; }
        .signal-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .signal-long { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .signal-short { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .signal-flat { background: rgba(156, 163, 175, 0.2); color: #9ca3af; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Trading Bot Dashboard</h1>
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h3>BALANCE</h3>
                <div class="value" id="balance">$10,000.00</div>
            </div>
            <div class="card">
                <h3>P&L</h3>
                <div class="value" id="pnl">$0.00</div>
            </div>
            <div class="card">
                <h3>LAST PRICE</h3>
                <div class="value" id="price">$0.00</div>
            </div>
            <div class="card">
                <h3>POSITION</h3>
                <div class="value" id="position">FLAT</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="priceChart"></canvas>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">Recent Signals</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Price</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody id="signalsTable"></tbody>
            </table>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        let chart = null;
        
        ws.onopen = () => {
            document.getElementById('statusDot').style.background = '#22c55e';
            document.getElementById('statusText').textContent = 'Connected';
        };
        
        ws.onclose = () => {
            document.getElementById('statusDot').style.background = '#ef4444';
            document.getElementById('statusText').textContent = 'Disconnected';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('balance').textContent = '$' + data.balance.toLocaleString('en-US', {minimumFractionDigits: 2});
            
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = (data.pnl >= 0 ? '+' : '') + '$' + data.pnl.toLocaleString('en-US', {minimumFractionDigits: 2});
            pnlEl.className = 'value ' + (data.pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('price').textContent = '$' + data.last_price.toLocaleString('en-US', {minimumFractionDigits: 2});
            
            const posEl = document.getElementById('position');
            if (data.current_position) {
                posEl.textContent = data.current_position.side.toUpperCase();
                posEl.className = 'value ' + (data.current_position.side === 'long' ? 'positive' : 'negative');
            } else {
                posEl.textContent = 'FLAT';
                posEl.className = 'value';
            }
            
            // Update chart
            if (data.candles && data.candles.length > 0) {
                updateChart(data.candles);
            }
            
            // Update signals table
            if (data.signals) {
                updateSignalsTable(data.signals);
            }
        }
        
        function updateChart(candles) {
            const labels = candles.map(c => new Date(c.timestamp).toLocaleTimeString());
            const prices = candles.map(c => c.close);
            
            if (!chart) {
                const ctx = document.getElementById('priceChart').getContext('2d');
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Price',
                            data: prices,
                            borderColor: '#3a7bd5',
                            backgroundColor: 'rgba(58, 123, 213, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { 
                                grid: { color: 'rgba(255,255,255,0.05)' },
                                ticks: { color: '#888' }
                            },
                            y: { 
                                grid: { color: 'rgba(255,255,255,0.05)' },
                                ticks: { color: '#888' }
                            }
                        }
                    }
                });
            } else {
                chart.data.labels = labels;
                chart.data.datasets[0].data = prices;
                chart.update('none');
            }
        }
        
        function updateSignalsTable(signals) {
            const tbody = document.getElementById('signalsTable');
            tbody.innerHTML = signals.slice(-10).reverse().map(s => `
                <tr>
                    <td>${new Date(s.timestamp).toLocaleTimeString()}</td>
                    <td>$${s.price?.toLocaleString('en-US', {minimumFractionDigits: 2}) || '-'}</td>
                    <td><span class="signal-badge signal-${s.direction.toLowerCase()}">${s.direction}</span></td>
                    <td>${(s.confidence * 100).toFixed(0)}%</td>
                    <td>${s.reason}</td>
                </tr>
            `).join('');
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_json(state.to_dict())
        
        while True:
            # Keep connection alive, data pushed from trading loop
            await asyncio.sleep(1)
            await websocket.send_json(state.to_dict())
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/state")
async def get_state():
    return state.to_dict()


def run_dashboard(host: str = "127.0.0.1", port: int = 8888):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
