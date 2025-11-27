import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Chart from './components/Chart'
import './App.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [status, setStatus] = useState(null)
  const [trades, setTrades] = useState([])
  const [equity, setEquity] = useState([])
  const [livePrice, setLivePrice] = useState(null)
  const [isStarting, setIsStarting] = useState(false)
  const [symbol, setSymbol] = useState('BTC/USDT')

  // Polling pour les données
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Convertir BTC/USDT en BTC/USDT pour l'API price
        const [base, quote] = symbol.split('/')
        const [statusRes, tradesRes, equityRes, priceRes] = await Promise.all([
          axios.get(`${API_BASE}/api/status`),
          axios.get(`${API_BASE}/api/trades?limit=20`),
          axios.get(`${API_BASE}/api/equity`),
          axios.get(`${API_BASE}/api/price/${base}/${quote}`)
        ])
        setStatus(statusRes.data)
        setTrades(tradesRes.data)
        setEquity(equityRes.data.data)
        if (priceRes.data.price) {
          setLivePrice(priceRes.data)
        }
      } catch (err) {
        console.error('Erreur fetch:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 2000) // Refresh toutes les 2 secondes
    return () => clearInterval(interval)
  }, [symbol])

  const startBot = async () => {
    setIsStarting(true)
    try {
      const res = await axios.post(`${API_BASE}/api/start`, {
        symbol,
        mode: 'paper'
      })
      console.log(res.data.message)
    } catch (err) {
      console.error('Erreur démarrage:', err)
    } finally {
      setIsStarting(false)
    }
  }

  const stopBot = async () => {
    try {
      const res = await axios.post(`${API_BASE}/api/stop`)
      console.log(res.data.message)
    } catch (err) {
      console.error('Erreur arrêt:', err)
    }
  }

  const formatPnL = (pnl) => {
    const num = parseFloat(pnl) || 0
    const sign = num >= 0 ? '+' : ''
    return `${sign}$${num.toFixed(2)}`
  }

  // Formater le temps du trade (heure réelle)
  const formatTradeTime = (timestamp) => {
    try {
      const date = new Date(timestamp)
      const now = new Date()
      const diffMs = now - date
      const diffMin = Math.floor(diffMs / 60000)
      
      if (diffMin < 1) return "À l'instant"
      if (diffMin < 60) return `Il y a ${diffMin} min`
      
      const diffHours = Math.floor(diffMin / 60)
      if (diffHours < 24) return `Il y a ${diffHours}h`
      
      return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    } catch {
      return "—"
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>⚡ Trading Bot Dashboard</h1>
        <div className="header-status">
          <div className="status-badge" data-running={status?.running}>
            {status?.running ? '🟢 EN COURS' : '🔴 ARRÊTÉ'}
          </div>
          <div className="status-badge" data-connected={status?.connected_to_exchange}>
            {status?.connected_to_exchange ? '🔗 Binance Connecté' : '⚪ Non connecté'}
          </div>
          <div className="mode-badge" data-mode={status?.env?.toLowerCase()}>
            {status?.env === 'LIVE' ? '🔴 LIVE' : status?.env === 'PAPER' ? '📝 PAPER' : '🔧 DEV'}
          </div>
        </div>
      </header>

      {/* Prix Live Banner */}
      {livePrice && (
        <div className="live-price-banner">
          <div className="price-main">
            <span className="price-symbol">{symbol}</span>
            <span className="price-value">${livePrice.price?.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            <span className={`price-change ${livePrice.change_24h >= 0 ? 'positive' : 'negative'}`}>
              {livePrice.change_24h >= 0 ? '▲' : '▼'} {Math.abs(livePrice.change_24h || 0).toFixed(2)}%
            </span>
          </div>
          <div className="price-details">
            <span>24h High: ${livePrice.high_24h?.toLocaleString()}</span>
            <span>24h Low: ${livePrice.low_24h?.toLocaleString()}</span>
            <span>Vol: {(livePrice.volume_24h || 0).toFixed(2)} BTC</span>
          </div>
        </div>
      )}

      <div className="container">
        {/* Contrôles */}
        <div className="controls-card">
          <h2>🎮 Contrôles</h2>
          <div className="controls">
            <div className="input-group">
              <label>Symbole</label>
              <select value={symbol} onChange={(e) => setSymbol(e.target.value)} disabled={status?.running}>
                <option value="BTC/USDT">BTC/USDT</option>
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
              </select>
            </div>
            <div className="button-group">
              <button 
                onClick={startBot} 
                disabled={status?.running || isStarting}
                className="btn btn-start"
              >
                {isStarting ? '⏳ Démarrage...' : '▶️ Démarrer'}
              </button>
              <button 
                onClick={stopBot} 
                disabled={!status?.running}
                className="btn btn-stop"
              >
                ⏹️ Arrêter
              </button>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Balance</div>
            <div className="stat-value">${(status?.balance || 0).toFixed(2)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">PnL Total</div>
            <div className="stat-value" data-positive={status?.pnl >= 0}>
              {formatPnL(status?.pnl)}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Trades</div>
            <div className="stat-value">{status?.total_trades || 0}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Win Rate</div>
            <div className="stat-value">{(status?.win_rate || 0).toFixed(1)}%</div>
          </div>
        </div>

        {/* Position actuelle */}
        {status?.current_position && (
          <div className="position-card">
            <h3>📊 Position Actuelle</h3>
            <div className="position-details">
              <span className={`position-side ${status.current_position.side}`}>
                {status.current_position.side.toUpperCase()}
              </span>
              <span>Prix d'entrée: ${status.current_position.entry_price.toFixed(2)}</span>
              <span>Quantité: {status.current_position.quantity.toFixed(6)}</span>
            </div>
          </div>
        )}

        {/* Graphique TradingView - Prix */}
        <Chart 
          symbol={symbol}
          apiBaseUrl={API_BASE}
          height={450}
          showVolume={true}
          showSMA={true}
          refreshInterval={2000}
        />

        {/* Graphique Equity */}
        {equity.length > 0 && (
          <div className="chart-card">
            <h2>📈 Courbe d'Equity</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={equity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2e45" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#8884d8"
                  tick={false}
                />
                <YAxis stroke="#8884d8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1e35', border: '1px solid #2a2e45' }}
                  labelStyle={{ color: '#ffffff' }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="equity" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
                  name="Balance"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Historique des trades */}
        <div className="trades-card">
          <h2>📜 Historique des Trades</h2>
          <div className="trades-table">
            {trades.length === 0 ? (
              <div className="no-trades">Aucun trade pour le moment...</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Heure</th>
                    <th>Symbole</th>
                    <th>Direction</th>
                    <th>Prix</th>
                    <th>Quantité</th>
                    <th>PnL</th>
                    <th>Raison</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.slice().reverse().map((trade, idx) => (
                    <tr key={idx}>
                      <td>{formatTradeTime(trade.timestamp)}</td>
                      <td>{trade.symbol}</td>
                      <td>
                        <span className={`trade-side ${trade.side.includes('long') ? 'long' : 'short'}`}>
                          {trade.side.toUpperCase()}
                        </span>
                      </td>
                      <td>${trade.price.toFixed(2)}</td>
                      <td>{trade.quantity.toFixed(6)}</td>
                      <td data-positive={trade.pnl >= 0}>
                        {trade.pnl !== null ? formatPnL(trade.pnl) : '-'}
                      </td>
                      <td className="trade-reason">{trade.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
