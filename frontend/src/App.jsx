import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
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

  // Fetch data loop
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [base, quote] = symbol.split('/')
        const [statusRes, tradesRes, equityRes, priceRes] = await Promise.all([
          axios.get(`${API_BASE}/api/status`),
          axios.get(`${API_BASE}/api/trades?limit=50`),
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
        console.error('Data fetch error:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 1000)
    return () => clearInterval(interval)
  }, [symbol])

  const startBot = async () => {
    setIsStarting(true)
    try {
      await axios.post(`${API_BASE}/api/start`, { symbol, mode: 'paper' })
    } catch (err) {
      console.error('Start error:', err)
    } finally {
      setIsStarting(false)
    }
  }

  const stopBot = async () => {
    try {
      await axios.post(`${API_BASE}/api/stop`)
    } catch (err) {
      console.error('Stop error:', err)
    }
  }

  const formatMoney = (val) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(val || 0)
  }

  const formatPct = (val) => {
    return `${(val || 0).toFixed(2)}%`
  }

  const formatTime = (ts) => {
    return new Date(ts).toLocaleTimeString()
  }

  return (
    <div className="app">
      {/* Header Bar */}
      <header className="header">
        <h1>
          <span style={{color: '#f0b90b'}}>⚡</span> AlgoTrader Pro
        </h1>
        <div className="header-status">
          <div className="status-badge" data-running={status?.running}>
            {status?.running ? '● Running' : '● Stopped'}
          </div>
          <div className="status-badge" data-connected={status?.connected_to_exchange}>
            {status?.connected_to_exchange ? 'Exchange Connected' : 'Exchange Offline'}
          </div>
          <div className="status-badge" data-connected="false">
            {status?.env || 'DEV'}
          </div>
        </div>
      </header>

      <div className="container">
        {/* Left Sidebar: Controls & Stats */}
        <aside className="sidebar">
          {/* Controls Section */}
          <div className="sidebar-section">
            <h3>Configuration</h3>
            <div className="control-group">
              <label>Trading Pair</label>
              <select 
                className="select-input"
                value={symbol} 
                onChange={(e) => setSymbol(e.target.value)} 
                disabled={status?.running}
              >
                <option value="BTC/USDT">BTC/USDT</option>
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
              </select>
            </div>
            <div className="btn-group">
              <button 
                onClick={startBot} 
                disabled={status?.running || isStarting}
                className="btn btn-primary"
              >
                {isStarting ? 'Starting...' : 'Start Engine'}
              </button>
              <button 
                onClick={stopBot} 
                disabled={!status?.running}
                className="btn btn-danger"
              >
                Stop Engine
              </button>
            </div>
          </div>

          {/* Wallet Stats */}
          <div className="sidebar-section">
            <h3>Portfolio</h3>
            <div className="stat-row">
              <span className="stat-label">Total Balance</span>
              <span className="stat-value">{formatMoney(status?.balance)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Unrealized PnL</span>
              <span className={`stat-value ${status?.current_position ? ((livePrice?.price || 0) - status.current_position.entry_price > 0 ? 'positive' : 'negative') : ''}`}>
                {status?.current_position 
                  ? formatMoney(((livePrice?.price || 0) - status.current_position.entry_price) * status.current_position.quantity * (status.current_position.side === 'long' ? 1 : -1)) 
                  : '--'}
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Total Realized PnL</span>
              <span className={`stat-value ${(status?.pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                {formatMoney(status?.pnl)}
              </span>
            </div>
          </div>

          {/* Performance Stats */}
          <div className="sidebar-section">
            <h3>Performance</h3>
            <div className="stat-row">
              <span className="stat-label">Win Rate</span>
              <span className="stat-value">{formatPct(status?.win_rate)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Total Trades</span>
              <span className="stat-value">{status?.total_trades || 0}</span>
            </div>
          </div>

          {/* Active Position */}
          {status?.current_position && (
            <div className="sidebar-section">
              <h3>Active Position</h3>
              <div className="stat-row">
                <span className="stat-label">Side</span>
                <span className={`side-badge ${status.current_position.side.toLowerCase()}`}>
                  {status.current_position.side.toUpperCase()}
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Entry Price</span>
                <span className="stat-value">${status.current_position.entry_price?.toFixed(2)}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Size</span>
                <span className="stat-value">{status.current_position.quantity?.toFixed(6)}</span>
              </div>
              {status.current_position.stop_loss && (
                <div className="stat-row">
                  <span className="stat-label">Stop Loss</span>
                  <span className="stat-value negative">${status.current_position.stop_loss?.toFixed(2)}</span>
                </div>
              )}
              {status.current_position.take_profit && (
                <div className="stat-row">
                  <span className="stat-label">Take Profit</span>
                  <span className="stat-value positive">${status.current_position.take_profit?.toFixed(2)}</span>
                </div>
              )}
            </div>
          )}
        </aside>

        {/* Main Content: Charts & Logs */}
        <main className="main-content">
          {/* Ticker Tape */}
          {livePrice && (
            <div className="ticker-bar">
              <div className="ticker-symbol">{livePrice.symbol}</div>
              <div className={`ticker-price ${(livePrice.change_24h || 0) < 0 ? 'down' : ''}`}>
                ${livePrice.price?.toLocaleString()}
              </div>
              <div className={`ticker-change ${(livePrice.change_24h || 0) < 0 ? 'negative' : 'positive'}`}>
                {(livePrice.change_24h || 0) > 0 ? '+' : ''}{(livePrice.change_24h || 0).toFixed(2)}%
              </div>
              
              <div className="ticker-meta">
                <div className="meta-item">24h High <span>${livePrice.high_24h?.toLocaleString()}</span></div>
                <div className="meta-item">24h Low <span>${livePrice.low_24h?.toLocaleString()}</span></div>
                <div className="meta-item">24h Vol <span>{Number(livePrice.volume_24h || 0).toFixed(0)}</span></div>
              </div>
            </div>
          )}

          {/* TradingView Chart */}
          <div className="chart-container-wrapper">
            <Chart 
              symbol={symbol}
              apiBaseUrl={API_BASE}
              height={500}
              showVolume={true}
              showSMA={true}
              refreshInterval={1000}
            />
          </div>

          {/* Bottom Panel: Trade History & Equity Curve */}
          <div className="bottom-panel">
            <div className="panel-section">
              <div className="panel-header">Trade History</div>
              <div className="table-container">
                <table className="trade-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Type</th>
                      <th>Price</th>
                      <th>Amount</th>
                      <th>PnL</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice().reverse().map((t, i) => (
                      <tr key={i}>
                        <td>{formatTime(t.timestamp)}</td>
                        <td>
                          <span className={`side-badge ${t.side?.includes('buy') || t.side?.includes('long') ? 'long' : 'short'}`}>
                            {t.side?.toUpperCase().replace('CLOSE_', 'EXIT ')}
                          </span>
                        </td>
                        <td>${t.price?.toFixed(2)}</td>
                        <td>{t.quantity?.toFixed(6)}</td>
                        <td className={t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : ''}>
                          {t.pnl ? formatMoney(t.pnl) : '-'}
                        </td>
                        <td style={{color: '#848e9c', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{t.reason}</td>
                      </tr>
                    ))}
                    {trades.length === 0 && (
                      <tr><td colSpan="6" style={{textAlign: 'center', padding: '20px', color: '#848e9c'}}>No trades yet</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel-section">
              <div className="panel-header">Equity Curve</div>
              <div className="equity-chart-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equity}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2b3139" />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis 
                      domain={['auto', 'auto']} 
                      orientation="right" 
                      tick={{fontSize: 10, fill: '#848e9c'}}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e2329', border: '1px solid #2b3139', borderRadius: '4px'}}
                      labelStyle={{color: '#848e9c'}}
                      formatter={(value) => [`$${value.toFixed(2)}`, 'Equity']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="equity" 
                      stroke="#f0b90b" 
                      strokeWidth={2} 
                      dot={false} 
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
