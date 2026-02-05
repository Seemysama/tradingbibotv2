import { useState, useEffect, useRef } from 'react'
import { createChart } from 'lightweight-charts'
import './App.css'

const API = 'http://localhost:8000/api'

function App() {
  const [status, setStatus] = useState(null)
  const [trades, setTrades] = useState([])
  const [price, setPrice] = useState(null)
  const [loading, setLoading] = useState(false)
  const [runParams, setRunParams] = useState({
    data: 'data/massive/BTC_USDT_5m_FULL.parquet',
    epochs: 20,
    batch_size: 512,
    seq_length: 64,
    n_splits: 3,
    device: ''
  })
  const [runStatus, setRunStatus] = useState(null)
  const [runLoading, setRunLoading] = useState(false)
  const [symbol] = useState('BTC/USDT')
  const chartRef = useRef(null)
  const chartInstanceRef = useRef(null)

  // Fetch data
  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [statusRes, tradesRes, priceRes] = await Promise.all([
          fetch(`${API}/status`).then(r => r.json()),
          fetch(`${API}/trades?limit=50`).then(r => r.json()),
          fetch(`${API}/price/BTC/USDT`).then(r => r.json())
        ])
        setStatus(statusRes)
        setTrades(tradesRes || [])
        setPrice(priceRes)
      } catch (e) {
        console.log('API error:', e)
      }
    }
    
    fetchAll()
    const interval = setInterval(fetchAll, 2000)
    return () => clearInterval(interval)
  }, [])

  // Chart
  useEffect(() => {
    if (!chartRef.current || chartInstanceRef.current) return

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 400,
      layout: { background: { color: '#0b0e11' }, textColor: '#848e9c' },
      grid: { 
        vertLines: { color: 'rgba(255,255,255,0.05)' }, 
        horzLines: { color: 'rgba(255,255,255,0.05)' } 
      },
      timeScale: { borderColor: '#2b3139', timeVisible: true }
    })

    const series = chart.addCandlestickSeries({
      upColor: '#0ecb81',
      downColor: '#f6465d',
      borderUpColor: '#0ecb81',
      borderDownColor: '#f6465d',
      wickUpColor: '#0ecb81',
      wickDownColor: '#f6465d'
    })

    chartInstanceRef.current = { chart, series }

    // Fetch candles
    fetch(`${API}/candles?limit=100`)
      .then(r => r.json())
      .then(data => {
        if (data.candles) {
          const formatted = data.candles.map(c => ({
            time: Math.floor(new Date(c.timestamp).getTime() / 1000),
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close
          }))
          series.setData(formatted)
        }
      })

    const handleResize = () => {
      chart.applyOptions({ width: chartRef.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)
    
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Actions
  const handleStart = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/start`, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ symbol, mode: 'paper' }) 
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Erreur serveur')
      }
      console.log('Bot démarré:', data)
    } catch (e) { 
      console.error('Start error:', e)
      alert('Erreur: ' + e.message) 
    }
    setLoading(false)
  }

  const handleStop = async () => {
    try {
      const res = await fetch(`${API}/stop`, { method: 'POST' })
      const data = await res.json()
      console.log('Bot arrêté:', data)
    } catch (e) { 
      console.error('Stop error:', e)
      alert('Erreur: ' + e.message) 
    }
  }

  // Format
  const fmt = (v) => '$' + (v || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  const pnl = status?.pnl || 0
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString() : '—')

  // --- Lab functions ---
  const handleRunParamChange = (key, value) => {
    setRunParams(prev => ({ ...prev, [key]: value }))
  }

  const estimateTime = () => {
    // Heuristique simple pour donner un ordre de grandeur
    const base = 2 // minutes de base
    const scale = (runParams.epochs / 20) * (runParams.n_splits / 3) * (runParams.seq_length / 64) * (runParams.batch_size / 512) ** -0.5
    const model = Math.max(1, (runParams.d_model || 64) / 64) * Math.max(1, (runParams.n_layers || 2) / 2)
    return Math.max(1, Math.round(base * scale * model))
  }

  const refreshRunStatus = async () => {
    try {
      const res = await fetch(`${API}/runs/status`)
      const data = await res.json()
      setRunStatus(data)
    } catch (e) {
      console.error('Run status error:', e)
    }
  }

  const startRun = async () => {
    setRunLoading(true)
    try {
      const payload = { ...runParams }
      if (!payload.device) delete payload.device
      const res = await fetch(`${API}/runs/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const data = await res.json()
      setRunStatus(data)
    } catch (e) {
      console.error('Run start error:', e)
      alert('Erreur lancement run: ' + e.message)
    } finally {
      setRunLoading(false)
    }
  }

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <h1>⚡ TRADING BOT V2</h1>
        <div className="status-badges">
          <span className={`badge ${status?.connected_to_exchange ? 'green' : 'red'}`}>
            {status?.connected_to_exchange ? 'BINANCE ✓' : 'OFFLINE'}
          </span>
          <span className={`badge ${status?.running ? 'green' : 'red'}`}>
            {status?.running ? 'RUNNING' : 'STOPPED'}
          </span>
          <span className="badge yellow">{status?.env || 'PAPER'}</span>
        </div>
      </header>

      {/* STATS */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="label">Prix BTC</div>
          <div className="value">{fmt(price?.price)}</div>
        </div>
        <div className="stat-card">
          <div className="label">Balance</div>
          <div className="value">{fmt(status?.balance)}</div>
        </div>
        <div className="stat-card">
          <div className="label">PnL Total</div>
          <div className={`value ${pnl >= 0 ? 'up' : 'down'}`}>{fmt(pnl)}</div>
        </div>
        <div className="stat-card">
          <div className="label">Win Rate</div>
          <div className="value">{(status?.win_rate || 0).toFixed(1)}%</div>
        </div>
      </div>

      {/* MAIN */}
      <div className="main-grid">
        {/* CHART */}
        <div className="chart-section">
          <div className="chart-header">
            <span>{symbol} • 1m</span>
            <span>Live</span>
          </div>
          <div className="chart-container" ref={chartRef}></div>
        </div>

        {/* CONTROLS */}
        <div className="controls-section">
          <div className="control-card">
            <h3>Contrôles</h3>
            <select disabled={status?.running}>
              <option>BTC/USDT</option>
              <option>ETH/USDT</option>
              <option>SOL/USDT</option>
            </select>
            {status?.running ? (
              <button className="btn btn-stop" onClick={handleStop}>Arrêter le Bot</button>
            ) : (
              <button className="btn btn-start" onClick={handleStart} disabled={loading}>
                {loading ? 'Démarrage...' : 'Démarrer le Bot'}
              </button>
            )}
          </div>

          {status?.current_position && (
            <div className={`control-card position-card ${status.current_position.side}`}>
              <h3>Position Active</h3>
              <div style={{fontSize: '1.2rem', fontWeight: 'bold', color: status.current_position.side === 'long' ? '#0ecb81' : '#f6465d'}}>
                {status.current_position.side.toUpperCase()}
              </div>
              <div className="position-info">
                <span>Entrée:</span>
                <span>{fmt(status.current_position.entry_price)}</span>
              </div>
              <div className="position-info">
                <span>Quantité:</span>
                <span>{status.current_position.quantity?.toFixed(4)}</span>
              </div>
            </div>
          )}

          <div className="control-card">
            <h3>Statistiques</h3>
            <div className="position-info">
              <span>Trades:</span>
              <span>{status?.total_trades || 0}</span>
            </div>
            <div className="position-info">
              <span>Positions:</span>
              <span>{status?.current_position ? '1' : '0'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* TRADES */}
      <div className="trades-section">
        <div className="trades-header">Historique des Trades</div>
        <div className="trades-table">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Type</th>
                <th>Prix</th>
                <th>Quantité</th>
                <th>PnL</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 ? (
                <tr>
                  <td colSpan="5" style={{textAlign: 'center', color: '#848e9c', padding: '30px'}}>
                    Aucun trade pour le moment
                  </td>
                </tr>
              ) : (
                trades.slice().reverse().map((t, i) => (
                  <tr key={i}>
                    <td>{new Date(t.timestamp).toLocaleTimeString()}</td>
                    <td>
                      <span className={`tag ${t.side?.includes('buy') ? 'long' : 'short'}`}>
                        {t.side?.toUpperCase()}
                      </span>
                    </td>
                    <td>{fmt(t.price || t.entry_price)}</td>
                    <td>{(t.quantity || t.amount || 0).toFixed(4)}</td>
                    <td style={{color: t.pnl > 0 ? '#0ecb81' : t.pnl < 0 ? '#f6465d' : '#848e9c'}}>
                      {t.pnl ? fmt(t.pnl) : '-'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* LAB PANEL */}
      <div className="lab-section">
        <div className="lab-card">
          <div className="lab-header">
            <div>
              <div className="lab-title">Console Lab (Entraînement)</div>
              <div className="lab-sub">Lance un run d'entraînement et suis le statut en direct.</div>
            </div>
            <div className="lab-badges">
              <span className="badge">{runStatus?.status || 'idle'}</span>
              <span className="badge">{runStatus?.run_id || '—'}</span>
            </div>
          </div>

          <div className="lab-grid">
            <div className="lab-form">
              <label>Dataset (Parquet)</label>
              <input value={runParams.data} onChange={e => handleRunParamChange('data', e.target.value)} />

              <div className="lab-two">
                <div>
                  <label>Epochs</label>
                  <input type="number" min="1" value={runParams.epochs} onChange={e => handleRunParamChange('epochs', Number(e.target.value))} />
                </div>
                <div>
                  <label>Batch size</label>
                  <input type="number" min="32" step="32" value={runParams.batch_size} onChange={e => handleRunParamChange('batch_size', Number(e.target.value))} />
                </div>
              </div>

              <div className="lab-two">
                <div>
                  <label>Seq length</label>
                  <input type="number" min="16" step="8" value={runParams.seq_length} onChange={e => handleRunParamChange('seq_length', Number(e.target.value))} />
                </div>
                <div>
                  <label>Folds (walk-forward)</label>
                  <input type="number" min="2" max="6" value={runParams.n_splits} onChange={e => handleRunParamChange('n_splits', Number(e.target.value))} />
                </div>
              </div>

              <label>Device (optionnel)</label>
              <input placeholder="auto (laisser vide) ou cuda/cpu" value={runParams.device} onChange={e => handleRunParamChange('device', e.target.value)} />

              <div className="lab-actions">
                <button className="btn btn-start" onClick={startRun} disabled={runLoading}>{runLoading ? 'Lancement...' : 'Lancer un run'}</button>
                <button className="btn btn-stop" onClick={refreshRunStatus}>Rafraîchir statut</button>
              </div>

              <div className="lab-estimate">
                <div>Temps estimé (heuristique) : ~{estimateTime()} min</div>
              </div>
            </div>

            <div className="lab-status">
              <div className="lab-status-row">
                <span>Run ID :</span>
                <span className="mono">{runStatus?.run_id || '—'}</span>
              </div>
              <div className="lab-status-row">
                <span>Statut :</span>
                <span className="mono">{runStatus?.status || 'idle'}</span>
              </div>
              <div className="lab-status-row">
                <span>PID :</span>
                <span className="mono">{runStatus?.pid || '—'}</span>
              </div>
              <div className="lab-status-row">
                <span>Début :</span>
                <span className="mono">{fmtTime(runStatus?.start_time)}</span>
              </div>
              <div className="lab-status-row">
                <span>Fin :</span>
                <span className="mono">{fmtTime(runStatus?.end_time)}</span>
              </div>
              <div className="lab-status-row">
                <span>Log :</span>
                <span className="mono">{runStatus?.log_path || '—'}</span>
              </div>
              <div className="lab-status-row">
                <span>Métriques :</span>
                <span className="mono">{runStatus?.metrics_path || '—'}</span>
              </div>

              <div className="lab-log">
                <div className="lab-log-title">Dernières lignes du log</div>
                <div className="lab-log-body">
                  {(runStatus?.last_lines || []).map((l, i) => (
                    <div key={i} className="mono">{l}</div>
                  ))}
                  {(runStatus?.last_lines || []).length === 0 && <div className="mono" style={{color:'#848e9c'}}>—</div>}
                </div>
              </div>

              <div className="lab-history">
                <div className="lab-log-title">Derniers runs</div>
                <div className="lab-history-list">
                  {(runStatus?.history || []).map((h, i) => (
                    <div key={i} className="lab-history-item">
                      <div className="mono">{h.run_id}</div>
                      <div className="lab-history-meta">
                        <span className="badge">{h.status}</span>
                        <span className="badge">epochs {h.params?.epochs}</span>
                        <span className="badge">batch {h.params?.batch_size}</span>
                      </div>
                    </div>
                  ))}
                  {(runStatus?.history || []).length === 0 && <div className="mono" style={{color:'#848e9c'}}>Aucun run</div>}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
