/**
 * Chart Component - TradingView Lightweight Charts Integration
 * Graphique interactif avec bougies, volumes et indicateurs
 */
import React, { useEffect, useRef, useState, useCallback, memo } from 'react';
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts';

// Configuration par défaut du graphique
const CHART_CONFIG = {
  layout: {
    background: { type: 'solid', color: '#1a1a2e' },
    textColor: '#d1d5db',
  },
  grid: {
    vertLines: { color: '#2d2d44', style: LineStyle.Dotted },
    horzLines: { color: '#2d2d44', style: LineStyle.Dotted },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      width: 1,
      color: '#6366f1',
      style: LineStyle.Dashed,
      labelBackgroundColor: '#6366f1',
    },
    horzLine: {
      width: 1,
      color: '#6366f1',
      style: LineStyle.Dashed,
      labelBackgroundColor: '#6366f1',
    },
  },
  rightPriceScale: {
    borderColor: '#3d3d5c',
    scaleMargins: {
      top: 0.1,
      bottom: 0.2,
    },
  },
  timeScale: {
    borderColor: '#3d3d5c',
    timeVisible: true,
    secondsVisible: false,
  },
};

// Couleurs des bougies
const CANDLE_COLORS = {
  upColor: '#22c55e',
  downColor: '#ef4444',
  borderUpColor: '#16a34a',
  borderDownColor: '#dc2626',
  wickUpColor: '#22c55e',
  wickDownColor: '#ef4444',
};

/**
 * Composant principal du graphique
 */
const Chart = memo(function Chart({
  symbol = 'BTC/USDT',
  apiBaseUrl = 'http://localhost:8000',
  height = 500,
  showVolume = true,
  showSMA = true,
  refreshInterval = 1000,
}) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const smaSeriesRef = useRef(null);
  const markersRef = useRef([]);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastPrice, setLastPrice] = useState(null);
  const [priceChange, setPriceChange] = useState({ value: 0, percent: 0 });

  /**
   * Initialise le graphique
   */
  const initChart = useCallback(() => {
    if (!chartContainerRef.current) return;

    // Créer le graphique
    const chart = createChart(chartContainerRef.current, {
      ...CHART_CONFIG,
      width: chartContainerRef.current.clientWidth,
      height: height,
    });

    chartRef.current = chart;

    // Série des bougies
    const candleSeries = chart.addCandlestickSeries({
      ...CANDLE_COLORS,
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });
    candleSeriesRef.current = candleSeries;

    // Série des volumes
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#6366f1',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
      volumeSeriesRef.current = volumeSeries;

      // Configurer l'échelle de volume
      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
    }

    // SMA 50
    if (showSMA) {
      const smaSeries = chart.addLineSeries({
        color: '#f59e0b',
        lineWidth: 2,
        title: 'SMA 50',
        priceLineVisible: false,
        lastValueVisible: false,
      });
      smaSeriesRef.current = smaSeries;
    }

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height, showVolume, showSMA]);

  /**
   * Calcule la SMA
   */
  const calculateSMA = (data, period = 50) => {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data
        .slice(i - period + 1, i + 1)
        .reduce((acc, d) => acc + d.close, 0);
      sma.push({
        time: data[i].time,
        value: sum / period,
      });
    }
    return sma;
  };

  /**
   * Charge les données initiales
   */
  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${apiBaseUrl}/api/candles?symbol=${encodeURIComponent(symbol)}&limit=500`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.candles || data.candles.length === 0) {
        throw new Error('Pas de données disponibles');
      }

      // Formater les données
      const candles = data.candles.map(c => ({
        time: Math.floor(new Date(c.timestamp).getTime() / 1000),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

      const volumes = data.candles.map(c => ({
        time: Math.floor(new Date(c.timestamp).getTime() / 1000),
        value: c.volume,
        color: c.close >= c.open ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)',
      }));

      // Mettre à jour les séries
      if (candleSeriesRef.current) {
        candleSeriesRef.current.setData(candles);
      }

      if (volumeSeriesRef.current && showVolume) {
        volumeSeriesRef.current.setData(volumes);
      }

      if (smaSeriesRef.current && showSMA) {
        const smaData = calculateSMA(candles, 50);
        smaSeriesRef.current.setData(smaData);
      }

      // Mettre à jour le dernier prix
      if (candles.length > 0) {
        const lastCandle = candles[candles.length - 1];
        const firstCandle = candles[0];
        setLastPrice(lastCandle.close);
        setPriceChange({
          value: lastCandle.close - firstCandle.open,
          percent: ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100,
        });
      }

      // Ajuster la vue
      chartRef.current?.timeScale().fitContent();
      
      setIsLoading(false);
    } catch (err) {
      console.error('Erreur chargement données:', err);
      setError(err.message);
      setIsLoading(false);
    }
  }, [apiBaseUrl, symbol, showVolume, showSMA]);

  /**
   * Charge les trades pour les markers
   */
  const loadTrades = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/trades?limit=50`);
      if (!response.ok) return;

      const trades = await response.json();
      
      if (!candleSeriesRef.current) return;

      // Convertir les trades en markers
      const markers = trades.map(trade => ({
        time: Math.floor(new Date(trade.timestamp).getTime() / 1000),
        position: trade.side.includes('long') || trade.side === 'buy' ? 'belowBar' : 'aboveBar',
        color: trade.side.includes('long') || trade.side === 'buy' ? '#22c55e' : '#ef4444',
        shape: trade.side.includes('long') || trade.side === 'buy' ? 'arrowUp' : 'arrowDown',
        text: trade.pnl ? `${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}` : trade.side,
      }));

      // Filtrer les markers invalides et trier par temps
      const validMarkers = markers
        .filter(m => m.time && !isNaN(m.time))
        .sort((a, b) => a.time - b.time);

      candleSeriesRef.current.setMarkers(validMarkers);
      markersRef.current = validMarkers;
    } catch (err) {
      console.error('Erreur chargement trades:', err);
    }
  }, [apiBaseUrl]);

  /**
   * Met à jour les données en temps réel
   */
  const updateRealtime = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/candles?symbol=${encodeURIComponent(symbol)}&limit=1`);
      if (!response.ok) return;

      const data = await response.json();
      if (!data.candles || data.candles.length === 0) return;

      const latestCandle = data.candles[0];
      const candleData = {
        time: Math.floor(new Date(latestCandle.timestamp).getTime() / 1000),
        open: latestCandle.open,
        high: latestCandle.high,
        low: latestCandle.low,
        close: latestCandle.close,
      };

      // Mise à jour incrémentale
      if (candleSeriesRef.current) {
        candleSeriesRef.current.update(candleData);
      }

      if (volumeSeriesRef.current && showVolume) {
        volumeSeriesRef.current.update({
          time: candleData.time,
          value: latestCandle.volume,
          color: latestCandle.close >= latestCandle.open 
            ? 'rgba(34, 197, 94, 0.5)' 
            : 'rgba(239, 68, 68, 0.5)',
        });
      }

      setLastPrice(latestCandle.close);
    } catch (err) {
      // Silencieux pour les erreurs de refresh
    }
  }, [apiBaseUrl, symbol, showVolume]);

  // Initialisation
  useEffect(() => {
    const cleanup = initChart();
    return cleanup;
  }, [initChart]);

  // Chargement des données
  useEffect(() => {
    if (chartRef.current) {
      loadData();
      loadTrades();
    }
  }, [loadData, loadTrades]);

  // Rafraîchissement périodique
  useEffect(() => {
    const interval = setInterval(() => {
      updateRealtime();
      loadTrades();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [updateRealtime, loadTrades, refreshInterval]);

  return (
    <div className="chart-container">
      {/* Header avec infos prix */}
      <div className="chart-header">
        <div className="chart-symbol">
          <span className="symbol-name">{symbol}</span>
          {lastPrice && (
            <span className="last-price">${lastPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          )}
        </div>
        {priceChange.value !== 0 && (
          <div className={`price-change ${priceChange.value >= 0 ? 'positive' : 'negative'}`}>
            <span>{priceChange.value >= 0 ? '+' : ''}{priceChange.value.toFixed(2)}</span>
            <span>({priceChange.percent >= 0 ? '+' : ''}{priceChange.percent.toFixed(2)}%)</span>
          </div>
        )}
      </div>

      {/* Loading / Error states */}
      {isLoading && (
        <div className="chart-loading">
          <div className="spinner"></div>
          <span>Chargement du graphique...</span>
        </div>
      )}

      {error && (
        <div className="chart-error">
          <span>⚠️ {error}</span>
          <button onClick={loadData}>Réessayer</button>
        </div>
      )}

      {/* Chart container */}
      <div 
        ref={chartContainerRef} 
        className="chart-canvas"
        style={{ 
          visibility: isLoading ? 'hidden' : 'visible',
          height: `${height}px`,
        }}
      />

      {/* Légende */}
      <div className="chart-legend">
        {showSMA && <span className="legend-item sma">━ SMA 50</span>}
        <span className="legend-item buy">▲ Achat</span>
        <span className="legend-item sell">▼ Vente</span>
      </div>

      <style>{`
        .chart-container {
          background: #1a1a2e;
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .chart-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
          padding-bottom: 12px;
          border-bottom: 1px solid #3d3d5c;
        }

        .chart-symbol {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .symbol-name {
          font-size: 1.25rem;
          font-weight: 600;
          color: #fff;
        }

        .last-price {
          font-size: 1.5rem;
          font-weight: 700;
          color: #fff;
        }

        .price-change {
          display: flex;
          gap: 8px;
          font-weight: 500;
        }

        .price-change.positive {
          color: #22c55e;
        }

        .price-change.negative {
          color: #ef4444;
        }

        .chart-loading,
        .chart-error {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 300px;
          color: #9ca3af;
        }

        .chart-error {
          color: #f87171;
        }

        .chart-error button {
          margin-top: 12px;
          padding: 8px 16px;
          background: #6366f1;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid #3d3d5c;
          border-top-color: #6366f1;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 12px;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .chart-canvas {
          border-radius: 8px;
          overflow: hidden;
        }

        .chart-legend {
          display: flex;
          gap: 20px;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #3d3d5c;
        }

        .legend-item {
          font-size: 0.875rem;
          color: #9ca3af;
        }

        .legend-item.sma {
          color: #f59e0b;
        }

        .legend-item.buy {
          color: #22c55e;
        }

        .legend-item.sell {
          color: #ef4444;
        }
      `}</style>
    </div>
  );
});

export default Chart;
