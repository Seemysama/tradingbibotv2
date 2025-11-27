# 🚀 Trading Engine v2 - Production Grade

Architecture de trading bot modulaire supportant **LIVE**, **PAPER** et **BACKTEST** via configuration.

## 📐 Architecture

```
trading_engine/
├── main.py                 # Orchestrateur unifié
├── api/
│   └── app.py              # API FastAPI + WebSocket
├── src/
│   ├── config.py           # Configuration Pydantic (MODE, etc.)
│   ├── feed.py             # Abstraction MarketFeed (Parquet/Live)
│   ├── database.py         # Persistance SQLite/QuestDB
│   ├── strategy.py         # HybridStrategy (Golden Cross + ML)
│   ├── execution.py        # Risk Management & Execution
│   └── ai/
│       ├── model.py        # LSTM + Attention, GRU (upgraded)
│       └── inference.py    # Moteur d'inférence ML
├── state/
│   └── models.py           # SQLModel: Orders, Positions, Trades
├── frontend/
│   └── src/
│       ├── App.jsx         # Dashboard React
│       └── components/
│           └── Chart.jsx   # TradingView Lightweight Charts
└── data/
    ├── historical/         # Fichiers Parquet
    └── trading_state.db    # SQLite persistance
```

## 🔧 Configuration

Copiez `.env.example` vers `.env` et configurez:

```bash
# Mode de fonctionnement
MODE=PAPER          # LIVE | PAPER | BACKTEST

# Pour mode LIVE
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
USE_SANDBOX=true

# ML
ML_MODEL_TYPE=lstm_attention  # lstm | lstm_attention | gru
ML_ENABLED=true

# Trading
PAIRS=["BTC/USDT"]
INITIAL_BALANCE=10000.0
```

## 🚀 Démarrage

### Backend (API + Trading)
```bash
# Mode démo (API uniquement)
cd trading_engine
source ../.venv/bin/activate
python run_backend.py

# Mode production (orchestrateur complet)
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Accéder à http://localhost:3000

## 📊 Modes de Fonctionnement

### PAPER (défaut)
- Utilise les données historiques Parquet
- Timestamps en temps réel
- Parfait pour les démos

### BACKTEST
- Lecture Parquet à vitesse maximale
- Pas de délai entre les bougies
- Pour l'optimisation de stratégie

### LIVE
- WebSocket via ccxt.pro
- Reconnexion automatique
- Nécessite les clés API

## 🏗️ Composants Clés

### MarketFeed (`src/feed.py`)
```python
from src.feed import create_feed, FeedConfig

config = FeedConfig(symbol="BTC/USDT", timeframe="1m")
feed = create_feed("PAPER", config)  # ou "LIVE", "BACKTEST"

await feed.start()
async for candle in feed:
    # Traiter la bougie
    pass
```

### Database (`src/database.py`)
```python
from src.database import init_database, get_db

db = await init_database()

# Créer une position
position = await db.create_position(Position(...))

# Récupérer positions ouvertes
positions = await db.get_open_positions("BTC/USDT")
```

### ML Models (`src/ai/model.py`)
```python
from src.ai.model import create_model, save_checkpoint, load_checkpoint

# Créer un modèle
model = create_model("lstm_attention", input_dim=6, hidden_dim=128)

# Sauvegarder avec métadonnées
save_checkpoint(
    model, 
    Path("models/v2.pth"),
    feature_names=["close", "volume", "RSI_14", ...],
    scaler_params={"close_min": 0, "close_max": 100000, ...},
    training_metrics={"accuracy": 0.68, "loss": 0.42}
)

# Charger
model, metadata = load_checkpoint(Path("models/v2.pth"), device="mps")
```

## 🎨 Frontend Features

- **TradingView Charts**: Graphique interactif avec bougies OHLCV
- **Markers de Trade**: Visualisation des entrées/sorties
- **SMA 50**: Indicateur de tendance
- **Courbe d'Equity**: Performance en temps réel
- **Historique des Trades**: Avec PnL et raisons

## 🔒 Persistance

### SQLite (état transactionnel)
- Orders (statut, prix, timestamps)
- Positions (entry, exit, PnL)
- Bot State (balance, métriques)

### QuestDB (optionnel - séries temporelles)
- Candles haute fréquence
- Métriques de trading

## 📈 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/status` | GET | État du bot |
| `/api/trades` | GET | Historique trades |
| `/api/candles` | GET | Bougies OHLCV |
| `/api/equity` | GET | Courbe equity |
| `/api/start` | POST | Démarrer le bot |
| `/api/stop` | POST | Arrêter le bot |
| `/ws` | WebSocket | Updates temps réel |

## 🧪 Tests

```bash
# Lancer le backtest
MODE=BACKTEST python main.py

# Vérifier le modèle ML
python -c "from src.ai.model import load_checkpoint; m, meta = load_checkpoint('models/lstm_v1.pth'); print(meta)"
```

## 📦 Dépendances

```bash
pip install -r requirements.txt
```

Key packages:
- `fastapi`, `uvicorn` - API
- `sqlmodel`, `sqlalchemy` - Database
- `ccxt` - Exchange connectivity
- `torch` - ML
- `pandas`, `pyarrow` - Data

## 🔄 Migration depuis v1

1. Copier `.env.example` vers `.env`
2. Ajouter `MODE=PAPER`
3. Le bot utilisera automatiquement les nouveaux composants

---

Built with ❤️ for crypto trading
