# 🚀 Trading Bot avec Dashboard - Guide de Démarrage Rapide

## 📋 Vue d'ensemble

Bot de trading automatisé avec interface web moderne, combinant analyse technique (Golden/Death Cross, RSI) et machine learning (LSTM) pour générer des signaux de trading.

**Fonctionnalités principales:**
- ✅ Trading Paper/Backtest sur données historiques (BTC, ETH, SOL)
- ✅ Dashboard en temps réel avec graphiques
- ✅ Stratégie hybride: Technical Analysis + ML
- ✅ Gestion des risques (stop loss, take profit, exposition maximale)
- ✅ API REST + WebSocket pour monitoring live

## 🎯 Démarrage Rapide (Démo Paper Trading)

### Prérequis

- Python 3.9+
- Node.js 18+ et npm
- Git

### Installation & Lancement

```bash
# 1. Cloner le projet (si nécessaire)
cd trading_engine_v2/trading_engine

# 2. Créer l'environnement virtuel Python
python3 -m venv ../.venv
source ../.venv/bin/activate  # Sur Windows: ..\.venv\Scripts\activate

# 3. Installer les dépendances Python
pip install -r requirements.txt

# 4. Lancer la démo complète (backend + frontend)
chmod +x start_demo.sh
./start_demo.sh
```

Le script va:
1. Démarrer l'API backend sur `http://localhost:8000`
2. Installer les dépendances frontend (première fois seulement)
3. Lancer le dashboard sur `http://localhost:3000`

### Lancement Manuel (Alternative)

**Terminal 1 - Backend:**
```bash
cd trading_engine_v2/trading_engine
source ../.venv/bin/activate
python api/app.py
```

**Terminal 2 - Frontend:**
```bash
cd trading_engine_v2/trading_engine/frontend
npm install  # Première fois seulement
npm run dev
```

## 🎮 Utilisation du Dashboard

1. **Ouvrir** `http://localhost:3000` dans votre navigateur
2. **Sélectionner** le symbole (BTC/USDT, ETH/USDT, SOL/USDT)
3. **Cliquer** sur "▶️ Démarrer" pour lancer le bot en mode paper
4. **Observer** les trades en temps réel, les statistiques et la courbe d'equity

### Indicateurs affichés

- **Balance**: Capital actuel
- **PnL Total**: Profit/Perte cumulé
- **Total Trades**: Nombre de positions ouvertes/fermées
- **Win Rate**: Pourcentage de trades gagnants
- **Position Actuelle**: Détails de la position ouverte (si applicable)
- **Historique**: 20 derniers trades avec raisons d'entrée/sortie

## 🏗️ Architecture

```
trading_engine/
├── api/
│   └── app.py              # Backend FastAPI avec endpoints REST + WebSocket
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Composant principal du dashboard
│   │   ├── App.css         # Styles modernes (glassmorphism)
│   │   └── main.jsx        # Point d'entrée React
│   ├── package.json
│   └── vite.config.js      # Configuration Vite avec proxy API
├── src/
│   ├── strategy.py         # Stratégie hybride (Golden Cross + ML)
│   ├── execution.py        # Gestion des ordres et risques
│   ├── config.py           # Configuration globale
│   └── ai/
│       ├── model.py        # Architecture LSTM
│       └── inference.py    # Moteur d'inférence ML
├── data/historical/        # Données parquet (BTC, ETH, SOL)
├── models/
│   └── lstm_v1.pth         # Modèle ML pré-entraîné
└── start_demo.sh           # Script de lancement automatique
```

## 📡 API Endpoints

### REST API

- `GET /api/status` - État du bot (running, balance, PnL, etc.)
- `GET /api/trades?limit=N` - Historique des N derniers trades
- `GET /api/equity` - Courbe d'equity (500 derniers points)
- `POST /api/start` - Démarrer le bot (body: `{symbol, mode}`)
- `POST /api/stop` - Arrêter le bot
- `GET /docs` - Documentation Swagger interactive

### WebSocket

- `ws://localhost:8000/ws` - Streaming temps réel des mises à jour

## ⚙️ Configuration

Créer un fichier `.env` à la racine pour personnaliser:

```env
ENV=PAPER                           # DEV | PAPER | LIVE
ML_ENABLED=true                     # Activer/désactiver le filtre ML
ML_CONFIDENCE_THRESHOLD=0.65        # Seuil de confiance ML (0-1)
MAX_EXPOSURE=0.5                    # Exposition max (50% du capital)
COOLDOWN_SEC=30                     # Délai minimum entre trades
SPREAD_LIMIT=0.002                  # Spread max acceptable (0.2%)
LOGGER_LEVEL=INFO                   # DEBUG | INFO | WARNING
```

## 🧪 Mode de Fonctionnement

### Paper Trading (par défaut)

- Utilise les données historiques en parquet
- Simule l'exécution en temps accéléré (10ms par bougie)
- Capital initial: $10,000
- Taille de position: 5% du capital par trade
- Stop Loss: -1% | Take Profit: +2%

### Backtest

- Traite l'intégralité du dataset historique
- Affiche les résultats finaux (PnL, Sharpe, Drawdown max)

### Live (production)

- Nécessite `BINANCE_API_KEY` et `BINANCE_API_SECRET` dans `.env`
- Se connecte au WebSocket Binance pour les données en temps réel
- Exécute de vrais ordres via l'API Binance

## 🔍 Stratégie de Trading

**Signaux techniques:**
- **Long**: Golden Cross (SMA 50 > SMA 200) + RSI < 70
- **Short**: Death Cross (SMA 50 < SMA 200) + RSI > 30

**Filtre ML (LSTM):**
- Prédit la probabilité de hausse sur les 60 prochaines bougies
- Confirme (✅) ou véto (🚫) les signaux techniques
- Seuil de confiance configurable (défaut: 65%)

**Gestion des risques:**
- Exposition maximale: 50% du capital
- Cool-down: 30s entre trades
- Stop Loss automatique: -1%
- Take Profit automatique: +2%
- Vérification du spread avant exécution

## 📊 Données Disponibles

- `BTC_USDT_1m_2Y.parquet` - Bitcoin 1min (2 ans)
- `ETH_USDT_1m_2Y.parquet` - Ethereum 1min (2 ans)
- `SOL_USDT_1m_2Y.parquet` - Solana 1min (2 ans)

**Total:** ~1M de bougies par symbole

## 🐛 Dépannage

**Erreur: "ModuleNotFoundError: No module named 'fastapi'"**
```bash
source ../.venv/bin/activate
pip install -r requirements.txt
```

**Frontend ne démarre pas**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Pas de données trouvées**
- Vérifier que les fichiers `.parquet` existent dans `data/historical/`
- Utiliser `tools/backfill.py` pour télécharger les données manquantes

**Bot ne génère pas de signaux**
- Vérifier `ML_ENABLED=true` dans `.env`
- S'assurer que `models/lstm_v1.pth` existe
- Mettre `DEBUG_SIGNALS=true` pour logs détaillés

## 🚀 Prochaines Étapes

- [ ] Ajouter support multi-symboles simultanés
- [ ] Implémenter WebSocket Binance pour mode live
- [ ] Créer des backtests paramétrisables depuis le dashboard
- [ ] Ajouter notifications (Discord, Telegram)
- [ ] Système de logs persistant avec base de données

## 📝 Licence

Ce projet est à usage éducatif. Tradez à vos propres risques.

---

**Bon trading ! 📈**
