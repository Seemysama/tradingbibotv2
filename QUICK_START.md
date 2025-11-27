# 🚀 Trading Bot - Démarrage Rapide

## ✅ Solution Complète Prête

Votre bot de trading avec dashboard est maintenant **100% opérationnel** !

### 📦 Ce qui a été créé

1. **Backend API FastAPI** (`api/app.py`)
   - Endpoints REST pour contrôler le bot
   - WebSocket pour mises à jour temps réel
   - Moteur de trading paper avec données historiques
   - Stratégie hybride (Golden Cross + ML LSTM)

2. **Frontend React** (`frontend/`)
   - Dashboard moderne avec graphiques
   - Contrôles start/stop en temps réel
   - Affichage des trades et statistiques
   - Design glassmorphism responsive

3. **Scripts et Documentation**
   - `run_backend.py` - Lance l'API
   - `README_DEMO.md` - Documentation complète
   - `.env.example` - Template de configuration

## 🎯 Démarrage en 2 Étapes

### Actuellement en Cours

✅ **Backend API** : `http://localhost:8000` (PID: visible dans terminal)  
✅ **Frontend Dashboard** : `http://localhost:3000` (prêt à l'emploi)

### Pour Démarrer la Démo

1. **Ouvrir le Dashboard** : http://localhost:3000
2. **Sélectionner** BTC/USDT, ETH/USDT ou SOL/USDT
3. **Cliquer** sur "▶️ Démarrer"
4. **Observer** les trades en temps réel

Le bot va:
- Charger les données historiques (2 ans)
- Préchauffer la stratégie (500 bougies)
- Simuler le trading en mode accéléré
- Afficher les résultats en temps réel

## 🔄 Redémarrage Manuel (si nécessaire)

### Terminal 1 - Backend
```bash
cd /Users/semy/trading_engine_v2/trading_engine
/Users/semy/trading_engine_v2/.venv/bin/python run_backend.py
```

### Terminal 2 - Frontend
```bash
cd /Users/semy/trading_engine_v2/trading_engine/frontend
/Users/semy/.local/lib/node_modules/npm/bin/npm-cli.js run dev
```

## 📊 Fonctionnalités Disponibles

### Dashboard
- ⚡ Contrôles Start/Stop
- 💰 Balance et PnL en temps réel
- 📈 Courbe d'equity interactive
- 📜 Historique des 20 derniers trades
- 🎯 Position actuelle avec détails
- 📊 Statistiques (Win Rate, Total Trades)

### API Endpoints
- `GET /api/status` - État du bot
- `GET /api/trades` - Historique des trades
- `GET /api/equity` - Courbe d'equity
- `POST /api/start` - Démarrer (body: `{symbol, mode}`)
- `POST /api/stop` - Arrêter
- `GET /docs` - Documentation Swagger

### Stratégie de Trading
- **Golden Cross** : Long quand SMA 50 > SMA 200 + RSI < 70
- **Death Cross** : Short quand SMA 50 < SMA 200 + RSI > 30
- **Filtre ML** : LSTM confirme ou véto les signaux
- **Risk Management** : Stop Loss -1%, Take Profit +2%

## 🎨 Personnalisation

### Configuration (`.env`)
```env
ENV=PAPER                    # Mode paper trading
ML_ENABLED=true              # Filtre ML activé
ML_CONFIDENCE_THRESHOLD=0.65 # Seuil de confiance
MAX_EXPOSURE=0.5             # 50% du capital max
COOLDOWN_SEC=30              # Délai entre trades
```

### Données Disponibles
- `BTC_USDT_1m_2Y.parquet` - ~1M bougies Bitcoin
- `ETH_USDT_1m_2Y.parquet` - ~1M bougies Ethereum  
- `SOL_USDT_1m_2Y.parquet` - ~1M bougies Solana

## 🐛 Troubleshooting

**Backend ne démarre pas**
```bash
cd /Users/semy/trading_engine_v2
/Users/semy/trading_engine_v2/.venv/bin/pip install -r trading_engine/requirements.txt websockets
```

**Frontend ne charge pas**
```bash
cd /Users/semy/trading_engine_v2/trading_engine/frontend
rm -rf node_modules package-lock.json
/Users/semy/.local/lib/node_modules/npm/bin/npm-cli.js install
```

**Pas de signaux générés**
- Vérifier que le modèle `models/lstm_v1.pth` existe
- S'assurer que les fichiers `.parquet` sont dans `data/historical/`
- Mettre `DEBUG_SIGNALS=true` dans `.env` pour logs détaillés

## 📚 Documentation Complète

Voir `README_DEMO.md` pour :
- Architecture détaillée
- Configuration avancée
- Mode live avec Binance
- Développement et extensions

## 🎯 Prochaines Étapes Suggérées

- [ ] Tester différents symboles (BTC, ETH, SOL)
- [ ] Ajuster les paramètres de risque dans `.env`
- [ ] Observer les performances sur différentes périodes
- [ ] Analyser les raisons des trades (colonne "Raison")
- [ ] Comparer stratégie pure technique vs hybride ML

---

**Bon trading ! 📈 Le système est 100% fonctionnel.**
