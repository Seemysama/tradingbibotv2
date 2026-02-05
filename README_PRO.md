# Trading Engine V2 - Hedge Fund Grade

## Architecture

```
trading_engine/
├── tools/
│   └── download_massive.py    # Téléchargement Binance Futures 2020→now
├── src/
│   ├── features.py            # 53+ features (volatilité, momentum, volume)
│   └── ai/
│       └── transformer_model.py  # Transformer + TCN SOTA
├── train_pro.py               # Entraînement Walk-Forward
└── models/                    # Modèles sauvegardés
```

## Quick Start

### 1. Télécharger les données (2020→maintenant)

```bash
cd trading_engine
source ../.venv/bin/activate

# Téléchargement massif BTC, ETH, SOL (1m + 5m)
python tools/download_massive.py \
    --symbols BTC/USDT ETH/USDT SOL/USDT \
    --timeframes 1m 5m \
    --start 2020-01-01 \
    --consolidate
```

**Temps estimé:** ~30-60 min pour ~5 ans de données 1-minute

### 2. Entraîner le modèle Transformer

```bash
python train_pro.py \
    --data data/massive/BTC_USDT_1m_FULL.parquet \
    --model transformer \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-4 \
    --d-model 128 \
    --n-layers 4 \
    --patience 15 \
    --output models/transformer_btc.pth
```

### 3. Utiliser les données existantes

Si vous avez déjà des données Parquet :

```bash
python train_pro.py --data data/historical/BTC_USDT_1m_FULL.parquet --epochs 50
```

## Features Générées (53 features)

### Volatilité
- `feat_atr` - ATR normalisé
- `feat_realized_vol_10/20/50` - Volatilité réalisée
- `feat_parkinson_vol_10/20` - Volatilité Parkinson (high-low)
- `feat_vol_ratio` - Ratio vol court/long

### Momentum
- `feat_rsi_7/14/21` - RSI multi-périodes
- `feat_roc_1/5/10/20` - Rate of Change
- `feat_macd/signal/hist` - MACD
- `feat_stoch_14/21` - Stochastique

### Trend
- `feat_ema_5/10/20/50/100` - EMAs
- `feat_dist_ema_*` - Distance au prix
- `feat_ema_slope_20/50` - Pente des EMAs
- `feat_adx` - Average Directional Index
- `feat_trend_score` - Score de tendance composite

### Volume
- `feat_vol_ratio_10/20/50` - Volume vs MA
- `feat_vwap_dist` - Distance au VWAP
- `feat_obv_slope` - Pente OBV
- `feat_mfv_sum_20` - Money Flow Volume

### Microstructure
- `feat_close_position` - Position du close dans le range
- `feat_body_ratio` - Ratio corps/range
- `feat_upper/lower_shadow` - Shadows
- `feat_candle_dir` - Direction de la bougie
- `feat_consecutive` - Bougies consécutives
- `feat_gap` - Gap d'ouverture

### Targets
- `target_dir_5/15/30/60` - Classification (-1, 0, 1)
- `target_ret_5/15/30/60` - Returns (régression)
- `target_binary_*` - Classification binaire

## Modèle Transformer

Architecture inspirée de "Attention Is All You Need" adaptée pour time series :

- **Input Projection** : features → d_model
- **Positional Encoding** : sinusoïdal
- **N Transformer Layers** : Multi-Head Self-Attention + FFN
- **Dual Head** : Classification (3 classes) + Régression (return)
- **Confidence Head** : Estimation d'incertitude

**Paramètres par défaut:**
- d_model = 128
- n_heads = 8
- n_layers = 4
- d_ff = 512
- dropout = 0.1

## Métriques de Trading

L'entraînement optimise sur :

1. **Sharpe Ratio** (cible principale)
2. **Max Drawdown** (risque)
3. **Win Rate** (fréquence)
4. **Profit Factor** (qualité)

Early stopping basé sur le Sharpe Ratio du validation set.

## Docker (Production)

```bash
docker-compose up -d
```

## Licence

MIT
