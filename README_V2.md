# 🏦 Trading Engine V2 - Industrial Grade

## Vue d'ensemble

Système de trading algorithmique **Hedge Fund Grade** avec:
- **Triple Barrier Method** pour les labels (pas de RSI naïf)
- **Features stationnaires** (log-returns, z-scores, spreads normalisés)
- **Walk-Forward Validation** (pas de data leakage)
- **Transformer Architecture** avec attention multi-têtes

## Architecture

```
trading_engine/
├── tools/
│   └── massive_ingest.py     # Téléchargement multi-paires, multi-timeframes
├── src/
│   ├── features_pro.py       # 39 features stationnaires + Triple Barrier
│   └── ai/
│       └── transformer_pro.py # Transformer + Walk-Forward Validation
└── train_v2.py               # Pipeline d'entraînement industriel
```

## Changements clés vs V1

| Aspect | V1 (Échec) | V2 (Industriel) |
|--------|------------|-----------------|
| Timeframe | 1m (trop de bruit) | **5m/15m** |
| Target | Return simple | **Triple Barrier** (TP/SL) |
| Features | Chained assignment warnings | **Vectorisé strict** |
| Validation | Random split (leakage!) | **Walk-Forward** |
| Sharpe | -18 🔴 | Cible: > 1.0 ✅ |

## Installation

```bash
cd /Users/semy/trading_engine_v2/trading_engine
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Télécharger les données (5m, moins de bruit)

```bash
# BTC seulement
python tools/massive_ingest.py --pairs BTC/USDT --timeframes 5m --consolidate

# Top 10 paires
python tools/massive_ingest.py --pairs all --timeframes 5m 15m --consolidate
```

### 2. Entraîner le modèle

```bash
# Avec les nouvelles données 5m
python train_v2.py \
    --data data/futures/BTC_USDT_5m_FULL.parquet \
    --epochs 100 \
    --batch-size 256 \
    --seq-length 128 \
    --n-splits 5

# Ou avec les données 1m existantes (pour test rapide)
python train_v2.py \
    --data data/massive/BTC_USDT_1m_FULL.parquet \
    --epochs 50
```

### 3. Évaluer les métriques

Les métriques sont sauvées dans `models/metrics_v2.json`:

- **Sharpe Ratio**: Cible > 1.0 (vs -18 avant)
- **Sortino Ratio**: Pénalise seulement la volatilité négative
- **Max Drawdown**: Cible < 20%
- **Win Rate**: Cible > 55%
- **Profit Factor**: Cible > 1.5

## Features (39 stationnaires)

### Log-Returns (5 features)
- `feat_logret_1`: Return 1 période
- `feat_logret_5/15/30/60`: Returns cumulés

### Volatilité Normalisée (6 features)
- `feat_vol_ratio_20/50/100`: Ratio vol court/long terme
- `feat_vol_20/50/100`: Volatilité annualisée

### Z-Scores (3 features)
- `feat_zscore_20/50/100`: Position relative au prix moyen

### RSI Normalisé (2 features)
- `feat_rsi_14/28`: RSI normalisé entre -1 et 1

### EMA Spreads (6 features)
- `feat_ema_spread_9_21/21_50/50_200`: Spreads normalisés
- `feat_price_vs_ema_21/50/100`: Position vs EMAs

### ATR (2 features)
- `feat_atr_ratio`: ATR court/long terme
- `feat_atr_pct`: ATR % du prix

### Volume (3 features)
- `feat_vol_zscore_20/50`: Z-score du volume
- `feat_dollar_vol_zscore`: Z-score du volume en dollars

### Microstructure (5 features)
- `feat_hl_range`: High-Low normalisé
- `feat_body_ratio`: Ratio corps/range
- `feat_upper_shadow/lower_shadow`: Mèches
- `feat_gap`: Gap d'ouverture

### Momentum (3 features)
- `feat_roc_zscore_5/15/30`: Z-score du Rate of Change

### Time (4 features)
- `feat_hour_sin/cos`: Heure (cyclique)
- `feat_dow_sin/cos`: Jour de semaine (cyclique)

## Triple Barrier Method

Au lieu de prédire simplement la direction, on prédit le résultat d'un trade:

```
              TP (2.0 × ATR)
              ═══════════════════
             ╱
            ╱
Entry ─────●──────────────────────→ Timeout (60 périodes)
            ╲
             ╲
              ═══════════════════
              SL (1.0 × ATR)
```

- **Label 2 (LONG)**: TP touché avant SL
- **Label 1 (NEUTRAL)**: Timeout sans toucher TP/SL
- **Label 0 (SHORT)**: SL touché avant TP

## Walk-Forward Validation

Évite le data leakage avec un split temporel strict:

```
Fold 1: [====Train====][gap][Val]
Fold 2: [=========Train=========][gap][Val]
Fold 3: [================Train================][gap][Val]
Fold 4: [=====================Train=====================][gap][Val]
Fold 5: [===========================Train===========================][gap][Val]
```

Le `gap` (purge) évite que les features de validation ne contaminent le train.

## Modèle Transformer

```
Input (batch, 128, 39) 
    ↓
Linear Projection (39 → 128)
    ↓
Positional Encoding (sinusoïdal)
    ↓
4× Transformer Encoder Block
    │ ├── LayerNorm
    │ ├── Multi-Head Attention (8 heads)
    │ └── FFN (128 → 512 → 128)
    ↓
Global Average Pooling
    ↓
├── Classification Head → 3 classes (LONG/NEUTRAL/SHORT)
├── Regression Head → Return prédit
└── Confidence Head → Confiance [0, 1]
```

**Paramètres**: ~811,000

## Objectifs de performance

| Métrique | V1 (Échec) | V2 Cible | Signification |
|----------|------------|----------|---------------|
| Sharpe Ratio | -18.08 | > 1.0 | Rentabilité ajustée au risque |
| Max Drawdown | 100% | < 20% | Perte maximale |
| Win Rate | 48.8% | > 55% | Trades gagnants |
| Profit Factor | 0.82 | > 1.5 | Gains / Pertes |

## Dépannage

### "ModuleNotFoundError"
```bash
pip install numba pyarrow tqdm torch pandas numpy
```

### Données non trouvées
```bash
# Vérifier le téléchargement
ls -la data/futures/
```

### GPU non détecté
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Prochaines étapes

1. **Télécharger données 5m** (en cours)
2. **Entraîner avec train_v2.py**
3. **Évaluer les métriques**
4. **Intégrer en production** si Sharpe > 1.0
5. **Ajouter multi-paires** (ETH, SOL, etc.)

---

*Version 2.0 - Industrial Grade Trading System*
