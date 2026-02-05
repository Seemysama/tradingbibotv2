#!/usr/bin/env python3
"""
FEATURES PRO - VERSION RÉALISTE
================================
Feature engineering corrigé avec seuils RÉALISTES:
- FEE_THRESHOLD = 0.001 (0.1% minimum pour couvrir les frais)
- Labels triple-barrier tenant compte des frais
- Normalisation rolling anti-leakage

Author: Lead Quant Researcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from numba import njit

log = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES RÉALISTES
# ============================================================================

# Frais Binance Futures (taker)
TAKER_FEE: float = 0.0004  # 0.04%
SPREAD_ESTIMATE: float = 0.0002  # 0.02%

# SEUIL MINIMUM: mouvement doit dépasser frais aller-retour + marge
FEE_THRESHOLD: float = 0.001  # 0.1% (couvre frais + spread + marge sécurité)
PROFIT_TARGET: float = 0.002  # 0.2% cible de profit après frais


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FeatureConfig:
    """Configuration du feature engineering."""
    
    # Timeframes pour les indicateurs
    short_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    long_windows: List[int] = field(default_factory=lambda: [50, 100, 200])
    
    # Triple Barrier RÉALISTE
    horizon: int = 12  # 12 * 5min = 1 heure
    fee_threshold: float = FEE_THRESHOLD
    profit_target: float = PROFIT_TARGET
    stop_loss: float = 0.003  # 0.3% stop loss
    
    # Normalisation
    rolling_window: int = 500  # Fenêtre de normalisation
    min_samples: int = 100  # Minimum pour calcul stats


# ============================================================================
# NUMBA ACCELERATED FUNCTIONS
# ============================================================================

@njit
def _log_returns(close: np.ndarray) -> np.ndarray:
    """Log-returns."""
    n = len(close)
    result = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] > 0:
            result[i] = np.log(close[i] / close[i - 1])
    return result


@njit
def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Moyenne mobile."""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        total = 0.0
        for j in range(window):
            total += values[i - j]
        result[i] = total / window
    
    return result


@njit
def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Écart-type mobile."""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        mean = 0.0
        for j in range(window):
            mean += values[i - j]
        mean /= window
        
        var = 0.0
        for j in range(window):
            diff = values[i - j] - mean
            var += diff * diff
        result[i] = np.sqrt(var / window)
    
    return result


@njit
def _rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI optimisé."""
    n = len(close)
    result = np.full(n, np.nan)
    
    deltas = np.zeros(n)
    for i in range(1, n):
        deltas[i] = close[i] - close[i - 1]
    
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(n):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = -deltas[i]
    
    # Première moyenne
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period
    
    if avg_loss > 0:
        result[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    else:
        result[period] = 100.0
    
    # EMA style RSI
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss > 0:
            result[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            result[i] = 100.0
    
    return result


@njit
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR optimisé."""
    n = len(high)
    result = np.full(n, np.nan)
    
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Première ATR
    atr = 0.0
    for i in range(1, period + 1):
        atr += tr[i]
    atr /= period
    result[period] = atr
    
    # EMA style ATR
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + tr[i]) / period
        result[i] = atr
    
    return result


@njit
def _triple_barrier_labels_realistic(
    close: np.ndarray,
    horizon: int,
    profit_target: float,
    stop_loss: float,
    fee_threshold: float,
) -> np.ndarray:
    """
    Triple Barrier RÉALISTE: Label 1 seulement si le gain > fee_threshold.
    
    - Le modèle ne doit prédire UP que si le mouvement couvre les frais
    - Évite de trader le bruit
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int64)
    
    for i in range(n - horizon):
        entry_price = close[i]
        best_up = 0.0
        worst_down = 0.0
        
        for j in range(1, horizon + 1):
            future_price = close[i + j]
            ret = (future_price - entry_price) / entry_price
            
            if ret > best_up:
                best_up = ret
            if ret < worst_down:
                worst_down = ret
            
            # Profit target atteint ET supérieur au fee_threshold
            if ret >= profit_target and ret >= fee_threshold:
                labels[i] = 1
                break
            
            # Stop loss touché
            if ret <= -stop_loss:
                labels[i] = 0
                break
        else:
            # Fin d'horizon: label basé sur le mouvement final
            final_ret = (close[i + horizon] - entry_price) / entry_price
            
            # UP seulement si le gain NET (après frais) est positif
            if final_ret > fee_threshold:
                labels[i] = 1
            else:
                labels[i] = 0
    
    return labels


# ============================================================================
# FEATURE ENGINEER PRO
# ============================================================================

class FeatureEngineerPro:
    """Feature engineering RÉALISTE avec seuils au-dessus des frais."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ utils
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule toutes les features."""
        log.info(f"📊 Calcul features sur {len(df):,} lignes")
        
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        features = {}
        
        # Returns
        features['log_return'] = _log_returns(close)
        features['high_low_range'] = (high - low) / close
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # Volume
        for w in self.config.short_windows:
            vol_ma = _rolling_mean(volume, w)
            features[f'volume_ratio_{w}'] = volume / (vol_ma + 1e-10)
        
        # Moving averages
        for w in self.config.short_windows + self.config.long_windows:
            ma = _rolling_mean(close, w)
            features[f'ma_ratio_{w}'] = close / (ma + 1e-10) - 1
        
        # Volatility
        for w in [10, 20, 50]:
            features[f'volatility_{w}'] = _rolling_std(_log_returns(close), w)
        
        # RSI
        features['rsi_14'] = _rsi_numba(close, 14) / 100 - 0.5
        features['rsi_7'] = _rsi_numba(close, 7) / 100 - 0.5
        
        # ATR
        atr = _atr_numba(high, low, close, 14)
        features['atr_ratio'] = atr / close
        
        # Momentum
        for lag in [5, 10, 20]:
            features[f'momentum_{lag}'] = np.roll(close, -lag) / close - 1
            features[f'momentum_{lag}'][-lag:] = 0
        
        # VWAP proxy
        typical_price = (high + low + close) / 3
        cum_vol = np.cumsum(volume)
        cum_vwap = np.cumsum(typical_price * volume)
        vwap = cum_vwap / (cum_vol + 1e-10)
        features['vwap_ratio'] = close / vwap - 1
        
        result_df = pd.DataFrame(features, index=df.index)
        
        self.feature_names = list(features.keys())
        log.info(f"   ✅ {len(self.feature_names)} features générées")
        
        return result_df
    
    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule les labels RÉALISTES (tenant compte des frais)."""
        close = df['close'].values.astype(np.float64)
        
        labels = _triple_barrier_labels_realistic(
            close=close,
            horizon=self.config.horizon,
            profit_target=self.config.profit_target,
            stop_loss=self.config.stop_loss,
            fee_threshold=self.config.fee_threshold,
        )
        
        up_pct = labels.mean() * 100
        log.info(f"   📈 Labels: UP={up_pct:.1f}%, DOWN={100-up_pct:.1f}%")
        log.info(f"   💰 Fee threshold: {self.config.fee_threshold:.2%}")
        
        return labels
    
    def compute_returns(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Calcule les returns forward."""
        close = df['close'].values
        returns = np.zeros(len(close))
        returns[:-horizon] = close[horizon:] / close[:-horizon] - 1
        return returns

    # ------------------------------------------------------------------ high level
    def engineer(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Calcule features, labels et returns, et assemble dans un DataFrame unique.
        """
        feat_df = self.compute_features(df)
        if include_target:
            labels = self.compute_labels(df)
            target_returns = self.compute_returns(df, horizon=self.config.horizon)
            feat_df["target"] = labels
            feat_df["target_return"] = target_returns
            self.feature_names = [c for c in feat_df.columns if c not in ("target", "target_return")]
        else:
            self.feature_names = list(feat_df.columns)
        
        return feat_df


# ============================================================================
# NORMALISATION ROLLING (ANTI-LEAKAGE)
# ============================================================================

def normalize_features_rolling(
    features: pd.DataFrame | np.ndarray,
    window: int = 500,
    min_periods: int = 100,
) -> np.ndarray:
    """
    Normalisation rolling: utilise UNIQUEMENT les données passées.
    
    Évite le leakage: à chaque point t, on normalise avec les stats de [t-window, t-1].
    """
    log.info(f"🔄 Normalisation rolling (window={window})")
    
    # Assure une DataFrame pour le calcul, mais retourne un ndarray
    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    else:
        df = features.copy()
    
    result = df.copy()
    
    for col in df.columns:
        rolling_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(window=window, min_periods=min_periods).std()
        
        # Shift de 1 pour éviter le leakage (on utilise stats jusqu'à t-1)
        rolling_mean = rolling_mean.shift(1)
        rolling_std = rolling_std.shift(1)
        
        # Normalisation
        result[col] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
    
    # Clip des valeurs extrêmes
    result = result.clip(-5, 5)
    
    # Remplir les NaN du début
    result = result.fillna(0)
    
    log.info(f"   ✅ Normalisation terminée")
    
    return result.values.astype(np.float32)


# ============================================================================
# PIPELINE COMPLET
# ============================================================================

def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    seq_length: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crée les séquences pour le Transformer."""
    n_samples = len(features) - seq_length
    n_features = features.shape[1]
    
    X = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    r = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = features[i:i + seq_length]
        y[i] = labels[i + seq_length]
        r[i] = returns[i + seq_length]
    
    log.info(f"   📦 Séquences: {n_samples:,} x {seq_length} x {n_features}")
    
    return X, y, r


def prepare_dataset(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    seq_length: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Pipeline complet: features + labels + normalisation + séquences.
    """
    config = config or FeatureConfig()
    engineer = FeatureEngineerPro(config)
    
    # Features
    features_df = engineer.compute_features(df)
    
    # Normalisation rolling
    features_norm = normalize_features_rolling(
        features_df,
        window=config.rolling_window,
        min_periods=config.min_samples,
    )
    
    # Labels RÉALISTES
    labels = engineer.compute_labels(df)
    
    # Returns forward
    returns = engineer.compute_returns(df, horizon=1)
    
    # Séquences
    X, y, r = create_sequences(
        features_norm,
        labels,
        returns,
        seq_length=seq_length,
    )
    
    # Suppression des samples avec NaN
    valid_mask = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid_mask]
    y = y[valid_mask]
    r = r[valid_mask]
    
    log.info(f"✅ Dataset prêt: {len(X):,} samples, {X.shape[2]} features")
    
    return X, y, r, engineer.feature_names


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("FEATURES PRO - VERSION RÉALISTE")
    print("=" * 60)
    print(f"FEE_THRESHOLD: {FEE_THRESHOLD:.2%}")
    print(f"PROFIT_TARGET: {PROFIT_TARGET:.2%}")
    print()
    
    # Test avec données synthétiques
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 101,
        'low': np.random.randn(n).cumsum() + 99,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.rand(n) * 1000,
    })
    
    config = FeatureConfig()
    X, y, r, names = prepare_dataset(df, config, seq_length=32)
    
    print(f"\n✅ Test réussi!")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features: {len(names)}")
