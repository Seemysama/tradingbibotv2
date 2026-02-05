#!/usr/bin/env python3
"""
FEATURE ENGINEERING PIPELINE - Hedge Fund Grade
================================================
Pipeline de génération de features avancées pour ML:
- Volatilité (GARCH-like, ATR normalisé)
- Momentum (RSI, ROC multi-fenêtres)
- Volume Profile (VWAP, Volume MA ratio)
- Microstructure (Spread proxy, Order Flow)
- Targets pour classification et régression

Usage:
    from src.features import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.transform(raw_df)
"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats

log = logging.getLogger("features")


# ============================================================================
# NUMBA-ACCELERATED COMPUTATIONS
# ============================================================================

@jit(nopython=True, cache=True)
def _ewm_numba(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential Weighted Mean optimisé avec Numba."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


@jit(nopython=True, cache=True)
def _rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    """RSI optimisé avec Numba."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True, cache=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR optimisé avec Numba."""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.empty(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Premier ATR = moyenne simple
    atr[period - 1] = np.mean(tr[:period])

    # Suite = EMA
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True, cache=True)
def _rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation optimisé."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1:i + 1])
    
    return result


# ============================================================================
# FEATURE ENGINEER CLASS
# ============================================================================

class FeatureEngineer:
    """
    Pipeline de feature engineering pour séries temporelles financières.
    
    Features générées:
    - Volatilité: ATR, Realized Vol, Parkinson Vol
    - Momentum: RSI, ROC, MACD
    - Trend: EMAs, ADX
    - Volume: VWAP, Volume ratio, OBV
    - Microstructure: High-Low ratio, Close position
    - Targets: Classification (direction) et Régression (returns)
    """

    # Paramètres par défaut
    DEFAULT_WINDOWS = [5, 10, 20, 50, 100]
    RSI_PERIODS = [7, 14, 21]
    ROC_PERIODS = [1, 5, 10, 20]
    ATR_PERIOD = 14
    ADX_PERIOD = 14

    # Seuil pour classification (en % après frais)
    CLASSIFICATION_THRESHOLD = 0.001  # 0.1% = ~10 bps
    TRADING_FEES = 0.0004  # 4 bps maker

    # Horizons de prédiction
    PREDICTION_HORIZONS = [5, 15, 30, 60]  # En nombre de barres

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        rsi_periods: Optional[List[int]] = None,
        prediction_horizons: Optional[List[int]] = None,
        include_targets: bool = True,
        normalize: bool = True,
    ):
        self.windows = windows or self.DEFAULT_WINDOWS
        self.rsi_periods = rsi_periods or self.RSI_PERIODS
        self.prediction_horizons = prediction_horizons or self.PREDICTION_HORIZONS
        self.include_targets = include_targets
        self.normalize = normalize

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme un DataFrame OHLCV en features ML.
        
        Args:
            df: DataFrame avec colonnes [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame avec features et targets
        """
        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Vérification des colonnes requises
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        log.info(f"Feature engineering sur {len(df):,} lignes...")

        # 1. Returns basiques
        df = self._add_returns(df)

        # 2. Volatilité
        df = self._add_volatility(df)

        # 3. Momentum
        df = self._add_momentum(df)

        # 4. Trend
        df = self._add_trend(df)

        # 5. Volume
        df = self._add_volume_features(df)

        # 6. Microstructure / Price Action
        df = self._add_microstructure(df)

        # 7. Targets
        if self.include_targets:
            df = self._add_targets(df)

        # 8. Normalisation (optionnelle)
        if self.normalize:
            df = self._normalize_features(df)

        # 9. Clean up
        df = df.replace([np.inf, -np.inf], np.nan)

        log.info(f"Features générées: {len([c for c in df.columns if c.startswith('feat_')])} features")

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-returns sur plusieurs horizons."""
        close = df["close"].values

        # Returns simples
        df["feat_ret_1"] = np.log(close / np.roll(close, 1))
        df["feat_ret_1"].iloc[0] = 0

        for w in [5, 10, 20]:
            df[f"feat_ret_{w}"] = np.log(close / np.roll(close, w))
            df[f"feat_ret_{w}"].iloc[:w] = 0

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilité."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # ATR normalisé
        atr = _atr_numba(high, low, close, self.ATR_PERIOD)
        df["feat_atr"] = atr / close  # Normalisé par prix

        # Realized volatility (rolling std of returns)
        returns = np.log(close / np.roll(close, 1))
        returns[0] = 0
        
        for w in [10, 20, 50]:
            df[f"feat_realized_vol_{w}"] = _rolling_std_numba(returns, w) * np.sqrt(w)

        # Parkinson volatility (high-low based)
        parkinson = np.log(high / low) ** 2 / (4 * np.log(2))
        for w in [10, 20]:
            df[f"feat_parkinson_vol_{w}"] = pd.Series(parkinson).rolling(w).mean().values

        # Volatility ratio (short/long)
        df["feat_vol_ratio"] = df["feat_realized_vol_10"] / (df["feat_realized_vol_50"] + 1e-8)

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum."""
        close = df["close"].values

        # RSI multi-périodes
        for period in self.rsi_periods:
            rsi = _rsi_numba(close, period)
            df[f"feat_rsi_{period}"] = rsi / 100.0  # Normalisé 0-1

        # ROC (Rate of Change)
        for period in self.ROC_PERIODS:
            roc = (close - np.roll(close, period)) / (np.roll(close, period) + 1e-8)
            df[f"feat_roc_{period}"] = roc
            df[f"feat_roc_{period}"].iloc[:period] = 0

        # MACD
        ema12 = _ewm_numba(close, 12)
        ema26 = _ewm_numba(close, 26)
        macd = ema12 - ema26
        signal = _ewm_numba(macd, 9)
        
        df["feat_macd"] = macd / close  # Normalisé
        df["feat_macd_signal"] = signal / close
        df["feat_macd_hist"] = (macd - signal) / close

        # Stochastic
        for w in [14, 21]:
            lowest = pd.Series(df["low"]).rolling(w).min()
            highest = pd.Series(df["high"]).rolling(w).max()
            df[f"feat_stoch_{w}"] = (close - lowest) / (highest - lowest + 1e-8)

        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de tendance."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # EMAs et distance au prix
        for w in self.windows:
            ema = _ewm_numba(close, w)
            df[f"feat_ema_{w}"] = ema
            df[f"feat_dist_ema_{w}"] = (close - ema) / ema  # Distance normalisée

        # Pente des EMAs
        for w in [20, 50]:
            ema = df[f"feat_ema_{w}"].values
            slope = (ema - np.roll(ema, 5)) / (np.roll(ema, 5) + 1e-8)
            df[f"feat_ema_slope_{w}"] = slope
            df[f"feat_ema_slope_{w}"].iloc[:5] = 0

        # ADX (Average Directional Index)
        df = self._add_adx(df, high, low, close)

        # Trend strength (prix au-dessus/dessous des EMAs)
        df["feat_trend_score"] = (
            (close > df["feat_ema_10"].values).astype(float) +
            (close > df["feat_ema_20"].values).astype(float) +
            (close > df["feat_ema_50"].values).astype(float) +
            (close > df["feat_ema_100"].values).astype(float)
        ) / 4.0

        return df

    def _add_adx(self, df: pd.DataFrame, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Calcul ADX."""
        period = self.ADX_PERIOD
        n = len(close)

        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth
        atr = _ewm_numba(tr, period)
        plus_di = 100 * _ewm_numba(plus_dm, period) / (atr + 1e-8)
        minus_di = 100 * _ewm_numba(minus_dm, period) / (atr + 1e-8)

        # DX et ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = _ewm_numba(dx, period)

        df["feat_adx"] = adx / 100.0  # Normalisé 0-1
        df["feat_plus_di"] = plus_di / 100.0
        df["feat_minus_di"] = minus_di / 100.0
        df["feat_di_diff"] = (plus_di - minus_di) / 100.0

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volume."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # Volume MA ratio
        for w in [10, 20, 50]:
            vol_ma = pd.Series(volume).rolling(w).mean().values
            df[f"feat_vol_ratio_{w}"] = volume / (vol_ma + 1e-8)

        # VWAP
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)
        vwap = cumulative_tp_vol / (cumulative_vol + 1e-8)
        
        df["feat_vwap_dist"] = (close - vwap) / vwap

        # OBV (On Balance Volume) normalisé
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope
        obv_ema = _ewm_numba(obv, 20)
        df["feat_obv_slope"] = (obv_ema - np.roll(obv_ema, 5)) / (np.abs(np.roll(obv_ema, 5)) + 1e-8)
        df["feat_obv_slope"].iloc[:5] = 0

        # Volume profile (accumulation/distribution)
        mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
        mfv = mfm * volume
        df["feat_mfv_sum_20"] = pd.Series(mfv).rolling(20).sum().values / (pd.Series(volume).rolling(20).sum().values + 1e-8)

        return df

    def _add_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de microstructure."""
        open_p = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Close position in range
        df["feat_close_position"] = (close - low) / (high - low + 1e-8)

        # Body ratio (corps de la bougie vs range total)
        body = np.abs(close - open_p)
        range_total = high - low
        df["feat_body_ratio"] = body / (range_total + 1e-8)

        # Upper/Lower shadow
        upper_shadow = high - np.maximum(open_p, close)
        lower_shadow = np.minimum(open_p, close) - low
        df["feat_upper_shadow"] = upper_shadow / (range_total + 1e-8)
        df["feat_lower_shadow"] = lower_shadow / (range_total + 1e-8)

        # Candle direction
        df["feat_candle_dir"] = np.sign(close - open_p)

        # Consecutive candles
        directions = np.sign(close - open_p)
        consecutive = np.zeros(len(close))
        for i in range(1, len(close)):
            if directions[i] == directions[i - 1]:
                consecutive[i] = consecutive[i - 1] + directions[i]
            else:
                consecutive[i] = directions[i]
        df["feat_consecutive"] = consecutive / 10.0  # Normalisé

        # Gap (overnight ou intraday)
        gap = (open_p - np.roll(close, 1)) / np.roll(close, 1)
        gap[0] = 0
        df["feat_gap"] = gap

        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les targets pour ML."""
        close = df["close"].values
        threshold = self.CLASSIFICATION_THRESHOLD + self.TRADING_FEES * 2  # Aller-retour

        for horizon in self.prediction_horizons:
            # Return futur (régression target)
            future_return = np.roll(close, -horizon) / close - 1
            future_return[-horizon:] = np.nan
            df[f"target_ret_{horizon}"] = future_return

            # Log-return futur
            df[f"target_logret_{horizon}"] = np.log(1 + future_return)

            # Classification (direction avec seuil)
            df[f"target_dir_{horizon}"] = np.where(
                future_return > threshold, 1,  # Long
                np.where(future_return < -threshold, -1, 0)  # Short ou Hold
            )

            # Classification binaire (Up/Down seulement)
            df[f"target_binary_{horizon}"] = (future_return > 0).astype(int)

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise les features avec rolling z-score."""
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        
        for col in feature_cols:
            # Skip already normalized features
            if any(x in col for x in ["_ratio", "_dist", "_slope", "_dir", "_position"]):
                continue
            
            # Rolling z-score (lookback 100)
            rolling_mean = df[col].rolling(100, min_periods=20).mean()
            rolling_std = df[col].rolling(100, min_periods=20).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        return df

    def get_feature_names(self) -> List[str]:
        """Retourne la liste des noms de features générées."""
        # Simulation sur un petit df pour obtenir les noms
        dummy = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=200, freq="1min"),
            "open": np.random.randn(200).cumsum() + 100,
            "high": np.random.randn(200).cumsum() + 101,
            "low": np.random.randn(200).cumsum() + 99,
            "close": np.random.randn(200).cumsum() + 100,
            "volume": np.abs(np.random.randn(200)) * 1000,
        })
        df = self.transform(dummy)
        return [c for c in df.columns if c.startswith("feat_")]


def prepare_training_data(
    parquet_path: Path,
    target_col: str = "target_dir_15",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sequence_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prépare les données pour l'entraînement avec séquences.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    log.info(f"Chargement des données depuis {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.transform(df)

    # Drop NaN
    df = df.dropna()
    log.info(f"Données après nettoyage: {len(df):,} lignes")

    # Feature columns
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    
    # Préparation des séquences
    X = df[feature_cols].values
    y = df[target_col].values

    # Créer séquences
    n_samples = len(X) - sequence_length
    X_seq = np.zeros((n_samples, sequence_length, len(feature_cols)), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X_seq[i] = X[i:i + sequence_length]
        y_seq[i] = y[i + sequence_length]

    # Split temporel (pas de shuffle pour éviter look-ahead)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    log.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    fe = FeatureEngineer()
    features = fe.get_feature_names()
    print(f"Nombre de features: {len(features)}")
    for f in sorted(features):
        print(f"  - {f}")
