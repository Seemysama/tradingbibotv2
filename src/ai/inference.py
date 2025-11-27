import logging
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.ai.model import CryptoLSTM
from src.config import settings

log = logging.getLogger(__name__)

FEATURES = ["close", "volume", "RSI_14", "ATR_14", "BBP_20_2.0", "ADX_14"]


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule RSI, ATR, Bollinger %B et ADX."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    roll_up = gain.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Bollinger %B (20, 2)
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BBP_20_2.0"] = (close - lower) / (upper - lower)

    # ADX 14
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX_14"] = (dx * 100).ewm(alpha=1 / 14, adjust=False).mean()

    return df


def _scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df[FEATURES].values)


class InferenceEngine:
    """Moteur d'inférence robuste, ne bloque jamais le thread."""

    def __init__(
        self,
        model_path: Union[Path, str] = settings.ML_MODEL_PATH,
        device: str = settings.DEVICE,
        seq_length: int = settings.ML_SEQ_LENGTH,
        buffer_size: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.seq_length = seq_length
        self.model: Optional[CryptoLSTM] = None
        self.loaded = False
        self.input_dim = len(FEATURES)
        self.buffer: Deque[dict] = deque(maxlen=buffer_size)
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning("Modèle ML introuvable à %s (mode neutre).", self.model_path)
            return

        try:
            self.model = CryptoLSTM(input_dim=self.input_dim).to(self.device)
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.loaded = True
            log.info("Modèle ML chargé depuis %s sur %s", self.model_path, self.device)
        except Exception as exc:  # pragma: no cover - robust load
            log.exception("Échec de chargement du modèle ML: %s", exc)
            self.model = None
            self.loaded = False

    def update_buffer(self, candle: Union[dict, pd.Series]) -> None:
        """Alimente le buffer interne avec une bougie (dict ou Series)."""
        if isinstance(candle, pd.Series):
            candle = candle.to_dict()
        self.buffer.append(candle)

    def _buffer_dataframe(self) -> pd.DataFrame:
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(self.buffer)

    def predict(self, candles: Optional[pd.DataFrame] = None) -> Tuple[float, bool]:
        """Retourne (proba, ready). Ne lève pas d'exception."""
        if not self.loaded or self.model is None:
            return 0.5, False

        if candles is None:
            candles = self._buffer_dataframe()

        if len(candles) < self.seq_length + 20:
            return 0.5, False

        try:
            df = candles.tail(self.seq_length + 200)
            df = _compute_indicators(df)
            df = df.dropna()
            if len(df) < self.seq_length:
                return 0.5, False

            window = df.tail(self.seq_length)
            feats = _scale_features(window)
            x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(x)
                prob = torch.sigmoid(out).squeeze().item()
                return float(prob), True
        except Exception as exc:  # pragma: no cover - robust inference
            log.exception("Erreur d'inférence ML: %s", exc)
            return 0.5, False
