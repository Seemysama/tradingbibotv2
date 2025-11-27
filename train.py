import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Hack path so imports work when running from repo root
sys.path.append(os.getcwd())
from config import settings
from core.ai_models import CryptoLSTM  # rétrocompatibilité
from src.ai.advanced_model import AttentionLSTM, train_with_validation

# --- Hyperparameters ---
SEQ_LENGTH = 60  # Look-back window
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
TARGET_HORIZON = 5  # Predict 5 minutes ahead


def _find_parquet(symbol: str) -> Optional[Path]:
    """Locate the parquet file for a symbol, trying common slugs."""
    slug_underscore = symbol.replace("/", "_")
    slug_compact = symbol.replace("/", "")
    candidates = [
        settings.DATA_PATH / f"{slug_underscore}_{settings.TIMEFRAME}.parquet",
        settings.DATA_PATH / f"{slug_compact}_{settings.TIMEFRAME}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few classic indicators (RSI, ATR, Bollinger %B, ADX)."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI 14 (Wilder's smoothing)
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

    # Bollinger Bands %B (20, 2.0)
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BBP_20_2.0"] = (close - lower) / (upper - lower)

    # ADX 14 (simplified Wilder)
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


def load_and_prep_data(symbol: str) -> Tuple[np.ndarray, np.ndarray, int]:
    path = _find_parquet(symbol)
    if not path:
        raise FileNotFoundError(f"Parquet introuvable pour {symbol} dans {settings.DATA_PATH}")

    print(f"📂 Chargement {path}...")
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = compute_indicators(df)

    # Target: log-return at horizon
    df["target"] = np.log(df["close"].shift(-TARGET_HORIZON) / df["close"])
    df = df.dropna().reset_index(drop=True)

    feature_names = ["close", "volume", "RSI_14", "ATR_14", "BBP_20_2.0", "ADX_14"]
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après préparation: {missing}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = scaler.fit_transform(df[feature_names].values)
    target = df["target"].values.reshape(-1, 1)
    return features_scaled, target, len(feature_names)


def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)


def train() -> None:
    symbol = settings.PAIRS[0]
    data, target, input_dim = load_and_prep_data(symbol)

    if len(data) <= SEQ_LENGTH:
        raise ValueError("Pas assez de données pour créer des séquences.")

    print("✂️  Création des séquences...")
    X, y = create_sequences(data, target, SEQ_LENGTH)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    device = torch.device(settings.DEVICE)
    print(f"🧠 Entraînement sur : {device}")

    # Entraînement modèle avancé avec validation/early stopping
    model, best_val = train_with_validation(
        X_tensor,
        y_tensor,
        input_dim=input_dim,
        device=device,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=5,
        model_path=Path("models/best_model.pth"),
    )
    print(f"✅ Modèle avancé sauvegardé (val_loss={best_val:.6f}) : models/best_model.pth")


if __name__ == "__main__":
    train()
