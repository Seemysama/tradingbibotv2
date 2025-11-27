#!/usr/bin/env python3
"""
🧠 MASSIVE TRAINING SCRIPT
Entraînement sur données massives (2+ ans de données 1min)

Features:
- Chargement par chunks pour gérer la RAM
- Architecture LSTM + Attention
- Support MPS (Mac M1/M2) / CUDA / CPU
- Sauvegarde avec métadonnées scaler
- Early stopping et gradient clipping

Usage:
    python train_massive.py
    python train_massive.py --epochs 50 --batch 2048
    python train_massive.py --symbol ETH/USDT --lr 0.0005
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.model import CryptoLSTMAttention, save_checkpoint
from src.config import settings


# ============================================================================
# HYPERPARAMÈTRES PAR DÉFAUT
# ============================================================================
DEFAULT_CONFIG = {
    "seq_length": 60,       # Fenêtre temporelle (60 bougies = 1h en 1min)
    "batch_size": 1024,     # Gros batch pour gros dataset
    "epochs": 30,
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "weight_decay": 1e-5,
    "clip_grad": 1.0,
    "patience": 5,          # Early stopping patience
    "target_horizon": 5,    # Prédiction à N bougies
}

# Features à utiliser
FEATURES = ['close', 'volume', 'RSI_14', 'ATR_14', 'ADX_14', 'BBP_20_2.0']


# ============================================================================
# DATA PREPARATION
# ============================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques de manière vectorisée."""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Bollinger %B (20, 2)
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BBP_20_2.0'] = (close - lower) / (upper - lower)

    # ADX 14
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['ADX_14'] = (dx * 100).ewm(alpha=1/14, adjust=False).mean()

    return df


def prepare_data(
    filepath: Path,
    seq_length: int,
    target_horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray, int, int, MinMaxScaler, List[str]]:
    """
    Prépare les données pour l'entraînement.
    
    Returns:
        X_raw, y_raw, split_idx, input_dim, scaler, feature_names
    """
    print(f"\n📂 Chargement: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"   {len(df):,} bougies chargées")

    # Calculer les indicateurs
    print("⚙️ Calcul des indicateurs techniques...")
    df = compute_indicators(df)
    
    # Target: Log returns à horizon N
    print(f"🎯 Création target (horizon={target_horizon} bougies)...")
    df['target'] = np.log(df['close'].shift(-target_horizon) / df['close'])
    
    # Supprimer NaN
    df = df.dropna().reset_index(drop=True)
    print(f"   {len(df):,} bougies après nettoyage")

    # Sélectionner les features disponibles
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"🧠 Features: {available_features}")

    # Scaling
    print("📊 Normalisation...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(df[available_features].values)
    y = df['target'].values.reshape(-1, 1)

    # Split Train/Val (80/20) - PAS de shuffle (série temporelle!)
    split_idx = int(len(X) * 0.8)
    print(f"✂️ Split: Train={split_idx:,} | Val={len(X)-split_idx:,}")

    return X, y, split_idx, len(available_features), scaler, available_features


def create_sequences_vectorized(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crée les séquences de manière vectorisée (rapide).
    Utilise numpy stride_tricks pour éviter les boucles.
    """
    num_samples = len(data) - seq_length
    
    # Créer les fenêtres glissantes
    shape = (num_samples, seq_length, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    # Aligner les targets
    y_seq = target[seq_length:seq_length + num_samples]

    # Copier pour éviter les problèmes de mémoire avec stride_tricks
    X_seq = np.array(X_seq)
    
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


# ============================================================================
# TRAINING
# ============================================================================
class EarlyStopping:
    """Early stopping pour éviter l'overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(config: Dict) -> None:
    """Boucle d'entraînement principale."""
    
    # Device
    device = torch.device(settings.DEVICE)
    print(f"\n⚡ Device: {device}")
    if device.type == "mps":
        print("   🍎 Apple Silicon MPS activé")
    elif device.type == "cuda":
        print(f"   🎮 CUDA: {torch.cuda.get_device_name(0)}")

    # Data path
    symbol = config.get("symbol", "BTC/USDT").replace("/", "_")
    data_path = settings.DATA_PATH / f"{symbol}_BULK.parquet"
    
    if not data_path.exists():
        # Fallback sur les données existantes
        alt_paths = [
            settings.DATA_PATH / f"{symbol}_1m_2Y.parquet",
            settings.DATA_PATH / f"{symbol}_1m_FULL.parquet",
            settings.DATA_PATH / f"{symbol}_1m.parquet",
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            print(f"❌ Aucune donnée trouvée pour {symbol}")
            print("   Lancez d'abord: python tools/download_bulk.py")
            return

    # Préparation des données
    X_raw, y_raw, split_idx, input_dim, scaler, feature_names = prepare_data(
        data_path,
        config["seq_length"],
        config["target_horizon"]
    )

    # Création des séquences
    print("\n🔧 Création des séquences...")
    X_train, y_train = create_sequences_vectorized(
        X_raw[:split_idx], 
        y_raw[:split_idx], 
        config["seq_length"]
    )
    X_val, y_val = create_sequences_vectorized(
        X_raw[split_idx:], 
        y_raw[split_idx:], 
        config["seq_length"]
    )
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # MPS ne supporte pas bien le multiprocessing
        pin_memory=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config["batch_size"],
        shuffle=False
    )

    # Modèle
    print(f"\n🧠 Création du modèle LSTM+Attention...")
    model = CryptoLSTMAttention(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=1,
        dropout=config["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {total_params:,}")

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=config["patience"])

    # Training loop
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    print(f"\n{'='*60}")
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"{'='*60}\n")

    for epoch in range(config["epochs"]):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = train_loss / len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Historique
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Affichage
        print(f"\n📉 Epoch {epoch+1}: Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f}")

        # Scheduler
        scheduler.step(avg_val_loss)

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Préparer les métadonnées du scaler
            scaler_params = {
                feature_names[i]: {
                    "min": float(scaler.data_min_[i]),
                    "max": float(scaler.data_max_[i]),
                    "scale": float(scaler.scale_[i])
                }
                for i in range(len(feature_names))
            }
            
            # Sauvegarder avec la fonction du modèle
            model_path = Path("models") / "best_massive_v2.pth"
            save_checkpoint(
                model=model,
                filepath=model_path,
                feature_names=feature_names,
                scaler_params=scaler_params,
                seq_length=config["seq_length"],
                training_metrics={
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                },
                version="2.0.0"
            )
            print(f"💾 Nouveau meilleur modèle sauvegardé! (val_loss={best_val_loss:.6f})")

        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"\n⏹️ Early stopping déclenché après {epoch+1} epochs")
            break

    # === FIN ===
    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"   Meilleur Val Loss: {best_val_loss:.6f}")
    print(f"   Modèle sauvegardé: models/best_massive_v2.pth")
    
    # Sauvegarder l'historique
    history_path = Path("models") / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"   Historique: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Massive Training Script")
    
    parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Symbole à entraîner")
    parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch", "-b", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--seq-length", type=int, default=DEFAULT_CONFIG["seq_length"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    
    args = parser.parse_args()
    
    config = {
        **DEFAULT_CONFIG,
        "symbol": args.symbol,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "seq_length": args.seq_length,
        "hidden_dim": args.hidden_dim,
        "patience": args.patience,
    }
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           🧠 MASSIVE TRAINING - LSTM + ATTENTION             ║
║                    Trading Engine v2                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    train(config)


if __name__ == "__main__":
    main()
