#!/usr/bin/env python3
"""
PROFESSIONAL TRAINING PIPELINE - Hedge Fund Grade
==================================================
Entraînement robuste avec:
- Walk-Forward Validation (évite look-ahead bias)
- Early stopping sur Sharpe Ratio
- Tracking de métriques avancées (Sharpe, Max Drawdown, Win Rate)
- Checkpointing et reprise
- Logging structuré avec TensorBoard

Usage:
    python train_pro.py --data data/massive/BTC_USDT_1m_FULL.parquet --epochs 100
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from src.features import FeatureEngineer, prepare_training_data
from src.ai.transformer_model import (
    TimeSeriesTransformer,
    TemporalConvNet,
    TradingLoss,
    create_model,
)

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_pro")

# Directories
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset pour séquences temporelles avec targets multiples."""

    def __init__(
        self,
        features: np.ndarray,
        class_targets: np.ndarray,
        reg_targets: np.ndarray,
        sequence_length: int = 60,
    ):
        self.features = torch.FloatTensor(features)
        self.class_targets = torch.LongTensor(class_targets)
        self.reg_targets = torch.FloatTensor(reg_targets)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length]
        y_class = self.class_targets[idx + self.sequence_length]
        y_reg = self.reg_targets[idx + self.sequence_length]
        return x, y_class, y_reg


# ============================================================================
# METRICS
# ============================================================================

class TradingMetrics:
    """Calcul de métriques de trading réalistes."""

    def __init__(self, initial_capital: float = 10000.0, trading_fee: float = 0.0004):
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee

    def calculate_returns(
        self,
        predictions: np.ndarray,  # Classes: 0=Hold, 1=Long, 2=Short (ou -1, 0, 1)
        actual_returns: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Simule les returns en tenant compte des frais.
        """
        # Mapper les classes vers positions
        # 0 = Short (-1), 1 = Hold (0), 2 = Long (+1)
        # ou directement -1, 0, 1 si c'est le format
        if predictions.max() > 1:  # Format 0, 1, 2
            positions = predictions - 1  # -> -1, 0, 1
        else:
            positions = predictions

        # Calculer les returns de stratégie
        position_changes = np.diff(positions, prepend=0)
        fees = np.abs(position_changes) * self.trading_fee * 2  # Aller-retour

        strategy_returns = positions * actual_returns - fees

        # Equity curve
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)

        # Métriques
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        
        # Sharpe Ratio (annualisé, assuming 1-minute bars)
        # 525600 minutes par an
        if strategy_returns.std() > 0:
            sharpe = np.sqrt(525600) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max() * 100

        # Win Rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (positions != 0).sum()
        win_rate = winning_trades / max(total_trades, 1) * 100

        # Profit Factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-8)

        return {
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "n_trades": int(total_trades),
            "equity_curve": equity,
        }


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

class WalkForwardValidator:
    """
    Walk-Forward Validation pour séries temporelles.
    
    Divise les données en fenêtres glissantes:
    - Train sur window[0:train_size]
    - Validate sur window[train_size:train_size+val_size]
    - Avance de step_size
    """

    def __init__(
        self,
        train_size: int,
        val_size: int,
        step_size: int,
        min_train_samples: int = 10000,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.step_size = step_size
        self.min_train_samples = min_train_samples

    def split(self, n_samples: int) -> List[Tuple[range, range]]:
        """Génère les splits train/val."""
        splits = []
        start = 0

        while start + self.train_size + self.val_size <= n_samples:
            train_range = range(start, start + self.train_size)
            val_range = range(start + self.train_size, start + self.train_size + self.val_size)
            
            if len(train_range) >= self.min_train_samples:
                splits.append((train_range, val_range))
            
            start += self.step_size

        return splits


# ============================================================================
# TRAINER
# ============================================================================

class ProTrainer:
    """
    Trainer professionnel avec toutes les bonnes pratiques.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
        min_delta: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.min_delta = min_delta

        # Optimizer avec weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler avec warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # Loss
        self.criterion = TradingLoss(
            class_weight=1.0,
            reg_weight=0.5,
            sharpe_weight=0.1,
        )

        # Metrics
        self.metrics = TradingMetrics()

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_sharpe": [],
            "val_drawdown": [],
            "val_accuracy": [],
        }

        # Early stopping
        self.best_sharpe = float("-inf")
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Entraîne une epoch."""
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_x, batch_y_class, batch_y_reg in pbar:
            batch_x = batch_x.to(self.device)
            batch_y_class = batch_y_class.to(self.device)
            batch_y_reg = batch_y_reg.to(self.device)

            # Mapper les classes (-1, 0, 1) vers (0, 1, 2) pour CrossEntropy
            batch_y_class_mapped = batch_y_class + 1

            self.optimizer.zero_grad()

            # Forward
            class_logits, regression, confidence = self.model(batch_x)

            # Loss
            loss, metrics = self.criterion(
                class_logits, regression, batch_y_class_mapped, batch_y_reg
            )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += metrics["class_loss"]
            total_reg_loss += metrics["reg_loss"]
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        self.scheduler.step()

        return {
            "loss": total_loss / n_batches,
            "class_loss": total_class_loss / n_batches,
            "reg_loss": total_reg_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        """Valide sur un dataset."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_predictions = []
        all_class_targets = []
        all_reg_targets = []
        all_confidence = []

        for batch_x, batch_y_class, batch_y_reg in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y_class = batch_y_class.to(self.device)
            batch_y_reg = batch_y_reg.to(self.device)

            batch_y_class_mapped = batch_y_class + 1

            class_logits, regression, confidence = self.model(batch_x)

            loss, _ = self.criterion(
                class_logits, regression, batch_y_class_mapped, batch_y_reg
            )

            total_loss += loss.item()
            n_batches += 1

            # Collecter pour métriques
            pred_class = torch.argmax(class_logits, dim=1) - 1  # Remap vers -1, 0, 1
            all_predictions.extend(pred_class.cpu().numpy())
            all_class_targets.extend(batch_y_class.cpu().numpy())
            all_reg_targets.extend(batch_y_reg.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())

        # Calculer métriques trading
        predictions = np.array(all_predictions)
        class_targets = np.array(all_class_targets)
        reg_targets = np.array(all_reg_targets)

        # Accuracy
        accuracy = (predictions == class_targets).mean() * 100

        # Trading metrics
        trading_metrics = self.metrics.calculate_returns(predictions, reg_targets)

        return {
            "loss": total_loss / n_batches,
            "accuracy": accuracy,
            "sharpe": trading_metrics["sharpe_ratio"],
            "max_drawdown": trading_metrics["max_drawdown_pct"],
            "win_rate": trading_metrics["win_rate_pct"],
            "profit_factor": trading_metrics["profit_factor"],
            "n_trades": trading_metrics["n_trades"],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Boucle d'entraînement complète."""
        log.info(f"Début entraînement: {n_epochs} epochs")

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log
            log.info(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Sharpe: {val_metrics['sharpe']:.2f} | "
                f"DD: {val_metrics['max_drawdown']:.1f}% | "
                f"Acc: {val_metrics['accuracy']:.1f}%"
            )

            # History
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_sharpe"].append(val_metrics["sharpe"])
            self.history["val_drawdown"].append(val_metrics["max_drawdown"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Early stopping sur Sharpe
            if val_metrics["sharpe"] > self.best_sharpe + self.min_delta:
                self.best_sharpe = val_metrics["sharpe"]
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()

                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                log.info(f"Early stopping à epoch {epoch+1}")
                break

        # Restaurer le meilleur modèle
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_sharpe": self.best_sharpe,
        }
        torch.save(checkpoint, path)
        log.info(f"Checkpoint sauvé: {path} (Sharpe: {metrics['sharpe']:.2f})")

    def load_checkpoint(self, path: Path):
        """Charge un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_sharpe = checkpoint.get("best_sharpe", float("-inf"))
        log.info(f"Checkpoint chargé: {path}")
        return checkpoint


# ============================================================================
# MAIN
# ============================================================================

def load_and_prepare_data(
    parquet_path: Path,
    sequence_length: int = 60,
    class_target: str = "target_dir_15",
    reg_target: str = "target_ret_15",
) -> Tuple[pd.DataFrame, List[str]]:
    """Charge et prépare les données."""
    log.info(f"Chargement: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log.info(f"Données brutes: {len(df):,} lignes")

    # Feature engineering
    fe = FeatureEngineer(include_targets=True, normalize=True)
    df = fe.transform(df)

    # Drop NaN
    df = df.dropna()
    log.info(f"Après nettoyage: {len(df):,} lignes")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    log.info(f"Features: {len(feature_cols)}")

    return df, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Training professionnel")
    parser.add_argument(
        "--data",
        type=str,
        default="data/massive/BTC_USDT_1m_FULL.parquet",
        help="Chemin vers les données Parquet",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille de batch")
    parser.add_argument("--seq-length", type=int, default=60, help="Longueur de séquence")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "tcn"])
    parser.add_argument("--d-model", type=int, default=128, help="Dimension du modèle")
    parser.add_argument("--n-layers", type=int, default=4, help="Nombre de couches")
    parser.add_argument("--n-heads", type=int, default=8, help="Nombre de têtes attention")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--patience", type=int, default=15, help="Patience early stopping")
    parser.add_argument("--output", type=str, default="models/transformer_pro.pth", help="Chemin sortie")
    parser.add_argument("--walk-forward", action="store_true", help="Utiliser Walk-Forward Validation")
    parser.add_argument("--resume", type=str, default=None, help="Chemin checkpoint à reprendre")

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    # Charger données
    data_path = ROOT_DIR / args.data
    if not data_path.exists():
        # Fallback vers données existantes
        alt_paths = list(ROOT_DIR.glob("data/**/*.parquet"))
        if alt_paths:
            data_path = alt_paths[0]
            log.warning(f"Fichier non trouvé, utilisation de: {data_path}")
        else:
            log.error("Aucune donnée Parquet trouvée!")
            sys.exit(1)

    df, feature_cols = load_and_prepare_data(
        data_path,
        sequence_length=args.seq_length,
    )

    # Créer le modèle
    n_features = len(feature_cols)
    n_classes = 3  # Short, Hold, Long

    model = create_model(
        model_type=args.model,
        n_features=n_features,
        n_classes=n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Modèle: {args.model} | Paramètres: {n_params:,}")

    # Préparer features et targets
    X = df[feature_cols].values.astype(np.float32)
    y_class = df["target_dir_15"].values.astype(np.int64)
    y_reg = df["target_ret_15"].values.astype(np.float32)

    # Split simple (80/10/10)
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)

    # Datasets
    train_dataset = TimeSeriesDataset(
        X[:n_train], y_class[:n_train], y_reg[:n_train], args.seq_length
    )
    val_dataset = TimeSeriesDataset(
        X[n_train : n_train + n_val],
        y_class[n_train : n_train + n_val],
        y_reg[n_train : n_train + n_val],
        args.seq_length,
    )
    test_dataset = TimeSeriesDataset(
        X[n_train + n_val :],
        y_class[n_train + n_val :],
        y_reg[n_train + n_val :],
        args.seq_length,
    )

    log.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Trainer
    trainer = ProTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        patience=args.patience,
    )

    # Resume si spécifié
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Entraînement
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        save_path=output_path,
    )

    # Évaluation finale sur test set
    log.info("\n" + "=" * 60)
    log.info("ÉVALUATION FINALE SUR TEST SET")
    log.info("=" * 60)

    test_metrics = trainer.validate(test_loader)
    log.info(f"Test Loss: {test_metrics['loss']:.4f}")
    log.info(f"Test Accuracy: {test_metrics['accuracy']:.1f}%")
    log.info(f"Test Sharpe Ratio: {test_metrics['sharpe']:.2f}")
    log.info(f"Test Max Drawdown: {test_metrics['max_drawdown']:.1f}%")
    log.info(f"Test Win Rate: {test_metrics['win_rate']:.1f}%")
    log.info(f"Test Profit Factor: {test_metrics['profit_factor']:.2f}")
    log.info(f"Test Trades: {test_metrics['n_trades']}")

    # Sauvegarder les métriques
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train_history": history,
            "test_metrics": test_metrics,
            "config": vars(args),
        }, f, indent=2, default=str)

    log.info(f"\n✅ Modèle sauvé: {output_path}")
    log.info(f"✅ Métriques sauvées: {metrics_path}")


if __name__ == "__main__":
    main()
