#!/usr/bin/env python3
"""
TRAIN V2 - REALISTIC TRAINING PIPELINE
=======================================
Pipeline corrigé avec:
- Frais de transaction intégrés (0.06% par trade)
- Early stopping sur Profit Factor (pas loss)
- Sharpe et PnL NETS de frais
- Normalisation rolling anti-leakage

Author: Lead Quant Researcher
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# Local imports
from src.features_pro import FeatureEngineerPro, FeatureConfig, normalize_features_rolling
from src.ai.transformer_pro import (
    TransformerPro,
    TransformerConfig,
    WalkForwardTrainer,
    TimeSeriesDataset,
)

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_v2.log"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES RÉALISTES
# ============================================================================

# Binance Futures fees
TAKER_FEE = 0.0004  # 0.04%
MAKER_FEE = 0.0002  # 0.02%
SPREAD_ESTIMATE = 0.0001  # 0.01%

# Total cost per round-trip trade
TRANSACTION_COST = 0.0006  # 0.06% (taker + spread)


# ============================================================================
# DATA LOADING
# ============================================================================

class ParquetLoader:
    """Chargeur de données robuste."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        """Charge un fichier Parquet."""
        log.info(f"📂 Chargement: {filepath}")
        
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        log.info(f"   Lignes: {len(df):,}")
        
        return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    sequence_length: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prépare les features avec normalisation ANTI-LEAKAGE.
    """
    log.info(f"🔧 Feature engineering sur {len(df):,} lignes...")
    
    fe = FeatureEngineerPro(config)
    df = fe.engineer(df, include_target=True)
    
    feature_names = fe.get_feature_names()
    log.info(f"   Features générées: {len(feature_names)}")
    
    # Nettoyer
    df = df.dropna(subset=feature_names + ["target"]).reset_index(drop=True)
    log.info(f"   Après nettoyage: {len(df):,} lignes")
    
    # Extraire
    X = df[feature_names].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    y_returns = df["target_return"].values.astype(np.float32)
    
    # Normalisation ROLLING (anti-leakage)
    log.info("   Normalisation rolling...")
    X = normalize_features_rolling(X, window=288, min_periods=50)  # 1 jour en 5min
    
    # Créer séquences
    X_seq, y_seq, ret_seq = _create_sequences(X, y, y_returns, sequence_length)
    
    log.info(f"   Séquences: {X_seq.shape}")
    
    # Distribution
    unique, counts = np.unique(y_seq, return_counts=True)
    dist = dict(zip(unique, counts))
    log.info(f"   Distribution: {dist}")
    
    return X_seq, y_seq, ret_seq, feature_names


def _create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    y_returns: np.ndarray,
    seq_len: int,
    stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crée des séquences avec stride."""
    indices = list(range(0, len(X) - seq_len, stride))
    n = len(indices)
    
    X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.int64)
    ret_seq = np.zeros(n, dtype=np.float32)
    
    for j, i in enumerate(indices):
        X_seq[j] = X[i:i+seq_len]
        y_seq[j] = y[i+seq_len-1]  # Label à la fin de la séquence
        ret_seq[j] = y_returns[i+seq_len-1]
    
    return X_seq, y_seq, ret_seq


# ============================================================================
# TRADING METRICS (AVEC FRAIS)
# ============================================================================

class TradingMetrics:
    """Métriques de trading RÉALISTES avec frais."""
    
    @staticmethod
    def calculate_all(
        predictions: np.ndarray,
        targets: np.ndarray,
        target_returns: np.ndarray,
        transaction_cost: float = TRANSACTION_COST,
    ) -> Dict[str, float]:
        """
        Calcule les métriques NETTES DE FRAIS.
        
        CRITIQUE: Le Sharpe et le PnL incluent les coûts de transaction!
        """
        # Positions: pred=1 → LONG (+1), pred=0 → SHORT (-1)
        positions = np.where(predictions == 1, 1, -1)
        
        # Returns bruts
        strategy_returns = positions * target_returns
        
        # Coûts de transaction (sur changements de position)
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * transaction_cost
        
        # Returns NETS
        net_returns = strategy_returns - costs
        
        # Accuracy
        accuracy = float((predictions == targets).mean())
        
        # Sharpe NET
        if len(net_returns) > 1 and net_returns.std() > 1e-10:
            # Annualisé (5min bars)
            sharpe = float(net_returns.mean() / net_returns.std() * np.sqrt(252 * 24 * 12))
        else:
            sharpe = 0.0
        
        # Sortino NET
        downside = net_returns[net_returns < 0]
        if len(downside) > 0 and downside.std() > 1e-10:
            sortino = float(net_returns.mean() / downside.std() * np.sqrt(252 * 24 * 12))
        else:
            sortino = sharpe
        
        # Max Drawdown
        cumulative = np.cumsum(net_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Calmar
        annual_return = net_returns.mean() * 252 * 24 * 12
        calmar = float(annual_return / (max_drawdown + 1e-10))
        
        # Win Rate
        trades = net_returns[positions != 0]
        win_rate = float((trades > 0).mean()) if len(trades) > 0 else 0.5
        
        # Profit Factor
        gains = net_returns[net_returns > 0].sum()
        losses = abs(net_returns[net_returns < 0].sum())
        profit_factor = float(gains / (losses + 1e-10))
        
        # Stats
        n_trades = int(position_changes.sum() // 2)
        total_return = float(np.sum(net_returns))
        avg_trade = float(net_returns.mean()) if len(net_returns) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "total_return": total_return,
            "avg_trade_return": avg_trade,
            "transaction_cost_used": transaction_cost,
        }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_pipeline(
    data_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 512,
    sequence_length: int = 64,
    n_splits: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pipeline d'entraînement avec métriques réalistes.
    """
    log.info("=" * 70)
    log.info("🚀 TRAINING PIPELINE V2 - REALISTIC")
    log.info("=" * 70)
    log.info(f"   Transaction cost: {TRANSACTION_COST:.4%} per trade")
    
    # Device
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    
    log.info(f"📱 Device: {dev}")
    
    # Charger les données
    loader = ParquetLoader(data_path)
    df = loader.load_single_file(data_path)
    
    # Feature engineering avec config réaliste
    feature_config = FeatureConfig()
    X, y, y_returns, feature_names = prepare_features(
        df, feature_config, sequence_length
    )
    
    n_classes = 2  # Binary
    log.info(f"📊 Classes: {n_classes} (Binary UP/DOWN)")
    
    # Configuration modèle RÉGULARISÉ
    model_config = TransformerConfig(
        n_features=len(feature_names),
        n_classes=n_classes,
        seq_length=sequence_length,
        
        # Architecture SIMPLE (évite overfitting)
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        
        # Régularisation FORTE
        dropout=0.4,
        attention_dropout=0.3,
        label_smoothing=0.1,
        weight_decay=1e-3,
        
        # Training
        learning_rate=5e-4,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=10,
        grad_clip=0.5,
        
        # Walk-Forward
        n_splits=n_splits,
        purge_gap=sequence_length,
        
        # COÛTS
        transaction_cost=TRANSACTION_COST,
    )
    
    log.info(f"\n📋 Config modèle:")
    log.info(f"   d_model: {model_config.d_model}")
    log.info(f"   n_layers: {model_config.n_layers}")
    log.info(f"   dropout: {model_config.dropout}")
    log.info(f"   weight_decay: {model_config.weight_decay}")
    
    # Créer répertoire
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "transformer_v2.pth"
    
    # Entraînement Walk-Forward
    trainer = WalkForwardTrainer(model_config, dev)
    results = trainer.train(
        X, y, y_returns,
        save_path=str(model_path),
    )
    
    # Évaluation finale
    log.info("\n" + "=" * 70)
    log.info("📊 ÉVALUATION FINALE (Out-of-Sample)")
    log.info("=" * 70)
    
    # Test sur 20% final
    model = TransformerPro(model_config).to(dev)
    if results["best_model_state"] is not None:
        model.load_state_dict(results["best_model_state"])
    model.eval()
    
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:]
    y_test = y[test_start:]
    ret_test = y_returns[test_start:]
    
    test_ds = TimeSeriesDataset(X_test, y_test, ret_test)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_returns = []
    
    with torch.no_grad():
        for x, y_batch, ret in test_loader:
            x = x.to(dev)
            out = model(x)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            all_returns.extend(ret.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)
    
    # Métriques RÉALISTES
    final_metrics = TradingMetrics.calculate_all(
        all_preds, all_targets, all_returns,
        transaction_cost=TRANSACTION_COST,
    )
    
    log.info("\n📊 MÉTRIQUES TEST SET (NETTES DE FRAIS):")
    log.info("-" * 40)
    for k, v in final_metrics.items():
        if isinstance(v, float):
            if "return" in k or "drawdown" in k:
                log.info(f"   {k}: {v:.4%}")
            else:
                log.info(f"   {k}: {v:.4f}")
        else:
            log.info(f"   {k}: {v}")
    
    # Sauvegarder
    metrics_path = output_dir / "metrics_v2.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": asdict(model_config),
            "feature_config": asdict(feature_config),
            "feature_names": feature_names,
            "walk_forward_results": results["folds"],
            "final_metrics": final_metrics,
            "training_date": datetime.now().isoformat(),
            "transaction_cost": TRANSACTION_COST,
        }, f, indent=2, default=str)
    
    log.info(f"\n✅ Modèle sauvé: {model_path}")
    log.info(f"✅ Métriques sauvées: {metrics_path}")
    
    # Résumé
    log.info("\n" + "=" * 70)
    log.info("🎯 RÉSUMÉ RÉALISTE")
    log.info("=" * 70)
    
    # Évaluation du modèle
    pf = final_metrics['profit_factor']
    sharpe = final_metrics['sharpe_ratio']
    
    if pf > 1.2 and sharpe > 1.5:
        verdict = "✅ MODÈLE PROMETTEUR - À tester en paper trading"
    elif pf > 1.0 and sharpe > 0.5:
        verdict = "⚠️ MODÈLE MARGINAL - Amélioration nécessaire"
    else:
        verdict = "❌ MODÈLE NON VIABLE - Retour à la R&D"
    
    log.info(f"\n{verdict}")
    log.info(f"\n   Profit Factor: {pf:.2f} (objectif > 1.2)")
    log.info(f"   Sharpe Ratio: {sharpe:.2f} (objectif > 1.5)")
    log.info(f"   Win Rate: {final_metrics['win_rate']:.1%}")
    log.info(f"   Max Drawdown: {final_metrics['max_drawdown']:.2%}")
    
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "final_metrics": final_metrics,
        "verdict": verdict,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training Pipeline V2 - Realistic")
    
    parser.add_argument("--data", type=str, required=True, help="Chemin données")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=64, help="Longueur séquence")
    parser.add_argument("--n-splits", type=int, default=5, help="Folds Walk-Forward")
    parser.add_argument("--output", type=str, default="models", help="Output dir")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        log.error(f"❌ Fichier non trouvé: {data_path}")
        sys.exit(1)
    
    try:
        results = train_pipeline(
            data_path=data_path,
            output_dir=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            n_splits=args.n_splits,
            device=args.device,
        )
        
        log.info("\n" + "=" * 70)
        log.info("🎉 ENTRAÎNEMENT TERMINÉ")
        log.info("=" * 70)
        
    except Exception as e:
        log.exception(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
