#!/usr/bin/env python3
"""
OPTUNA HYPERPARAMETER OPTIMIZATION
===================================
Optimisation bayésienne des hyperparamètres du Transformer.
Utilise la RTX 3080 Ti pour trouver la meilleure architecture.

Usage:
    python optimize_hyperparams.py --trials 50 --epochs 10
    python optimize_hyperparams.py --trials 100 --epochs 5 --study-name btc_optim
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from src.features_pro import FeatureEngineerPro, FeatureConfig
from src.ai.transformer_pro import TransformerPro, TransformerConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# DATA LOADING
# ============================================================================

class OptimizationDataLoader:
    """Charge et prépare les données une seule fois."""
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 128,
        val_ratio: float = 0.2,
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.val_ratio = val_ratio
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.n_features = None
        
        self._load_and_prepare()
    
    def _load_and_prepare(self):
        """Charge les données et prépare les séquences."""
        log.info(f"📂 Chargement: {self.data_path}")
        
        # Charger
        df = pd.read_parquet(self.data_path)
        log.info(f"   Données brutes: {len(df):,} lignes")
        
        # Feature Engineering
        config = FeatureConfig()
        fe = FeatureEngineerPro(config)
        df = fe.engineer(df, include_target=True)
        
        # Récupérer features et target
        feature_cols = fe.get_feature_names()
        self.n_features = len(feature_cols)
        log.info(f"   Features: {self.n_features}")
        
        # Nettoyer
        df = df.dropna()
        log.info(f"   Après nettoyage: {len(df):,} lignes")
        
        # Extraire
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.int64)
        
        # Créer séquences
        X_seq, y_seq = self._create_sequences(X, y)
        log.info(f"   Séquences: {X_seq.shape}")
        
        # Split train/val
        split_idx = int(len(X_seq) * (1 - self.val_ratio))
        
        self.X_train = torch.from_numpy(X_seq[:split_idx])
        self.y_train = torch.from_numpy(y_seq[:split_idx])
        self.X_val = torch.from_numpy(X_seq[split_idx:])
        self.y_val = torch.from_numpy(y_seq[split_idx:])
        
        log.info(f"   Train: {len(self.X_train):,} | Val: {len(self.X_val):,}")
        
        # Distribution
        unique, counts = np.unique(y_seq, return_counts=True)
        dist = dict(zip(unique, counts))
        log.info(f"   Distribution: {dist}")
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        stride: int = 4,
    ) -> tuple:
        """Crée des séquences avec stride."""
        n_samples = (len(X) - self.sequence_length) // stride
        
        X_seq = np.zeros((n_samples, self.sequence_length, X.shape[1]), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int64)
        
        for i in range(n_samples):
            start = i * stride
            end = start + self.sequence_length
            X_seq[i] = X[start:end]
            y_seq[i] = y[end - 1]
        
        return X_seq, y_seq
    
    def get_loaders(self, batch_size: int) -> tuple:
        """Retourne les DataLoaders."""
        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

class TransformerObjective:
    """Objective Optuna pour optimiser le Transformer."""
    
    def __init__(
        self,
        data_loader: OptimizationDataLoader,
        device: torch.device,
        epochs_per_trial: int = 10,
        n_classes: int = 2,
    ):
        self.data_loader = data_loader
        self.device = device
        self.epochs_per_trial = epochs_per_trial
        self.n_classes = n_classes
        self.n_features = data_loader.n_features
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Fonction objective pour Optuna."""
        
        # === LIBÉRER LA MÉMOIRE GPU ===
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        try:
            return self._run_trial(trial)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Gérer les erreurs OOM en retournant un mauvais score
            if "out of memory" in str(e).lower():
                log.warning(f"⚠️ OOM détecté, pruning trial...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                raise optuna.exceptions.TrialPruned()
            raise
    
    def _run_trial(self, trial: optuna.Trial) -> float:
        """Exécution réelle du trial."""
        
        # === HYPERPARAMÈTRES À OPTIMISER ===
        
        # Architecture (réduit pour éviter OOM)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024])
        
        # Régularisation
        dropout = trial.suggest_float("dropout", 0.05, 0.5)
        attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.3)
        
        # Optimisation
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 768])
        
        # Label smoothing
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        
        # Contrainte: d_model doit être divisible par n_heads
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()
        
        # === CRÉER MODÈLE ===
        config = TransformerConfig(
            n_features=self.n_features,
            n_classes=self.n_classes,
            seq_length=self.data_loader.sequence_length,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_dropout=attention_dropout,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
            learning_rate=lr,
            batch_size=batch_size,
        )
        
        try:
            model = TransformerPro(config).to(self.device)
        except Exception as e:
            log.warning(f"Échec création modèle: {e}")
            raise optuna.exceptions.TrialPruned()
        
        # === TRAINING ===
        train_loader, val_loader = self.data_loader.get_loaders(batch_size)
        
        # Class weights pour l'imbalance
        y_train = self.data_loader.y_train.numpy()
        class_counts = np.bincount(y_train, minlength=self.n_classes)
        class_weights = len(y_train) / (self.n_classes * class_counts + 1)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.epochs_per_trial,
        )
        
        best_sharpe = -float('inf')
        
        for epoch in range(self.epochs_per_trial):
            # Train
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                
                # Récupérer logits
                if isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output
                
                loss = criterion(logits, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Métrique combinée: Sharpe simulé
            sharpe = val_metrics["sharpe"]
            
            # Report pour pruning
            trial.report(sharpe, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
        
        # Nettoyage mémoire GPU après chaque trial
        del model, optimizer, scheduler, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_sharpe
    
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Validation avec métriques de trading."""
        model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                output = model(X_batch)
                
                if isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output
                
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Accuracy
        accuracy = (all_preds == all_targets).mean()
        
        # Sharpe simulé (basé sur les prédictions vs targets)
        # Position: +1 si UP prédit, -1 si DOWN
        positions = np.where(all_preds == 1, 1, -1)
        correct = positions * np.where(all_targets == 1, 1, -1)
        
        # Simuler returns
        simulated_returns = correct * 0.001  # 0.1% par trade correct
        
        if len(simulated_returns) > 1 and simulated_returns.std() > 0:
            sharpe = simulated_returns.mean() / simulated_returns.std() * np.sqrt(252 * 24 * 12)
        else:
            sharpe = 0.0
        
        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy,
            "sharpe": sharpe,
        }


# ============================================================================
# MAIN
# ============================================================================

def optimize(
    data_path: str,
    n_trials: int = 50,
    epochs_per_trial: int = 10,
    study_name: str = "transformer_optim",
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    Lance l'optimisation Optuna.
    
    Args:
        data_path: Chemin vers les données
        n_trials: Nombre d'essais
        epochs_per_trial: Epochs par essai
        study_name: Nom de l'étude
        output_dir: Répertoire de sortie
    
    Returns:
        Meilleurs paramètres
    """
    log.info("=" * 70)
    log.info("🚀 OPTUNA HYPERPARAMETER OPTIMIZATION")
    log.info("=" * 70)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("🍎 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("💻 Device: CPU")
    
    # Charger données
    data_loader = OptimizationDataLoader(
        data_path=data_path,
        sequence_length=128,
        val_ratio=0.2,
    )
    
    # Créer objective
    objective = TransformerObjective(
        data_loader=data_loader,
        device=device,
        epochs_per_trial=epochs_per_trial,
        n_classes=2,  # Binary classification
    )
    
    # Créer étude
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximiser Sharpe
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )
    
    # Optimiser
    log.info(f"\n🔬 Lancement de {n_trials} essais ({epochs_per_trial} epochs chacun)...")
    log.info("-" * 70)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Résultats
    log.info("\n" + "=" * 70)
    log.info("🏆 RÉSULTATS OPTIMISATION")
    log.info("=" * 70)
    
    # Stats
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == TrialState.FAIL])
    
    log.info(f"   Essais terminés: {completed}")
    log.info(f"   Essais pruned: {pruned}")
    log.info(f"   Essais échoués: {failed}")
    
    # Meilleurs paramètres
    log.info(f"\n📊 Meilleur Sharpe: {study.best_value:.4f}")
    log.info("\n🎯 MEILLEURS HYPERPARAMÈTRES:")
    
    best_params = study.best_params
    for key, value in sorted(best_params.items()):
        log.info(f"   {key}: {value}")
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "completed_trials": completed,
        "optimization_date": datetime.now().isoformat(),
        "data_path": str(data_path),
        "n_features": data_loader.n_features,
    }
    
    # Top 5 trials
    top_trials = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )[:5]
    
    results["top_5_trials"] = [
        {"value": t.value, "params": t.params}
        for t in top_trials
    ]
    
    # Sauvegarder JSON
    json_path = output_path / "best_hyperparams.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"\n✅ Paramètres sauvegardés: {json_path}")
    
    # Afficher config finale
    log.info("\n📋 CONFIG POUR train_v2.py:")
    log.info("-" * 40)
    log.info(f"TransformerConfig(")
    log.info(f"    n_features={data_loader.n_features},")
    log.info(f"    n_classes=2,")
    log.info(f"    seq_length=128,")
    log.info(f"    d_model={best_params['d_model']},")
    log.info(f"    n_heads={best_params['n_heads']},")
    log.info(f"    n_layers={best_params['n_layers']},")
    log.info(f"    d_ff={best_params['d_ff']},")
    log.info(f"    dropout={best_params['dropout']:.4f},")
    log.info(f"    attention_dropout={best_params['attention_dropout']:.4f},")
    log.info(f"    label_smoothing={best_params['label_smoothing']:.4f},")
    log.info(f"    weight_decay={best_params['weight_decay']:.6f},")
    log.info(f"    learning_rate={best_params['lr']:.6f},")
    log.info(f"    batch_size={best_params['batch_size']},")
    log.info(f")")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimisation Hyperparamètres Optuna")
    parser.add_argument(
        "--data",
        type=str,
        default="data/futures/BTC_USDT_5m_FULL.parquet",
        help="Chemin vers les données",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Nombre d'essais Optuna",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs par essai",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="transformer_optim",
        help="Nom de l'étude",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Répertoire de sortie",
    )
    
    args = parser.parse_args()
    
    try:
        optimize(
            data_path=args.data,
            n_trials=args.trials,
            epochs_per_trial=args.epochs,
            study_name=args.study_name,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        log.info("\n⏹️ Optimisation interrompue par l'utilisateur")
    except Exception as e:
        log.error(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
