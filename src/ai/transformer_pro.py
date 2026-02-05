#!/usr/bin/env python3
"""
TRANSFORMER PRO - REGULARIZED & REALISTIC
==========================================
Architecture corrigée pour éviter l'overfitting:
- Dropout élevé (0.3-0.5) par défaut
- Modèle plus petit (moins de paramètres)
- Early stopping sur Profit Factor (pas loss)
- Métriques nettes de frais

Author: Lead Quant Researcher
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION RÉALISTE
# ============================================================================

@dataclass
class TransformerConfig:
    """Configuration du Transformer avec régularisation forte."""
    
    # Architecture SIMPLIFIÉE
    n_features: int = 20
    n_classes: int = 2
    seq_length: int = 64
    
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    
    # Régularisation FORTE
    dropout: float = 0.4
    attention_dropout: float = 0.3
    label_smoothing: float = 0.1
    weight_decay: float = 1e-3
    
    # Training
    learning_rate: float = 5e-4
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 0.5
    
    # Walk-Forward
    n_splits: int = 5
    purge_gap: int = 48
    
    # COÛTS
    transaction_cost: float = 0.0006


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal."""
    
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================================

class TransformerEncoderBlock(nn.Module):
    """Bloc Encoder avec Pre-LayerNorm et dropout élevé."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.3,
        attention_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + self.dropout1(attn_out)
        
        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        
        return x


# ============================================================================
# TRANSFORMER PRO MODEL
# ============================================================================

class TransformerPro(nn.Module):
    """Transformer RÉGULARISÉ pour séries temporelles financières."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_features, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )
        
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.seq_length * 2, config.dropout
        )
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                config.attention_dropout,
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        
        return {"logits": logits, "probs": probs}
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset pour séries temporelles."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, returns: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.returns = torch.from_numpy(returns).float() if returns is not None else torch.zeros(len(y))
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.returns[idx]


# ============================================================================
# TRADING LOSS
# ============================================================================

class TradingLoss(nn.Module):
    """Loss avec label smoothing."""
    
    def __init__(self, n_classes: int = 2, label_smoothing: float = 0.1, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ce_loss(logits, targets)


# ============================================================================
# WALK-FORWARD
# ============================================================================

class WalkForwardSplitter:
    """Splitter Walk-Forward avec purge gap."""
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 48):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = []
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            
            splits.append((train_idx, val_idx))
        
        return splits


class WalkForwardTrainer:
    """Trainer Walk-Forward avec métriques réalistes."""
    
    def __init__(self, config: TransformerConfig, device: torch.device):
        self.config = config
        self.device = device
        self.splitter = WalkForwardSplitter(config.n_splits, config.purge_gap)
    
    def train(self, X: np.ndarray, y: np.ndarray, returns: np.ndarray, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Entraînement Walk-Forward."""
        splits = self.splitter.split(len(X))
        
        fold_results = []
        best_model_state = None
        best_profit_factor = 0.0
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            log.info(f"\n{'='*60}")
            log.info(f"FOLD {fold_idx + 1}/{len(splits)}")
            log.info(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")
            log.info("=" * 60)
            
            train_ds = TimeSeriesDataset(X[train_idx], y[train_idx], returns[train_idx])
            val_ds = TimeSeriesDataset(X[val_idx], y[val_idx], returns[val_idx])
            
            train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
            
            model = TransformerPro(self.config).to(self.device)
            log.info(f"Paramètres: {model.count_parameters():,}")
            
            class_counts = np.bincount(y[train_idx], minlength=self.config.n_classes)
            class_weights = len(train_idx) / (self.config.n_classes * class_counts + 1)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            criterion = TradingLoss(n_classes=self.config.n_classes, label_smoothing=self.config.label_smoothing, class_weights=class_weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            
            best_fold_pf = 0.0
            best_fold_state = None
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
                scheduler.step()
                
                val_metrics = self._validate(model, val_loader)
                
                if epoch % 3 == 0 or epoch == self.config.max_epochs - 1:
                    log.info(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Acc: {val_metrics['accuracy']:.1%} | PF: {val_metrics['profit_factor']:.2f} | Sharpe: {val_metrics['sharpe']:.2f}")
                
                if val_metrics['profit_factor'] > best_fold_pf:
                    best_fold_pf = val_metrics['profit_factor']
                    best_fold_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    log.info(f"Early stopping à epoch {epoch + 1}")
                    break
            
            if best_fold_state is not None:
                model.load_state_dict(best_fold_state)
            
            final_metrics = self._validate(model, val_loader)
            fold_results.append({"fold": fold_idx + 1, "profit_factor": final_metrics['profit_factor'], "sharpe": final_metrics['sharpe'], "accuracy": final_metrics['accuracy']})
            
            log.info(f"\n🎯 Fold {fold_idx+1} Final: PF={final_metrics['profit_factor']:.2f}, Sharpe={final_metrics['sharpe']:.2f}")
            
            if final_metrics['profit_factor'] > best_profit_factor:
                best_profit_factor = final_metrics['profit_factor']
                best_model_state = model.state_dict().copy()
                log.info(f"✅ Nouveau meilleur modèle (PF: {best_profit_factor:.2f})")
        
        if save_path and best_model_state is not None:
            torch.save({"model_state": best_model_state, "config": self.config.__dict__, "fold_results": fold_results}, save_path)
            log.info(f"\n✅ Modèle sauvé: {save_path}")
        
        return {"folds": fold_results, "best_profit_factor": best_profit_factor, "best_model_state": best_model_state}
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        
        for x, y, returns in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output["logits"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        
        all_preds = []
        all_targets = []
        all_returns = []
        
        with torch.no_grad():
            for x, y, returns in loader:
                x = x.to(self.device)
                output = model(x)
                preds = output["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                all_returns.extend(returns.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_returns = np.array(all_returns)
        
        return self._calculate_trading_metrics(all_preds, all_targets, all_returns)
    
    def _calculate_trading_metrics(self, predictions: np.ndarray, targets: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        positions = np.where(predictions == 1, 1, -1)
        strategy_returns = positions * returns
        
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * self.config.transaction_cost
        net_returns = strategy_returns - costs
        
        accuracy = float((predictions == targets).mean())
        
        if len(net_returns) > 1 and net_returns.std() > 1e-10:
            sharpe = float(net_returns.mean() / net_returns.std() * np.sqrt(252 * 24 * 12))
        else:
            sharpe = 0.0
        
        gains = net_returns[net_returns > 0].sum()
        losses = abs(net_returns[net_returns < 0].sum())
        profit_factor = float(gains / (losses + 1e-10))
        
        trades = net_returns[net_returns != 0]
        win_rate = float((trades > 0).mean()) if len(trades) > 0 else 0.5
        
        return {"accuracy": accuracy, "sharpe": sharpe, "profit_factor": profit_factor, "win_rate": win_rate}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = TransformerConfig(n_features=20, n_classes=2, seq_length=64)
    model = TransformerPro(config)
    
    print(f"✅ Modèle créé: {model.count_parameters():,} paramètres")
    
    x = torch.randn(4, 64, 20)
    out = model(x)
    print(f"✅ Output keys: {list(out.keys())}")
