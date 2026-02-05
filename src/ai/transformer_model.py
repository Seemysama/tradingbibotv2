#!/usr/bin/env python3
"""
TRANSFORMER MODEL - Hedge Fund Grade
=====================================
Architecture Transformer pour séries temporelles financières.
Basé sur "Attention Is All You Need" adapté pour time series.

Features:
- Positional Encoding (sinusoïdal)
- Multi-Head Self-Attention
- Feed-Forward Network avec GELU
- Layer Normalization (Pre-LN pour stabilité)
- Dropout régularisation
- Dual Head: Classification + Régression

Reference:
- Vaswani et al., "Attention Is All You Need" (2017)
- Zerveas et al., "A Transformer-based Framework for Multivariate Time Series" (2021)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoïdal pour injecter l'information de position.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Créer la matrice de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention avec masque causal optionnel.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projections Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Masque causal (pour ne pas regarder le futur)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.W_o(attn_output)


class FeedForward(nn.Module):
    """
    Feed-Forward Network avec GELU activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Un bloc Transformer Encoder avec Pre-LN (plus stable).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture (plus stable pour l'entraînement)
        # Attention
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_out)

        return x


class TimeSeriesTransformer(nn.Module):
    """
    Transformer complet pour séries temporelles financières.
    
    Architecture:
    1. Input Projection (features -> d_model)
    2. Positional Encoding
    3. N x Transformer Encoder Layers
    4. Global Average Pooling ou dernière position
    5. Dual Head: Classification + Régression
    
    Args:
        n_features: Nombre de features d'entrée
        d_model: Dimension du modèle (embedding)
        n_heads: Nombre de têtes d'attention
        n_layers: Nombre de couches Transformer
        d_ff: Dimension du feed-forward (généralement 4 * d_model)
        n_classes: Nombre de classes pour classification
        dropout: Taux de dropout
        max_len: Longueur maximale de séquence
        pooling: 'last' (dernière position) ou 'mean' (moyenne)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        n_classes: int = 3,  # Long, Short, Hold
        dropout: float = 0.1,
        max_len: int = 256,
        pooling: str = "last",
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.pooling = pooling

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, causal=True)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Confidence head (uncertainty estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            return_attention: Si True, retourne aussi les poids d'attention
            
        Returns:
            class_logits: (batch, n_classes)
            regression: (batch, 1)
            confidence: (batch, 1)
        """
        # Input projection
        x = self.input_projection(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Pooling
        if self.pooling == "last":
            x = x[:, -1, :]  # Dernière position
        elif self.pooling == "mean":
            x = x.mean(dim=1)  # Global average pooling
        else:
            raise ValueError(f"Pooling inconnu: {self.pooling}")

        # Output heads
        class_logits = self.classification_head(x)
        regression = self.regression_head(x)
        confidence = self.confidence_head(x)

        return class_logits, regression, confidence

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de classification."""
        class_logits, _, _ = self.forward(x)
        return F.softmax(class_logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les prédictions de classe."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=-1)


class TemporalConvNet(nn.Module):
    """
    Alternative TCN (Temporal Convolutional Network).
    Plus rapide à entraîner que le Transformer, bonne baseline.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(n_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_features if i == 0 else n_channels[i - 1]
            out_channels = n_channels[i]

            layers.append(
                self._make_layer(
                    in_channels, out_channels, kernel_size, dilation, dropout
                )
            )

        self.network = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels[-1], n_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels[-1] // 2, n_classes),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels[-1], 1),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float
    ):
        padding = (kernel_size - 1) * dilation // 2
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 1),  # Pointwise
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.network(x)
        
        class_logits = self.classifier(x)
        regression = self.regressor(x)
        
        return class_logits, regression


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre de classes.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class TradingLoss(nn.Module):
    """
    Loss combinée pour trading:
    - Classification Loss (Focal)
    - Regression Loss (Huber)
    - Sharpe-aware penalty
    """

    def __init__(
        self,
        class_weight: float = 1.0,
        reg_weight: float = 0.5,
        sharpe_weight: float = 0.1,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.sharpe_weight = sharpe_weight
        self.focal_loss = FocalLoss(gamma=gamma)
        self.huber_loss = nn.SmoothL1Loss()

    def forward(
        self,
        class_logits: torch.Tensor,
        regression: torch.Tensor,
        class_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # Classification loss
        class_loss = self.focal_loss(class_logits, class_targets)

        # Regression loss
        reg_loss = self.huber_loss(regression.squeeze(), reg_targets)

        # Sharpe-aware: pénalise les prédictions qui ne correspondent pas à la direction
        pred_direction = torch.sign(regression.squeeze())
        actual_direction = torch.sign(reg_targets)
        direction_match = (pred_direction == actual_direction).float()
        
        # Pondérer par l'amplitude du return
        sharpe_penalty = (1 - direction_match) * torch.abs(reg_targets)
        sharpe_loss = sharpe_penalty.mean()

        # Total loss
        total_loss = (
            self.class_weight * class_loss
            + self.reg_weight * reg_loss
            + self.sharpe_weight * sharpe_loss
        )

        metrics = {
            "class_loss": class_loss.item(),
            "reg_loss": reg_loss.item(),
            "sharpe_loss": sharpe_loss.item(),
            "direction_accuracy": direction_match.mean().item(),
        }

        return total_loss, metrics


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(
    model_type: str,
    n_features: int,
    n_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory pour créer les modèles.
    
    Args:
        model_type: 'transformer' ou 'tcn'
        n_features: Nombre de features d'entrée
        n_classes: Nombre de classes
        **kwargs: Arguments spécifiques au modèle
    """
    if model_type == "transformer":
        return TimeSeriesTransformer(
            n_features=n_features,
            n_classes=n_classes,
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 8),
            n_layers=kwargs.get("n_layers", 4),
            d_ff=kwargs.get("d_ff", 512),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_type == "tcn":
        return TemporalConvNet(
            n_features=n_features,
            n_classes=n_classes,
            n_channels=kwargs.get("n_channels", [64, 128, 256]),
            dropout=kwargs.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Model type inconnu: {model_type}")


if __name__ == "__main__":
    # Test rapide
    batch_size = 32
    seq_len = 60
    n_features = 50
    n_classes = 3

    # Test Transformer
    model = TimeSeriesTransformer(n_features=n_features, n_classes=n_classes)
    x = torch.randn(batch_size, seq_len, n_features)
    class_out, reg_out, conf_out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Classification output: {class_out.shape}")
    print(f"Regression output: {reg_out.shape}")
    print(f"Confidence output: {conf_out.shape}")
    
    # Nombre de paramètres
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres: {n_params:,}")
