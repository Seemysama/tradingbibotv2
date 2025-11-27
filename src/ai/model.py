"""
Advanced ML Models - LSTM avec Attention, GRU, et Transformer
Production-grade avec save/load checkpoint complet
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Métadonnées d'un checkpoint de modèle."""
    model_name: str
    version: str
    created_at: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    output_dim: int
    seq_length: int
    feature_names: List[str]
    scaler_params: Dict[str, Any]  # min/max par feature
    training_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "seq_length": self.seq_length,
            "feature_names": self.feature_names,
            "scaler_params": self.scaler_params,
            "training_metrics": self.training_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelCheckpoint":
        return cls(**data)


class AttentionLayer(nn.Module):
    """
    Mécanisme d'attention pour séquences temporelles.
    Permet au modèle de se concentrer sur les timesteps les plus pertinents.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            context: (batch, hidden_dim) - Contexte pondéré
            weights: (batch, seq_len) - Poids d'attention
        """
        # Calcul des scores d'attention
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=1)  # (batch, seq_len)
        
        # Contexte pondéré
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)
        
        return context, attn_weights


class CryptoLSTMAttention(nn.Module):
    """
    LSTM bidirectionnel avec mécanisme d'attention.
    Architecture production-grade pour prédiction directionnelle crypto.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        attention_dim: int = 64,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection avec normalisation
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Attention
        lstm_output_dim = hidden_dim * self.num_directions
        self.attention = AttentionLayer(lstm_output_dim, attention_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier/He pour une meilleure convergence."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            return_attention: Si True, retourne aussi les poids d'attention
        
        Returns:
            output: (batch, output_dim) - Logits
            attention_weights: (batch, seq_len) - Optionnel
        """
        # Input normalization et projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * directions)
        
        # Attention
        context, attn_weights = self.attention(lstm_out)
        
        # Classification
        output = self.head(context)
        
        if return_attention:
            return output, attn_weights
        return output


class CryptoGRU(nn.Module):
    """
    GRU plus léger et rapide que LSTM.
    Bon compromis performance/vitesse pour trading haute fréquence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # GRU avec skip connections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            output: (batch, output_dim)
        """
        # Input projection
        x_proj = F.gelu(self.input_proj(x))
        
        # GRU
        gru_out, hidden = self.gru(x_proj)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim)
        
        # Residual de la dernière entrée
        residual = self.residual_proj(x_proj[:, -1, :])
        combined = last_hidden + residual
        
        return self.head(combined)


class CryptoLSTM(nn.Module):
    """
    LSTM simple (legacy) avec améliorations.
    Maintenu pour compatibilité avec les anciens checkpoints.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]  # dernière timestep
        return self.head(last_hidden)


# ----- Checkpoint Management -----

def save_checkpoint(
    model: nn.Module,
    filepath: Path,
    feature_names: List[str],
    scaler_params: Dict[str, Any],
    seq_length: int = 60,
    training_metrics: Optional[Dict[str, float]] = None,
    version: str = "1.0.0",
) -> None:
    """
    Sauvegarde un checkpoint complet incluant le modèle et ses métadonnées.
    
    Args:
        model: Le modèle PyTorch
        filepath: Chemin de sauvegarde (.pth)
        feature_names: Liste des noms de features dans l'ordre
        scaler_params: Paramètres du scaler (min/max par feature)
        seq_length: Longueur de séquence utilisée
        training_metrics: Métriques d'entraînement optionnelles
        version: Version du modèle
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Déterminer les dimensions du modèle
    input_dim = getattr(model, "input_dim", len(feature_names))
    hidden_dim = getattr(model, "hidden_dim", 64)
    num_layers = getattr(model, "num_layers", 2)
    output_dim = getattr(model, "output_dim", 1)
    
    # Créer les métadonnées
    metadata = ModelCheckpoint(
        model_name=model.__class__.__name__,
        version=version,
        created_at=datetime.now().isoformat(),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        seq_length=seq_length,
        feature_names=feature_names,
        scaler_params=scaler_params,
        training_metrics=training_metrics or {},
    )
    
    # Sauvegarder
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
    }
    
    torch.save(checkpoint, filepath)
    
    # Sauvegarder aussi les métadonnées en JSON pour lisibilité
    json_path = filepath.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    
    log.info(f"Checkpoint sauvegardé: {filepath}")
    log.info(f"Métadonnées: {json_path}")


def load_checkpoint(
    filepath: Path,
    device: str = "cpu",
    model_class: Optional[type] = None,
) -> Tuple[nn.Module, ModelCheckpoint]:
    """
    Charge un checkpoint complet.
    
    Args:
        filepath: Chemin du checkpoint (.pth)
        device: Device cible (cpu, cuda, mps)
        model_class: Classe du modèle (si None, déduit depuis les métadonnées)
    
    Returns:
        model: Le modèle chargé
        metadata: Les métadonnées du checkpoint
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint non trouvé: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Extraire les métadonnées
    if "metadata" in checkpoint:
        metadata = ModelCheckpoint.from_dict(checkpoint["metadata"])
    else:
        # Ancien format sans métadonnées
        log.warning("Ancien format de checkpoint détecté, métadonnées par défaut")
        metadata = ModelCheckpoint(
            model_name="CryptoLSTM",
            version="legacy",
            created_at="unknown",
            input_dim=6,
            hidden_dim=64,
            num_layers=2,
            output_dim=1,
            seq_length=60,
            feature_names=["close", "volume", "RSI_14", "ATR_14", "BBP_20_2.0", "ADX_14"],
            scaler_params={},
            training_metrics={},
        )
        # L'ancien format avait juste les state_dict
        checkpoint = {"model_state_dict": checkpoint, "metadata": metadata.to_dict()}
    
    # Déterminer la classe du modèle
    if model_class is None:
        model_classes = {
            "CryptoLSTMAttention": CryptoLSTMAttention,
            "CryptoGRU": CryptoGRU,
            "CryptoLSTM": CryptoLSTM,
        }
        model_class = model_classes.get(metadata.model_name, CryptoLSTM)
    
    # Instancier le modèle
    model = model_class(
        input_dim=metadata.input_dim,
        hidden_dim=metadata.hidden_dim,
        num_layers=metadata.num_layers,
        output_dim=metadata.output_dim,
    )
    
    # Charger les poids
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    log.info(f"Modèle chargé: {metadata.model_name} v{metadata.version}")
    log.info(f"Dimensions: input={metadata.input_dim}, hidden={metadata.hidden_dim}, layers={metadata.num_layers}")
    
    return model, metadata


# ----- Factory Function -----

def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    output_dim: int = 1,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory pour créer différents types de modèles.
    
    Args:
        model_type: "lstm", "lstm_attention", ou "gru"
        input_dim: Nombre de features en entrée
        hidden_dim: Dimension cachée
        num_layers: Nombre de couches
        output_dim: Dimension de sortie
        dropout: Taux de dropout
    
    Returns:
        Modèle instancié
    """
    model_type = model_type.lower()
    
    if model_type == "lstm_attention":
        return CryptoLSTMAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            **kwargs,
        )
    elif model_type == "gru":
        return CryptoGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
    else:  # Default: lstm
        return CryptoLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
