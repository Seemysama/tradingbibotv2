"""
INFERENCE ENGINE V2 - TransformerPro Production Inference
===========================================================
Charge le modèle TransformerPro binaire et fait des prédictions en temps réel.
Synchronisé avec features_pro.py pour garantir la cohérence des features.

Usage:
    from src.ai.inference import InferenceEngine
    engine = InferenceEngine()
    prob_up, confidence, ready = engine.predict(candles_df)
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.features_pro import FeatureEngineerPro, FeatureConfig
from src.ai.transformer_pro import TransformerPro, TransformerConfig
from src.config import settings

log = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfigV2:
    """Configuration du moteur d'inférence V2."""
    model_path: str = "models/transformer_v2.pth"
    config_path: str = "models/metrics_v2.json"
    sequence_length: int = 128
    warmup_candles: int = 250  # Bougies pour warmup EMAs
    confidence_threshold: float = 0.55  # Minimum pour trader
    high_confidence_threshold: float = 0.65  # Signal fort


class InferenceEngine:
    """
    Moteur d'inférence V2 avec TransformerPro.
    
    Charge le modèle Transformer binaire (UP/DOWN) et fait des prédictions
    sur des données OHLCV en temps réel.
    
    Compatible avec l'ancienne API pour HybridStrategy.
    """

    def __init__(
        self,
        model_path: Union[Path, str] = None,
        device: str = None,
        seq_length: int = None,
        buffer_size: int = 5000,
    ) -> None:
        # Configuration
        self.config = InferenceConfigV2()
        
        if model_path:
            self.config.model_path = str(model_path)
        if seq_length:
            self.config.sequence_length = seq_length
        
        self.model_path = Path(self.config.model_path)
        
        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.seq_length = self.config.sequence_length
        
        # Model components
        self.model: Optional[TransformerPro] = None
        self.model_config: Optional[TransformerConfig] = None
        self.feature_engineer: Optional[FeatureEngineerPro] = None
        self.feature_names: List[str] = []
        
        # Buffer pour données
        self.buffer: Deque[dict] = deque(maxlen=buffer_size)
        self.loaded = False
        
        # State
        self._last_features: Optional[np.ndarray] = None
        self._last_prob: float = 0.5
        
        self._load()

    def _load(self) -> None:
        """Charge le modèle TransformerPro."""
        if not self.model_path.exists():
            log.warning("Modèle V2 introuvable à %s (mode neutre).", self.model_path)
            self._setup_fallback()
            return

        try:
            # 1. Charger la configuration sauvegardée
            config_path = Path(self.config.config_path)
            if config_path.exists():
                with open(config_path) as f:
                    saved = json.load(f)
                
                model_cfg = saved.get("config", {})
                self.feature_names = saved.get("feature_names", [])
                
                self.model_config = TransformerConfig(
                    n_features=model_cfg.get("n_features", 39),
                    n_classes=model_cfg.get("n_classes", 2),
                    seq_length=model_cfg.get("seq_length", 128),
                    d_model=model_cfg.get("d_model", 128),
                    n_heads=model_cfg.get("n_heads", 8),
                    n_layers=model_cfg.get("n_layers", 4),
                    d_ff=model_cfg.get("d_ff", 512),
                    dropout=model_cfg.get("dropout", 0.1),
                )
                self.seq_length = self.model_config.seq_length
                log.info("✅ Config V2 chargée: %d features, %d classes", 
                        self.model_config.n_features, self.model_config.n_classes)
            else:
                # Config par défaut
                self.model_config = TransformerConfig(
                    n_features=39,
                    n_classes=2,
                    seq_length=128,
                )
                log.warning("⚠️ Config non trouvée, utilisation des défauts")
            
            # 2. Feature Engineer
            self.feature_engineer = FeatureEngineerPro(FeatureConfig())
            
            # 3. Créer et charger le modèle
            self.model = TransformerPro(self.model_config)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            log.info("✅ TransformerPro V2 chargé depuis %s sur %s (%d params)", 
                    self.model_path, self.device, self.model.count_parameters())
            
        except Exception as exc:
            log.exception("❌ Échec de chargement du modèle V2: %s", exc)
            self._setup_fallback()
    
    def _setup_fallback(self) -> None:
        """Configure le mode fallback (sans modèle)."""
        self.model = None
        self.loaded = False
        self.feature_engineer = FeatureEngineerPro(FeatureConfig())

    def update_buffer(self, candle: Union[dict, pd.Series]) -> None:
        """Alimente le buffer interne avec une bougie (dict ou Series)."""
        if isinstance(candle, pd.Series):
            candle = candle.to_dict()
        self.buffer.append(candle)

    def _buffer_dataframe(self) -> pd.DataFrame:
        """Retourne le buffer sous forme de DataFrame."""
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(self.buffer)

    def predict(self, candles: Optional[pd.DataFrame] = None) -> Tuple[float, bool]:
        """
        Prédit la probabilité de hausse (UP).
        
        Args:
            candles: DataFrame OHLCV optionnel. Si None, utilise le buffer.
        
        Returns:
            (prob_up, ready): Probabilité UP [0-1] et flag de validité
        """
        if not self.loaded or self.model is None:
            return 0.5, False

        if candles is None:
            candles = self._buffer_dataframe()

        min_candles = self.config.warmup_candles
        if len(candles) < min_candles:
            return 0.5, False

        try:
            # 1. Feature Engineering avec FeatureEngineerPro
            df = candles.copy()
            df = self.feature_engineer.engineer(df, include_target=False)
            
            # Récupérer les noms de features
            feature_cols = self.feature_engineer.get_feature_names()
            
            # Limiter au nombre de features du modèle
            n_feat = self.model_config.n_features
            feature_cols = feature_cols[:n_feat]
            
            # Nettoyer
            df = df.dropna()
            if len(df) < self.seq_length:
                return 0.5, False

            # 2. Extraire la dernière séquence
            window = df.iloc[-self.seq_length:]
            X = window[feature_cols].values.astype(np.float32)
            
            # Store pour debug
            self._last_features = X
            
            # 3. Tensor
            X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)  # [1, seq, feat]

            # 4. Inférence
            with torch.no_grad():
                output = self.model(X_tensor)
                
                if isinstance(output, dict):
                    probs = output["probs"]
                else:
                    probs = torch.softmax(output, dim=-1)
                
                # Probabilité UP (classe 1)
                prob_up = probs[0, 1].item()
            
            self._last_prob = prob_up
            return float(prob_up), True
            
        except Exception as exc:
            log.exception("⚠️ Erreur d'inférence V2: %s", exc)
            return 0.5, False
    
    def get_signal(
        self,
        candles_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Retourne un signal de trading structuré.
        
        Args:
            candles_df: DataFrame OHLCV (optionnel)
        
        Returns:
            Dict avec signal, probability, confidence, strength, ready
        """
        prob_up, ready = self.predict(candles_df)
        
        if not ready:
            return {
                "signal": "NEUTRAL",
                "probability": 0.5,
                "confidence": 0.0,
                "strength": "NONE",
                "ready": False,
            }
        
        # Confiance = distance à 0.5
        confidence = abs(prob_up - 0.5) * 2  # 0 à 1
        
        # Déterminer le signal
        if prob_up >= self.config.confidence_threshold:
            signal = "LONG"
        elif prob_up <= (1 - self.config.confidence_threshold):
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        # Force du signal
        if confidence >= 0.3:  # prob > 0.65 ou < 0.35
            strength = "STRONG"
        elif confidence >= 0.1:  # prob > 0.55 ou < 0.45
            strength = "MEDIUM"
        else:
            strength = "WEAK"
        
        return {
            "signal": signal,
            "probability": prob_up,
            "confidence": confidence,
            "strength": strength,
            "ready": True,
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retourne des informations de diagnostic."""
        return {
            "loaded": self.loaded,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "model_type": "TransformerPro" if self.loaded else "None",
            "n_features": self.model_config.n_features if self.model_config else None,
            "n_classes": self.model_config.n_classes if self.model_config else None,
            "seq_length": self.seq_length,
            "warmup_candles": self.config.warmup_candles,
            "buffer_size": len(self.buffer),
            "last_prob": self._last_prob,
        }


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    
    print("🧪 Test du moteur d'inférence V2...")
    
    # Créer données test
    n = 500
    np.random.seed(42)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": 50000 + np.cumsum(np.random.randn(n) * 50),
        "high": 50000 + np.cumsum(np.random.randn(n) * 50) + np.random.rand(n) * 100,
        "low": 50000 + np.cumsum(np.random.randn(n) * 50) - np.random.rand(n) * 100,
        "close": 50000 + np.cumsum(np.random.randn(n) * 50),
        "volume": np.random.exponential(1000, n),
    })
    
    # Fix high/low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    
    # Test engine
    engine = InferenceEngine()
    
    print(f"\n📊 Diagnostics: {engine.get_diagnostics()}")
    
    if engine.loaded:
        signal = engine.get_signal(df)
        print(f"\n🎯 Signal: {signal}")
    else:
        print("\n⚠️ Modèle non chargé. Créez d'abord models/transformer_v2.pth")
