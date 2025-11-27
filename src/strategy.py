import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import pandas as pd

from src.ai.inference import InferenceEngine
from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: Optional[str]  # "long", "short", None
    confidence: float
    reason: str


class HybridStrategy:
    """Stratégie de trading hybride RSI + ML pour la démo."""

    def __init__(self, inference: Optional[InferenceEngine] = None, max_candles: int = 5000) -> None:
        self.inference = inference or InferenceEngine()
        self.buffer: Deque[dict] = deque(maxlen=max_candles)

    def update(self, candle: dict) -> Signal:
        """Met à jour l'état avec une bougie et produit un signal éventuel."""
        self.buffer.append(candle)
        self.inference.update_buffer(candle)

        df = pd.DataFrame(self.buffer)
        
        # Minimum 30 bougies pour calculer RSI
        if len(df) < 30:
            return Signal(direction=None, confidence=0.0, reason=f"Warmup ({len(df)}/30)")

        df = df.reset_index(drop=True)
        close = df["close"]
        
        # RSI calculation (14 periods)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll_up = gain.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]
        
        # Prix actuel et variation
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2] if len(close) > 1 else current_price
        price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        direction = None
        confidence = 0.0
        reason = ""
        
        # DEMO MODE: Signaux TRÈS agressifs pour montrer le système
        # Force un signal à chaque minute pour la démo
        
        # LONG si RSI < 50 (en dessous de la médiane)
        if curr_rsi < 50:
            direction = "long"
            confidence = 0.60 + (50 - curr_rsi) / 100
            reason = f"RSI={curr_rsi:.0f} < 50 | LONG"
        
        # SHORT si RSI >= 50 (au dessus de la médiane)
        else:
            direction = "short"
            confidence = 0.60 + (curr_rsi - 50) / 100
            reason = f"RSI={curr_rsi:.0f} >= 50 | SHORT"
        
        # ML optionnel - juste un boost si activé
        if settings.ML_ENABLED:
            try:
                prob, ready = self.inference.predict()
                if ready:
                    reason += f" | ML={prob:.1%}"
            except:
                pass
        
        return Signal(direction=direction, confidence=confidence, reason=reason)
