import logging
import pandas as pd
import numpy as np
from typing import Optional, Any
from collections import deque
from src.interfaces import IStrategy
from src.strategies.polymarket_strategy import Signal # Reusing or redefining signal? Let's genericize later.
# For now, let's stick to the existing Signal class structure or a compatible dictionary.

from src.config import settings

log = logging.getLogger(__name__)

# We need a compatible Signal class or dict
from src.strategy import Signal # Reuse existing Signal class for compatibility

class PolymarketStrategy(IStrategy):
    """
    Strategy for Polymarket Binary Tokens (0-1 range).
    Focuses on Mean Reversion and Trend Following within probabilities.
    """
    
    def __init__(self, max_candles: int = 100):
        self.buffer = deque(maxlen=max_candles)
        
    def update(self, candle: dict) -> Signal:
        self.buffer.append(candle)
        
        if len(self.buffer) < 20:
            return Signal(direction=None, confidence=0.0, reason="Warmup")
            
        df = pd.DataFrame(self.buffer)
        price = candle["close"]
        
        # Polymarket specific logic
        # 1. Extreme probabilities (Reversion)
        if price > 0.95:
             # Too expensive, likely to resolve YES or pullback. 
             # Buying NO (Short YES) might be good if time remains.
             # For now, let's say we don't buy top.
             pass
        elif price < 0.05:
            # Too cheap.
            pass
            
        # 2. Simple Moving Average Crossover
        short_window = 5
        long_window = 20
        
        df["sma_short"] = df["close"].rolling(window=short_window).mean()
        df["sma_long"] = df["close"].rolling(window=long_window).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        direction = None
        confidence = 0.0
        reason = ""
        
        # Golden Cross
        if current["sma_short"] > current["sma_long"] and prev["sma_short"] <= prev["sma_long"]:
            direction = "long"
            confidence = 0.6
            reason = "SMA Golden Cross"
            
        # Death Cross
        elif current["sma_short"] < current["sma_long"] and prev["sma_short"] >= prev["sma_long"]:
            direction = "short"
            confidence = 0.6
            reason = "SMA Death Cross"
            
        # Filter weak signals in the middle
        if direction and (0.4 < price < 0.6):
            confidence += 0.1 # Higher confidence in uncertain middle ground if trend emerges
            
        if direction:
            return Signal(
                direction=direction,
                confidence=confidence,
                reason=reason,
                price=price,
                stop_loss=price * 0.9 if direction == "long" else price * 1.1,
                take_profit=price * 1.2 if direction == "long" else price * 0.8,
                regime="RANGE" # Todo: implement regime detection
            )
            
        return Signal(direction=None, confidence=0.0, reason="Wait", price=price)
