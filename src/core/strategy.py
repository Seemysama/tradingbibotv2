"""
Abstract Strategy Interface - Core of the Alpha Engine.
All trading strategies MUST inherit from this base class.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

log = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """
    Trading signal produced by a strategy.
    """
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    reason: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger a trade."""
        return self.direction != SignalDirection.FLAT and self.confidence > 0


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


class IStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Implement this interface to create new strategies.
    The strategy receives market data and produces trading signals.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self._is_warmed_up = False
        self._candle_count = 0
        self._warmup_period = 50
        
    @property
    def is_ready(self) -> bool:
        """Check if strategy has enough data to produce signals."""
        return self._is_warmed_up
        
    @abstractmethod
    def update(self, candle: Dict[str, Any]) -> Signal:
        """
        Process a new candle and generate a trading signal.
        
        Args:
            candle: Dict with keys: timestamp, open, high, low, close, volume
            
        Returns:
            Signal with direction, confidence, and reason
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state (clear buffers, indicators, etc.)."""
        pass
        
    def warmup(self, candles: List[Dict[str, Any]]) -> None:
        """
        Warm up the strategy with historical data.
        
        Args:
            candles: List of historical candles
        """
        for candle in candles:
            self.update(candle)
        # Do not force set _is_warmed_up = True here.
        # Let the strategy strategy implementation determine readiness.
        log.info(f"{self.name}: Warmed up with {len(candles)} candles")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        return {
            "name": self.name,
            "is_ready": self.is_ready,
            "candle_count": self._candle_count
        }


class SMAStrategy(IStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    - Long when fast SMA > slow SMA
    - Short when fast SMA < slow SMA
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(name=f"SMA({fast_period},{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._warmup_period = slow_period + 5
        self._prices: List[float] = []
        self._prev_signal = SignalDirection.FLAT
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        self._prices.append(candle["close"])
        
        # Keep buffer limited
        if len(self._prices) > self.slow_period * 2:
            self._prices = self._prices[-self.slow_period * 2:]
            
        # Not enough data
        if len(self._prices) < self.slow_period:
            return Signal(SignalDirection.FLAT, 0.0, "Warming up")
            
        self._is_warmed_up = True
        
        # Calculate SMAs
        fast_sma = sum(self._prices[-self.fast_period:]) / self.fast_period
        slow_sma = sum(self._prices[-self.slow_period:]) / self.slow_period
        
        # Generate signal
        if fast_sma > slow_sma:
            direction = SignalDirection.LONG
            reason = f"Fast SMA ({fast_sma:.2f}) > Slow SMA ({slow_sma:.2f})"
        elif fast_sma < slow_sma:
            direction = SignalDirection.SHORT
            reason = f"Fast SMA ({fast_sma:.2f}) < Slow SMA ({slow_sma:.2f})"
        else:
            direction = SignalDirection.FLAT
            reason = "SMAs equal"
            
        # Confidence based on divergence
        divergence = abs(fast_sma - slow_sma) / slow_sma
        confidence = min(divergence * 10, 1.0)  # Scale to 0-1
        
        # Only signal on crossover (direction change)
        if direction == self._prev_signal:
            confidence = 0.0  # No new signal
            
        self._prev_signal = direction
        
        return Signal(
            direction=direction,
            confidence=confidence,
            reason=reason,
            metadata={"fast_sma": fast_sma, "slow_sma": slow_sma}
        )
        
    def reset(self):
        self._prices = []
        self._prev_signal = SignalDirection.FLAT
        self._is_warmed_up = False
        self._candle_count = 0


class RSIStrategy(IStrategy):
    """
    RSI Mean Reversion Strategy.
    
    - Long when RSI < oversold (buy the dip)
    - Short when RSI > overbought (sell the top)
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(name=f"RSI({period})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._warmup_period = period + 5
        self._prices: List[float] = []
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        self._prices.append(candle["close"])
        
        if len(self._prices) > self.period * 3:
            self._prices = self._prices[-self.period * 3:]
            
        if len(self._prices) < self.period + 1:
            return Signal(SignalDirection.FLAT, 0.0, "Warming up")
            
        self._is_warmed_up = True
        
        # Calculate RSI
        rsi = self._calculate_rsi()
        
        if rsi < self.oversold:
            direction = SignalDirection.LONG
            confidence = (self.oversold - rsi) / self.oversold
            reason = f"RSI oversold ({rsi:.1f} < {self.oversold})"
        elif rsi > self.overbought:
            direction = SignalDirection.SHORT
            confidence = (rsi - self.overbought) / (100 - self.overbought)
            reason = f"RSI overbought ({rsi:.1f} > {self.overbought})"
        else:
            direction = SignalDirection.FLAT
            confidence = 0.0
            reason = f"RSI neutral ({rsi:.1f})"
            
        return Signal(
            direction=direction,
            confidence=min(confidence, 1.0),
            reason=reason,
            metadata={"rsi": rsi}
        )
        
    def _calculate_rsi(self) -> float:
        """Calculate RSI from price history."""
        deltas = [self._prices[i] - self._prices[i-1] for i in range(1, len(self._prices))]
        
        gains = [d if d > 0 else 0 for d in deltas[-self.period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-self.period:]]
        
        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def reset(self):
        self._prices = []
        self._is_warmed_up = False
        self._candle_count = 0


class CombinedStrategy(IStrategy):
    """
    Combines multiple strategies with voting.
    Signal is generated when strategies agree.
    """
    
    def __init__(self, strategies: List[IStrategy], min_agreement: float = 0.5):
        super().__init__(name="Combined")
        self.strategies = strategies
        self.min_agreement = min_agreement
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        
        signals = [s.update(candle) for s in self.strategies]
        
        # Count votes
        long_votes = sum(1 for s in signals if s.direction == SignalDirection.LONG and s.confidence > 0)
        short_votes = sum(1 for s in signals if s.direction == SignalDirection.SHORT and s.confidence > 0)
        total = len(self.strategies)
        
        if long_votes / total >= self.min_agreement:
            avg_conf = sum(s.confidence for s in signals if s.direction == SignalDirection.LONG) / max(long_votes, 1)
            return Signal(SignalDirection.LONG, avg_conf, f"{long_votes}/{total} strategies agree LONG")
        elif short_votes / total >= self.min_agreement:
            avg_conf = sum(s.confidence for s in signals if s.direction == SignalDirection.SHORT) / max(short_votes, 1)
            return Signal(SignalDirection.SHORT, avg_conf, f"{short_votes}/{total} strategies agree SHORT")
        else:
            return Signal(SignalDirection.FLAT, 0.0, "No consensus")
            
    def reset(self):
        for s in self.strategies:
            s.reset()
        self._is_warmed_up = False
        self._candle_count = 0
        
    @property
    def is_ready(self) -> bool:
        return all(s.is_ready for s in self.strategies)
