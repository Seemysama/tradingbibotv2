"""
Alpha Engine - Profitable Trading Strategies.
Includes Mean Reversion, Momentum, and Polymarket Arbitrage.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.strategy import IStrategy, Signal, SignalDirection

log = logging.getLogger(__name__)


# =============================================================================
# MEAN REVERSION STRATEGY (RSI + Bollinger Bands)
# =============================================================================

class MeanReversionStrategy(IStrategy):
    """
    Mean Reversion Strategy using RSI and Bollinger Bands.
    
    Entry Conditions:
    - LONG: RSI < oversold AND price < lower BB
    - SHORT: RSI > overbought AND price > upper BB
    
    Exit:
    - Price returns to middle BB (mean)
    
    This works best in RANGE-BOUND markets.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        oversold: int = 30,
        overbought: int = 70
    ):
        super().__init__(name=f"MeanReversion(RSI{rsi_period},BB{bb_period})")
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.oversold = oversold
        self.overbought = overbought
        self._warmup_period = max(rsi_period, bb_period) + 5
        
        self._prices: List[float] = []
        self._in_position = False
        self._position_side: Optional[str] = None
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        self._prices.append(candle["close"])
        
        # Limit buffer
        max_len = max(self.rsi_period, self.bb_period) * 3
        if len(self._prices) > max_len:
            self._prices = self._prices[-max_len:]
            
        # Warmup check
        if len(self._prices) < self._warmup_period:
            return Signal(SignalDirection.FLAT, 0.0, "Warming up")
            
        self._is_warmed_up = True
        
        # Calculate indicators
        rsi = self._calculate_rsi()
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger()
        price = self._prices[-1]
        
        # Exit logic (if in position)
        if self._in_position:
            if self._position_side == "long" and price >= bb_middle:
                self._in_position = False
                self._position_side = None
                return Signal(
                    SignalDirection.FLAT, 0.8,
                    f"Exit LONG: Price ${price:.2f} reached middle BB ${bb_middle:.2f}",
                    metadata={"rsi": rsi, "bb_middle": bb_middle}
                )
            elif self._position_side == "short" and price <= bb_middle:
                self._in_position = False
                self._position_side = None
                return Signal(
                    SignalDirection.FLAT, 0.8,
                    f"Exit SHORT: Price ${price:.2f} reached middle BB ${bb_middle:.2f}",
                    metadata={"rsi": rsi, "bb_middle": bb_middle}
                )
            # Stay in position
            return Signal(SignalDirection.FLAT, 0.0, "Holding position")
            
        # Entry logic
        if rsi < self.oversold and price < bb_lower:
            # LONG: Oversold + below lower band
            confidence = min((self.oversold - rsi) / self.oversold, 1.0)
            self._in_position = True
            self._position_side = "long"
            return Signal(
                SignalDirection.LONG, confidence,
                f"LONG: RSI={rsi:.1f} < {self.oversold}, Price < BB lower",
                metadata={"rsi": rsi, "bb_lower": bb_lower, "bb_upper": bb_upper}
            )
            
        elif rsi > self.overbought and price > bb_upper:
            # SHORT: Overbought + above upper band
            confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
            self._in_position = True
            self._position_side = "short"
            return Signal(
                SignalDirection.SHORT, confidence,
                f"SHORT: RSI={rsi:.1f} > {self.overbought}, Price > BB upper",
                metadata={"rsi": rsi, "bb_lower": bb_lower, "bb_upper": bb_upper}
            )
            
        return Signal(SignalDirection.FLAT, 0.0, f"No signal (RSI={rsi:.1f})")
        
    def _calculate_rsi(self) -> float:
        prices = self._prices[-self.rsi_period - 1:]
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))
        
    def _calculate_bollinger(self) -> Tuple[float, float, float]:
        prices = self._prices[-self.bb_period:]
        middle = sum(prices) / len(prices)
        variance = sum((p - middle) ** 2 for p in prices) / len(prices)
        std = variance ** 0.5
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        return upper, middle, lower
        
    def reset(self):
        self._prices = []
        self._in_position = False
        self._position_side = None
        self._is_warmed_up = False
        self._candle_count = 0


# =============================================================================
# MOMENTUM + VOLUME CONFIRMATION
# =============================================================================

class MomentumVolumeStrategy(IStrategy):
    """
    Momentum Strategy with Volume Confirmation.
    
    Entry:
    - LONG: Price breaks above N-period high with above-average volume
    - SHORT: Price breaks below N-period low with above-average volume
    
    Exit:
    - Trailing stop or opposite signal
    """
    
    def __init__(
        self,
        breakout_period: int = 20,
        volume_mult: float = 1.5,
        atr_period: int = 14,
        trailing_atr_mult: float = 2.0
    ):
        super().__init__(name=f"MomentumVol({breakout_period})")
        self.breakout_period = breakout_period
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.trailing_atr_mult = trailing_atr_mult
        self._warmup_period = max(breakout_period, atr_period) + 5
        
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        self._volumes: List[float] = []
        
        self._position_side: Optional[str] = None
        self._entry_price: float = 0.0
        self._trailing_stop: float = 0.0
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        
        self._highs.append(candle["high"])
        self._lows.append(candle["low"])
        self._closes.append(candle["close"])
        self._volumes.append(candle.get("volume", 0))
        
        # Limit buffers
        max_len = self.breakout_period * 3
        if len(self._closes) > max_len:
            self._highs = self._highs[-max_len:]
            self._lows = self._lows[-max_len:]
            self._closes = self._closes[-max_len:]
            self._volumes = self._volumes[-max_len:]
            
        if len(self._closes) < self._warmup_period:
            return Signal(SignalDirection.FLAT, 0.0, "Warming up")
            
        self._is_warmed_up = True
        
        price = self._closes[-1]
        high = self._highs[-1]
        low = self._lows[-1]
        volume = self._volumes[-1]
        
        # Calculate indicators
        period_high = max(self._highs[-self.breakout_period:-1])
        period_low = min(self._lows[-self.breakout_period:-1])
        avg_volume = sum(self._volumes[-self.breakout_period:-1]) / (self.breakout_period - 1)
        atr = self._calculate_atr()
        
        volume_confirmed = volume > avg_volume * self.volume_mult
        
        # Check trailing stop if in position
        if self._position_side == "long":
            new_stop = price - atr * self.trailing_atr_mult
            self._trailing_stop = max(self._trailing_stop, new_stop)
            
            if price < self._trailing_stop:
                self._position_side = None
                return Signal(
                    SignalDirection.FLAT, 0.9,
                    f"Exit LONG: Trailing stop hit at ${self._trailing_stop:.2f}"
                )
                
        elif self._position_side == "short":
            new_stop = price + atr * self.trailing_atr_mult
            self._trailing_stop = min(self._trailing_stop, new_stop)
            
            if price > self._trailing_stop:
                self._position_side = None
                return Signal(
                    SignalDirection.FLAT, 0.9,
                    f"Exit SHORT: Trailing stop hit at ${self._trailing_stop:.2f}"
                )
                
        # Entry signals
        if self._position_side is None:
            if high > period_high and volume_confirmed:
                self._position_side = "long"
                self._entry_price = price
                self._trailing_stop = price - atr * self.trailing_atr_mult
                return Signal(
                    SignalDirection.LONG, 0.8,
                    f"LONG: Breakout above ${period_high:.2f} with {volume/avg_volume:.1f}x volume",
                    metadata={"breakout_level": period_high, "volume_ratio": volume/avg_volume}
                )
                
            elif low < period_low and volume_confirmed:
                self._position_side = "short"
                self._entry_price = price
                self._trailing_stop = price + atr * self.trailing_atr_mult
                return Signal(
                    SignalDirection.SHORT, 0.8,
                    f"SHORT: Breakdown below ${period_low:.2f} with {volume/avg_volume:.1f}x volume",
                    metadata={"breakdown_level": period_low, "volume_ratio": volume/avg_volume}
                )
                
        return Signal(SignalDirection.FLAT, 0.0, "No signal")
        
    def _calculate_atr(self) -> float:
        """Average True Range."""
        trs = []
        for i in range(-self.atr_period, 0):
            high = self._highs[i]
            low = self._lows[i]
            prev_close = self._closes[i - 1]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            
        return sum(trs) / len(trs) if trs else 0
        
    def reset(self):
        self._highs = []
        self._lows = []
        self._closes = []
        self._volumes = []
        self._position_side = None
        self._is_warmed_up = False
        self._candle_count = 0


# =============================================================================
# POLYMARKET ARBITRAGE STRATEGY
# =============================================================================

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    market_id: str
    question: str
    yes_price: float
    no_price: float
    total_cost: float
    profit_margin: float  # 1.0 - total_cost
    timestamp: datetime


class PolymarketArbitrageStrategy:
    """
    Complete-Set Arbitrage for Polymarket.
    
    Logic:
    - If YES_price + NO_price < 1.0, buy both for guaranteed profit
    - One outcome ALWAYS resolves to $1.00
    - Profit = 1.0 - (YES_price + NO_price) - fees
    
    Example:
    - YES = $0.45, NO = $0.52 → Total = $0.97
    - Buy both for $0.97, one resolves to $1.00
    - Profit = $0.03 (3%) - fees
    
    This is RISK-FREE arbitrage if executed atomically.
    """
    
    def __init__(
        self,
        min_profit_margin: float = 0.01,  # 1% minimum
        max_position_usd: float = 100.0,
        fee_pct: float = 0.002  # 0.2% per side
    ):
        self.min_profit_margin = min_profit_margin
        self.max_position_usd = max_position_usd
        self.fee_pct = fee_pct
        self.name = "PolymarketArbitrage"
        
    def scan_markets(self, markets: pd.DataFrame) -> List[ArbitrageOpportunity]:
        """
        Scan markets for arbitrage opportunities.
        
        Args:
            markets: DataFrame with columns: condition_id, question, yes_price, no_price
            
        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []
        
        for _, row in markets.iterrows():
            yes_price = row.get("yes_price", 0.5)
            no_price = row.get("no_price", 0.5)
            
            total_cost = yes_price + no_price
            gross_profit = 1.0 - total_cost
            
            # Account for fees (buy both sides)
            net_profit = gross_profit - (2 * self.fee_pct)
            
            if net_profit > self.min_profit_margin:
                opportunities.append(ArbitrageOpportunity(
                    market_id=row.get("condition_id", ""),
                    question=row.get("question", "")[:100],
                    yes_price=yes_price,
                    no_price=no_price,
                    total_cost=total_cost,
                    profit_margin=net_profit,
                    timestamp=datetime.utcnow()
                ))
                
        # Sort by profit margin descending
        opportunities.sort(key=lambda x: x.profit_margin, reverse=True)
        
        return opportunities
        
    def generate_orders(
        self,
        opportunity: ArbitrageOpportunity
    ) -> List[Dict[str, Any]]:
        """
        Generate orders to capture arbitrage.
        
        Returns list of orders to execute atomically.
        """
        # Calculate position size
        position_usd = min(
            self.max_position_usd,
            self.max_position_usd / opportunity.total_cost
        )
        
        orders = [
            {
                "market_id": opportunity.market_id,
                "side": "YES",
                "price": opportunity.yes_price,
                "size_usd": position_usd * opportunity.yes_price,
                "order_type": "LIMIT",
                "time_in_force": "IOC"  # Immediate or Cancel
            },
            {
                "market_id": opportunity.market_id,
                "side": "NO",
                "price": opportunity.no_price,
                "size_usd": position_usd * opportunity.no_price,
                "order_type": "LIMIT",
                "time_in_force": "IOC"
            }
        ]
        
        expected_profit = position_usd * opportunity.profit_margin
        log.info(f"Arbitrage opportunity: {opportunity.question[:50]}...")
        log.info(f"  YES: ${opportunity.yes_price:.4f}, NO: ${opportunity.no_price:.4f}")
        log.info(f"  Expected profit: ${expected_profit:.2f} ({opportunity.profit_margin:.1%})")
        
        return orders


# =============================================================================
# VECTORIZED SIGNAL FUNCTIONS (for backtester)
# =============================================================================

def mean_reversion_signal(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    oversold: int = 30,
    overbought: int = 70
) -> pd.Series:
    """
    Vectorized Mean Reversion signal for backtesting.
    """
    close = df["close"]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_middle = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_middle + bb_std * bb_std_val
    bb_lower = bb_middle - bb_std * bb_std_val
    
    # Signals
    signal = pd.Series(0, index=df.index)
    
    # Long when oversold AND below lower band
    long_condition = (rsi < oversold) & (close < bb_lower)
    # Short when overbought AND above upper band
    short_condition = (rsi > overbought) & (close > bb_upper)
    
    signal[long_condition] = 1
    signal[short_condition] = -1
    
    return signal


def momentum_volume_signal(
    df: pd.DataFrame,
    breakout_period: int = 20,
    volume_mult: float = 1.5
) -> pd.Series:
    """
    Vectorized Momentum + Volume signal for backtesting.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df.get("volume", pd.Series(1, index=df.index))
    
    # Period high/low (excluding current bar)
    period_high = high.shift(1).rolling(breakout_period).max()
    period_low = low.shift(1).rolling(breakout_period).min()
    
    # Average volume
    avg_volume = volume.shift(1).rolling(breakout_period).mean()
    volume_confirmed = volume > avg_volume * volume_mult
    
    # Signals
    signal = pd.Series(0, index=df.index)
    
    # Long on breakout above period high with volume
    long_condition = (high > period_high) & volume_confirmed
    # Short on breakdown below period low with volume
    short_condition = (low < period_low) & volume_confirmed
    
    signal[long_condition] = 1
    signal[short_condition] = -1
    
    return signal


# =============================================================================
# ADAPTIVE REGIME STRATEGY
# =============================================================================

def adaptive_regime_signal(
    df: pd.DataFrame,
    atr_period: int = 14,
    regime_period: int = 50,
    rsi_period: int = 7,
    bb_period: int = 14
) -> pd.Series:
    """
    Adaptive strategy that detects market regime and applies appropriate logic.
    
    Regime Detection:
    - TRENDING: High volatility or strong directional move
    - RANGING: Low volatility or price oscillating around SMA
    
    Strategy:
    - In TREND: Follow the trend (momentum)
    - In RANGE: Mean reversion (RSI + BB)
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # Calculate ATR for volatility
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    # Regime detection: Compare current volatility to historical
    atr_ma = atr.rolling(regime_period).mean()
    atr_std = atr.rolling(regime_period).std()
    
    # High volatility = trending, Low volatility = ranging
    volatility_zscore = (atr - atr_ma) / atr_std.replace(0, 1)
    
    # Price momentum
    sma_50 = close.rolling(50).mean()
    sma_20 = close.rolling(20).mean()
    
    # Trend strength: distance from SMA normalized by ATR
    trend_strength = abs(close - sma_50) / atr
    
    # Regime classification
    is_trending = (volatility_zscore > 0.5) | (trend_strength > 2)
    is_ranging = ~is_trending
    
    # --- TREND REGIME: Follow momentum ---
    momentum_long = (close > sma_50) & (sma_20 > sma_50)
    momentum_short = (close < sma_50) & (sma_20 < sma_50)
    
    # --- RANGE REGIME: Mean reversion ---
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger
    bb_middle = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_middle + 2 * bb_std_val
    bb_lower = bb_middle - 2 * bb_std_val
    
    mr_long = (rsi < 25) & (close < bb_lower)
    mr_short = (rsi > 75) & (close > bb_upper)
    
    # Combine based on regime
    signal = pd.Series(0, index=df.index)
    
    # Trending regime
    signal[(is_trending) & (momentum_long)] = 1
    signal[(is_trending) & (momentum_short)] = -1
    
    # Ranging regime
    signal[(is_ranging) & (mr_long)] = 1
    signal[(is_ranging) & (mr_short)] = -1
    
    return signal


class AdaptiveRegimeStrategy(IStrategy):
    """
    Class-based adaptive regime strategy for live trading.
    """
    
    def __init__(self, atr_period: int = 14, regime_period: int = 50):
        super().__init__(name="AdaptiveRegime")
        self.atr_period = atr_period
        self.regime_period = regime_period
        self._warmup_period = max(atr_period, regime_period) + 10
        
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        self._current_regime = "unknown"
        
    def update(self, candle: Dict[str, Any]) -> Signal:
        self._candle_count += 1
        self._highs.append(candle["high"])
        self._lows.append(candle["low"])
        self._closes.append(candle["close"])
        
        max_len = self.regime_period * 3
        if len(self._closes) > max_len:
            self._highs = self._highs[-max_len:]
            self._lows = self._lows[-max_len:]
            self._closes = self._closes[-max_len:]
            
        if len(self._closes) < self._warmup_period:
            return Signal(SignalDirection.FLAT, 0.0, "Warming up")
            
        self._is_warmed_up = True
        
        # Build mini dataframe for signal calculation
        df = pd.DataFrame({
            "high": self._highs,
            "low": self._lows,
            "close": self._closes
        })
        
        signals = adaptive_regime_signal(df)
        current_signal = signals.iloc[-1]
        
        # Detect regime for logging
        price = self._closes[-1]
        sma_50 = sum(self._closes[-50:]) / 50 if len(self._closes) >= 50 else price
        
        if abs(price - sma_50) / sma_50 > 0.02:
            self._current_regime = "TREND"
        else:
            self._current_regime = "RANGE"
        
        if current_signal == 1:
            return Signal(
                SignalDirection.LONG, 0.7,
                f"LONG ({self._current_regime} regime)",
                metadata={"regime": self._current_regime}
            )
        elif current_signal == -1:
            return Signal(
                SignalDirection.SHORT, 0.7,
                f"SHORT ({self._current_regime} regime)",
                metadata={"regime": self._current_regime}
            )
        else:
            return Signal(
                SignalDirection.FLAT, 0.0,
                f"No signal ({self._current_regime} regime)"
            )
            
    def reset(self):
        self._highs = []
        self._lows = []
        self._closes = []
        self._is_warmed_up = False
        self._candle_count = 0
