"""
Vectorized Backtesting Engine - High Performance.
NO for-loops. Pure NumPy/Pandas vectorized operations.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_balance: float = 10000.0
    position_size_pct: float = 0.05      # 5% per trade
    commission_pct: float = 0.001         # 0.1% per trade
    slippage_pct: float = 0.0005          # 0.05% slippage
    max_positions: int = 1                # Single position at a time
    

@dataclass
class BacktestResult:
    """Backtest results summary."""
    initial_balance: float
    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    max_consecutive_losses: int
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


class VectorizedBacktester:
    """
    High-performance vectorized backtesting engine.
    
    Key principles:
    - No Python loops over data
    - Pure NumPy/Pandas operations
    - Signal generation is user-provided
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series],
    ) -> BacktestResult:
        """
        Run vectorized backtest.
        
        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            signal_func: Function that takes df and returns Series of signals
                         (+1 = long, -1 = short, 0 = flat)
                         
        Returns:
            BacktestResult with metrics and equity curve
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        required_cols = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        # Ensure sorted
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Generate signals (vectorized)
        log.info("Generating signals...")
        signals = signal_func(df)
        
        # Calculate returns
        log.info("Computing returns...")
        result = self._compute_returns(df, signals)
        
        return result
        
    def _compute_returns(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> BacktestResult:
        """
        Vectorized return computation.
        This is the performance-critical section.
        """
        n = len(df)
        close = df["close"].values
        
        # Price returns (log returns for accuracy)
        log_returns = np.log(close[1:] / close[:-1])
        log_returns = np.concatenate([[0], log_returns])
        
        # Position from signals (with 1-bar delay for realistic execution)
        position = np.zeros(n)
        position[1:] = signals.values[:-1]  # Signal at t -> Position at t+1
        
        # Strategy returns = position * market return - costs
        strategy_returns = position * log_returns
        
        # Transaction costs (when position changes)
        position_change = np.abs(np.diff(position, prepend=0))
        transaction_costs = position_change * (self.config.commission_pct + self.config.slippage_pct)
        strategy_returns -= transaction_costs
        
        # Cumulative returns -> Equity curve
        cumulative_returns = np.exp(np.cumsum(strategy_returns))
        equity = self.config.initial_balance * cumulative_returns
        
        # Build equity DataFrame
        equity_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "close": close,
            "signal": signals.values,
            "position": position,
            "return": strategy_returns,
            "equity": equity
        })
        
        # Extract trades
        trades_df = self._extract_trades(equity_df, close)
        
        # Compute metrics
        final_balance = equity[-1]
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance
        
        # Max Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.abs(drawdown.min())
        
        # Sharpe Ratio (annualized, assuming 1-min bars)
        # Adjust for different timeframes
        if len(strategy_returns) > 1:
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            if std_return > 0:
                # Annualize: sqrt(252 * 24 * 60) for 1-min bars
                sharpe = mean_return / std_return * np.sqrt(252 * 24 * 60)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
            
        # Trade statistics
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning = trades_df[trades_df["pnl"] > 0]
            losing = trades_df[trades_df["pnl"] <= 0]
            
            winning_trades = len(winning)
            losing_trades = len(losing)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            gross_profit = winning["pnl"].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_trade_pnl = trades_df["pnl"].mean()
            
            # Max consecutive losses
            is_loss = (trades_df["pnl"] <= 0).astype(int)
            loss_groups = (is_loss != is_loss.shift()).cumsum()
            max_consecutive_losses = is_loss.groupby(loss_groups).sum().max()
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_trade_pnl = 0
            max_consecutive_losses = 0
            
        return BacktestResult(
            initial_balance=self.config.initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return * 100,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            max_consecutive_losses=int(max_consecutive_losses),
            equity_curve=equity_df,
            trades=trades_df
        )
        
    def _extract_trades(
        self,
        equity_df: pd.DataFrame,
        close: np.ndarray
    ) -> pd.DataFrame:
        """Extract individual trades from position changes."""
        position = equity_df["position"].values
        timestamp = equity_df["timestamp"].values
        
        trades = []
        current_trade = None
        
        # Iterate through positions to identify trades
        # Skip first element (0) as we diff against it effectively
        for i in range(1, len(position)):
            prev_pos = position[i-1]
            curr_pos = position[i]
            
            if curr_pos == prev_pos:
                continue
                
            price = close[i]
            ts = timestamp[i]
            
            # 1. Close existing position if any
            if prev_pos != 0:
                if current_trade:
                    direction = 1 if current_trade["side"] == "long" else -1
                    entry_price = current_trade["entry_price"]
                    
                    pnl_pct = direction * (price - entry_price) / entry_price
                    pnl = pnl_pct * self.config.initial_balance * self.config.position_size_pct
                    
                    trades.append({
                        "entry_time": current_trade["entry_time"],
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "direction": current_trade["side"],
                        "pnl_pct": pnl_pct * 100,
                        "pnl": pnl,
                        "bars_held": i - current_trade["entry_idx"]
                    })
                    current_trade = None
            
            # 2. Open new position if applicable
            if curr_pos != 0:
                current_trade = {
                    "entry_time": ts,
                    "entry_idx": i,
                    "entry_price": price,
                    "side": "long" if curr_pos > 0 else "short"
                }
                
        return pd.DataFrame(trades)


def print_results(result: BacktestResult):
    """Pretty print backtest results."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Initial Balance:    ${result.initial_balance:,.2f}")
    print(f"Final Balance:      ${result.final_balance:,.2f}")
    print(f"Total Return:       {result.total_return_pct:+.2f}%")
    print(f"Max Drawdown:       {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print("-" * 50)
    print(f"Total Trades:       {result.total_trades}")
    print(f"Win Rate:           {result.win_rate:.1%}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")
    print(f"Avg Trade PnL:      ${result.avg_trade_pnl:,.2f}")
    print(f"Max Consec. Losses: {result.max_consecutive_losses}")
    print("=" * 50)


# Example signal functions (vectorized)
def sma_crossover_signal(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """
    Simple SMA crossover signal.
    Fully vectorized - no loops.
    """
    sma_fast = df["close"].rolling(window=fast).mean()
    sma_slow = df["close"].rolling(window=slow).mean()
    
    signal = pd.Series(0, index=df.index)
    signal[sma_fast > sma_slow] = 1   # Long when fast > slow
    signal[sma_fast < sma_slow] = -1  # Short when fast < slow
    
    return signal


def rsi_signal(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    RSI mean-reversion signal.
    Fully vectorized.
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0, index=df.index)
    signal[rsi < oversold] = 1    # Long when oversold
    signal[rsi > overbought] = -1  # Short when overbought
    
    return signal
