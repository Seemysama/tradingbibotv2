"""Backtesting Engine - Vectorized performance testing."""
from src.backtest.engine import (
    VectorizedBacktester,
    BacktestConfig,
    BacktestResult,
    print_results,
    sma_crossover_signal,
    rsi_signal
)

__all__ = [
    "VectorizedBacktester",
    "BacktestConfig",
    "BacktestResult",
    "print_results",
    "sma_crossover_signal",
    "rsi_signal"
]
