"""Core module - Domain logic and abstract interfaces."""
from src.core.strategy import (
    IStrategy,
    Signal,
    SignalDirection,
    SMAStrategy,
    RSIStrategy,
    CombinedStrategy
)

from src.core.alpha import (
    MeanReversionStrategy,
    MomentumVolumeStrategy,
    PolymarketArbitrageStrategy,
    mean_reversion_signal,
    momentum_volume_signal
)

__all__ = [
    "IStrategy",
    "Signal",
    "SignalDirection",
    "SMAStrategy",
    "RSIStrategy",
    "CombinedStrategy",
    "MeanReversionStrategy",
    "MomentumVolumeStrategy",
    "PolymarketArbitrageStrategy",
    "mean_reversion_signal",
    "momentum_volume_signal"
]
