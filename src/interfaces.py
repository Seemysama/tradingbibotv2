from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class TradeCheck:
    allowed: bool
    reason: str

class IExecutionEngine(ABC):
    """Interface for execution engines (CEX, DEX, Polymarket)."""
    
    @abstractmethod
    def pre_trade_checks(self, spread: float, desired_exposure: float, available_balance: float, order_notional: float) -> TradeCheck:
        pass
        
    @abstractmethod
    def should_exit(self, pnl_pct: float, stop_loss: float, take_profit: float) -> Optional[str]:
        pass
        
    @abstractmethod
    def record_trade(self, exposure_change: float):
        pass

class IStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    def update(self, candle: dict) -> Any:
        """Process a new candle and return a Signal."""
        pass
