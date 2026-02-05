"""
Risk Management Kill Switch - Hard limits and circuit breakers.
CRITICAL: This module protects capital. All trades MUST pass through here.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

log = logging.getLogger("risk")


class RiskCheckResult(Enum):
    """Risk check outcomes."""
    APPROVED = "approved"
    REJECTED_MAX_DRAWDOWN = "max_drawdown_exceeded"
    REJECTED_DAILY_LOSS = "daily_loss_exceeded"
    REJECTED_POSITION_SIZE = "position_size_exceeded"
    REJECTED_MAX_POSITIONS = "max_positions_exceeded"
    REJECTED_COOLDOWN = "trade_cooldown_active"
    REJECTED_KILL_SWITCH = "kill_switch_engaged"
    REJECTED_EXPOSURE = "exposure_limit_exceeded"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Hard limits
    max_daily_loss_pct: float = 0.05          # 5% max daily loss
    max_drawdown_pct: float = 0.10            # 10% max drawdown from peak
    max_position_size_pct: float = 0.10       # 10% max per position
    max_total_exposure_pct: float = 0.50      # 50% max total exposure
    max_open_positions: int = 5
    
    # Trade cooldowns
    min_trade_interval_sec: int = 30          # Min seconds between trades
    max_trades_per_hour: int = 20
    
    # Kill switch triggers
    kill_on_consecutive_losses: int = 5       # Auto-stop after X losses
    kill_on_api_errors: int = 3               # Stop after X API errors


@dataclass
class RiskState:
    """Current risk state tracking."""
    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    peak_balance: float = 10000.0
    daily_pnl: float = 0.0
    daily_start_balance: float = 10000.0
    daily_reset_time: datetime = field(default_factory=datetime.utcnow)
    
    open_positions: int = 0
    total_exposure: float = 0.0
    
    last_trade_time: Optional[datetime] = None
    trades_this_hour: int = 0
    hour_reset_time: datetime = field(default_factory=datetime.utcnow)
    
    consecutive_losses: int = 0
    api_error_count: int = 0
    
    kill_switch_engaged: bool = False
    kill_switch_reason: str = ""


class RiskManager:
    """
    Central risk management and kill switch.
    All trading decisions MUST pass through pre_trade_check().
    """
    
    def __init__(self, config: RiskConfig = None, initial_balance: float = 10000.0):
        self.config = config or RiskConfig()
        self.state = RiskState(
            initial_balance=initial_balance,
            current_balance=initial_balance,
            peak_balance=initial_balance,
            daily_start_balance=initial_balance
        )
        log.info(f"RiskManager initialized with balance={initial_balance}")
        
    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight UTC."""
        now = datetime.utcnow()
        if now.date() > self.state.daily_reset_time.date():
            self.state.daily_pnl = 0.0
            self.state.daily_start_balance = self.state.current_balance
            self.state.daily_reset_time = now
            log.info("Daily risk counters reset")
            
    def _reset_hourly_if_needed(self):
        """Reset hourly trade counter."""
        now = datetime.utcnow()
        if (now - self.state.hour_reset_time) > timedelta(hours=1):
            self.state.trades_this_hour = 0
            self.state.hour_reset_time = now
            
    def pre_trade_check(
        self,
        order_notional: float,
        is_close: bool = False
    ) -> RiskCheckResult:
        """
        Pre-trade risk validation. MUST be called before every order.
        
        Args:
            order_notional: Dollar value of the order
            is_close: True if this is closing a position
            
        Returns:
            RiskCheckResult indicating approval or rejection reason
        """
        self._reset_daily_if_needed()
        self._reset_hourly_if_needed()
        
        # 1. Kill switch check
        if self.state.kill_switch_engaged:
            log.warning(f"Trade rejected: Kill switch engaged ({self.state.kill_switch_reason})")
            return RiskCheckResult.REJECTED_KILL_SWITCH
            
        # Closing positions is always allowed (to reduce exposure)
        if is_close:
            return RiskCheckResult.APPROVED
            
        # 2. Daily loss check
        daily_loss_pct = abs(min(0, self.state.daily_pnl)) / self.state.daily_start_balance
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            log.warning(f"Trade rejected: Daily loss {daily_loss_pct:.1%} >= {self.config.max_daily_loss_pct:.1%}")
            return RiskCheckResult.REJECTED_DAILY_LOSS
            
        # 3. Drawdown check
        drawdown_pct = (self.state.peak_balance - self.state.current_balance) / self.state.peak_balance
        if drawdown_pct >= self.config.max_drawdown_pct:
            log.warning(f"Trade rejected: Drawdown {drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}")
            self.engage_kill_switch("Max drawdown exceeded")
            return RiskCheckResult.REJECTED_MAX_DRAWDOWN
            
        # 4. Position size check
        position_size_pct = order_notional / self.state.current_balance
        if position_size_pct > self.config.max_position_size_pct:
            log.warning(f"Trade rejected: Position size {position_size_pct:.1%} > {self.config.max_position_size_pct:.1%}")
            return RiskCheckResult.REJECTED_POSITION_SIZE
            
        # 5. Max positions check
        if self.state.open_positions >= self.config.max_open_positions:
            log.warning(f"Trade rejected: Max positions {self.config.max_open_positions} reached")
            return RiskCheckResult.REJECTED_MAX_POSITIONS
            
        # 6. Total exposure check
        new_exposure = (self.state.total_exposure + order_notional) / self.state.current_balance
        if new_exposure > self.config.max_total_exposure_pct:
            log.warning(f"Trade rejected: Exposure {new_exposure:.1%} > {self.config.max_total_exposure_pct:.1%}")
            return RiskCheckResult.REJECTED_EXPOSURE
            
        # 7. Trade cooldown check
        if self.state.last_trade_time:
            elapsed = (datetime.utcnow() - self.state.last_trade_time).total_seconds()
            if elapsed < self.config.min_trade_interval_sec:
                log.debug(f"Trade rejected: Cooldown ({elapsed:.0f}s < {self.config.min_trade_interval_sec}s)")
                return RiskCheckResult.REJECTED_COOLDOWN
                
        # 8. Hourly trade limit
        if self.state.trades_this_hour >= self.config.max_trades_per_hour:
            log.warning(f"Trade rejected: Hourly limit {self.config.max_trades_per_hour} reached")
            return RiskCheckResult.REJECTED_COOLDOWN
            
        return RiskCheckResult.APPROVED
        
    def record_trade(self, pnl: float, order_notional: float, is_open: bool):
        """Record a completed trade for tracking."""
        self.state.last_trade_time = datetime.utcnow()
        self.state.trades_this_hour += 1
        
        if is_open:
            self.state.open_positions += 1
            self.state.total_exposure += order_notional
        else:
            self.state.open_positions = max(0, self.state.open_positions - 1)
            self.state.total_exposure = max(0, self.state.total_exposure - order_notional)
            
        # Update P&L
        self.state.current_balance += pnl
        self.state.daily_pnl += pnl
        
        # Update peak
        if self.state.current_balance > self.state.peak_balance:
            self.state.peak_balance = self.state.current_balance
            
        # Track consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.config.kill_on_consecutive_losses:
                self.engage_kill_switch(f"{self.state.consecutive_losses} consecutive losses")
        else:
            self.state.consecutive_losses = 0
            
        log.info(f"Trade recorded: PnL=${pnl:.2f}, Balance=${self.state.current_balance:.2f}, Daily=${self.state.daily_pnl:.2f}")
        
    def record_api_error(self):
        """Record an API error for circuit breaker."""
        self.state.api_error_count += 1
        if self.state.api_error_count >= self.config.kill_on_api_errors:
            self.engage_kill_switch(f"{self.state.api_error_count} API errors")
            
    def reset_api_errors(self):
        """Reset API error counter on successful call."""
        self.state.api_error_count = 0
        
    def engage_kill_switch(self, reason: str):
        """Emergency stop all trading."""
        self.state.kill_switch_engaged = True
        self.state.kill_switch_reason = reason
        log.critical(f"🛑 KILL SWITCH ENGAGED: {reason}")
        
    def disengage_kill_switch(self):
        """Manually reset kill switch (requires human intervention)."""
        self.state.kill_switch_engaged = False
        self.state.kill_switch_reason = ""
        self.state.consecutive_losses = 0
        self.state.api_error_count = 0
        log.warning("Kill switch manually disengaged")
        
    def get_status(self) -> dict:
        """Get current risk status for monitoring."""
        drawdown = (self.state.peak_balance - self.state.current_balance) / self.state.peak_balance
        daily_loss = abs(min(0, self.state.daily_pnl)) / self.state.daily_start_balance
        
        return {
            "balance": self.state.current_balance,
            "peak_balance": self.state.peak_balance,
            "drawdown_pct": drawdown,
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_pct": daily_loss,
            "open_positions": self.state.open_positions,
            "exposure_pct": self.state.total_exposure / self.state.current_balance if self.state.current_balance > 0 else 0,
            "trades_this_hour": self.state.trades_this_hour,
            "consecutive_losses": self.state.consecutive_losses,
            "kill_switch": self.state.kill_switch_engaged,
            "kill_reason": self.state.kill_switch_reason
        }
