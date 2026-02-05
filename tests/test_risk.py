"""
Unit tests for Risk Management (Kill Switch).
Testing hard limits and circuit breakers.
"""
import pytest
from src.risk import RiskManager, RiskConfig, RiskCheckResult

def test_max_daily_loss():
    """Test that max daily loss engages rejection."""
    config = RiskConfig(max_daily_loss_pct=0.05) # 5% max loss
    risk = RiskManager(config, initial_balance=10000.0)
    
    # Simulate loss
    risk.record_trade(pnl=-600, order_notional=1000, is_open=False)
    
    # Daily loss = 600 / 10000 = 6% > 5%
    result = risk.pre_trade_check(order_notional=100)
    assert result == RiskCheckResult.REJECTED_DAILY_LOSS

def test_max_consecutive_losses():
    """Test kill switch on consecutive losses."""
    config = RiskConfig(kill_on_consecutive_losses=3)
    risk = RiskManager(config)
    
    risk.record_trade(pnl=-10, order_notional=100, is_open=False)
    risk.record_trade(pnl=-10, order_notional=100, is_open=False)
    risk.record_trade(pnl=-10, order_notional=100, is_open=False)
    
    # 4th trade should be rejected by kill switch
    result = risk.pre_trade_check(order_notional=100)
    assert result == RiskCheckResult.REJECTED_KILL_SWITCH
    assert risk.state.kill_switch_engaged

def test_max_position_size():
    """Test position sizing limits."""
    config = RiskConfig(max_position_size_pct=0.1) # 10% max
    risk = RiskManager(config, initial_balance=10000.0)
    
    # Try 20% position
    result = risk.pre_trade_check(order_notional=2000)
    assert result == RiskCheckResult.REJECTED_POSITION_SIZE
    
    # Try 5% position
    result = risk.pre_trade_check(order_notional=500)
    assert result == RiskCheckResult.APPROVED

def test_api_error_circuit_breaker():
    """Test circuit breaker on API errors."""
    config = RiskConfig(kill_on_api_errors=2)
    risk = RiskManager(config)
    
    risk.record_api_error()
    risk.pre_trade_check(100) # Should be fine
    
    risk.record_api_error()
    # Now kill switch should be active
    result = risk.pre_trade_check(100)
    assert result == RiskCheckResult.REJECTED_KILL_SWITCH
