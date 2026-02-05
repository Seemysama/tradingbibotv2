"""
Live Trading Demo - Runs bot with dashboard.
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.core.strategy import IStrategy, SMAStrategy, RSIStrategy, Signal, SignalDirection
from src.adapters.binance import BinanceDataFetcher
from src.risk import RiskManager, RiskCheckResult
from src.dashboard.server import app, state, manager, run_dashboard

import ccxt
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("live_demo")


class LiveTradingDemo:
    """
    Live trading demo that streams real data and shows signals in dashboard.
    Uses PAPER mode - no real trades.
    """
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.strategy = SMAStrategy(fast_period=5, slow_period=15)
        self.risk_manager = RiskManager(initial_balance=10000.0)
        
        # Exchange for live data
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        
        self._running = False
        
    async def fetch_live_price(self) -> dict:
        """Fetch current price from Binance."""
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(
            None,
            lambda: self.exchange.fetch_ticker(self.symbol)
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "open": ticker["open"],
            "high": ticker["high"],
            "low": ticker["low"],
            "close": ticker["last"],
            "volume": ticker["baseVolume"]
        }
        
    async def run_trading_loop(self):
        """Main trading loop."""
        log.info(f"🚀 Starting live demo for {self.symbol}")
        self._running = True
        
        while self._running:
            try:
                # Fetch live price
                candle = await self.fetch_live_price()
                
                # Update state for dashboard
                state.candles.append(candle)
                state.last_price = candle["close"]
                
                # Generate signal
                signal = self.strategy.update(candle)
                
                signal_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "direction": signal.direction.value,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                    "price": candle["close"]
                }
                state.signals.append(signal_data)
                
                # Process signal
                if signal.is_actionable:
                    await self._process_signal(signal, candle)
                    
                # Update equity
                state.equity.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "equity": state.balance + state.pnl
                })
                
                # Broadcast to WebSocket clients
                await manager.broadcast(state.to_dict())
                
                log.info(f"Price: ${candle['close']:,.2f} | Signal: {signal.direction.value} ({signal.confidence:.0%}) | PnL: ${state.pnl:,.2f}")
                
                # Wait before next update
                await asyncio.sleep(5)  # 5 second updates
                
            except Exception as e:
                log.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
                
    async def _process_signal(self, signal: Signal, candle: dict):
        """Process trading signal (paper trading)."""
        price = candle["close"]
        position_size = 0.05  # 5% of balance
        notional = state.balance * position_size
        
        # Risk check
        check = self.risk_manager.pre_trade_check(notional)
        
        if check != RiskCheckResult.APPROVED:
            log.warning(f"Trade rejected: {check.value}")
            return
            
        # Paper trade simulation
        if signal.direction == SignalDirection.LONG and state.current_position is None:
            # Open long
            state.current_position = {
                "side": "long",
                "entry_price": price,
                "size": notional / price,
                "timestamp": datetime.utcnow().isoformat()
            }
            log.info(f"📈 LONG @ ${price:,.2f}")
            
        elif signal.direction == SignalDirection.SHORT and state.current_position is not None:
            # Close position
            pos = state.current_position
            if pos["side"] == "long":
                pnl = (price - pos["entry_price"]) * pos["size"]
            else:
                pnl = (pos["entry_price"] - price) * pos["size"]
                
            state.pnl += pnl
            state.balance += pnl
            
            state.trades.append({
                "entry_time": pos["timestamp"],
                "exit_time": datetime.utcnow().isoformat(),
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "exit_price": price,
                "pnl": pnl
            })
            
            log.info(f"📉 CLOSE @ ${price:,.2f} | PnL: ${pnl:,.2f}")
            state.current_position = None
            
            # Record trade in risk manager
            self.risk_manager.record_trade(pnl, notional, is_open=False)
            
    def stop(self):
        self._running = False


async def run_with_dashboard():
    """Run trading demo with dashboard."""
    demo = LiveTradingDemo()
    
    # Start dashboard server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8888, log_level="warning")
    server = uvicorn.Server(config)
    
    # Run both concurrently
    dashboard_task = asyncio.create_task(server.serve())
    trading_task = asyncio.create_task(demo.run_trading_loop())
    
    log.info("🌐 Dashboard available at http://127.0.0.1:8888")
    
    try:
        await asyncio.gather(dashboard_task, trading_task)
    except asyncio.CancelledError:
        demo.stop()


if __name__ == "__main__":
    asyncio.run(run_with_dashboard())
