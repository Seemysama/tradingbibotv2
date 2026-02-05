import logging
import uuid
from typing import Optional
from py_clob_client.client import ClobClient, OrderArgs, OrderType as PmOrderType
from py_clob_client.clob_types import ApiCreds, OrderSide as PmOrderSide

from src.interfaces import IExecutionEngine, TradeCheck
from src.state import OrderSide, OrderType
from src.config import settings

log = logging.getLogger(__name__)

class PolymarketExecution(IExecutionEngine):
    def __init__(self, client: ClobClient):
        self.client = client

    def pre_trade_checks(self, spread: float, desired_exposure: float, available_balance: float, order_notional: float) -> TradeCheck:
        # Polymarket constraints
        if order_notional < 1.0: # Minimum order size usually
            return TradeCheck(False, "Order value too low for Polymarket")
        return TradeCheck(True, "OK")

    def should_exit(self, pnl_pct: float, stop_loss: float, take_profit: float) -> Optional[str]:
        if pnl_pct <= stop_loss:
            return "Stop Loss (Polymarket)"
        if pnl_pct >= take_profit:
            return "Take Profit (Polymarket)"
        return None

    def record_trade(self, exposure_change: float):
        pass # Todo: update internal state

    async def execute_order(self, symbol: str, side: OrderSide, quantity: float, price: float):
        """
        Execute order on Polymarket CLOB.
        Symbol is Token ID.
        """
        try:
            pm_side = PmOrderSide.BUY if side == OrderSide.BUY else PmOrderSide.SELL
            
            # Polymarket accepts LIMIT orders mostly on CLOB
            order_args = OrderArgs(
                price=price,
                size=quantity,
                side=pm_side,
                token_id=symbol,
            )
            
            resp = self.client.create_order(order_args)
            log.info(f"Polymarket Order placed: {resp}")
            return resp
            
        except Exception as e:
            log.error(f"Polymarket Execution Failed: {e}")
            raise
