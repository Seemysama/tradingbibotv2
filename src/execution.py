import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str


class ExecutionEngine:
    """Gestion de l'exposition et garde-fous avant envoi d'ordres."""

    def __init__(self) -> None:
        self.last_trade_ts: float = 0.0
        self.current_exposure: float = 0.0  # fraction du capital engagé

    def _cooldown_ok(self) -> bool:
        if settings.COOLDOWN_SEC <= 0:
            return True
        return (time.time() - self.last_trade_ts) >= settings.COOLDOWN_SEC

    def pre_trade_checks(
        self,
        spread: float,
        desired_exposure: float,
        available_balance: float,
        order_notional: float,
    ) -> RiskCheckResult:
        if order_notional < settings.MIN_NOTIONAL:
            return RiskCheckResult(False, f"Notional trop faible (<{settings.MIN_NOTIONAL})")

        if available_balance < order_notional:
            return RiskCheckResult(False, "Solde insuffisant")

        if spread > settings.SPREAD_LIMIT:
            return RiskCheckResult(False, f"Spread trop élevé: {spread:.4f}")

        if not self._cooldown_ok():
            return RiskCheckResult(False, "Cool-down en cours")

        if (self.current_exposure + desired_exposure) > settings.MAX_EXPOSURE:
            return RiskCheckResult(False, "Exposition maximale atteinte")

        return RiskCheckResult(True, "OK")

    def record_trade(self, exposure_delta: float) -> None:
        """Met à jour l'exposition après un trade."""
        self.current_exposure = max(0.0, min(1.0, self.current_exposure + exposure_delta))
        self.last_trade_ts = time.time()
        log.info("Trade enregistré, exposition=%.3f", self.current_exposure)

    def should_exit(self, pnl_pct: float, stop_loss: float = -0.01, take_profit: float = 0.02) -> Optional[str]:
        if pnl_pct <= stop_loss:
            return "stop_loss"
        if pnl_pct >= take_profit:
            return "take_profit"
        return None
