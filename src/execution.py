"""
Execution Engine - Gestion Professionnelle des Risques

Implémente:
- Position Sizing basé sur la volatilité (Fixed Fractional Money Management)
- Stops dynamiques basés sur l'ATR
- Validation pré-trade (cooldown, spread, exposure)
- Calcul de quantité pour risquer un % fixe du capital par trade
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """Résultat de la validation pré-trade."""
    allowed: bool
    quantity: float
    reason: str


class ExecutionEngine:
    """
    Gère l'exécution des ordres et la validation des risques.
    Implémente le sizing de position basé sur la volatilité.
    """

    def __init__(self) -> None:
        self.last_trade_ts: float = 0.0
        self.current_exposure: float = 0.0

    def calculate_position_size(
        self, 
        balance: float, 
        entry_price: float, 
        stop_loss: float,
        risk_pct: float = 0.01  # 1% de risque par défaut
    ) -> float:
        """
        Calcule la taille de position pour risquer exactement X% du capital.
        
        Formule:
        - Risk = |Entry - SL|
        - Qty = (Balance * Risk_Pct) / Risk_Per_Unit
        
        Exemple:
        - Balance = $10,000
        - Entry = $91,000
        - SL = $90,500
        - Risk par share = $500
        - Risk max = $100 (1%)
        - Qty = 100 / 500 = 0.2 BTC
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
            
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
            
        # Risque max en $ (ex: 1% de 10,000$ = 100$)
        risk_amount = balance * risk_pct
        
        quantity = risk_amount / risk_per_share
        
        # Cap par le notional max (ex: pas plus de 50% du compte)
        max_qty_by_exposure = (balance * settings.MAX_EXPOSURE) / entry_price
        
        final_qty = min(quantity, max_qty_by_exposure)
        
        log.debug(f"Position sizing: balance=${balance:.2f}, risk_per_share=${risk_per_share:.2f}, qty={final_qty:.6f}")
        
        return final_qty

    def pre_trade_checks(
        self,
        current_balance: float,
        entry_price: float,
        stop_loss: float,
        spread: float = 0.0
    ) -> TradeSetup:
        """
        Vérifications complètes avant exécution d'un trade.
        
        Retourne un TradeSetup avec:
        - allowed: True si le trade est autorisé
        - quantity: La quantité calculée optimale
        - reason: Message explicatif
        """
        
        # 1. Cooldown - éviter l'overtrading
        elapsed = time.time() - self.last_trade_ts
        if elapsed < settings.COOLDOWN_SEC:
            remaining = settings.COOLDOWN_SEC - elapsed
            return TradeSetup(False, 0.0, f"Cooldown actif ({remaining:.0f}s)")

        # 2. Spread Check - protection slippage
        if spread > settings.SPREAD_LIMIT:
            return TradeSetup(False, 0.0, f"Spread trop élevé: {spread:.4f} > {settings.SPREAD_LIMIT}")

        # 3. Exposition maximale
        if self.current_exposure >= settings.MAX_EXPOSURE:
            return TradeSetup(False, 0.0, f"Exposition max atteinte: {self.current_exposure:.1%}")

        # 4. Position Sizing
        quantity = self.calculate_position_size(current_balance, entry_price, stop_loss)
        notional = quantity * entry_price
        
        if notional < settings.MIN_NOTIONAL:
            return TradeSetup(False, 0.0, f"Taille trop petite (${notional:.2f} < ${settings.MIN_NOTIONAL})")
            
        if quantity <= 0:
            return TradeSetup(False, 0.0, "Erreur calcul quantité")

        return TradeSetup(True, quantity, "OK")

    def record_trade(self, exposure_delta: float = 0.0) -> None:
        """Enregistre un trade et met à jour l'état."""
        self.last_trade_ts = time.time()
        self.current_exposure = max(0.0, min(1.0, self.current_exposure + exposure_delta))
        log.info(f"Trade enregistré, exposition={self.current_exposure:.1%}")

    def reset_exposure(self) -> None:
        """Réinitialise l'exposition (après fermeture de position)."""
        self.current_exposure = 0.0
        log.info("Exposition réinitialisée à 0%")

    def check_exit_conditions(
        self, 
        current_price: float, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float, 
        side: str
    ) -> Optional[str]:
        """
        Vérifie si le prix actuel touche le SL ou le TP.
        
        Args:
            current_price: Prix actuel du marché
            entry_price: Prix d'entrée de la position
            stop_loss: Niveau de stop loss
            take_profit: Niveau de take profit
            side: "long" ou "short"
            
        Returns:
            "STOP_LOSS", "TAKE_PROFIT", ou None
        """
        if side == "long":
            if current_price <= stop_loss:
                return "STOP_LOSS"
            if current_price >= take_profit:
                return "TAKE_PROFIT"
        elif side == "short":
            if current_price >= stop_loss:
                return "STOP_LOSS"
            if current_price <= take_profit:
                return "TAKE_PROFIT"
                
        return None

    def should_exit(
        self, 
        pnl_pct: float, 
        stop_loss_pct: float = -0.01, 
        take_profit_pct: float = 0.02
    ) -> Optional[str]:
        """
        Vérifie si on doit sortir basé sur le PnL en pourcentage.
        (Méthode legacy pour compatibilité)
        """
        if pnl_pct <= stop_loss_pct:
            return "stop_loss"
        if pnl_pct >= take_profit_pct:
            return "take_profit"
        return None
