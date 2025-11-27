"""
Trading Engine Orchestrator - Production Grade
Supporte LIVE, PAPER et BACKTEST via variable d'environnement MODE
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.inference import InferenceEngine
from src.config import settings
from src.database import init_database, get_db
from src.execution import ExecutionEngine
from src.feed import Candle, FeedConfig, MarketFeed, create_feed
from src.strategy import HybridStrategy
from state.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)


logging.basicConfig(
    level=getattr(logging, settings.LOGGER_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orchestrator")


class TradingOrchestrator:
    """
    Orchestrateur de trading unifié.
    Gère le cycle de vie complet: feed → strategy → execution → persistence
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        mode: Optional[str] = None,
    ):
        self.symbol = symbol
        self.mode = (mode or settings.MODE).upper()
        
        # Composants
        self.feed: Optional[MarketFeed] = None
        self.strategy: Optional[HybridStrategy] = None
        self.inference: Optional[InferenceEngine] = None
        self.execution: Optional[ExecutionEngine] = None
        self.db = None
        
        # État
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._current_position: Optional[Position] = None
        
        # Métriques
        self.stats = {
            "candles_processed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "start_time": None,
        }
    
    async def initialize(self) -> None:
        """Initialise tous les composants."""
        log.info(f"🚀 Initialisation TradingOrchestrator (mode={self.mode})")
        
        # Database
        self.db = await init_database()
        bot_state = await self.db.get_bot_state()
        log.info(f"État précédent: balance={bot_state.current_balance:.2f}, trades={bot_state.total_trades}")
        
        # Charger la position ouverte si existante
        open_positions = await self.db.get_open_positions(self.symbol)
        if open_positions:
            self._current_position = open_positions[0]
            log.info(f"Position ouverte trouvée: {self._current_position.side} @ {self._current_position.entry_price}")
        
        # Inference Engine
        self.inference = InferenceEngine()
        
        # Strategy
        self.strategy = HybridStrategy(inference=self.inference)
        
        # Execution Engine
        self.execution = ExecutionEngine()
        
        # Feed Configuration
        feed_config = FeedConfig(
            symbol=self.symbol,
            timeframe=settings.TIMEFRAME,
            warmup_candles=settings.WARMUP_CANDLES,
            replay_speed=settings.REPLAY_SPEED_MS if self.mode != "BACKTEST" else 0,
        )
        
        # Créer le feed approprié
        self.feed = create_feed(self.mode, feed_config)
        
        log.info("✅ Composants initialisés")
    
    async def warmup(self) -> None:
        """Préchauffe la stratégie avec les données de warmup."""
        if not self.feed:
            raise RuntimeError("Feed non initialisé")
        
        # Démarrer le feed pour charger les données
        await self.feed.start()
        
        # Attendre un peu que le warmup soit chargé
        await asyncio.sleep(0.1)
        
        # Récupérer les bougies de warmup
        warmup_candles = self.feed.get_warmup_candles()
        
        log.info(f"🔥 Warmup avec {len(warmup_candles)} bougies")
        
        for candle in warmup_candles:
            self.inference.update_buffer(candle.to_dict())
            self.strategy.update(candle.to_dict())
        
        log.info("✅ Warmup terminé")
    
    async def _process_candle(self, candle: Candle) -> None:
        """Traite une bougie: signal → risk check → execution."""
        self.stats["candles_processed"] += 1
        
        # Générer le signal
        signal = self.strategy.update(candle.to_dict())
        
        current_price = candle.close
        
        # Gestion de la position existante
        if self._current_position:
            await self._manage_open_position(candle, signal)
        
        # Ouverture de nouvelle position
        if not self._current_position and signal.direction and signal.confidence > 0:
            await self._open_position(candle, signal)
    
    async def _manage_open_position(self, candle: Candle, signal) -> None:
        """Gère une position ouverte (exit check)."""
        pos = self._current_position
        if not pos:
            return
        
        current_price = candle.close
        pnl_pct = pos.calculate_pnl_pct(current_price) / 100  # En ratio
        
        # Check stop loss / take profit
        exit_reason = self.execution.should_exit(
            pnl_pct,
            stop_loss=-settings.STOP_LOSS_PCT,
            take_profit=settings.TAKE_PROFIT_PCT
        )
        
        # Signal opposé = exit
        opposite_signal = (
            (signal.direction == "short" and pos.side == PositionSide.LONG) or
            (signal.direction == "long" and pos.side == PositionSide.SHORT)
        )
        
        if exit_reason or opposite_signal:
            reason = exit_reason or f"Signal opposé: {signal.reason}"
            await self._close_position(candle, reason)
    
    async def _open_position(self, candle: Candle, signal) -> None:
        """Ouvre une nouvelle position."""
        bot_state = await self.db.get_bot_state()
        balance = bot_state.current_balance
        
        # Calcul de la taille
        notional = balance * settings.POSITION_SIZE_PCT
        quantity = notional / candle.close
        
        # Vérifications
        check = self.execution.pre_trade_checks(
            spread=0.0005,  # Placeholder
            desired_exposure=settings.POSITION_SIZE_PCT,
            available_balance=balance,
            order_notional=notional
        )
        
        if not check.allowed:
            log.debug(f"Trade rejeté: {check.reason}")
            return
        
        # Créer l'ordre
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY if signal.direction == "long" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=candle.close,
            signal_reason=signal.reason,
            signal_confidence=signal.confidence,
        )
        
        # Créer la position
        position = Position(
            symbol=self.symbol,
            side=PositionSide.LONG if signal.direction == "long" else PositionSide.SHORT,
            size=quantity,
            entry_price=candle.close,
            leverage=settings.LEVERAGE,
            stop_loss=candle.close * (1 - settings.STOP_LOSS_PCT) if signal.direction == "long" else candle.close * (1 + settings.STOP_LOSS_PCT),
            take_profit=candle.close * (1 + settings.TAKE_PROFIT_PCT) if signal.direction == "long" else candle.close * (1 - settings.TAKE_PROFIT_PCT),
            entry_reason=signal.reason,
        )
        
        # Persister
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.average_fill_price = candle.close
        order.filled_at = datetime.now()
        
        await self.db.create_order(order)
        position = await self.db.create_position(position)
        
        self._current_position = position
        self.execution.record_trade(settings.POSITION_SIZE_PCT)
        self.stats["trades_executed"] += 1
        self.stats["signals_generated"] += 1
        
        log.info(f"🟢 Position ouverte: {signal.direction} @ ${candle.close:.2f} | Conf: {signal.confidence:.2%}")
    
    async def _close_position(self, candle: Candle, reason: str) -> None:
        """Ferme la position actuelle."""
        pos = self._current_position
        if not pos:
            return
        
        current_price = candle.close
        
        # Créer l'ordre de sortie
        order = Order(
            symbol=self.symbol,
            side=OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=pos.size,
            price=current_price,
            position_id=pos.id,
            signal_reason=reason,
        )
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = pos.size
        order.average_fill_price = current_price
        order.filled_at = datetime.now()
        
        await self.db.create_order(order)
        
        # Fermer la position
        closed_pos = await self.db.close_position(pos.id, current_price, reason)
        
        if closed_pos:
            pnl = closed_pos.realized_pnl
            
            # Mettre à jour l'état du bot
            bot_state = await self.db.get_bot_state()
            new_balance = bot_state.current_balance + pnl
            
            await self.db.update_bot_state(
                current_balance=new_balance,
                total_pnl=bot_state.total_pnl + pnl,
                total_trades=bot_state.total_trades + 1,
                winning_trades=bot_state.winning_trades + (1 if pnl > 0 else 0),
                losing_trades=bot_state.losing_trades + (1 if pnl < 0 else 0),
            )
            
            # Enregistrer l'equity
            await self.db.record_equity(
                equity=new_balance,
                balance=new_balance,
                unrealized_pnl=0,
            )
            
            log.info(f"🔴 Position fermée: {pos.side.value} | PnL: ${pnl:.2f} | Raison: {reason}")
        
        self._current_position = None
        self.execution.record_trade(-settings.POSITION_SIZE_PCT)
        self.stats["trades_executed"] += 1
    
    async def run(self) -> None:
        """Boucle principale de trading."""
        if not self.feed:
            await self.initialize()
        
        await self.warmup()
        
        self._running = True
        self.stats["start_time"] = datetime.now()
        
        # Mettre à jour l'état du bot
        await self.db.update_bot_state(
            is_running=True,
            mode=self.mode,
            active_symbols=self.symbol,
            started_at=datetime.now(),
        )
        
        log.info(f"🔄 Boucle de trading démarrée pour {self.symbol}")
        
        try:
            async for candle in self.feed:
                if not self._running:
                    break
                
                await self._process_candle(candle)
                
                # Log périodique
                if self.stats["candles_processed"] % 1000 == 0:
                    log.info(f"📊 {self.stats['candles_processed']} bougies | {self.stats['trades_executed']} trades")
        
        except asyncio.CancelledError:
            log.info("Trading interrompu")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Arrêt propre."""
        self._running = False
        
        if self.feed:
            await self.feed.stop()
        
        if self.db:
            await self.db.update_bot_state(
                is_running=False,
                last_activity_at=datetime.now(),
            )
        
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0
        
        log.info("🏁 Arrêt orchestrateur")
        log.info(f"📈 Stats: {self.stats['candles_processed']} bougies, {self.stats['trades_executed']} trades en {elapsed:.0f}s")


def setup_signal_handlers(orchestrator: TradingOrchestrator):
    """Configure les handlers pour arrêt propre."""
    def signal_handler(sig, frame):
        log.info(f"Signal {sig} reçu, arrêt en cours...")
        asyncio.create_task(orchestrator.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main() -> None:
    """Point d'entrée principal."""
    try:
        import uvloop
        uvloop.install()
        log.info("uvloop activé")
    except ImportError:
        pass
    
    log.info(f"🚀 Démarrage Trading Engine (MODE={settings.MODE}, DEVICE={settings.DEVICE})")
    
    # Créer l'orchestrateur
    orchestrator = TradingOrchestrator(
        symbol=settings.PAIRS[0],
        mode=settings.MODE,
    )
    
    # Signal handlers
    setup_signal_handlers(orchestrator)
    
    # Run
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
