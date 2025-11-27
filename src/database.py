"""
Database Layer - Gestion SQLite + Support QuestDB
Persistance transactionnelle pour ordres, positions et état du bot
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Type, TypeVar

from sqlmodel import Session, SQLModel, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from state.models import (
    BotState,
    EquitySnapshot,
    Order,
    OrderStatus,
    Position,
    PositionStatus,
    Trade,
)
from src.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=SQLModel)


class DatabaseManager:
    """
    Gestionnaire de base de données SQLite pour l'état transactionnel.
    
    Utilise:
    - SQLite pour ordres, positions, état du bot (transactionnel)
    - QuestDB pour les séries temporelles (prix, equity curve)
    """
    
    def __init__(self, db_path: Optional[Path] = None, echo: bool = False):
        self.db_path = db_path or Path("data/trading_state.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite synchrone (pour la plupart des opérations)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=echo,
            connect_args={"check_same_thread": False}
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False
        )
        
        self._initialized = False
        log.info(f"DatabaseManager initialisé avec {self.db_path}")
    
    def init_db(self) -> None:
        """Crée les tables si nécessaire."""
        if self._initialized:
            return
        
        SQLModel.metadata.create_all(self.engine)
        self._initialized = True
        log.info("Tables SQLite créées/vérifiées")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Session, None]:
        """Context manager pour session synchrone (utilisé dans asyncio)."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_sync_session(self) -> Session:
        """Retourne une session synchrone."""
        return self.SessionLocal()
    
    # ----- Bot State -----
    
    async def get_bot_state(self) -> BotState:
        """Récupère ou crée l'état du bot."""
        async with self.get_session() as session:
            state = session.get(BotState, "main")
            if not state:
                state = BotState(id="main")
                session.add(state)
                session.commit()
                session.refresh(state)
            return state
    
    async def update_bot_state(self, **kwargs) -> BotState:
        """Met à jour l'état du bot."""
        async with self.get_session() as session:
            state = session.get(BotState, "main")
            if not state:
                state = BotState(id="main")
                session.add(state)
            
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            
            state.last_activity_at = datetime.now()
            session.commit()
            session.refresh(state)
            return state
    
    # ----- Orders -----
    
    async def create_order(self, order: Order) -> Order:
        """Crée un nouvel ordre."""
        async with self.get_session() as session:
            session.add(order)
            session.commit()
            session.refresh(order)
            log.info(f"Ordre créé: {order.id} - {order.side} {order.symbol}")
            return order
    
    async def update_order(self, order_id: str, **kwargs) -> Optional[Order]:
        """Met à jour un ordre existant."""
        async with self.get_session() as session:
            order = session.get(Order, order_id)
            if not order:
                return None
            
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            order.updated_at = datetime.now()
            session.commit()
            session.refresh(order)
            return order
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Récupère un ordre par ID."""
        async with self.get_session() as session:
            return session.get(Order, order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Récupère les ordres ouverts."""
        async with self.get_session() as session:
            stmt = select(Order).where(
                Order.status.in_([OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED])
            )
            if symbol:
                stmt = stmt.where(Order.symbol == symbol)
            
            result = session.exec(stmt)
            return list(result.all())
    
    async def get_recent_orders(self, limit: int = 50) -> List[Order]:
        """Récupère les ordres récents."""
        async with self.get_session() as session:
            stmt = select(Order).order_by(Order.created_at.desc()).limit(limit)
            result = session.exec(stmt)
            return list(result.all())
    
    # ----- Positions -----
    
    async def create_position(self, position: Position) -> Position:
        """Crée une nouvelle position."""
        async with self.get_session() as session:
            session.add(position)
            session.commit()
            session.refresh(position)
            log.info(f"Position créée: {position.id} - {position.side} {position.symbol}")
            return position
    
    async def update_position(self, position_id: str, **kwargs) -> Optional[Position]:
        """Met à jour une position."""
        async with self.get_session() as session:
            position = session.get(Position, position_id)
            if not position:
                return None
            
            for key, value in kwargs.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            
            position.updated_at = datetime.now()
            session.commit()
            session.refresh(position)
            return position
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "manual"
    ) -> Optional[Position]:
        """Ferme une position."""
        async with self.get_session() as session:
            position = session.get(Position, position_id)
            if not position:
                return None
            
            position.status = PositionStatus.CLOSED
            position.exit_price = exit_price
            position.exit_reason = exit_reason
            position.closed_at = datetime.now()
            position.updated_at = datetime.now()
            
            # Calcul du PnL réalisé
            position.calculate_pnl(exit_price)
            position.realized_pnl = position.unrealized_pnl
            position.unrealized_pnl = 0.0
            
            session.commit()
            session.refresh(position)
            
            log.info(f"Position fermée: {position.id} - PnL: {position.realized_pnl:.2f}")
            return position
    
    async def get_position(self, position_id: str) -> Optional[Position]:
        """Récupère une position par ID."""
        async with self.get_session() as session:
            return session.get(Position, position_id)
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Récupère les positions ouvertes."""
        async with self.get_session() as session:
            stmt = select(Position).where(Position.status == PositionStatus.OPEN)
            if symbol:
                stmt = stmt.where(Position.symbol == symbol)
            
            result = session.exec(stmt)
            return list(result.all())
    
    async def get_position_for_symbol(self, symbol: str) -> Optional[Position]:
        """Récupère la position ouverte pour un symbole."""
        positions = await self.get_open_positions(symbol)
        return positions[0] if positions else None
    
    # ----- Trades -----
    
    async def record_trade(self, trade: Trade) -> Trade:
        """Enregistre un trade."""
        async with self.get_session() as session:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
    
    async def get_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Trade]:
        """Récupère les trades récents."""
        async with self.get_session() as session:
            stmt = select(Trade).order_by(Trade.executed_at.desc()).limit(limit)
            if symbol:
                stmt = stmt.where(Trade.symbol == symbol)
            
            result = session.exec(stmt)
            return list(result.all())
    
    # ----- Equity Snapshots -----
    
    async def record_equity(
        self,
        equity: float,
        balance: float,
        unrealized_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
        position_count: int = 0
    ) -> EquitySnapshot:
        """Enregistre un snapshot d'equity."""
        snapshot = EquitySnapshot(
            equity=equity,
            balance=balance,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            position_count=position_count
        )
        async with self.get_session() as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
            return snapshot
    
    async def get_equity_curve(self, limit: int = 500) -> List[EquitySnapshot]:
        """Récupère la courbe d'equity."""
        async with self.get_session() as session:
            stmt = select(EquitySnapshot).order_by(
                EquitySnapshot.timestamp.desc()
            ).limit(limit)
            result = session.exec(stmt)
            return list(reversed(result.all()))
    
    # ----- Cleanup -----
    
    async def cleanup_old_data(self, days: int = 30) -> int:
        """Nettoie les données anciennes."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        deleted = 0
        async with self.get_session() as session:
            # Equity snapshots
            old_snapshots = session.exec(
                select(EquitySnapshot).where(EquitySnapshot.timestamp < cutoff)
            ).all()
            for snap in old_snapshots:
                session.delete(snap)
                deleted += 1
            
            session.commit()
        
        log.info(f"Nettoyage: {deleted} enregistrements supprimés")
        return deleted


# ----- QuestDB Support (Optionnel) -----

class QuestDBClient:
    """
    Client QuestDB pour les séries temporelles haute fréquence.
    Utilisé pour les prix, candles, et métriques de performance.
    """
    
    def __init__(self, host: str = "localhost", port: int = 9009):
        self.host = host
        self.port = port
        self._socket = None
    
    async def connect(self) -> bool:
        """Établit la connexion ILP."""
        import socket
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
            log.info(f"Connecté à QuestDB {self.host}:{self.port}")
            return True
        except Exception as e:
            log.warning(f"Impossible de se connecter à QuestDB: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Ferme la connexion."""
        if self._socket:
            self._socket.close()
            self._socket = None
    
    async def send_line(self, line: str) -> None:
        """Envoie une ligne ILP."""
        if not self._socket:
            return
        try:
            self._socket.send((line + "\n").encode())
        except Exception as e:
            log.error(f"Erreur envoi QuestDB: {e}")
            await self.connect()
    
    async def record_candle(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float
    ) -> None:
        """Enregistre une bougie OHLCV."""
        ts_ns = int(timestamp.timestamp() * 1e9)
        line = f"candles,symbol={symbol.replace('/', '')} open={open_price},high={high},low={low},close={close},volume={volume} {ts_ns}"
        await self.send_line(line)
    
    async def record_trade_metric(
        self,
        symbol: str,
        side: str,
        price: float,
        pnl: float
    ) -> None:
        """Enregistre une métrique de trade."""
        ts_ns = int(datetime.now().timestamp() * 1e9)
        line = f"trades,symbol={symbol.replace('/', '')},side={side} price={price},pnl={pnl} {ts_ns}"
        await self.send_line(line)


# Singleton global
_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Récupère l'instance globale du gestionnaire de DB."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.init_db()
    return _db_manager


async def init_database() -> DatabaseManager:
    """Initialise la base de données (appelé au démarrage)."""
    db = get_db()
    # Créer l'état initial si nécessaire
    await db.get_bot_state()
    return db
