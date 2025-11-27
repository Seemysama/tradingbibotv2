"""
State Models - Modèles SQLModel pour persistance transactionnelle
Ordres, Positions et état du bot
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import field_validator
from sqlmodel import Field, Relationship, SQLModel


class OrderStatus(str, Enum):
    """Status d'un ordre."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(str, Enum):
    """Côté de l'ordre."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Type d'ordre."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class PositionSide(str, Enum):
    """Direction de la position."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Status de la position."""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


# ----- Modèles SQLModel -----

class Order(SQLModel, table=True):
    """
    Modèle d'ordre persisté.
    Contient tous les détails d'un ordre envoyé/exécuté.
    """
    __tablename__ = "orders"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    
    # Identifiants externes
    exchange_order_id: Optional[str] = Field(default=None, index=True)
    client_order_id: Optional[str] = Field(default=None)
    
    # Détails de l'ordre
    symbol: str = Field(index=True)
    side: OrderSide
    order_type: OrderType
    status: OrderStatus = Field(default=OrderStatus.PENDING, index=True)
    
    # Prix et quantités
    quantity: float
    price: Optional[float] = None  # None pour market orders
    stop_price: Optional[float] = None  # Pour stop orders
    filled_quantity: float = Field(default=0.0)
    average_fill_price: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, index=True)
    updated_at: datetime = Field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Métadonnées
    strategy_id: Optional[str] = Field(default=None, index=True)
    signal_reason: Optional[str] = None
    signal_confidence: Optional[float] = None
    
    # Relation avec Position
    position_id: Optional[str] = Field(default=None, foreign_key="positions.id")
    
    # Frais
    fee: float = Field(default=0.0)
    fee_currency: Optional[str] = None
    
    @field_validator("updated_at", mode="before")
    @classmethod
    def auto_update(cls, v):
        return datetime.now()


class Position(SQLModel, table=True):
    """
    Modèle de position persistée.
    Représente une position ouverte ou fermée.
    """
    __tablename__ = "positions"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    
    # Détails de la position
    symbol: str = Field(index=True)
    side: PositionSide
    status: PositionStatus = Field(default=PositionStatus.OPEN, index=True)
    
    # Tailles et prix
    size: float  # Quantité totale
    entry_price: float  # Prix moyen d'entrée
    current_price: Optional[float] = None  # Dernier prix connu
    exit_price: Optional[float] = None  # Prix moyen de sortie
    
    # Leverage et margin
    leverage: int = Field(default=1)
    margin_used: Optional[float] = None
    
    # PnL
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    total_fees: float = Field(default=0.0)
    
    # Stop Loss / Take Profit
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    
    # Timestamps
    opened_at: datetime = Field(default_factory=datetime.now, index=True)
    updated_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    # Métadonnées
    strategy_id: Optional[str] = Field(default=None, index=True)
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # Relations
    # orders: List[Order] = Relationship(back_populates="position")
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calcule le PnL latent."""
        if self.side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.size
        
        # Appliquer le leverage
        pnl *= self.leverage
        
        # Soustraire les frais
        pnl -= self.total_fees
        
        self.current_price = current_price
        self.unrealized_pnl = pnl
        return pnl
    
    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calcule le PnL en pourcentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == PositionSide.LONG:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        return pnl_pct * self.leverage * 100


class Trade(SQLModel, table=True):
    """
    Enregistrement d'un trade (fill).
    Un ordre peut avoir plusieurs trades (partial fills).
    """
    __tablename__ = "trades"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    
    # Références
    order_id: str = Field(foreign_key="orders.id", index=True)
    position_id: Optional[str] = Field(default=None, foreign_key="positions.id", index=True)
    exchange_trade_id: Optional[str] = None
    
    # Détails
    symbol: str = Field(index=True)
    side: OrderSide
    price: float
    quantity: float
    
    # Timestamps
    executed_at: datetime = Field(default_factory=datetime.now, index=True)
    
    # Frais
    fee: float = Field(default=0.0)
    fee_currency: Optional[str] = None
    
    # Métadonnées
    is_maker: bool = Field(default=False)


class BotState(SQLModel, table=True):
    """
    État global du bot pour persistance entre redémarrages.
    """
    __tablename__ = "bot_state"
    
    id: str = Field(default="main", primary_key=True)
    
    # État général
    is_running: bool = Field(default=False)
    mode: str = Field(default="PAPER")  # LIVE, PAPER, BACKTEST
    
    # Capital
    initial_balance: float = Field(default=10000.0)
    current_balance: float = Field(default=10000.0)
    total_pnl: float = Field(default=0.0)
    max_drawdown_pct: float = Field(default=0.0)
    
    # Statistiques
    total_trades: int = Field(default=0)
    winning_trades: int = Field(default=0)
    losing_trades: int = Field(default=0)
    win_rate: float = Field(default=0.0)
    
    # Timestamps
    started_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    last_candle_at: Optional[datetime] = None
    
    # Configuration active
    active_symbols: str = Field(default="BTC/USDT")  # JSON list or comma-separated
    active_timeframe: str = Field(default="1m")
    
    # Métriques ML
    ml_model_version: Optional[str] = None
    ml_last_prediction_at: Optional[datetime] = None
    ml_avg_confidence: float = Field(default=0.0)
    
    def update_win_rate(self) -> None:
        """Recalcule le win rate."""
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100


class EquitySnapshot(SQLModel, table=True):
    """
    Snapshot d'equity pour la courbe de performance.
    """
    __tablename__ = "equity_snapshots"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    equity: float
    balance: float
    unrealized_pnl: float = Field(default=0.0)
    
    # Métadonnées optionnelles
    drawdown_pct: float = Field(default=0.0)
    position_count: int = Field(default=0)
