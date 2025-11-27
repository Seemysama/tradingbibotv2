"""
Market Feed Abstraction Layer - Production Grade
Supporte LIVE (ccxt.pro WebSockets), PAPER et BACKTEST (Parquet)
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class Candle:
    """Représentation normalisée d'une bougie OHLCV."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_dict(cls, data: Dict, symbol: str = "") -> "Candle":
        """Crée une Candle depuis un dict (parquet ou API)."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            # Unix timestamp en ms
            ts = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
        elif not isinstance(ts, datetime):
            ts = datetime.now()
        
        return cls(
            timestamp=ts,
            symbol=symbol or data.get("symbol", "UNKNOWN"),
            open=float(data.get("open", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("close", 0)),
            volume=float(data.get("volume", 0)),
        )
    
    def to_dict(self) -> Dict:
        """Convertit en dict pour compatibilité avec le code existant."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class FeedConfig:
    """Configuration du feed."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    warmup_candles: int = 500
    replay_speed: float = 0.0  # 0 = max speed, sinon ms entre candles
    time_shift_hours: int = 0  # Décalage temporel pour simulation


class MarketFeed(ABC):
    """
    Classe abstraite pour l'alimentation en données de marché.
    Produit des Candles vers une asyncio.Queue standardisée.
    """
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.queue: asyncio.Queue[Candle] = asyncio.Queue(maxsize=10000)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    async def start(self) -> None:
        """Démarre le feed en background."""
        if self._running:
            log.warning("Feed déjà en cours")
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info(f"Feed démarré: {self.__class__.__name__} pour {self.config.symbol}")
    
    async def stop(self) -> None:
        """Arrête le feed proprement."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Feed arrêté")
    
    @abstractmethod
    async def _run(self) -> None:
        """Implémentation spécifique du feed."""
        pass
    
    async def get_candle(self, timeout: float = 5.0) -> Optional[Candle]:
        """Récupère la prochaine bougie de la queue."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def __aiter__(self) -> AsyncIterator[Candle]:
        return self
    
    async def __anext__(self) -> Candle:
        if not self._running and self.queue.empty():
            raise StopAsyncIteration
        candle = await self.get_candle(timeout=10.0)
        if candle is None:
            raise StopAsyncIteration
        return candle


class ParquetFeed(MarketFeed):
    """
    Feed basé sur des fichiers Parquet.
    Supporte:
    - Warmup (pré-chargement sans émission)
    - Time-shift (décalage temporel)
    - Replay speed control
    """
    
    def __init__(self, config: FeedConfig, data_path: Optional[Path] = None):
        super().__init__(config)
        self.data_path = data_path or settings.DATA_PATH
        self._df: Optional[pd.DataFrame] = None
        self._warmup_buffer: List[Candle] = []
    
    def _find_parquet(self) -> Optional[Path]:
        """Trouve le fichier parquet correspondant au symbole."""
        slug_underscore = self.config.symbol.replace("/", "_")
        slug_compact = self.config.symbol.replace("/", "")
        
        candidates = [
            self.data_path / f"{slug_underscore}_{self.config.timeframe}_2Y.parquet",
            self.data_path / f"{slug_underscore}_{self.config.timeframe}_FULL.parquet",
            self.data_path / f"{slug_underscore}_{self.config.timeframe}.parquet",
            self.data_path / f"{slug_compact}_{self.config.timeframe}_2Y.parquet",
            self.data_path / f"{slug_compact}_{self.config.timeframe}.parquet",
        ]
        
        for path in candidates:
            if path.exists():
                return path
        return None
    
    async def load_data(self) -> bool:
        """Charge les données parquet."""
        path = self._find_parquet()
        if not path:
            log.error(f"Aucun fichier parquet trouvé pour {self.config.symbol}")
            return False
        
        self._df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        log.info(f"Chargé {len(self._df)} bougies depuis {path}")
        return True
    
    def get_warmup_candles(self) -> List[Candle]:
        """Retourne les bougies de warmup (pré-chargées)."""
        return self._warmup_buffer.copy()
    
    async def _run(self) -> None:
        """Boucle principale de lecture parquet."""
        if self._df is None:
            if not await self.load_data():
                self._running = False
                return
        
        df = self._df
        if df is None or df.empty:
            self._running = False
            return
        
        # Warmup: les N premières bougies vont dans le buffer
        warmup_size = min(self.config.warmup_candles, len(df) // 2)
        
        for idx, row in df.head(warmup_size).iterrows():
            candle = Candle.from_dict(row.to_dict(), self.config.symbol)
            self._warmup_buffer.append(candle)
        
        log.info(f"Warmup: {len(self._warmup_buffer)} bougies pré-chargées")
        
        # Émission des bougies restantes
        for idx, row in df.iloc[warmup_size:].iterrows():
            if not self._running:
                break
            
            candle = Candle.from_dict(row.to_dict(), self.config.symbol)
            
            # Application du time-shift optionnel
            if self.config.time_shift_hours:
                from datetime import timedelta
                candle.timestamp = datetime.now() - timedelta(hours=self.config.time_shift_hours)
            
            await self.queue.put(candle)
            
            # Contrôle de vitesse
            if self.config.replay_speed > 0:
                await asyncio.sleep(self.config.replay_speed / 1000)
            else:
                await asyncio.sleep(0)  # Yield to event loop
        
        log.info("ParquetFeed terminé")
        self._running = False


class LiveFeed(MarketFeed):
    """
    Feed en temps réel via ccxt.pro (WebSockets).
    Implémente:
    - Reconnexion automatique
    - Gestion des erreurs réseau
    - Heartbeat monitoring
    """
    
    def __init__(
        self,
        config: FeedConfig,
        exchange_id: str = "binance",
        sandbox: bool = False,
    ):
        super().__init__(config)
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.exchange = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._last_candle_time: Optional[datetime] = None
    
    async def _create_exchange(self):
        """Crée l'instance d'exchange ccxt.pro."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError:
            log.error("ccxt.pro non installé. Installer avec: pip install ccxt")
            raise RuntimeError("ccxt.pro required for LiveFeed")
        
        exchange_class = getattr(ccxtpro, self.exchange_id)
        
        config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # Futures pour trading
            }
        }
        
        # Ajouter les clés API si disponibles
        if settings.BINANCE_API_KEY:
            config["apiKey"] = settings.BINANCE_API_KEY
        if settings.BINANCE_API_SECRET:
            config["secret"] = settings.BINANCE_API_SECRET
        
        self.exchange = exchange_class(config)
        
        if self.sandbox:
            self.exchange.set_sandbox_mode(True)
        
        log.info(f"Exchange {self.exchange_id} initialisé (sandbox={self.sandbox})")
    
    async def _watch_ohlcv(self) -> None:
        """Stream des bougies OHLCV via WebSocket."""
        symbol = self.config.symbol
        timeframe = self.config.timeframe
        
        while self._running:
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)
                
                if ohlcv:
                    # Dernière bougie complète
                    for item in ohlcv:
                        ts, o, h, l, c, v = item
                        candle = Candle(
                            timestamp=datetime.fromtimestamp(ts / 1000),
                            symbol=symbol,
                            open=o,
                            high=h,
                            low=l,
                            close=c,
                            volume=v,
                        )
                        
                        # Éviter les doublons
                        if self._last_candle_time and candle.timestamp <= self._last_candle_time:
                            continue
                        
                        self._last_candle_time = candle.timestamp
                        await self.queue.put(candle)
                
                # Reset reconnect delay on success
                self._reconnect_delay = 1.0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Erreur WebSocket: {e}")
                await self._handle_reconnect()
    
    async def _handle_reconnect(self) -> None:
        """Gestion de la reconnexion avec backoff exponentiel."""
        if not self._running:
            return
        
        log.warning(f"Reconnexion dans {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        
        # Backoff exponentiel
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay
        )
        
        # Recréer l'exchange
        try:
            if self.exchange:
                await self.exchange.close()
            await self._create_exchange()
        except Exception as e:
            log.error(f"Échec reconnexion: {e}")
    
    async def _run(self) -> None:
        """Boucle principale du LiveFeed."""
        try:
            await self._create_exchange()
            await self._watch_ohlcv()
        finally:
            if self.exchange:
                await self.exchange.close()
    
    async def stop(self) -> None:
        """Arrête le feed et ferme les connexions."""
        self._running = False
        if self.exchange:
            try:
                await self.exchange.close()
            except:
                pass
        await super().stop()


class PaperFeed(ParquetFeed):
    """
    Feed Paper Trading: données historiques mais avec timestamps en temps réel.
    Simule un environnement de trading avec des données passées.
    """
    
    def __init__(self, config: FeedConfig, data_path: Optional[Path] = None):
        super().__init__(config, data_path)
        self._start_time: Optional[datetime] = None
    
    async def _run(self) -> None:
        """Boucle principale avec timestamps simulés."""
        if self._df is None:
            if not await self.load_data():
                self._running = False
                return
        
        df = self._df
        if df is None or df.empty:
            self._running = False
            return
        
        self._start_time = datetime.now()
        
        # Warmup
        warmup_size = min(self.config.warmup_candles, len(df) // 2)
        for idx, row in df.head(warmup_size).iterrows():
            candle = Candle.from_dict(row.to_dict(), self.config.symbol)
            self._warmup_buffer.append(candle)
        
        log.info(f"PaperFeed warmup: {len(self._warmup_buffer)} bougies")
        
        # Émission avec timestamps en temps réel
        candle_interval_seconds = self._get_interval_seconds()
        
        for idx, row in df.iloc[warmup_size:].iterrows():
            if not self._running:
                break
            
            candle = Candle.from_dict(row.to_dict(), self.config.symbol)
            # Remplacer le timestamp par l'heure actuelle
            candle.timestamp = datetime.now()
            
            await self.queue.put(candle)
            
            # Simuler l'intervalle du timeframe (ou vitesse accélérée)
            if self.config.replay_speed > 0:
                await asyncio.sleep(self.config.replay_speed / 1000)
            else:
                # Mode accéléré mais pas instantané
                await asyncio.sleep(0.05)
        
        self._running = False
    
    def _get_interval_seconds(self) -> int:
        """Convertit le timeframe en secondes."""
        tf = self.config.timeframe
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        unit = tf[-1]
        value = int(tf[:-1]) if tf[:-1].isdigit() else 1
        return value * multipliers.get(unit, 60)


def create_feed(mode: str, config: FeedConfig) -> MarketFeed:
    """
    Factory function pour créer le feed approprié.
    
    Args:
        mode: "LIVE", "PAPER", ou "BACKTEST"
        config: Configuration du feed
    
    Returns:
        MarketFeed approprié pour le mode
    """
    mode = mode.upper()
    
    if mode == "LIVE":
        log.info("Création LiveFeed (WebSocket temps réel)")
        return LiveFeed(config)
    elif mode == "PAPER":
        log.info("Création PaperFeed (simulation paper trading)")
        return PaperFeed(config)
    elif mode == "BACKTEST":
        log.info("Création ParquetFeed (backtest rapide)")
        # Backtest = vitesse max
        config.replay_speed = 0
        return ParquetFeed(config)
    else:
        log.warning(f"Mode inconnu '{mode}', fallback sur PAPER")
        return PaperFeed(config)
