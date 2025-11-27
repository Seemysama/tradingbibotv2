from pathlib import Path
from typing import List, Literal, Optional

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def detect_device() -> str:
    """Détecte le meilleur device dispo (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mode de fonctionnement: LIVE | PAPER | BACKTEST
    MODE: Literal["LIVE", "PAPER", "BACKTEST"] = "PAPER"
    ENV: str = "DEV"  # DEV | STAGING | PROD (pour logging/monitoring)
    LOGGER_LEVEL: str = "INFO"
    
    # API Exchange
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    EXCHANGE: str = "binance"  # Exchange à utiliser
    USE_SANDBOX: bool = True  # Mode sandbox pour tests
    
    # Chemins données
    DATA_PATH: Path = Path("data/historical")
    DB_PATH: Path = Path("data/trading_state.db")
    MODELS_PATH: Path = Path("models")
    
    # Trading
    TIMEFRAME: str = "1m"
    PAIRS: List[str] = ["BTC/USDT"]
    INITIAL_BALANCE: float = 10000.0  # Capital initial paper trading
    
    # Feed configuration
    WARMUP_CANDLES: int = 500  # Bougies de warmup avant trading
    REPLAY_SPEED_MS: float = 50.0  # Vitesse de replay (ms entre bougies, 0=max)
    
    # ML Configuration
    ML_ENABLED: bool = True
    ML_CONFIDENCE_THRESHOLD: float = 0.65
    ML_MODEL_TYPE: str = "lstm_attention"  # lstm | lstm_attention | gru
    DEBUG_SIGNALS: bool = False
    ML_MODEL_PATH: Path = Path("models/lstm_v1.pth")
    ML_SEQ_LENGTH: int = 60

    # Risk Management
    MAX_DRAWDOWN: float = 0.1  # 10%
    MAX_EXPOSURE: float = 0.5  # 50% du capital
    LEVERAGE: int = 3
    COOLDOWN_SEC: int = 30
    SPREAD_LIMIT: float = 0.002  # 0.2%
    MIN_NOTIONAL: float = 5.0  # USDT minimal pour un ordre
    POSITION_SIZE_PCT: float = 0.05  # 5% du capital par trade
    STOP_LOSS_PCT: float = 0.01  # 1%
    TAKE_PROFIT_PCT: float = 0.02  # 2%

    # Indicateurs techniques
    SMA_TREND_WINDOW: int = 200
    SMA_FAST_WINDOW: int = 50
    BB_WINDOW: int = 20
    BB_STD: float = 2.0
    RSI_WINDOW: int = 14
    ATR_WINDOW: int = 14
    ADX_WINDOW: int = 14
    
    # Nouveaux paramètres - Stratégie Hybride
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    ADX_THRESHOLD: float = 25.0  # Seuil pour détecter un marché directionnel
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    ATR_MULTIPLIER_SL: float = 1.5  # Stop Loss = Entry ± (ATR * multiplier)
    RISK_REWARD_RATIO: float = 2.0  # TP = SL * ratio (ex: risque 1, gain 2)

    # QuestDB (optionnel)
    QUESTDB_ENABLED: bool = False
    QUESTDB_HOST: str = "localhost"
    QUESTDB_PORT: int = 9009
    
    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    DEVICE: str = detect_device()

    @field_validator("MODE")
    @classmethod
    def _upper_mode(cls, v: str) -> str:
        return v.upper()

    @field_validator("ENV")
    @classmethod
    def _upper_env(cls, v: str) -> str:
        return v.upper()

    @field_validator("BINANCE_API_KEY", "BINANCE_API_SECRET")
    @classmethod
    def _require_keys_for_live(cls, v: Optional[str], info):
        values = info.data
        if values.get("MODE", "PAPER").upper() == "LIVE" and not v:
            raise ValueError(f"Clé requise en mode LIVE: {info.field_name}")
        return v
    
    @property
    def is_live(self) -> bool:
        """Retourne True si en mode LIVE."""
        return self.MODE == "LIVE"
    
    @property
    def is_paper(self) -> bool:
        """Retourne True si en mode PAPER."""
        return self.MODE == "PAPER"
    
    @property
    def is_backtest(self) -> bool:
        """Retourne True si en mode BACKTEST."""
        return self.MODE == "BACKTEST"


settings = Settings()  # Chargement global
