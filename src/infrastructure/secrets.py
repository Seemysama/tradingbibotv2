"""
Secrets Manager - Secure credential handling.
Never hardcode API keys. All secrets loaded from environment variables.
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinanceCredentials:
    """Binance API credentials."""
    api_key: str
    api_secret: str
    testnet: bool = True


@dataclass(frozen=True)
class PolymarketCredentials:
    """Polymarket CLOB credentials."""
    api_key: str
    api_secret: str
    passphrase: str
    private_key: str  # For L1 signing


@dataclass(frozen=True)
class NotificationCredentials:
    """Notification service credentials."""
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None


class SecretsManager:
    """
    Centralized secrets management.
    Loads all credentials from environment variables only.
    """
    
    _instance: Optional["SecretsManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_dotenv()
        log.info("SecretsManager initialized")
    
    def _load_dotenv(self):
        """Load .env file if exists."""
        env_path = Path(".env")
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                log.debug(f"Loaded .env from {env_path.absolute()}")
            except ImportError:
                log.warning("python-dotenv not installed, using system env only")
    
    @staticmethod
    def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get environment variable with validation."""
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"Required environment variable not set: {key}")
        return value
    
    def get_binance_credentials(self, require: bool = False) -> Optional[BinanceCredentials]:
        """Get Binance API credentials."""
        try:
            api_key = self._get_env("BINANCE_API_KEY", required=require)
            api_secret = self._get_env("BINANCE_API_SECRET", required=require)
            testnet = self._get_env("BINANCE_TESTNET", "true").lower() == "true"
            
            if not api_key or not api_secret:
                return None
                
            return BinanceCredentials(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        except ValueError as e:
            log.error(f"Binance credentials error: {e}")
            raise
    
    def get_polymarket_credentials(self, require: bool = False) -> Optional[PolymarketCredentials]:
        """Get Polymarket CLOB credentials."""
        try:
            api_key = self._get_env("POLYMARKET_API_KEY", required=require)
            api_secret = self._get_env("POLYMARKET_SECRET", required=require)
            passphrase = self._get_env("POLYMARKET_PASSPHRASE", required=require)
            private_key = self._get_env("POLYMARKET_PRIVATE_KEY", required=require)
            
            if not all([api_key, api_secret, passphrase, private_key]):
                return None
                
            return PolymarketCredentials(
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                private_key=private_key
            )
        except ValueError as e:
            log.error(f"Polymarket credentials error: {e}")
            raise
    
    def get_notification_credentials(self) -> NotificationCredentials:
        """Get notification service credentials."""
        return NotificationCredentials(
            telegram_bot_token=self._get_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=self._get_env("TELEGRAM_CHAT_ID"),
            discord_webhook_url=self._get_env("DISCORD_WEBHOOK_URL")
        )
    
    def validate_all(self) -> dict:
        """Validate all credentials and return status."""
        status = {
            "binance": False,
            "polymarket": False,
            "telegram": False,
            "discord": False
        }
        
        if self.get_binance_credentials():
            status["binance"] = True
            
        if self.get_polymarket_credentials():
            status["polymarket"] = True
            
        notif = self.get_notification_credentials()
        if notif.telegram_bot_token and notif.telegram_chat_id:
            status["telegram"] = True
        if notif.discord_webhook_url:
            status["discord"] = True
            
        return status


# Singleton instance
secrets = SecretsManager()
