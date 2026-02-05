"""
Notification System - Alerts for critical trading events.
Supports Telegram and Discord webhooks.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import aiohttp

log = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    TRADE = "trade"
    RISK = "risk"


@dataclass
class Alert:
    """Alert message structure."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    data: dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class TelegramChannel(NotificationChannel):
    """Telegram bot notification channel."""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)
        
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
            
        emoji = self._get_emoji(alert.level)
        text = f"{emoji} *{alert.title}*\n\n{alert.message}"
        
        if alert.data:
            text += "\n\n```\n"
            for k, v in alert.data.items():
                text += f"{k}: {v}\n"
            text += "```"
            
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, json=payload) as resp:
                    if resp.status == 200:
                        log.debug("Telegram notification sent")
                        return True
                    else:
                        log.error(f"Telegram API error: {resp.status}")
                        return False
        except Exception as e:
            log.error(f"Failed to send Telegram notification: {e}")
            return False
            
    @staticmethod
    def _get_emoji(level: AlertLevel) -> str:
        return {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.TRADE: "💰",
            AlertLevel.RISK: "🛑"
        }.get(level, "📢")


class DiscordChannel(NotificationChannel):
    """Discord webhook notification channel."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def is_configured(self) -> bool:
        return bool(self.webhook_url)
        
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
            
        color = self._get_color(alert.level)
        
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color,
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": "Trading Bot Alert"}
        }
        
        if alert.data:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in alert.data.items()
            ]
            
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status in (200, 204):
                        log.debug("Discord notification sent")
                        return True
                    else:
                        log.error(f"Discord webhook error: {resp.status}")
                        return False
        except Exception as e:
            log.error(f"Failed to send Discord notification: {e}")
            return False
            
    @staticmethod
    def _get_color(level: AlertLevel) -> int:
        return {
            AlertLevel.INFO: 0x3498DB,      # Blue
            AlertLevel.WARNING: 0xF39C12,   # Orange
            AlertLevel.CRITICAL: 0xE74C3C,  # Red
            AlertLevel.TRADE: 0x2ECC71,     # Green
            AlertLevel.RISK: 0x9B59B6       # Purple
        }.get(level, 0x95A5A6)


class NotificationManager:
    """
    Centralized notification manager.
    Sends alerts to all configured channels.
    """
    
    _instance: Optional["NotificationManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._channels: list[NotificationChannel] = []
        self._rate_limit: dict[str, datetime] = {}
        self._min_interval = 60  # Minimum seconds between same alerts
        
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        if channel.is_configured():
            self._channels.append(channel)
            log.info(f"Added notification channel: {channel.__class__.__name__}")
            
    def configure_from_env(self):
        """Configure channels from environment variables."""
        from src.infrastructure.secrets import secrets
        
        creds = secrets.get_notification_credentials()
        
        if creds.telegram_bot_token and creds.telegram_chat_id:
            self.add_channel(TelegramChannel(
                creds.telegram_bot_token,
                creds.telegram_chat_id
            ))
            
        if creds.discord_webhook_url:
            self.add_channel(DiscordChannel(creds.discord_webhook_url))
            
    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        data: dict = None,
        dedupe_key: str = None
    ) -> bool:
        """Send alert to all configured channels."""
        
        # Rate limiting / deduplication
        if dedupe_key:
            last_sent = self._rate_limit.get(dedupe_key)
            if last_sent and (datetime.utcnow() - last_sent).seconds < self._min_interval:
                log.debug(f"Rate limited alert: {dedupe_key}")
                return False
            self._rate_limit[dedupe_key] = datetime.utcnow()
            
        alert = Alert(level=level, title=title, message=message, data=data)
        
        if not self._channels:
            log.warning("No notification channels configured")
            return False
            
        results = await asyncio.gather(
            *[channel.send(alert) for channel in self._channels],
            return_exceptions=True
        )
        
        success = any(r is True for r in results)
        return success
        
    # Convenience methods
    async def info(self, title: str, message: str, **kwargs):
        return await self.send_alert(AlertLevel.INFO, title, message, **kwargs)
        
    async def warning(self, title: str, message: str, **kwargs):
        return await self.send_alert(AlertLevel.WARNING, title, message, **kwargs)
        
    async def critical(self, title: str, message: str, **kwargs):
        return await self.send_alert(AlertLevel.CRITICAL, title, message, **kwargs)
        
    async def trade(self, title: str, message: str, **kwargs):
        return await self.send_alert(AlertLevel.TRADE, title, message, **kwargs)
        
    async def risk(self, title: str, message: str, **kwargs):
        return await self.send_alert(AlertLevel.RISK, title, message, **kwargs)


# Singleton instance
notifications = NotificationManager()
