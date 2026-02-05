"""
Async Rotating Logger - Production-grade logging for trading systems.
Features:
- Async file handlers (non-blocking I/O)
- Rotating logs (size + time based)
- Structured JSON logging option
- Separate logs per component
"""
import asyncio
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        if hasattr(record, "extra_data"):
            log_obj["data"] = record.extra_data
            
        return json.dumps(log_obj, default=str)


class TradingLogger:
    """
    Centralized logging configuration for the trading engine.
    """
    
    _configured = False
    
    def __init__(
        self,
        log_dir: Path = Path("logs"),
        level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        json_format: bool = False
    ):
        self.log_dir = log_dir
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.json_format = json_format
        
    def setup(self):
        """Configure logging for the entire application."""
        if TradingLogger._configured:
            return
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self._get_formatter(colored=True))
        root_logger.addHandler(console_handler)
        
        # Main rotating file handler
        main_handler = RotatingFileHandler(
            self.log_dir / "trading.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        main_handler.setLevel(self.level)
        main_handler.setFormatter(self._get_formatter())
        root_logger.addHandler(main_handler)
        
        # Error-only handler
        error_handler = RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self._get_formatter())
        root_logger.addHandler(error_handler)
        
        # Trade-specific handler
        self._setup_component_logger("trades", "trades.log")
        self._setup_component_logger("orders", "orders.log")
        self._setup_component_logger("risk", "risk.log")
        
        TradingLogger._configured = True
        logging.info("Logging system initialized")
        
    def _get_formatter(self, colored: bool = False) -> logging.Formatter:
        """Get appropriate formatter."""
        if self.json_format:
            return JSONFormatter()
            
        if colored:
            # ANSI colors for terminal
            fmt = "\033[90m%(asctime)s\033[0m [\033[1m%(levelname)s\033[0m] \033[36m%(name)s\033[0m: %(message)s"
        else:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        
    def _setup_component_logger(self, name: str, filename: str):
        """Setup a dedicated logger for a component."""
        logger = logging.getLogger(name)
        logger.setLevel(self.level)
        logger.propagate = True  # Also log to root
        
        handler = RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        handler.setLevel(self.level)
        handler.setFormatter(self._get_formatter())
        logger.addHandler(handler)


class AsyncLogBuffer:
    """
    Async buffer for high-frequency log events.
    Batches writes to reduce I/O overhead.
    """
    
    def __init__(self, filepath: Path, flush_interval: float = 1.0, max_buffer: int = 100):
        self.filepath = filepath
        self.flush_interval = flush_interval
        self.max_buffer = max_buffer
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the async flush loop."""
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        
    async def stop(self):
        """Stop and flush remaining buffer."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush()
        
    async def log(self, message: str):
        """Add message to buffer."""
        async with self._lock:
            timestamp = datetime.utcnow().isoformat()
            self._buffer.append(f"{timestamp} {message}\n")
            
            if len(self._buffer) >= self.max_buffer:
                await self._flush_unsafe()
                
    async def _flush_loop(self):
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()
            
    async def _flush(self):
        """Flush buffer to disk with lock."""
        async with self._lock:
            await self._flush_unsafe()
            
    async def _flush_unsafe(self):
        """Flush without acquiring lock."""
        if not self._buffer:
            return
            
        # Non-blocking write using executor
        loop = asyncio.get_event_loop()
        content = "".join(self._buffer)
        self._buffer.clear()
        
        await loop.run_in_executor(
            None,
            lambda: self.filepath.open("a").write(content)
        )


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    json_format: bool = False
):
    """Convenience function to setup logging."""
    logger = TradingLogger(
        log_dir=Path(log_dir),
        level=level,
        json_format=json_format
    )
    logger.setup()
    return logger
