"""
Binance Historical Data Fetcher - Production Grade.
Fetches OHLCV data with rate limiting and saves to Parquet.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import time

import ccxt
import pandas as pd
import numpy as np

from src.config import settings

log = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Fetches historical OHLCV data from Binance.
    Features:
    - Rate limiting (respects API limits)
    - Chunked fetching for large date ranges
    - Automatic Parquet storage
    - Resume capability
    """
    
    # Binance limits
    MAX_CANDLES_PER_REQUEST = 1000
    RATE_LIMIT_DELAY = 0.1  # seconds between requests
    
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    
    def __init__(
        self,
        data_dir: Path = None,
        use_testnet: bool = False
    ):
        self.data_dir = data_dir or Path(settings.DATA_PATH)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        exchange_config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        }
        
        if use_testnet:
            exchange_config["sandbox"] = True
            
        self.exchange = ccxt.binance(exchange_config)
        log.info(f"BinanceDataFetcher initialized (testnet={use_testnet})")
        
    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get Parquet file path for symbol/timeframe."""
        slug = symbol.replace("/", "_")
        return self.data_dir / f"{slug}_{timeframe}.parquet"
        
    def _load_existing_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load existing Parquet data if available."""
        path = self._get_parquet_path(symbol, timeframe)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                log.info(f"Loaded {len(df)} existing candles from {path}")
                return df
            except Exception as e:
                log.warning(f"Failed to load existing data: {e}")
        return None
        
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save DataFrame to Parquet."""
        path = self._get_parquet_path(symbol, timeframe)
        df.to_parquet(path, index=False, compression="snappy")
        log.info(f"Saved {len(df)} candles to {path}")
        
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: datetime = None,
        until: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start datetime (default: 30 days ago)
            until: End datetime (default: now)
            limit: Max candles to fetch (optional)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if timeframe not in self.TIMEFRAME_MS:
            raise ValueError(f"Invalid timeframe: {timeframe}")
            
        # Default date range
        if until is None:
            until = datetime.utcnow()
        if since is None:
            since = until - timedelta(days=30)
            
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        
        log.info(f"Fetching {symbol} {timeframe} from {since} to {until}")
        
        all_candles = []
        current_since = since_ms
        request_count = 0
        
        loop = asyncio.get_event_loop()
        
        while current_since < until_ms:
            try:
                # Fetch chunk
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=self.MAX_CANDLES_PER_REQUEST
                    )
                )
                
                if not ohlcv:
                    break
                    
                all_candles.extend(ohlcv)
                request_count += 1
                
                # Progress
                if request_count % 10 == 0:
                    progress_pct = (current_since - since_ms) / (until_ms - since_ms) * 100
                    log.info(f"Progress: {progress_pct:.1f}%, {len(all_candles)} candles")
                    
                # Move to next chunk
                last_ts = ohlcv[-1][0]
                current_since = last_ts + self.TIMEFRAME_MS[timeframe]
                
                # Check limit
                if limit and len(all_candles) >= limit:
                    all_candles = all_candles[:limit]
                    break
                    
                # Rate limiting
                await asyncio.sleep(self.RATE_LIMIT_DELAY)
                
            except ccxt.RateLimitExceeded:
                log.warning("Rate limit hit, waiting 60s...")
                await asyncio.sleep(60)
            except Exception as e:
                log.error(f"Fetch error: {e}")
                await asyncio.sleep(5)
                
        # Convert to DataFrame
        if not all_candles:
            log.warning("No data fetched")
            return pd.DataFrame()
            
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        log.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df
        
    async def fetch_and_save(
        self,
        symbol: str,
        timeframe: str = "1m",
        days: int = 30,
        update: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data and save to Parquet, with optional update mode.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to fetch
            update: If True, only fetch new data after existing
            
        Returns:
            Complete DataFrame
        """
        existing_df = None
        since = None
        
        if update:
            existing_df = self._load_existing_data(symbol, timeframe)
            if existing_df is not None and len(existing_df) > 0:
                # Start from last timestamp
                last_ts = existing_df["timestamp"].max()
                since = last_ts.to_pydatetime() + timedelta(milliseconds=self.TIMEFRAME_MS[timeframe])
                log.info(f"Updating from {since}")
                
        if since is None:
            since = datetime.utcnow() - timedelta(days=days)
            
        # Fetch new data
        new_df = await self.fetch_ohlcv(symbol, timeframe, since=since)
        
        # Merge with existing
        if existing_df is not None and len(new_df) > 0:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        elif existing_df is not None:
            df = existing_df
        else:
            df = new_df
            
        # Save
        if len(df) > 0:
            self._save_data(df, symbol, timeframe)
            
        return df
        
    async def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        days: int = 30
    ) -> dict:
        """Fetch multiple symbols in parallel."""
        results = {}
        
        for symbol in symbols:
            try:
                df = await self.fetch_and_save(symbol, timeframe, days)
                results[symbol] = df
                log.info(f"✅ {symbol}: {len(df)} candles")
            except Exception as e:
                log.error(f"❌ {symbol}: {e}")
                results[symbol] = pd.DataFrame()
                
        return results


async def main():
    """CLI entry point for data fetching."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch historical data from Binance")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--update", action="store_true", help="Update existing data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    fetcher = BinanceDataFetcher()
    df = await fetcher.fetch_and_save(
        args.symbol,
        args.timeframe,
        days=args.days,
        update=args.update
    )
    
    print(f"\nFetched {len(df)} candles")
    print(df.tail())


if __name__ == "__main__":
    asyncio.run(main())
