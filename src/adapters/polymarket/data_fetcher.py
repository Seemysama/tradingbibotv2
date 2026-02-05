"""
Polymarket Data Fetcher - Fetches prediction market data.
Uses CLOB API and Gamma API for historical probabilities.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import aiohttp
import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


class PolymarketDataFetcher:
    """
    Fetches market data from Polymarket.
    
    Data sources:
    - CLOB API: Order book, trades, markets
    - Gamma API: Historical odds, events
    """
    
    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(settings.DATA_PATH) / "polymarket"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"PolymarketDataFetcher initialized, data_dir={self.data_dir}")
        
    def _get_parquet_path(self, market_id: str, data_type: str = "prices") -> Path:
        """Get Parquet file path for market."""
        safe_id = market_id[:16].replace("/", "_").replace(":", "_")
        return self.data_dir / f"{safe_id}_{data_type}.parquet"
        
    async def _api_get(self, url: str, params: dict = None) -> Optional[Any]:
        """Make async GET request."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        log.warning(f"API error {resp.status}: {url}")
                        return None
        except Exception as e:
            log.error(f"Request failed: {e}")
            return None
            
    async def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active_only: bool = True
    ) -> pd.DataFrame:
        """
        Fetch list of available markets.
        
        Returns:
            DataFrame with market info (id, question, outcomes, etc.)
        """
        url = f"{self.CLOB_BASE_URL}/markets"
        params = {"limit": limit, "offset": offset}
        
        data = await self._api_get(url, params)
        
        if not data:
            return pd.DataFrame()
        
        # API returns a list directly or a dict with data key
        market_list = data if isinstance(data, list) else data.get("data", data)
        if not isinstance(market_list, list):
            market_list = [market_list] if market_list else []
            
        markets = []
        for m in market_list:
            if not isinstance(m, dict):
                continue
            markets.append({
                "condition_id": m.get("condition_id", ""),
                "question_id": m.get("question_id", ""),
                "question": m.get("question", ""),
                "description": m.get("description", "")[:200] if m.get("description") else "",
                "market_slug": m.get("market_slug", ""),
                "end_date_iso": m.get("end_date_iso", ""),
                "active": m.get("active", False),
                "closed": m.get("closed", False),
                "accepting_orders": m.get("accepting_orders", False),
                "minimum_order_size": m.get("minimum_order_size", 0),
            })
            
        df = pd.DataFrame(markets)
        
        if active_only and len(df) > 0 and "active" in df.columns:
            df = df[df["active"] == True]
            
        log.info(f"Fetched {len(df)} markets")
        return df
        
    async def fetch_order_book(self, token_id: str) -> Dict:
        """Fetch current order book for a token."""
        url = f"{self.CLOB_BASE_URL}/book"
        params = {"token_id": token_id}
        
        data = await self._api_get(url, params)
        
        if not data:
            return {"bids": [], "asks": []}
            
        return {
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
            "timestamp": datetime.utcnow()
        }
        
    async def fetch_midpoint_price(self, token_id: str) -> Optional[float]:
        """Get midpoint price from order book."""
        book = await self.fetch_order_book(token_id)
        
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        
        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            return (best_bid + best_ask) / 2
            
        return None
        
    async def fetch_price_history(
        self,
        token_id: str,
        interval: str = "1h",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a token.
        
        Note: Polymarket CLOB has limited historical data.
        For longer history, use Gamma API or build from trades.
        """
        url = f"{self.CLOB_BASE_URL}/prices-history"
        params = {
            "market": token_id,
            "interval": interval,
            "limit": limit
        }
        
        data = await self._api_get(url, params)
        
        if not data or "history" not in data:
            log.warning(f"No price history for {token_id}")
            return pd.DataFrame()
            
        history = data["history"]
        
        df = pd.DataFrame(history)
        
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s")
        if "p" in df.columns:
            df["price"] = df["p"].astype(float)
            
        df = df[["timestamp", "price"]].dropna()
        
        log.info(f"Fetched {len(df)} price points for {token_id}")
        return df
        
    async def fetch_recent_trades(
        self,
        token_id: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """Fetch recent trades for a market."""
        url = f"{self.CLOB_BASE_URL}/trades"
        params = {"asset_id": token_id, "limit": limit}
        
        data = await self._api_get(url, params)
        
        if not data:
            return pd.DataFrame()
            
        trades = []
        for t in data:
            trades.append({
                "id": t.get("id"),
                "timestamp": t.get("match_time"),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", 0)),
                "side": t.get("side"),
            })
            
        df = pd.DataFrame(trades)
        
        if len(df) > 0 and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            
        log.info(f"Fetched {len(df)} trades for {token_id}")
        return df
        
    async def build_ohlcv_from_trades(
        self,
        token_id: str,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Build OHLCV candles from trade data.
        Useful when direct OHLCV API is unavailable.
        """
        trades_df = await self.fetch_recent_trades(token_id, limit=1000)
        
        if trades_df.empty:
            return pd.DataFrame()
            
        # Resample to OHLCV
        trades_df.set_index("timestamp", inplace=True)
        
        ohlcv = trades_df["price"].resample(timeframe).ohlc()
        ohlcv["volume"] = trades_df["size"].resample(timeframe).sum()
        
        ohlcv = ohlcv.dropna().reset_index()
        ohlcv.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        return ohlcv
        
    async def snapshot_all_markets(self) -> pd.DataFrame:
        """
        Take a snapshot of all active market prices.
        Useful for screening and opportunity detection.
        """
        markets = await self.fetch_markets(limit=200, active_only=True)
        
        if markets.empty:
            return pd.DataFrame()
            
        snapshots = []
        
        for _, market in markets.iterrows():
            condition_id = market["condition_id"]
            
            # Fetch order book midpoint
            # Note: Need token_id, not condition_id for CLOB
            # This is simplified - real implementation needs token mapping
            
            snapshots.append({
                "condition_id": condition_id,
                "question": market["question"],
                "timestamp": datetime.utcnow(),
            })
            
        return pd.DataFrame(snapshots)
        
    async def save_market_data(
        self,
        token_id: str,
        df: pd.DataFrame,
        data_type: str = "prices"
    ):
        """Save market data to Parquet."""
        if df.empty:
            return
            
        path = self._get_parquet_path(token_id, data_type)
        df.to_parquet(path, index=False, compression="snappy")
        log.info(f"Saved {len(df)} rows to {path}")


async def main():
    """CLI entry point for data fetching."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Fetch Polymarket data")
    parser.add_argument("--markets", action="store_true", help="List active markets")
    parser.add_argument("--token-id", help="Token ID for price history")
    
    args = parser.parse_args()
    
    fetcher = PolymarketDataFetcher()
    
    if args.markets:
        df = await fetcher.fetch_markets(limit=50)
        print(df[["question", "active"]].to_string())
    elif args.token_id:
        df = await fetcher.fetch_price_history(args.token_id)
        print(df.tail(20))
    else:
        # Default: list markets
        df = await fetcher.fetch_markets(limit=10)
        print(df[["question", "active"]].head(10))


if __name__ == "__main__":
    asyncio.run(main())
