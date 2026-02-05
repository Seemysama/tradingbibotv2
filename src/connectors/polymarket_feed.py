import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from src.feed import MarketFeed, FeedConfig, Candle
from src.config import settings

log = logging.getLogger(__name__)

class PolymarketFeed(MarketFeed):
    """
    Feed for Polymarket using CLOB API.
    Fetches Order Book data or Last Trades to construct 'candles' (or price points).
    """
    
    def __init__(self, config: FeedConfig, api_key: str = None, secret: str = None, passphrase: str = None, private_key: str = None):
        super().__init__(config)
        self.api_key = api_key or settings.POLYMARKET_API_KEY
        self.secret = secret or settings.POLYMARKET_SECRET
        self.passphrase = passphrase or settings.POLYMARKET_PASSPHRASE
        self.private_key = private_key or settings.POLYMARKET_PRIVATE_KEY
        self.client: Optional[ClobClient] = None
        self.token_id = self.config.symbol  # For Polymarket, symbol is the Token ID (asset_id)
        self._last_price = 0.5
        
    async def _create_client(self):
        """Initializes the CLOB client."""
        try:
            # We assume L1/L2 auth if keys are provided, otherwise public access might be limited
            if self.api_key and self.private_key: 
                 # This is a simplification. Usually one needs to derive creds or use L1 key to sign.
                 # For read-only we might just need host. 
                 # But let's assume standard setup.
                 host = "https://clob.polymarket.com"
                 chain_id = 137 # Polygon
                 
                 creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.secret,
                    api_passphrase=self.passphrase,
                )
                 
                 self.client = ClobClient(
                     host=host,
                     key=self.private_key,
                     chain_id=chain_id,
                     creds=creds,
                     signature_type=1 # 0 or 1, check docs. 1 is usually EOA
                 )
            else:
                 # No auth - strictly for public data if supported by library, 
                 # otherwise this might fail for private endpoints.
                 self.client = ClobClient(host="https://clob.polymarket.com")
                 
            log.info(f"Polymarket Client initialized for token {self.token_id}")

        except Exception as e:
            log.error(f"Failed to create Polymarket client: {e}")
            raise

    async def _run(self) -> None:
        if not self.client:
            await self._create_client()
            
        while self._running:
            try:
                # Fetch midpoint or last trade
                # clob-client is synchronous, so we offload to thread or just block slightly
                # For high freq, we'd need websockets (which clob-client supports via another class)
                # Here we poll for simplicity in V1
                
                book = self.client.get_order_book(self.token_id)
                
                # Calculate mid price
                bids = book.bids
                asks = book.asks
                
                if bids and asks:
                    best_bid = float(bids[0].price)
                    best_ask = float(asks[0].price)
                    price = (best_bid + best_ask) / 2
                else:
                    price = self._last_price # Validation hold

                self._last_price = price
                
                candle = Candle(
                    timestamp=datetime.now(),
                    symbol=self.token_id,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0 # TODO: fetch volume
                )
                
                await self.queue.put(candle)
                
                # Sleep between polls
                await asyncio.sleep(1.0) 
                
            except Exception as e:
                log.error(f"Error fetching Polymarket data: {e}")
                await asyncio.sleep(5.0)
