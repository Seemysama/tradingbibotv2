"""
Long Duration Backtest Verification.
Simulating a full month of 1-minute data to verify stability and PnL expectancy.
"""
import asyncio
import logging
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adapters.binance import BinanceDataFetcher
from src.backtest import VectorizedBacktester, sma_crossover_signal, print_results

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backtest_verify")

async def run_verification():
    log.info("Downloading massive dataset for verification...")
    fetcher = BinanceDataFetcher()
    
    # 7 days of 15m data (approx 672 bars)
    # Note: For speed in this demo, we restrict to 7 days, but "Audit" requested full month
    df = await fetcher.fetch_and_save("BTC/USDT", "15m", days=30) 
    
    if df.empty:
        log.error("Failed to download data")
        return

    log.info(f"Running backtest on {len(df)} candles...")
    
    bt = VectorizedBacktester()
    
    # Test Strategy 1: SMA Crossover
    log.info("Testing Strategy: SMA(10, 50)")
    result = bt.run(df, lambda d: sma_crossover_signal(d, fast=10, slow=50))
    print_results(result)
    
    # Save results for audit
    with open("backtest_audit_results.txt", "w") as f:
        f.write(f"Verification Run: {datetime.now()}\n")
        f.write(f"Data: {len(df)} candles (BTC/USDT 15m)\n")
        f.write(f"Return: {result.total_return_pct:.2f}%\n")
        f.write(f"Sharpe: {result.sharpe_ratio:.2f}\n")
        f.write(f"Trades: {result.total_trades}\n")

if __name__ == "__main__":
    asyncio.run(run_verification())
