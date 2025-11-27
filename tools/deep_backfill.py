import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# Essaye d'utiliser pandas_ta si disponible (sinon fallback manuel)
try:
    import pandas_ta as ta  # type: ignore
except Exception:  # pragma: no cover - fallback
    ta = None


SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1m"
MONTHS = 24
DEST_DIR = Path("data/historical")
DEST_DIR.mkdir(parents=True, exist_ok=True)


def build_exchange() -> ccxt.binance:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    params: Dict[str, object] = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if api_key and api_secret:
        params.update({"apiKey": api_key, "secret": api_secret})
    return ccxt.binance(params)


def time_bounds(months: int, timeframe_ms: int) -> Tuple[int, Optional[int]]:
    since_ms = int((pd.Timestamp.utcnow() - pd.DateOffset(months=months)).timestamp() * 1000)
    return since_ms, None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute RSI, ATR, Bollinger %B, ADX, SMA50/200."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if ta:
        df["RSI_14"] = ta.rsi(close, length=14)
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None:
            df["BBP_20_2.0"] = (close - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])
        else:
            df["BBP_20_2.0"] = np.nan
        adx = ta.adx(high, low, close, length=14)
        df["ADX_14"] = adx["ADX_14"] if adx is not None else np.nan
        df["ATR_14"] = ta.atr(high, low, close, length=14)
    else:
        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll_up = gain.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down
        df["RSI_14"] = 100 - (100 / (1 + rs))
        # ATR 14
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        # Bollinger %B
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        df["BBP_20_2.0"] = (close - lower) / (upper - lower)
        # ADX 14
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["ADX_14"] = (dx * 100).ewm(alpha=1 / 14, adjust=False).mean()

    df["SMA_50"] = close.rolling(50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(200, min_periods=200).mean()
    return df


def fetch_symbol(ex: ccxt.binance, symbol: str, months: int) -> pd.DataFrame:
    tf_ms = int(ex.parse_timeframe(TIMEFRAME) * 1000)
    since_ms, until_ms = time_bounds(months, tf_ms)
    cursor = since_ms
    rows: List[List[float]] = []
    pbar = tqdm(desc=f"Download {symbol}", unit="req")
    while True:
        if until_ms and cursor >= until_ms:
            break
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=1000)
        except ccxt.RateLimitExceeded:
            time.sleep(ex.rateLimit / 1000)
            continue
        if not batch:
            break
        batch = [r for r in batch if r[0] >= cursor and (until_ms is None or r[0] < until_ms)]
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        pbar.update(1)
    pbar.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = compute_indicators(df)
    return df


def save_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)


async def main() -> None:
    ex = build_exchange()
    for symbol in SYMBOLS:
        df = fetch_symbol(ex, symbol, MONTHS)
        if df.empty:
            print(f"❌ Aucun data pour {symbol}")
            continue
        slug = symbol.replace("/", "_")
        dest = DEST_DIR / f"{slug}_{TIMEFRAME}_2Y.parquet"
        save_parquet(df, dest)
        print(f"✅ {symbol}: {len(df)} bougies sauvegardées dans {dest}")


if __name__ == "__main__":
    try:
        import uvloop

        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
