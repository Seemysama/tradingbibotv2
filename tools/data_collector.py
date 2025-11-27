import argparse
import os
from pathlib import Path
from typing import List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv


def build_exchange() -> ccxt.binance:
    """Create a Binance exchange client with optional API credentials."""
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    params = {"enableRateLimit": True}
    if api_key and api_secret:
        params.update({"apiKey": api_key, "secret": api_secret})

    return ccxt.binance(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from Binance.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1m", help="CCXT timeframe, e.g. 1m, 5m, 1h")
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="History depth in months when --since is not provided (default: 6).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start time in ISO format (UTC). Example: 2024-01-01T00:00:00",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="End time in ISO format (UTC). Defaults to now.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Candles per request (Binance max is 1000).",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=None,
        help="Stop after this many candles (useful to cap downloads).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output file path. Defaults to data/historical/<symbol>_<timeframe>.<ext>",
    )
    return parser.parse_args()


def compute_time_range(args: argparse.Namespace, timeframe_ms: int) -> tuple[int, Optional[int]]:
    """Return (since_ms, until_ms) for the requested history window."""
    if args.since:
        since_ms = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)
    else:
        months = max(args.months, 1)
        since_ms = int((pd.Timestamp.utcnow() - pd.DateOffset(months=months)).timestamp() * 1000)

    until_ms = None
    if args.until:
        until_ms = int(pd.Timestamp(args.until, tz="UTC").timestamp() * 1000)

    # Ensure at least one candle span to avoid tight loops
    if until_ms is not None and until_ms <= since_ms:
        until_ms = since_ms + timeframe_ms

    return since_ms, until_ms


def fetch_ohlcv_range(exchange: ccxt.binance, args: argparse.Namespace) -> pd.DataFrame:
    tf_seconds = exchange.parse_timeframe(args.timeframe)
    timeframe_ms = int(tf_seconds * 1000)
    since_ms, until_ms = compute_time_range(args, timeframe_ms)

    all_candles: List[List[float]] = []
    cursor = since_ms

    while True:
        if until_ms and cursor >= until_ms:
            break

        candles = exchange.fetch_ohlcv(
            args.symbol,
            timeframe=args.timeframe,
            since=cursor,
            limit=args.batch_size,
        )

        if not candles:
            break

        filtered = [c for c in candles if c[0] >= cursor and (until_ms is None or c[0] < until_ms)]
        if not filtered:
            break

        all_candles.extend(filtered)
        cursor = filtered[-1][0] + timeframe_ms

        if args.max_candles and len(all_candles) >= args.max_candles:
            all_candles = all_candles[: args.max_candles]
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def save_dataframe(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    exchange = build_exchange()

    print(
        f"Fetching {args.symbol} {args.timeframe} data "
        f"(months={args.months}, since={args.since}, until={args.until}, max_candles={args.max_candles})"
    )
    df = fetch_ohlcv_range(exchange, args)

    symbol_slug = args.symbol.replace("/", "_")
    if args.dest:
        output_path = args.dest
    else:
        ext = "parquet" if args.format == "parquet" else "csv"
        output_path = Path("data") / "historical" / f"{symbol_slug}_{args.timeframe}.{ext}"

    save_dataframe(df, output_path, args.format)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
