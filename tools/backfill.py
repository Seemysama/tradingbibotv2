import argparse
import os
import socket
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ccxt
import pandas as pd
from dotenv import load_dotenv


def build_futures_exchange() -> ccxt.binance:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if api_key and api_secret:
        params.update({"apiKey": api_key, "secret": api_secret})
    return ccxt.binance(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Binance Futures vers QuestDB/Parquet.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole Futures, ex: BTC/USDT")
    parser.add_argument("--timeframe", default="1m", help="Timeframe CCXT, ex: 1m,5m,1h")
    parser.add_argument("--months", type=int, default=12, help="Profondeur historique si --since absent")
    parser.add_argument("--since", type=str, default=None, help="Début ISO UTC, ex: 2021-01-01T00:00:00")
    parser.add_argument("--until", type=str, default=None, help="Fin ISO UTC (UTC).")
    parser.add_argument("--batch-size", type=int, default=1000, help="Taille des requêtes (max Binance: 1500/1000).")
    parser.add_argument("--dest", type=Path, default=Path("data/historical/futures.parquet"), help="Chemin Parquet de sortie.")
    parser.add_argument("--questdb-host", type=str, default=None, help="Host QuestDB ILP (ex: localhost).")
    parser.add_argument("--questdb-port", type=int, default=9009, help="Port ILP QuestDB (défaut 9009).")
    parser.add_argument("--questdb-table", type=str, default="futures_ohlcv", help="Nom de table QuestDB.")
    parser.add_argument("--max-candles", type=int, default=None, help="Limite totale de bougies (sécurité).")
    return parser.parse_args()


def time_bounds(args: argparse.Namespace, timeframe_ms: int) -> Tuple[int, Optional[int]]:
    if args.since:
        since_ms = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)
    else:
        since_ms = int((pd.Timestamp.utcnow() - pd.DateOffset(months=max(args.months, 1))).timestamp() * 1000)
    until_ms = None
    if args.until:
        until_ms = int(pd.Timestamp(args.until, tz="UTC").timestamp() * 1000)
        if until_ms <= since_ms:
            until_ms = since_ms + timeframe_ms
    return since_ms, until_ms


def ilp_lines(symbol: str, table: str, rows: Iterable[List]) -> str:
    """Construit les lignes ILP pour QuestDB."""
    lines = []
    sym_tag = symbol.replace("/", "_")
    for ts, o, h, l, c, v in rows:
        lines.append(f"{table},symbol={sym_tag} open={o},high={h},low={l},close={c},volume={v} {int(ts)*1_000_000}")
    return "\n".join(lines) + "\n"


def send_ilp(host: str, port: int, payload: str) -> None:
    with socket.create_connection((host, port)) as sock:
        sock.sendall(payload.encode())


def backfill() -> None:
    args = parse_args()
    ex = build_futures_exchange()
    tf_ms = int(ex.parse_timeframe(args.timeframe) * 1000)
    since_ms, until_ms = time_bounds(args, tf_ms)

    cursor = since_ms
    all_rows: List[List] = []
    print(f"Backfill {args.symbol} {args.timeframe} depuis {pd.to_datetime(cursor, unit='ms', utc=True)}")

    while True:
        if until_ms and cursor >= until_ms:
            break
        try:
            batch = ex.fetch_ohlcv(args.symbol, timeframe=args.timeframe, since=cursor, limit=args.batch_size)
        except ccxt.RateLimitExceeded:
            time.sleep(ex.rateLimit / 1000)
            continue
        if not batch:
            break

        filtered = [row for row in batch if row[0] >= cursor and (until_ms is None or row[0] < until_ms)]
        if not filtered:
            break

        all_rows.extend(filtered)
        cursor = filtered[-1][0] + tf_ms

        if args.max_candles and len(all_rows) >= args.max_candles:
            all_rows = all_rows[: args.max_candles]
            break

    if not all_rows:
        print("Aucune donnée téléchargée.")
        return

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.dest, index=False)
    print(f"{len(df)} bougies sauvegardées dans {args.dest}")

    if args.questdb_host:
        payload = ilp_lines(args.symbol, args.questdb_table, all_rows)
        send_ilp(args.questdb_host, args.questdb_port, payload)
        print(f"Données envoyées à QuestDB {args.questdb_host}:{args.questdb_port} table={args.questdb_table}")


if __name__ == "__main__":
    backfill()
