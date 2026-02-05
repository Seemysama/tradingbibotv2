#!/usr/bin/env python3
"""
MASSIVE DATA DOWNLOADER - Hedge Fund Grade
===========================================
Télécharge l'historique COMPLET de Binance Futures depuis 2020.
Stockage optimisé en Parquet partitionné par date.

Usage:
    python tools/download_massive.py --symbols BTC/USDT ETH/USDT SOL/USDT --timeframes 1m 5m
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import sys

import ccxt.async_support as ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("download_massive")

# Constants
DATA_DIR = Path(__file__).parent.parent / "data" / "massive"
BATCH_SIZE = 1000  # Candles per request (Binance limit)
RATE_LIMIT_DELAY = 0.1  # Seconds between requests
START_DATE = datetime(2020, 1, 1)  # Binance Futures launch ~Sept 2019


class MassiveDownloader:
    """
    Téléchargeur massif de données OHLCV avec:
    - Gestion robuste des erreurs et retry
    - Rate limiting intelligent
    - Stockage Parquet partitionné
    - Reprise après interruption
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime = START_DATE,
        end_date: Optional[datetime] = None,
        data_dir: Path = DATA_DIR,
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date or datetime.utcnow()
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exchange: Optional[ccxt.binance] = None

    async def init_exchange(self):
        """Initialise la connexion Binance Futures."""
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        await self.exchange.load_markets()
        log.info(f"Exchange initialisé: {len(self.exchange.markets)} marchés disponibles")

    async def close_exchange(self):
        """Ferme proprement la connexion."""
        if self.exchange:
            await self.exchange.close()

    def _get_partition_path(self, symbol: str, timeframe: str, date: datetime) -> Path:
        """Retourne le chemin du fichier Parquet partitionné."""
        symbol_clean = symbol.replace("/", "_")
        year_month = date.strftime("%Y-%m")
        partition_dir = self.data_dir / symbol_clean / timeframe / year_month
        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir / f"{date.strftime('%Y-%m-%d')}.parquet"

    def _get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Trouve le dernier timestamp téléchargé pour reprise."""
        symbol_clean = symbol.replace("/", "_")
        symbol_dir = self.data_dir / symbol_clean / timeframe
        
        if not symbol_dir.exists():
            return None

        # Trouver le fichier le plus récent
        parquet_files = sorted(symbol_dir.rglob("*.parquet"))
        if not parquet_files:
            return None

        # Lire le dernier fichier et obtenir le max timestamp
        try:
            df = pd.read_parquet(parquet_files[-1])
            if df.empty:
                return None
            last_ts = pd.to_datetime(df["timestamp"].max())
            log.info(f"Reprise pour {symbol} {timeframe} depuis {last_ts}")
            return last_ts.to_pydatetime()
        except Exception as e:
            log.warning(f"Erreur lecture dernier fichier: {e}")
            return None

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convertit un timeframe en millisecondes."""
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60_000)

    async def _fetch_ohlcv_batch(
        self, symbol: str, timeframe: str, since: int, limit: int = BATCH_SIZE
    ) -> List:
        """Fetch un batch de candles avec retry."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return ohlcv
            except ccxt.RateLimitExceeded:
                wait_time = 2 ** attempt
                log.warning(f"Rate limit atteint, attente {wait_time}s...")
                await asyncio.sleep(wait_time)
            except ccxt.NetworkError as e:
                log.warning(f"Erreur réseau (attempt {attempt+1}): {e}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                log.error(f"Erreur inattendue: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2)
        return []

    async def download_symbol_timeframe(self, symbol: str, timeframe: str):
        """Télécharge toutes les données pour un symbole et timeframe."""
        log.info(f"📥 Démarrage téléchargement: {symbol} {timeframe}")

        # Vérifier point de reprise
        last_ts = self._get_last_timestamp(symbol, timeframe)
        start_dt = last_ts + timedelta(minutes=1) if last_ts else self.start_date
        
        if start_dt >= self.end_date:
            log.info(f"✅ {symbol} {timeframe} déjà à jour")
            return

        # Calcul du nombre total de candles attendues
        tf_ms = self._timeframe_to_ms(timeframe)
        total_candles = int((self.end_date - start_dt).total_seconds() * 1000 / tf_ms)
        
        log.info(f"📊 Téléchargement de ~{total_candles:,} candles depuis {start_dt}")

        # Buffer pour accumulation avant écriture
        current_date = None
        day_buffer = []
        total_downloaded = 0

        since_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(self.end_date.timestamp() * 1000)

        pbar = tqdm(total=total_candles, desc=f"{symbol} {timeframe}", unit="candles")

        while since_ms < end_ms:
            ohlcv = await self._fetch_ohlcv_batch(symbol, timeframe, since_ms)
            
            if not ohlcv:
                break

            for candle in ohlcv:
                ts = datetime.utcfromtimestamp(candle[0] / 1000)
                candle_date = ts.date()

                # Nouveau jour -> flush le buffer précédent
                if current_date and candle_date != current_date and day_buffer:
                    self._write_day_partition(symbol, timeframe, current_date, day_buffer)
                    day_buffer = []

                current_date = candle_date
                day_buffer.append({
                    "timestamp": ts,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                })

            total_downloaded += len(ohlcv)
            pbar.update(len(ohlcv))

            # Avancer le curseur
            since_ms = ohlcv[-1][0] + tf_ms

        # Flush final
        if day_buffer:
            self._write_day_partition(symbol, timeframe, current_date, day_buffer)

        pbar.close()
        log.info(f"✅ {symbol} {timeframe}: {total_downloaded:,} candles téléchargées")

    def _write_day_partition(
        self, symbol: str, timeframe: str, date, data: List[dict]
    ):
        """Écrit les données d'un jour en Parquet."""
        if not data:
            return

        df = pd.DataFrame(data)
        
        # Schema optimisé
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["open"] = df["open"].astype("float32")
        df["high"] = df["high"].astype("float32")
        df["low"] = df["low"].astype("float32")
        df["close"] = df["close"].astype("float32")
        df["volume"] = df["volume"].astype("float64")

        # Dédupliquer et trier
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Chemin de partition
        filepath = self._get_partition_path(symbol, timeframe, datetime.combine(date, datetime.min.time()))

        # Écriture Parquet avec compression
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            filepath,
            compression="snappy",
            use_dictionary=True,
        )

    async def run(self):
        """Exécute le téléchargement complet."""
        await self.init_exchange()

        try:
            for symbol in self.symbols:
                # Vérifier que le symbole existe
                if symbol not in self.exchange.markets:
                    log.warning(f"⚠️ Symbole {symbol} non trouvé, skip")
                    continue

                for timeframe in self.timeframes:
                    await self.download_symbol_timeframe(symbol, timeframe)

        finally:
            await self.close_exchange()

        log.info("🎉 Téléchargement massif terminé!")
        self._print_summary()

    def _print_summary(self):
        """Affiche un résumé des données téléchargées."""
        log.info("\n" + "=" * 60)
        log.info("📊 RÉSUMÉ DES DONNÉES")
        log.info("=" * 60)

        for symbol in self.symbols:
            symbol_clean = symbol.replace("/", "_")
            symbol_dir = self.data_dir / symbol_clean

            if not symbol_dir.exists():
                continue

            for timeframe in self.timeframes:
                tf_dir = symbol_dir / timeframe
                if not tf_dir.exists():
                    continue

                parquet_files = list(tf_dir.rglob("*.parquet"))
                total_size = sum(f.stat().st_size for f in parquet_files)
                
                # Compter les lignes
                total_rows = 0
                for f in parquet_files:
                    try:
                        total_rows += pq.read_metadata(f).num_rows
                    except:
                        pass

                log.info(
                    f"  {symbol} {timeframe}: {total_rows:,} candles, "
                    f"{len(parquet_files)} fichiers, {total_size / 1024 / 1024:.1f} MB"
                )


def consolidate_to_single_parquet(symbol: str, timeframe: str, data_dir: Path = DATA_DIR) -> Path:
    """
    Consolide tous les fichiers partitionnés en un seul Parquet pour l'entraînement.
    """
    symbol_clean = symbol.replace("/", "_")
    tf_dir = data_dir / symbol_clean / timeframe
    
    if not tf_dir.exists():
        raise FileNotFoundError(f"Pas de données pour {symbol} {timeframe}")

    parquet_files = sorted(tf_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun fichier Parquet trouvé")

    log.info(f"Consolidation de {len(parquet_files)} fichiers pour {symbol} {timeframe}")

    dfs = []
    for f in tqdm(parquet_files, desc="Lecture"):
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    output_path = data_dir / f"{symbol_clean}_{timeframe}_FULL.parquet"
    df.to_parquet(output_path, compression="snappy", index=False)

    log.info(f"✅ Consolidé: {output_path} ({len(df):,} lignes)")
    return output_path


async def main():
    parser = argparse.ArgumentParser(description="Téléchargement massif Binance Futures")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Symboles à télécharger"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m"],
        help="Timeframes à télécharger"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Date de début (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolider en fichiers uniques après téléchargement"
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    downloader = MassiveDownloader(
        symbols=args.symbols,
        timeframes=args.timeframes,
        start_date=start_date,
    )

    await downloader.run()

    if args.consolidate:
        log.info("\n📦 Consolidation des fichiers...")
        for symbol in args.symbols:
            for tf in args.timeframes:
                try:
                    consolidate_to_single_parquet(symbol, tf)
                except Exception as e:
                    log.error(f"Erreur consolidation {symbol} {tf}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
