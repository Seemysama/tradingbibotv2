#!/usr/bin/env python3
"""
MASSIVE DATA INGEST - Industrial Grade
=======================================
Téléchargement multithreadé de Binance Futures avec:
- Top 10 paires crypto
- Timeframes 5m et 15m (moins de bruit que 1m)
- Gestion robuste des erreurs et rate limits
- Nettoyage des gaps et outliers
- Stockage Parquet partitionné

Usage:
    python tools/massive_ingest.py --pairs all --timeframes 5m 15m --start 2020-01-01
"""

from __future__ import annotations

import asyncio
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("massive_ingest")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IngestConfig:
    """Configuration du téléchargement."""
    
    # Top 10 paires par volume
    TOP_PAIRS: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"
    ])
    
    # Timeframes recommandés (5m/15m = moins de bruit que 1m)
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["5m", "15m"])
    
    # Dates
    START_DATE: datetime = datetime(2020, 1, 1)
    END_DATE: Optional[datetime] = None
    
    # API
    BATCH_SIZE: int = 1000
    MAX_RETRIES: int = 5
    RATE_LIMIT_DELAY: float = 0.05  # 50ms entre requêtes
    MAX_CONCURRENT: int = 3  # Requêtes parallèles par paire
    
    # Stockage
    DATA_DIR: Path = Path(__file__).parent.parent / "data" / "futures"
    
    # Nettoyage
    MAX_GAP_MINUTES: int = 60  # Gap max avant interpolation
    OUTLIER_ZSCORE: float = 5.0  # Seuil pour outliers


# ============================================================================
# DATA CLEANER
# ============================================================================

class DataCleaner:
    """Nettoyage robuste des données OHLCV."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
    
    def clean(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Nettoie un DataFrame OHLCV:
        1. Supprime les doublons
        2. Gère les gaps temporels
        3. Supprime les outliers
        4. Valide l'intégrité OHLCV
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # 1. Doublons
        df = df.drop_duplicates(subset=["timestamp"]).copy()
        
        # 2. Tri chronologique
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 3. Conversion timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # 4. Détection des gaps
        df = self._handle_gaps(df, timeframe)
        
        # 5. Outliers sur returns
        df = self._remove_outliers(df)
        
        # 6. Validation OHLCV
        df = self._validate_ohlcv(df)
        
        # 7. Types optimisés
        df = self._optimize_types(df)
        
        cleaned_len = len(df)
        if original_len != cleaned_len:
            log.debug(f"Nettoyage: {original_len} -> {cleaned_len} lignes")
        
        return df
    
    def _handle_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Gère les gaps temporels."""
        tf_minutes = self._timeframe_to_minutes(timeframe)
        expected_delta = timedelta(minutes=tf_minutes)
        
        # Calculer les gaps
        df["time_diff"] = df["timestamp"].diff()
        
        # Marquer les gaps excessifs
        max_gap = timedelta(minutes=self.config.MAX_GAP_MINUTES)
        large_gaps = df["time_diff"] > max_gap
        
        if large_gaps.any():
            n_gaps = large_gaps.sum()
            log.warning(f"Détecté {n_gaps} gaps > {self.config.MAX_GAP_MINUTES} min")
        
        # Petits gaps: forward fill
        small_gaps = (df["time_diff"] > expected_delta) & (df["time_diff"] <= max_gap)
        # On ne fait rien pour les petits gaps, c'est normal en crypto
        
        df = df.drop(columns=["time_diff"])
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les outliers basés sur le z-score des returns."""
        if len(df) < 100:
            return df
        
        # Log-returns
        returns = np.log(df["close"] / df["close"].shift(1))
        
        # Z-score rolling (fenêtre 100)
        rolling_mean = returns.rolling(100, min_periods=20).mean()
        rolling_std = returns.rolling(100, min_periods=20).std()
        zscore = (returns - rolling_mean) / (rolling_std + 1e-8)
        
        # Filtrer
        mask = np.abs(zscore) <= self.config.OUTLIER_ZSCORE
        mask = mask.fillna(True)  # Garder les premières lignes
        
        n_outliers = (~mask).sum()
        if n_outliers > 0:
            log.debug(f"Suppression de {n_outliers} outliers")
        
        return df[mask].reset_index(drop=True)
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la cohérence OHLCV."""
        # High >= max(Open, Close)
        # Low <= min(Open, Close)
        valid = (
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"]) &
            (df["high"] >= df["low"]) &
            (df["volume"] >= 0) &
            (df["close"] > 0)
        )
        
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            log.warning(f"Suppression de {n_invalid} lignes OHLCV invalides")
        
        return df[valid].reset_index(drop=True)
    
    def _optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimise les types pour réduire la mémoire."""
        df["open"] = df["open"].astype("float32")
        df["high"] = df["high"].astype("float32")
        df["low"] = df["low"].astype("float32")
        df["close"] = df["close"].astype("float32")
        df["volume"] = df["volume"].astype("float64")
        return df
    
    @staticmethod
    def _timeframe_to_minutes(tf: str) -> int:
        """Convertit un timeframe en minutes."""
        unit = tf[-1]
        value = int(tf[:-1])
        multipliers = {"m": 1, "h": 60, "d": 1440}
        return value * multipliers.get(unit, 1)


# ============================================================================
# ASYNC DOWNLOADER
# ============================================================================

class AsyncDownloader:
    """Téléchargeur asynchrone avec gestion des erreurs."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.cleaner = DataCleaner(config)
        self._exchange = None
    
    async def init(self):
        """Initialise la connexion exchange."""
        import ccxt.async_support as ccxt
        
        self._exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        await self._exchange.load_markets()
        log.info(f"Exchange initialisé: {len(self._exchange.markets)} marchés")
    
    async def close(self):
        """Ferme la connexion."""
        if self._exchange:
            await self._exchange.close()
    
    async def download_pair_timeframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Télécharge toutes les données pour une paire/timeframe."""
        
        if symbol not in self._exchange.markets:
            log.warning(f"Symbole {symbol} non trouvé")
            return pd.DataFrame()
        
        log.info(f"📥 Début: {symbol} {timeframe} ({start_date.date()} -> {end_date.date()})")
        
        all_data = []
        since_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        tf_ms = self._timeframe_to_ms(timeframe)
        
        total_expected = (end_ms - since_ms) // tf_ms
        downloaded = 0
        
        while since_ms < end_ms:
            try:
                ohlcv = await self._fetch_with_retry(symbol, timeframe, since_ms)
                
                if not ohlcv:
                    break
                
                for candle in ohlcv:
                    if candle[0] >= end_ms:
                        break
                    all_data.append({
                        "timestamp": datetime.utcfromtimestamp(candle[0] / 1000),
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5],
                    })
                
                downloaded += len(ohlcv)
                since_ms = ohlcv[-1][0] + tf_ms
                
                # Progress log tous les 10%
                progress = downloaded / max(total_expected, 1) * 100
                if downloaded % 10000 < self.config.BATCH_SIZE:
                    log.debug(f"{symbol} {timeframe}: {progress:.0f}% ({downloaded:,} candles)")
                
                await asyncio.sleep(self.config.RATE_LIMIT_DELAY)
                
            except Exception as e:
                log.error(f"Erreur fatale {symbol} {timeframe}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = self.cleaner.clean(df, timeframe)
        
        log.info(f"✅ {symbol} {timeframe}: {len(df):,} candles nettoyées")
        return df
    
    async def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int,
    ) -> list:
        """Fetch avec retry exponentiel."""
        import ccxt
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                return await self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=self.config.BATCH_SIZE
                )
            except ccxt.RateLimitExceeded:
                wait = 2 ** attempt
                log.warning(f"Rate limit, attente {wait}s...")
                await asyncio.sleep(wait)
            except ccxt.NetworkError as e:
                wait = 2 ** attempt
                log.warning(f"Erreur réseau (attempt {attempt+1}): {e}")
                await asyncio.sleep(wait)
            except Exception as e:
                log.error(f"Erreur inattendue: {e}")
                if attempt == self.config.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(1)
        
        return []
    
    @staticmethod
    def _timeframe_to_ms(tf: str) -> int:
        """Convertit timeframe en millisecondes."""
        unit = tf[-1]
        value = int(tf[:-1])
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        return value * multipliers.get(unit, 60_000)


# ============================================================================
# PARQUET WRITER
# ============================================================================

class PartitionedWriter:
    """Écrit les données en Parquet partitionné par symbole/année/mois."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def write(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Écrit un DataFrame partitionné."""
        if df.empty:
            return
        
        symbol_clean = symbol.replace("/", "_")
        
        # Grouper par année-mois
        df["year_month"] = df["timestamp"].dt.to_period("M")
        
        for period, group in df.groupby("year_month"):
            year = period.year
            month = period.month
            
            # Chemin de partition
            partition_dir = (
                self.config.DATA_DIR / symbol_clean / timeframe / 
                f"{year}" / f"{month:02d}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = partition_dir / "data.parquet"
            
            # Préparer le DataFrame
            write_df = group.drop(columns=["year_month"]).copy()
            
            # Si fichier existe, merger
            if filepath.exists():
                existing = pd.read_parquet(filepath)
                write_df = pd.concat([existing, write_df], ignore_index=True)
                write_df = write_df.drop_duplicates(subset=["timestamp"])
                write_df = write_df.sort_values("timestamp")
            
            # Écrire
            table = pa.Table.from_pandas(write_df, preserve_index=False)
            pq.write_table(table, filepath, compression="snappy")
        
        log.debug(f"Écrit {len(df)} lignes pour {symbol} {timeframe}")
    
    def consolidate(self, symbol: str, timeframe: str) -> Path:
        """Consolide toutes les partitions en un fichier unique."""
        symbol_clean = symbol.replace("/", "_")
        tf_dir = self.config.DATA_DIR / symbol_clean / timeframe
        
        if not tf_dir.exists():
            raise FileNotFoundError(f"Pas de données pour {symbol} {timeframe}")
        
        parquet_files = sorted(tf_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Aucun fichier Parquet")
        
        log.info(f"Consolidation de {len(parquet_files)} fichiers pour {symbol} {timeframe}")
        
        dfs = []
        for f in parquet_files:
            dfs.append(pd.read_parquet(f))
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        
        output_path = self.config.DATA_DIR / f"{symbol_clean}_{timeframe}_FULL.parquet"
        df.to_parquet(output_path, compression="snappy", index=False)
        
        log.info(f"✅ Consolidé: {output_path} ({len(df):,} lignes)")
        return output_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des données."""
        stats = {}
        
        for symbol_dir in self.config.DATA_DIR.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name.replace("_", "/")
            stats[symbol] = {}
            
            for tf_dir in symbol_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                
                parquet_files = list(tf_dir.rglob("*.parquet"))
                total_rows = sum(
                    pq.read_metadata(f).num_rows 
                    for f in parquet_files
                )
                total_size = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
                
                stats[symbol][tf_dir.name] = {
                    "files": len(parquet_files),
                    "rows": total_rows,
                    "size_mb": round(total_size, 1),
                }
        
        return stats


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class IngestOrchestrator:
    """Orchestre le téléchargement massif."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.downloader = AsyncDownloader(config)
        self.writer = PartitionedWriter(config)
    
    async def run(
        self,
        pairs: List[str],
        timeframes: List[str],
        consolidate: bool = True,
    ):
        """Exécute le téléchargement complet."""
        await self.downloader.init()
        
        end_date = self.config.END_DATE or datetime.utcnow()
        
        try:
            for symbol in pairs:
                for timeframe in timeframes:
                    try:
                        df = await self.downloader.download_pair_timeframe(
                            symbol, timeframe, self.config.START_DATE, end_date
                        )
                        
                        if not df.empty:
                            self.writer.write(df, symbol, timeframe)
                        
                    except Exception as e:
                        log.error(f"Erreur {symbol} {timeframe}: {e}")
                        continue
            
            if consolidate:
                log.info("\n📦 Consolidation des fichiers...")
                for symbol in pairs:
                    for timeframe in timeframes:
                        try:
                            self.writer.consolidate(symbol, timeframe)
                        except Exception as e:
                            log.error(f"Erreur consolidation {symbol} {timeframe}: {e}")
        
        finally:
            await self.downloader.close()
        
        # Afficher stats
        self._print_stats()
    
    def _print_stats(self):
        """Affiche les statistiques finales."""
        stats = self.writer.get_stats()
        
        log.info("\n" + "=" * 70)
        log.info("📊 RÉSUMÉ DES DONNÉES TÉLÉCHARGÉES")
        log.info("=" * 70)
        
        for symbol, tfs in stats.items():
            for tf, data in tfs.items():
                log.info(
                    f"  {symbol:12} {tf:4} | "
                    f"{data['rows']:>12,} rows | "
                    f"{data['files']:>4} files | "
                    f"{data['size_mb']:>8.1f} MB"
                )
        
        log.info("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Téléchargement massif Binance Futures")
    
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["all"],
        help="Paires à télécharger (ou 'all' pour le Top 10)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "15m"],
        help="Timeframes à télécharger"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Date de début (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Date de fin (YYYY-MM-DD, défaut: maintenant)"
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        default=True,
        help="Consolider en fichiers uniques"
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_false",
        dest="consolidate",
        help="Ne pas consolider"
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = IngestConfig()
    config.START_DATE = datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        config.END_DATE = datetime.strptime(args.end, "%Y-%m-%d")
    config.TIMEFRAMES = args.timeframes
    
    # Paires
    if "all" in args.pairs:
        pairs = config.TOP_PAIRS
    else:
        pairs = args.pairs
    
    log.info(f"🚀 Démarrage téléchargement massif")
    log.info(f"   Paires: {pairs}")
    log.info(f"   Timeframes: {args.timeframes}")
    log.info(f"   Période: {config.START_DATE.date()} -> {config.END_DATE or 'maintenant'}")
    
    # Exécution
    orchestrator = IngestOrchestrator(config)
    await orchestrator.run(pairs, args.timeframes, args.consolidate)
    
    log.info("🎉 Téléchargement massif terminé!")


if __name__ == "__main__":
    asyncio.run(main())
