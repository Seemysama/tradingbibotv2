#!/usr/bin/env python3
"""
Téléchargeur Massif de Données Binance Vision
Contourne les limites de l'API en téléchargeant les dumps mensuels officiels.
100x plus rapide que ccxt pour l'historique.

Usage:
    python tools/download_bulk.py --symbol BTC/USDT --years 2
    python tools/download_bulk.py --symbol ETH/USDT --start 2022-01 --end 2024-11
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# Ajout du path pour la config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import settings

# URL de base de Binance Vision (Données publiques officielles)
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Colonnes standard Binance Vision
BINANCE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count',
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]

# Colonnes qu'on garde
KEEP_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


def download_monthly_data(
    symbol: str,
    year: int,
    month: int,
    timeframe: str = "1m"
) -> Optional[pd.DataFrame]:
    """
    Télécharge et extrait un mois de données depuis Binance Vision.
    
    Args:
        symbol: Paire de trading (ex: "BTC/USDT" ou "BTCUSDT")
        year: Année
        month: Mois (1-12)
        timeframe: Intervalle (1m, 5m, 15m, 1h, 4h, 1d)
    
    Returns:
        DataFrame avec les bougies ou None si non disponible
    """
    # Format Binance Vision: BTCUSDT (pas de slash)
    sym_clean = symbol.replace('/', '').upper()
    filename = f"{sym_clean}-{timeframe}-{year}-{month:02d}"
    url = f"{BASE_URL}/{sym_clean}/{timeframe}/{filename}.zip"

    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Pas de header dans les CSV Binance Vision
                df = pd.read_csv(f, header=None)
                df.columns = BINANCE_COLUMNS

                # Nettoyage et renommage
                df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
                df.rename(columns={'open_time': 'timestamp'}, inplace=True)
                
                # Conversion des types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                return df
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur réseau pour {filename}: {e}")
        return None
    except Exception as e:
        print(f"❌ Erreur inattendue pour {filename}: {e}")
        return None


def download_bulk_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1m",
    output_path: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Télécharge plusieurs mois de données et les fusionne.
    
    Args:
        symbol: Paire de trading
        start_date: Date de début
        end_date: Date de fin
        timeframe: Intervalle
        output_path: Chemin de sauvegarde (optionnel)
    
    Returns:
        DataFrame complet ou None si échec
    """
    all_dfs: List[pd.DataFrame] = []
    current = start_date

    # Calculer le nombre de mois pour la progress bar
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    print(f"🚀 Téléchargement BULK pour {symbol} ({timeframe})")
    print(f"📅 Période: {start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}")
    print(f"📦 {total_months} mois à télécharger\n")

    with tqdm(total=total_months, desc="Téléchargement", unit="mois") as pbar:
        while current <= end_date:
            df = download_monthly_data(symbol, current.year, current.month, timeframe)
            
            if df is not None:
                all_dfs.append(df)
                pbar.set_postfix({"bougies": sum(len(d) for d in all_dfs)})
            else:
                pbar.set_postfix({"status": f"⚠️ {current.strftime('%Y-%m')} N/A"})
            
            pbar.update(1)
            current += relativedelta(months=1)

    if not all_dfs:
        print("❌ Aucune donnée récupérée.")
        return None

    # Fusion et tri
    print("\n🔧 Fusion des données...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    # Stats
    print(f"\n📊 Statistiques:")
    print(f"   - Bougies totales: {len(full_df):,}")
    print(f"   - Première bougie: {full_df['timestamp'].iloc[0]}")
    print(f"   - Dernière bougie: {full_df['timestamp'].iloc[-1]}")
    print(f"   - Taille mémoire: {full_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Sauvegarde
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(output_path, index=False)
        print(f"\n💾 Sauvegardé: {output_path}")
        print(f"   Taille fichier: {output_path.stat().st_size / 1024**2:.1f} MB")

    return full_df


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargeur Massif de Données Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python download_bulk.py --symbol BTC/USDT --years 2
    python download_bulk.py --symbol ETH/USDT --start 2022-01 --end 2024-11
    python download_bulk.py --symbol SOL/USDT --years 1 --timeframe 5m
        """
    )
    
    parser.add_argument(
        "--symbol", "-s",
        default="BTC/USDT",
        help="Paire de trading (défaut: BTC/USDT)"
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=None,
        help="Nombre d'années à télécharger depuis aujourd'hui"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Date de début (format: YYYY-MM)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Date de fin (format: YYYY-MM)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="1m",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Intervalle de temps (défaut: 1m)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Chemin de sortie personnalisé"
    )

    args = parser.parse_args()

    # Déterminer les dates
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m")
        end_date = datetime.strptime(args.end, "%Y-%m")
    elif args.years:
        end_date = datetime.now() - relativedelta(months=1)  # Mois courant pas dispo
        start_date = end_date - relativedelta(years=args.years)
    else:
        # Défaut: 2 ans
        end_date = datetime.now() - relativedelta(months=1)
        start_date = end_date - relativedelta(years=2)

    # Chemin de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        sym_clean = args.symbol.replace('/', '_')
        output_path = settings.DATA_PATH / f"{sym_clean}_BULK.parquet"

    # Téléchargement
    download_bulk_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        output_path=output_path
    )

    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
