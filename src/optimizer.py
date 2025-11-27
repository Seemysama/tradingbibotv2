import itertools
import math
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Assure l'import local quand lancé via python src/optimizer.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import settings


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if "RSI_14" not in df.columns:
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll_up = gain.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down
        df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def simulate_pnl(
    df: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    rsi_low: int,
    rsi_high: int,
    ml_conf: float,
    fee: float = 0.0004,
) -> Tuple[float, float, float]:
    """Retourne (pnl, max_dd, sharpe) pour la combinaison."""
    close = df["close"]
    rsi = df["RSI_14"]
    fast = close.rolling(sma_fast, min_periods=sma_fast).mean()
    slow = close.rolling(sma_slow, min_periods=sma_slow).mean()

    long_signal = (fast > slow) & (rsi < rsi_high)
    short_signal = (fast < slow) & (rsi > rsi_low)

    position = np.where(long_signal, 1, np.where(short_signal, -1, 0)).astype(float)
    # ML proxy: on réduit l'exposition proportionnellement au seuil (simulation simple)
    position = position * (ml_conf)

    # Retours
    rets = close.pct_change().fillna(0).values
    pnl_series = position * rets - fee * np.abs(np.diff(position, prepend=0))
    pnl = pnl_series.sum()

    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) else 0.0
    sharpe = pnl_series.mean() / (pnl_series.std() + 1e-9) * math.sqrt(365 * 24 * 60) if len(pnl_series) else 0.0
    return pnl, max_dd, sharpe


def worker(args: Tuple[int, int, Tuple[int, int], float, Path]) -> Dict[str, float]:
    sma_fast, sma_slow, rsi_range, ml_conf, path = args
    df = load_data(path)
    pnl, max_dd, sharpe = simulate_pnl(df, sma_fast, sma_slow, rsi_range[0], rsi_range[1], ml_conf)
    return {
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rsi_low": rsi_range[0],
        "rsi_high": rsi_range[1],
        "ml_conf": ml_conf,
        "pnl": pnl,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


def run_grid(data_path: Path, out_csv: Path) -> None:
    params = list(
        itertools.product(
            [10, 20, 50],
            [100, 200],
            [(30, 70), (20, 80)],
            [0.55, 0.60, 0.65, 0.70, 0.75],
        )
    )
    jobs = [(f, s, rsi, ml, data_path) for (f, s, rsi, ml) in params]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(worker, jobs)

    out_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Optimisation terminée. Résultats: {out_csv}")
    print(out_df.head(5))


if __name__ == "__main__":
    data_path = Path(settings.DATA_PATH) / "BTC_USDT_1m.parquet"
    out_csv = Path("optimization_results.csv")
    run_grid(data_path, out_csv)
