import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ai.inference import InferenceEngine
from src.config import settings
from src.strategy import HybridStrategy, Signal

# Force ML actif pour le test
settings.ML_ENABLED = True


async def run_backtest() -> None:
    print("🧪 Backtest Golden Cross + ML...")

    symbol = "BTC/USDT"
    slug = symbol.replace("/", "_")
    path = Path(settings.DATA_PATH) / f"{slug}_{settings.TIMEFRAME}.parquet"

    if not path.exists():
        print(f"❌ Données introuvables : {path}")
        return

    print(f"📂 Chargement {path}...")
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    print(f"📊 {len(df)} bougies chargées.")

    inference = InferenceEngine()
    strategy = HybridStrategy(inference=inference)

    tech_signals = 0
    ml_validated = 0
    ml_veto = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        candle = row.to_dict()
        sig: Signal = strategy.update(candle)
        if sig.direction is None:
            continue

        tech_signals += 1
        if "VETO" in sig.reason:
            ml_veto += 1
        else:
            ml_validated += 1
            print(f"🚀 TRADE {sig.direction} @ {candle.get('close')} | {sig.reason}")

        await asyncio.sleep(0)

    print("\n" + "=" * 40)
    print(f"🏁 RÉSULTATS SUR {len(df)} BOUGIES")
    print(f"🔎 Signaux techniques détectés : {tech_signals}")
    print(f"✅ Validés par ML              : {ml_validated}")
    print(f"🛡️  Veto ML                    : {ml_veto}")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(run_backtest())
