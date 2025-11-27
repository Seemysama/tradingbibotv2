import subprocess
import sys
from pathlib import Path


def run_step(cmd: list, desc: str) -> bool:
    print(f"\n=== {desc} ===")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Échec: {exc}")
        return False


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    env_python = repo_root / ".venv" / "bin" / "python"
    data_path = repo_root / "data" / "historical" / "BTC_USDT_1m.parquet"

    # 1. Backfill
    ok = run_step(
        [str(env_python), "tools/deep_backfill.py"],
        "Backfill massif (24 mois BTC/ETH/SOL)",
    )
    if not ok:
        sys.exit(1)

    # 2. Entraînement avancé (50 epochs, early stopping dans le trainer)
    ok = run_step(
        [str(env_python), "train.py"],
        "Entraînement modèle avancé",
    )
    if not ok:
        sys.exit(1)

    # 3. Optimisation grille
    ok = run_step(
        [str(env_python), "src/optimizer.py"],
        "Optimisation brute (grid search)",
    )
    if not ok:
        sys.exit(1)

    # 4. Résumé
    print("\n=== RÉSUMÉ ===")
    print(f"Données utilisées : {data_path}")
    print("Modèle : models/best_model.pth")
    print("Résultats optimisation : optimization_results.csv")


if __name__ == "__main__":
    main()
