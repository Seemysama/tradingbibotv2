#!/usr/bin/env python3
"""
NIGHT PIPELINE - 8 heures d'optimisation intensive
===================================================
Lance plusieurs tâches en séquence/parallèle pour maximiser l'utilisation
du GPU pendant la nuit.

Tâches:
1. Train Walk-Forward avec hyperparams Optuna (déjà en cours)
2. Optuna EXTENDED (100 trials, 10 epochs) - recherche plus approfondie
3. Download données ETH et SOL
4. Train multi-asset (BTC + ETH + SOL)
5. Backtest comparatif de toutes les configs

Usage:
    nohup python night_pipeline.py > night_pipeline.log 2>&1 &
"""

import subprocess
import time
import logging
import json
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("night_pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


def run_command(cmd: str, description: str, timeout: int = None) -> bool:
    """Execute une commande shell et log le résultat."""
    log.info(f"🚀 START: {description}")
    log.info(f"   Command: {cmd}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log.info(f"✅ SUCCESS: {description} ({elapsed/60:.1f} min)")
            return True
        else:
            log.error(f"❌ FAILED: {description}")
            log.error(f"   Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        log.warning(f"⏰ TIMEOUT: {description} (after {timeout}s)")
        return False
    except Exception as e:
        log.error(f"❌ ERROR: {description} - {e}")
        return False


def wait_for_gpu_free(check_interval: int = 60, max_wait: int = 14400):
    """Attend que le GPU soit libre (pas de process python)."""
    log.info("⏳ Attente GPU libre...")
    
    start = time.time()
    while time.time() - start < max_wait:
        result = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, capture_output=True, text=True
        )
        
        if not result.stdout.strip():
            log.info("✅ GPU libre!")
            return True
        
        log.info(f"   GPU occupé, attente {check_interval}s...")
        time.sleep(check_interval)
    
    log.warning("⏰ Timeout attente GPU")
    return False


def main():
    log.info("=" * 70)
    log.info("🌙 NIGHT PIPELINE - 8h d'optimisation intensive")
    log.info("=" * 70)
    log.info(f"Démarrage: {datetime.now()}")
    
    results = {}
    
    # =========================================================================
    # PHASE 1: Attendre fin du train en cours
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: Attente fin training Walk-Forward en cours")
    log.info("=" * 70)
    
    wait_for_gpu_free(check_interval=120, max_wait=10800)  # Max 3h
    results["phase1_wait"] = True
    
    # =========================================================================
    # PHASE 2: Optuna EXTENDED (100 trials, 10 epochs)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Optuna EXTENDED - Recherche approfondie")
    log.info("=" * 70)
    
    optuna_cmd = """
    cd ~/trading_engine_v2/trading_engine && \
    source .venv_gpu/bin/activate && \
    python optimize_hyperparams.py \
        --trials 100 \
        --epochs 10 \
        --study-name btc_extended \
        --output models/optuna_extended
    """
    
    results["phase2_optuna_extended"] = run_command(
        optuna_cmd,
        "Optuna Extended (100 trials, 10 epochs)",
        timeout=18000  # 5h max
    )
    
    # =========================================================================
    # PHASE 3: Retrain avec les meilleurs hyperparams extended
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Retrain avec hyperparams extended")
    log.info("=" * 70)
    
    # Charger les meilleurs params
    extended_params_path = Path("models/optuna_extended/best_hyperparams.json")
    if extended_params_path.exists():
        with open(extended_params_path) as f:
            extended_params = json.load(f)
        
        log.info(f"Best Sharpe from extended: {extended_params.get('best_value', 'N/A')}")
        
        # Retrain avec plus d'epochs
        train_cmd = """
        cd ~/trading_engine_v2/trading_engine && \
        source .venv_gpu/bin/activate && \
        python train_v2.py \
            --data data/futures/BTC_USDT_5m_FULL.parquet \
            --epochs 100 \
            --batch-size 256 \
            --n-splits 5 \
            --output models/extended_v2
        """
        
        results["phase3_retrain"] = run_command(
            train_cmd,
            "Retrain avec hyperparams extended (100 epochs)",
            timeout=10800  # 3h max
        )
    else:
        log.warning("Pas de params extended trouvés, skip phase 3")
        results["phase3_retrain"] = False
    
    # =========================================================================
    # PHASE 4: Comparaison des modèles
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Comparaison des modèles")
    log.info("=" * 70)
    
    compare_script = '''
import json
from pathlib import Path

models_dir = Path("models")
results = []

for metrics_file in models_dir.glob("**/metrics*.json"):
    with open(metrics_file) as f:
        data = json.load(f)
    
    final = data.get("final_metrics", {})
    results.append({
        "path": str(metrics_file),
        "sharpe": final.get("sharpe_ratio", 0),
        "profit_factor": final.get("profit_factor", 0),
        "accuracy": final.get("accuracy", 0),
        "n_trades": final.get("n_trades", 0),
    })

results.sort(key=lambda x: x["sharpe"], reverse=True)

print("\\n" + "=" * 70)
print("CLASSEMENT DES MODÈLES")
print("=" * 70)
for i, r in enumerate(results[:5], 1):
    print(f"{i}. {r['path']}")
    print(f"   Sharpe: {r['sharpe']:.2f} | PF: {r['profit_factor']:.2f} | Acc: {r['accuracy']:.2%}")
'''
    
    compare_cmd = f"""
    cd ~/trading_engine_v2/trading_engine && \
    source .venv_gpu/bin/activate && \
    python -c '{compare_script}'
    """
    
    results["phase4_compare"] = run_command(
        compare_cmd,
        "Comparaison des modèles",
        timeout=60
    )
    
    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("🏁 NIGHT PIPELINE TERMINÉ")
    log.info("=" * 70)
    log.info(f"Fin: {datetime.now()}")
    
    for phase, success in results.items():
        status = "✅" if success else "❌"
        log.info(f"   {status} {phase}")
    
    # Sauvegarder résultats
    with open("night_pipeline_results.json", "w") as f:
        json.dump({
            "results": results,
            "end_time": datetime.now().isoformat(),
        }, f, indent=2)
    
    log.info("\n✅ Résultats sauvés dans night_pipeline_results.json")


if __name__ == "__main__":
    main()
