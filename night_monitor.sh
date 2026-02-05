#!/bin/bash
#
# NIGHT MONITOR - Surveillance et enchainement des tâches
# ========================================================
# Ce script surveille le train en cours et lance Optuna extended quand il finit.
# Crée des fichiers de statut pour vérification facile.
#
# Usage: nohup bash night_monitor.sh > night_monitor.log 2>&1 &
#

cd ~/trading_engine_v2/trading_engine
source .venv_gpu/bin/activate

LOG_FILE="night_monitor.log"
STATUS_FILE="night_status.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" > "$STATUS_FILE"
}

# ============================================================================
# PHASE 1: Attendre fin du train_v2.py en cours
# ============================================================================
log "🌙 NIGHT MONITOR DÉMARRÉ"
log "📊 Phase 1: Surveillance train_v2.py en cours..."

while pgrep -f "train_v2.py" > /dev/null; do
    # Extraire les dernières métriques
    LAST_LINE=$(tail -1 train_night.log 2>/dev/null | grep -oP "Sharpe: [\d\.\-]+" | tail -1)
    FOLD=$(grep -oP "FOLD \d/\d" train_night.log 2>/dev/null | tail -1)
    
    log "⏳ Train en cours - $FOLD - $LAST_LINE"
    echo "Phase: TRAINING | $FOLD | $LAST_LINE | $(date)" > "$STATUS_FILE"
    
    sleep 120  # Check toutes les 2 minutes
done

log "✅ Phase 1 TERMINÉE - train_v2.py fini!"

# Copier les résultats
cp train_night.log "train_night_BACKUP_$(date +%Y%m%d_%H%M%S).log"

# Vérifier si le modèle a été sauvé
if [ -f "models/transformer_v2.pth" ]; then
    log "✅ Modèle transformer_v2.pth trouvé!"
else
    log "⚠️ ATTENTION: Pas de modèle trouvé!"
fi

# ============================================================================
# PHASE 2: Optuna Extended (100 trials, 10 epochs)
# ============================================================================
log "📊 Phase 2: Lancement Optuna Extended (100 trials, 10 epochs)..."
echo "Phase: OPTUNA_EXTENDED | Starting... | $(date)" > "$STATUS_FILE"

python optimize_hyperparams.py \
    --trials 100 \
    --epochs 10 \
    --study-name btc_extended_night \
    --output models 2>&1 | tee optuna_extended.log

OPTUNA_EXIT=$?

if [ $OPTUNA_EXIT -eq 0 ]; then
    log "✅ Phase 2 TERMINÉE - Optuna Extended réussi!"
    
    # Afficher les meilleurs params
    if [ -f "models/best_hyperparams.json" ]; then
        log "📋 Meilleurs hyperparamètres:"
        cat models/best_hyperparams.json | head -20
    fi
else
    log "❌ Phase 2 ÉCHOUÉE - Optuna exit code: $OPTUNA_EXIT"
fi

# ============================================================================
# PHASE 3: Retrain avec les nouveaux hyperparams (si Optuna OK)
# ============================================================================
if [ $OPTUNA_EXIT -eq 0 ]; then
    log "📊 Phase 3: Retrain avec hyperparams optimisés (100 epochs)..."
    echo "Phase: RETRAIN_EXTENDED | Starting... | $(date)" > "$STATUS_FILE"
    
    python train_v2.py \
        --data data/futures/BTC_USDT_5m_FULL.parquet \
        --epochs 100 \
        --batch-size 256 \
        --n-splits 5 2>&1 | tee train_extended.log
    
    TRAIN_EXIT=$?
    
    if [ $TRAIN_EXIT -eq 0 ]; then
        log "✅ Phase 3 TERMINÉE - Retrain réussi!"
    else
        log "❌ Phase 3 ÉCHOUÉE - Train exit code: $TRAIN_EXIT"
    fi
else
    log "⏭️ Phase 3 SKIPPÉE (Optuna a échoué)"
fi

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
log "========================================"
log "🏁 NIGHT MONITOR TERMINÉ"
log "========================================"

# Créer un résumé
cat > night_summary.txt << EOF
NIGHT PIPELINE SUMMARY
======================
Date: $(date)

PHASE 1 - Train Walk-Forward:
  Status: COMPLETED
  Log: train_night.log
  Model: models/transformer_v2.pth

PHASE 2 - Optuna Extended:
  Status: $([ $OPTUNA_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")
  Log: optuna_extended.log
  Best params: models/best_hyperparams.json

PHASE 3 - Retrain Extended:
  Status: $([ ${TRAIN_EXIT:-1} -eq 0 ] && echo "SUCCESS" || echo "SKIPPED/FAILED")
  Log: train_extended.log

FILES CREATED:
$(ls -la models/*.pth models/*.json 2>/dev/null)

BEST METRICS (if available):
$(tail -20 train_extended.log 2>/dev/null | grep -E "Sharpe|Profit|Accuracy" || echo "N/A")
EOF

log "📄 Résumé sauvé dans night_summary.txt"
echo "Phase: COMPLETED | Check night_summary.txt | $(date)" > "$STATUS_FILE"

log "🌙 Bonne nuit! Tout est terminé."
