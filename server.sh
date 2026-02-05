#!/bin/bash
# =============================================================================
# GPU SERVER LAUNCHER - Scripts de lancement sur PC Fixe
# =============================================================================
# À utiliser sur le serveur Ubuntu après le déploiement
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv_gpu/bin/activate

case "$1" in
    # =========================================================================
    # TRAINING
    # =========================================================================
    train)
        echo "🧠 Lancement de l'entraînement GPU..."
        
        # Paramètres par défaut ou custom
        DATA=${2:-"data/futures/BTC_USDT_5m_FULL.parquet"}
        EPOCHS=${3:-100}
        BATCH=${4:-1024}
        
        echo "  📊 Data: $DATA"
        echo "  🔄 Epochs: $EPOCHS"
        echo "  📦 Batch size: $BATCH"
        
        python train_v2.py \
            --data "$DATA" \
            --epochs $EPOCHS \
            --batch-size $BATCH \
            --n-splits 5 \
            --device cuda
        ;;
    
    # =========================================================================
    # DATA DOWNLOAD
    # =========================================================================
    download)
        echo "📥 Téléchargement des données massives..."
        
        PAIRS=${2:-"BTC/USDT,ETH/USDT,SOL/USDT"}
        TF=${3:-"5m"}
        
        echo "  💹 Paires: $PAIRS"
        echo "  ⏱️  Timeframe: $TF"
        
        python tools/massive_ingest.py \
            --pairs $PAIRS \
            --timeframes $TF \
            --start 2020-01-01 \
            --consolidate
        ;;
    
    # =========================================================================
    # API BACKEND
    # =========================================================================
    api)
        echo "🌐 Lancement de l'API Backend..."
        echo "  🔗 URL: http://0.0.0.0:8000"
        
        uvicorn api.app:app \
            --host 0.0.0.0 \
            --port 8000 \
            --reload
        ;;
    
    api-prod)
        echo "🌐 Lancement de l'API Backend (PRODUCTION)..."
        echo "  🔗 URL: http://0.0.0.0:8000"
        
        uvicorn api.app:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 4
        ;;
    
    # =========================================================================
    # GPU INFO
    # =========================================================================
    gpu)
        echo "🖥️ Information GPU:"
        nvidia-smi
        echo ""
        python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'VRAM Libre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB réservée')
"
        ;;
    
    # =========================================================================
    # TEST
    # =========================================================================
    test)
        echo "🧪 Test des modules..."
        python -c "
from src.features_pro import FeatureEngineerPro
from src.ai.transformer_pro import TransformerPro, TransformerConfig
import torch

print(f'✅ CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')

config = TransformerConfig(n_features=39, n_classes=3, seq_length=128)
model = TransformerPro(config).cuda()
print(f'✅ Modèle sur GPU: {next(model.parameters()).device}')

x = torch.randn(32, 128, 39).cuda()
out = model(x)
print(f'✅ Forward pass OK: {out[\"logits\"].shape}')
"
        ;;
    
    # =========================================================================
    # STATUS
    # =========================================================================
    status)
        echo "📊 Status du système:"
        echo ""
        echo "📁 Données disponibles:"
        ls -lh data/futures/*.parquet 2>/dev/null || echo "  (aucune)"
        ls -lh data/massive/*.parquet 2>/dev/null || echo "  (aucune)"
        echo ""
        echo "🧠 Modèles disponibles:"
        ls -lh models/*.pth 2>/dev/null || echo "  (aucun)"
        echo ""
        echo "🖥️ GPU:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
        ;;
    
    # =========================================================================
    # TRAIN ALL (Multi-GPU / Multi-Pair)
    # =========================================================================
    train-all)
        echo "🚀 Entraînement sur toutes les paires..."
        
        for PAIR in BTC_USDT ETH_USDT SOL_USDT; do
            DATA="data/futures/${PAIR}_5m_FULL.parquet"
            if [ -f "$DATA" ]; then
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "Training $PAIR..."
                python train_v2.py \
                    --data "$DATA" \
                    --epochs 100 \
                    --batch-size 1024 \
                    --output "models/${PAIR}"
            else
                echo "⚠️ Données non trouvées: $DATA"
            fi
        done
        ;;
    
    # =========================================================================
    # HELP
    # =========================================================================
    *)
        echo "🏦 Trading Engine V2 - GPU Server Commands"
        echo ""
        echo "Usage: ./server.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  train [data] [epochs] [batch]  - Entraîner le modèle"
        echo "  train-all                      - Entraîner sur toutes les paires"
        echo "  download [pairs] [timeframe]   - Télécharger les données"
        echo "  api                            - Lancer l'API (dev)"
        echo "  api-prod                       - Lancer l'API (production)"
        echo "  gpu                            - Afficher les infos GPU"
        echo "  test                           - Tester les modules"
        echo "  status                         - Afficher le status"
        echo ""
        echo "Exemples:"
        echo "  ./server.sh train"
        echo "  ./server.sh train data/futures/BTC_USDT_5m_FULL.parquet 200 2048"
        echo "  ./server.sh download 'BTC/USDT,ETH/USDT,SOL/USDT' 5m"
        echo "  ./server.sh api"
        ;;
esac
