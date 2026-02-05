#!/bin/bash
# =============================================================================
# DEPLOY TO GPU SERVER - Script de déploiement sur PC Fixe NVIDIA
# =============================================================================
# Usage: ./deploy_to_server.sh user@ip_pc_fixe
#
# Ce script:
# 1. Transfère le code et les données vers le serveur
# 2. Configure l'environnement Python avec CUDA
# 3. Prépare les scripts de lancement
# =============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 TRADING ENGINE V2 - GPU Server Deployment            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Vérifier les arguments
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 user@ip_serveur${NC}"
    echo -e "${YELLOW}Exemple: $0 semy@192.168.1.100${NC}"
    exit 1
fi

SERVER=$1
REMOTE_DIR="~/trading_engine_v2"
LOCAL_DIR="/Users/semy/trading_engine_v2/trading_engine"

echo -e "\n${GREEN}📡 Serveur cible: ${SERVER}${NC}"
echo -e "${GREEN}📁 Dossier distant: ${REMOTE_DIR}${NC}"

# =============================================================================
# 1. TEST CONNEXION SSH
# =============================================================================
echo -e "\n${YELLOW}[1/5] Test de connexion SSH...${NC}"
if ssh -o ConnectTimeout=5 $SERVER "echo 'SSH OK'" 2>/dev/null; then
    echo -e "${GREEN}✅ Connexion SSH réussie${NC}"
else
    echo -e "${RED}❌ Impossible de se connecter à $SERVER${NC}"
    exit 1
fi

# =============================================================================
# 2. VÉRIFIER NVIDIA/CUDA SUR LE SERVEUR
# =============================================================================
echo -e "\n${YELLOW}[2/5] Vérification GPU NVIDIA...${NC}"
ssh $SERVER "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader" 2>/dev/null || {
    echo -e "${RED}❌ nvidia-smi non disponible. Vérifie l'installation CUDA.${NC}"
    exit 1
}
echo -e "${GREEN}✅ GPU NVIDIA détecté${NC}"

# =============================================================================
# 3. CRÉER LA STRUCTURE SUR LE SERVEUR
# =============================================================================
echo -e "\n${YELLOW}[3/5] Création de la structure distante...${NC}"
ssh $SERVER "mkdir -p $REMOTE_DIR/trading_engine/{data/{futures,massive,historical},models,logs,src/ai,tools,api,frontend}"

# =============================================================================
# 4. TRANSFERT DES FICHIERS
# =============================================================================
echo -e "\n${YELLOW}[4/5] Transfert des fichiers...${NC}"

# Fichiers de code (légers, rsync rapide)
echo -e "  📦 Code source..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.venv*' \
    --exclude 'data' \
    --exclude '*.log' \
    $LOCAL_DIR/ $SERVER:$REMOTE_DIR/trading_engine/

# Données Parquet (volumineux)
echo -e "  📊 Données 5m (futures)..."
rsync -avz --progress $LOCAL_DIR/data/futures/*.parquet $SERVER:$REMOTE_DIR/trading_engine/data/futures/ 2>/dev/null || true

echo -e "  📊 Données 1m (massive)..."
rsync -avz --progress $LOCAL_DIR/data/massive/*.parquet $SERVER:$REMOTE_DIR/trading_engine/data/massive/ 2>/dev/null || true

echo -e "  📊 Données historiques..."
rsync -avz --progress $LOCAL_DIR/data/historical/*.parquet $SERVER:$REMOTE_DIR/trading_engine/data/historical/ 2>/dev/null || true

# Modèles existants
echo -e "  🧠 Modèles..."
rsync -avz --progress $LOCAL_DIR/models/*.pth $SERVER:$REMOTE_DIR/trading_engine/models/ 2>/dev/null || true
rsync -avz --progress $LOCAL_DIR/models/*.json $SERVER:$REMOTE_DIR/trading_engine/models/ 2>/dev/null || true

echo -e "${GREEN}✅ Transfert terminé${NC}"

# =============================================================================
# 5. CONFIGURATION ENVIRONNEMENT PYTHON + CUDA
# =============================================================================
echo -e "\n${YELLOW}[5/5] Configuration de l'environnement Python + CUDA...${NC}"

ssh $SERVER << 'EOF'
cd ~/trading_engine_v2/trading_engine

# Détecter la version de CUDA
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | cut -d. -f1)
echo "CUDA Major Version: $CUDA_VERSION"

# Créer l'environnement virtuel
if [ ! -d ".venv_gpu" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv .venv_gpu
fi

source .venv_gpu/bin/activate

# Installer PyTorch avec la bonne version CUDA
echo "Installation de PyTorch pour CUDA $CUDA_VERSION..."
if [ "$CUDA_VERSION" -ge "12" ]; then
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" -ge "11" ]; then
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA version non supportée: $CUDA_VERSION"
    exit 1
fi

# Installer les dépendances
pip install pandas numpy numba pyarrow tqdm ccxt fastapi uvicorn websockets python-multipart pydantic

# Vérifier CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo "✅ Environnement configuré!"
EOF

echo -e "${GREEN}✅ Configuration terminée${NC}"

# =============================================================================
# RÉSUMÉ
# =============================================================================
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    🎉 DÉPLOIEMENT RÉUSSI!                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}📋 Prochaines étapes sur le serveur:${NC}"
echo -e "
${YELLOW}1. Connexion SSH:${NC}
   ssh $SERVER
   cd ~/trading_engine_v2/trading_engine
   source .venv_gpu/bin/activate

${YELLOW}2. Télécharger plus de données (optionnel):${NC}
   python tools/massive_ingest.py --pairs all --timeframes 5m 15m --consolidate

${YELLOW}3. Lancer l'entraînement GPU:${NC}
   python train_v2.py --data data/futures/BTC_USDT_5m_FULL.parquet --epochs 100 --batch-size 1024

${YELLOW}4. Lancer l'API Backend:${NC}
   uvicorn api.app:app --host 0.0.0.0 --port 8000

${YELLOW}5. Sur le Mac, modifier le frontend pour pointer vers le serveur:${NC}
   Dans frontend/src/App.jsx, changer:
   const API_BASE = 'http://$(echo $SERVER | cut -d@ -f2):8000'
"

echo -e "${GREEN}Pour VS Code Remote SSH:${NC}"
echo -e "   Cmd+Shift+P > Remote-SSH: Connect to Host > $SERVER"
