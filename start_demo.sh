#!/bin/bash

# Script de démarrage du Trading Bot avec Dashboard

echo "🚀 Démarrage du Trading Bot Dashboard..."

# Couleurs pour les logs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour arrêter les processus au signal CTRL+C
cleanup() {
    echo -e "\n${RED}🛑 Arrêt des services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Obtenir le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/../.venv"

# Vérifier si l'environnement virtuel existe
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Environnement virtuel non trouvé. Création...${NC}"
    cd "$SCRIPT_DIR/.."
    python3 -m venv .venv
    .venv/bin/pip install -r trading_engine/requirements.txt
    .venv/bin/pip install websockets
    echo -e "${GREEN}✅ Environnement virtuel créé${NC}"
else
    echo -e "${GREEN}✅ Environnement virtuel trouvé${NC}"
fi

# Démarrer le backend FastAPI
echo -e "\n${BLUE}🔧 Démarrage du backend API (port 8000)...${NC}"
cd "$SCRIPT_DIR"
"$VENV_PATH/bin/python" -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}✅ Backend démarré (PID: $BACKEND_PID)${NC}"

# Attendre que le backend soit prêt
sleep 3

# Vérifier si Node.js est installé
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js non installé. Veuillez installer Node.js et npm.${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Installer les dépendances frontend si nécessaire
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}📦 Installation des dépendances frontend...${NC}"
    cd frontend
    npm install
    cd ..
    echo -e "${GREEN}✅ Dépendances installées${NC}"
fi

# Démarrer le frontend
echo -e "\n${BLUE}🎨 Démarrage du frontend (port 3000)...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}✅ Frontend démarré (PID: $FRONTEND_PID)${NC}"

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Trading Bot Dashboard lancé avec succès !${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n📊 Dashboard: ${BLUE}http://localhost:3000${NC}"
echo -e "🔌 API Backend: ${BLUE}http://localhost:8000${NC}"
echo -e "📚 API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "\n${RED}Appuyez sur CTRL+C pour arrêter tous les services${NC}\n"

# Attendre que les processus tournent
wait $BACKEND_PID $FRONTEND_PID
