#!/usr/bin/env python3
"""
Script de démarrage du backend API
"""
import os
import sys
from pathlib import Path

# Ajouter le répertoire trading_engine au PYTHONPATH
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Changer le répertoire de travail
os.chdir(script_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
