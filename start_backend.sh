#!/bin/bash
cd /Users/semy/trading_engine_v2/trading_engine
/Users/semy/trading_engine_v2/.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
