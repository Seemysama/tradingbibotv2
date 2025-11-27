"""
API FastAPI pour le Trading Bot - Dashboard & Control
"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ai.inference import InferenceEngine
from src.config import settings
from src.execution import ExecutionEngine
from src.strategy import HybridStrategy

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# État global du bot
bot_state = {
    "running": False,
    "strategy": None,
    "execution": None,
    "inference": None,
    "trades": [],
    "balance": 10000.0,  # Capital initial pour paper trading
    "equity_curve": [],
    "current_position": None,
    "start_time": None,  # Heure de démarrage du bot
    "live_price": None,  # Prix temps réel
    "connected_to_exchange": False,  # Statut connexion Binance
    "stats": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "pnl": 0.0,
        "win_rate": 0.0,
    }
}

ws_connections: List[WebSocket] = []


# Models Pydantic
class BotStatus(BaseModel):
    running: bool
    balance: float
    pnl: float
    total_trades: int
    win_rate: float
    current_position: Optional[Dict]
    env: str
    live_price: Optional[float] = None
    connected_to_exchange: bool = False


class Trade(BaseModel):
    timestamp: str
    symbol: str
    side: str
    price: float
    quantity: float
    pnl: Optional[float] = None
    reason: str


class StartRequest(BaseModel):
    symbol: str = "BTC/USDT"
    mode: str = "paper"  # paper | backtest | live


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise et nettoie les ressources"""
    log.info("🚀 API FastAPI démarrée")
    yield
    log.info("🔌 Arrêt API FastAPI")


app = FastAPI(title="Trading Bot API", version="1.0.0", lifespan=lifespan)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Trading Bot API", "version": "1.0.0", "status": "online"}


@app.get("/api/status", response_model=BotStatus)
async def get_status():
    """Retourne l'état actuel du bot"""
    return BotStatus(
        running=bot_state["running"],
        balance=bot_state["balance"],
        pnl=bot_state["stats"]["pnl"],
        total_trades=bot_state["stats"]["total_trades"],
        win_rate=bot_state["stats"]["win_rate"],
        current_position=bot_state["current_position"],
        env=settings.ENV,
        live_price=bot_state.get("live_price"),
        connected_to_exchange=bot_state.get("connected_to_exchange", False)
    )


@app.get("/api/price/{base}/{quote}")
async def get_live_price(base: str = "BTC", quote: str = "USDT"):
    """Récupère le prix live depuis Binance"""
    import ccxt
    
    symbol = f"{base}/{quote}"
    try:
        exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        bot_state["live_price"] = price
        bot_state["connected_to_exchange"] = True
        return {
            "symbol": symbol,
            "price": price,
            "change_24h": ticker.get('percentage', 0),
            "high_24h": ticker.get('high'),
            "low_24h": ticker.get('low'),
            "volume_24h": ticker.get('baseVolume'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        bot_state["connected_to_exchange"] = False
        return {"error": str(e), "symbol": symbol, "price": None}


@app.get("/api/trades", response_model=List[Trade])
async def get_trades(limit: int = 50):
    """Retourne l'historique des trades"""
    return bot_state["trades"][-limit:]


@app.get("/api/candles")
async def get_candles(symbol: str = "BTC/USDT", limit: int = 500):
    """Retourne les bougies OHLCV pour le graphique"""
    import pandas as pd
    from pathlib import Path
    
    slug = symbol.replace("/", "_")
    data_path = Path(f"data/historical/{slug}_1m_2Y.parquet")
    
    if not data_path.exists():
        data_path = Path(f"data/historical/{slug}_1m.parquet")
    
    if not data_path.exists():
        return {"candles": [], "error": f"Pas de données pour {symbol}"}
    
    try:
        df = pd.read_parquet(data_path).sort_values("timestamp")
        
        # Prendre les dernières bougies simulées si le bot tourne
        if bot_state["running"] and "candle_index" in bot_state:
            idx = min(bot_state["candle_index"], len(df))
            df = df.iloc[max(0, idx - limit):idx]
        else:
            df = df.tail(limit)
        
        candles = []
        for _, row in df.iterrows():
            candles.append({
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        
        return {"candles": candles, "symbol": symbol}
    except Exception as e:
        log.error(f"Erreur lecture candles: {e}")
        return {"candles": [], "error": str(e)}


@app.get("/api/equity")
async def get_equity():
    """Retourne la courbe d'equity"""
    return {"data": bot_state["equity_curve"][-500:]}


@app.post("/api/start")
async def start_bot(request: StartRequest):
    """Démarre le bot de trading"""
    if bot_state["running"]:
        return {"success": False, "message": "Bot déjà en cours d'exécution"}
    
    log.info(f"Démarrage du bot - Symbol: {request.symbol}, Mode: {request.mode}")
    
    # Reset state
    bot_state["trades"] = []
    bot_state["equity_curve"] = []
    bot_state["balance"] = 10000.0
    bot_state["current_position"] = None
    bot_state["stats"] = {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "pnl": 0.0, "win_rate": 0.0}
    bot_state["start_time"] = datetime.now()
    
    # Initialisation des composants
    bot_state["inference"] = InferenceEngine()
    bot_state["strategy"] = HybridStrategy(inference=bot_state["inference"])
    bot_state["execution"] = ExecutionEngine()
    bot_state["running"] = True
    
    # Lancement de la boucle de trading en background
    asyncio.create_task(trading_loop(request.symbol))
    
    return {
        "success": True,
        "message": f"Bot démarré en mode {request.mode}",
        "symbol": request.symbol
    }


@app.post("/api/stop")
async def stop_bot():
    """Arrête le bot de trading"""
    if not bot_state["running"]:
        return {"success": False, "message": "Bot non actif"}
    
    bot_state["running"] = False
    log.info("Bot arrêté par l'utilisateur")
    
    return {"success": True, "message": "Bot arrêté"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket pour les mises à jour en temps réel"""
    await websocket.accept()
    ws_connections.append(websocket)
    log.info(f"WebSocket connecté. Total: {len(ws_connections)}")
    
    try:
        while True:
            # Envoie les mises à jour toutes les secondes
            await websocket.send_json({
                "type": "status",
                "data": {
                    "running": bot_state["running"],
                    "balance": bot_state["balance"],
                    "pnl": bot_state["stats"]["pnl"],
                    "trades": bot_state["stats"]["total_trades"]
                }
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
        log.info(f"WebSocket déconnecté. Total: {len(ws_connections)}")


async def broadcast_update(message: dict):
    """Envoie une mise à jour à tous les clients WebSocket"""
    for ws in ws_connections[:]:
        try:
            await ws.send_json(message)
        except:
            ws_connections.remove(ws)


async def trading_loop_live(symbol: str):
    """Boucle de trading en temps réel avec Binance API"""
    import ccxt
    
    log.info(f"🔴 Mode LIVE activé - Connexion à Binance...")
    
    try:
        # Initialisation exchange
        exchange = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'sandbox': False,
            'options': {'defaultType': 'spot'}
        })
        
        # Test connexion
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        log.info(f"✅ Connecté à Binance | USDT: ${usdt_balance:.2f}")
        bot_state["connected_to_exchange"] = True
        
        # Warmup avec les dernières bougies
        log.info("📊 Chargement des bougies récentes pour warmup...")
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=500)
        
        for candle in ohlcv:
            candle_dict = {
                'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            }
            bot_state["inference"].update_buffer(candle_dict)
            bot_state["strategy"].update(candle_dict)
        
        log.info(f"✅ Warmup terminé avec {len(ohlcv)} bougies")
        
        position = None
        last_candle_time = None
        
        while bot_state["running"]:
            try:
                # Fetch la dernière bougie
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=2)
                
                if len(ohlcv) < 2:
                    await asyncio.sleep(1)
                    continue
                
                # Prendre l'avant-dernière bougie (celle qui est complète)
                candle = ohlcv[-2]
                candle_time = candle[0]
                current_price = candle[4]  # close
                
                # Toujours mettre à jour le prix live
                bot_state["live_price"] = current_price
                
                # Skip si on a déjà traité cette bougie
                if candle_time == last_candle_time:
                    await asyncio.sleep(5)  # Attendre 5s avant de re-check
                    continue
                
                last_candle_time = candle_time
                
                candle_dict = {
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': current_price,
                    'volume': candle[5]
                }
                
                # Mise à jour de la stratégie
                signal = bot_state["strategy"].update(candle_dict)
                
                log.info(f"📊 {symbol} @ ${current_price:,.2f} | Signal: {signal.direction or 'HOLD'} ({signal.confidence:.1%})")
                
                # Gestion des positions (paper trading même en LIVE pour la démo)
                now = datetime.now().isoformat()
                
                if position:
                    if position["side"] == "long":
                        pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                    else:
                        pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
                    
                    exit_reason = bot_state["execution"].should_exit(pnl_pct)
                    
                    should_close = exit_reason or \
                        (signal.direction == "short" and position["side"] == "long") or \
                        (signal.direction == "long" and position["side"] == "short")
                    
                    if should_close:
                        pnl = position["quantity"] * (current_price - position["entry_price"]) * (1 if position["side"] == "long" else -1)
                        bot_state["balance"] += pnl
                        bot_state["stats"]["pnl"] += pnl
                        bot_state["stats"]["total_trades"] += 1
                        
                        if pnl > 0:
                            bot_state["stats"]["winning_trades"] += 1
                        else:
                            bot_state["stats"]["losing_trades"] += 1
                        
                        total = bot_state["stats"]["total_trades"]
                        bot_state["stats"]["win_rate"] = (bot_state["stats"]["winning_trades"] / total * 100) if total > 0 else 0
                        
                        trade = {
                            "timestamp": now,
                            "symbol": symbol,
                            "side": f"close_{position['side']}",
                            "price": current_price,
                            "quantity": position["quantity"],
                            "pnl": round(pnl, 2),
                            "reason": exit_reason or signal.reason
                        }
                        bot_state["trades"].append(trade)
                        bot_state["equity_curve"].append({"timestamp": now, "equity": bot_state["balance"]})
                        
                        log.info(f"🔴 CLOSE {position['side'].upper()} @ ${current_price:,.2f} | PnL: ${pnl:.2f}")
                        
                        position = None
                        bot_state["current_position"] = None
                        await broadcast_update({"type": "trade", "data": trade})
                
                if not position and signal.direction and signal.confidence > 0:
                    notional = bot_state["balance"] * 0.05
                    quantity = notional / current_price
                    
                    position = {
                        "side": signal.direction,
                        "entry_price": current_price,
                        "quantity": quantity,
                        "entry_time": now
                    }
                    bot_state["current_position"] = position
                    
                    trade = {
                        "timestamp": now,
                        "symbol": symbol,
                        "side": signal.direction,
                        "price": current_price,
                        "quantity": quantity,
                        "pnl": None,
                        "reason": signal.reason
                    }
                    bot_state["trades"].append(trade)
                    
                    log.info(f"🟢 OPEN {signal.direction.upper()} @ ${current_price:,.2f} | {signal.reason}")
                    await broadcast_update({"type": "trade", "data": trade})
                
                await asyncio.sleep(5)  # Check toutes les 5 secondes
                
            except Exception as e:
                log.error(f"Erreur dans la boucle LIVE: {e}")
                await asyncio.sleep(10)
        
    except Exception as e:
        log.error(f"❌ Erreur connexion Binance: {e}")
        bot_state["running"] = False
        bot_state["connected_to_exchange"] = False


async def trading_loop(symbol: str):
    """Boucle principale de trading avec simulation paper ou LIVE"""
    import pandas as pd
    from pathlib import Path
    
    is_live = settings.ENV.upper() == "LIVE"
    log.info(f"🔄 Boucle de trading démarrée pour {symbol} | Mode: {'LIVE 🔴' if is_live else 'PAPER 📝'}")
    
    if is_live:
        # Mode LIVE - utilise l'API Binance
        await trading_loop_live(symbol)
        return
    
    # Mode PAPER/BACKTEST - utilise les données Parquet
    slug = symbol.replace("/", "_")
    data_path = Path(f"data/historical/{slug}_BULK.parquet")
    
    if not data_path.exists():
        data_path = Path(f"data/historical/{slug}_1m_2Y.parquet")
    
    if not data_path.exists():
        data_path = Path(f"data/historical/{slug}_1m.parquet")
    
    if not data_path.exists():
        log.error(f"Aucune donnée trouvée pour {symbol}")
        bot_state["running"] = False
        return
    
    df = pd.read_parquet(data_path).sort_values("timestamp")
    log.info(f"Chargé {len(df)} bougies pour {symbol}")
    
    # Warmup - préchauffe le modèle avec 500 bougies
    warmup_size = min(500, len(df) // 2)
    for _, row in df.head(warmup_size).iterrows():
        candle = row.to_dict()
        bot_state["inference"].update_buffer(candle)
        bot_state["strategy"].update(candle)
    
    log.info(f"✅ Warmup terminé avec {warmup_size} bougies")
    
    # Simulation du trading sur les données restantes
    position = None  # {side: "long/short", entry_price: float, quantity: float, entry_time: str}
    
    for idx, row in df.iloc[warmup_size:].iterrows():
        if not bot_state["running"]:
            log.info("Bot arrêté par l'utilisateur")
            break
        
        candle = row.to_dict()
        current_price = candle["close"]
        timestamp = candle["timestamp"]
        
        # Mise à jour de la stratégie
        signal = bot_state["strategy"].update(candle)
        
        # Gestion des positions
        if position:
            # Calcul du PnL de la position en cours
            if position["side"] == "long":
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
            else:  # short
                pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
            
            # Check stop loss / take profit
            exit_reason = bot_state["execution"].should_exit(pnl_pct)
            
            # Ou signal opposé
            if (signal.direction == "short" and position["side"] == "long") or \
               (signal.direction == "long" and position["side"] == "short") or \
               exit_reason:
                
                # Fermeture de la position
                pnl = position["quantity"] * (current_price - position["entry_price"]) * (1 if position["side"] == "long" else -1)
                bot_state["balance"] += pnl
                bot_state["stats"]["pnl"] += pnl
                bot_state["stats"]["total_trades"] += 1
                
                if pnl > 0:
                    bot_state["stats"]["winning_trades"] += 1
                else:
                    bot_state["stats"]["losing_trades"] += 1
                
                # Calcul win rate
                total = bot_state["stats"]["total_trades"]
                bot_state["stats"]["win_rate"] = (bot_state["stats"]["winning_trades"] / total * 100) if total > 0 else 0
                
                # Enregistrement du trade avec heure actuelle
                now = datetime.now().isoformat()
                trade = {
                    "timestamp": now,
                    "symbol": symbol,
                    "side": f"close_{position['side']}",
                    "price": current_price,
                    "quantity": position["quantity"],
                    "pnl": round(pnl, 2),
                    "reason": exit_reason or signal.reason
                }
                bot_state["trades"].append(trade)
                
                log.info(f"🔴 Position fermée: {position['side']} | PnL: ${pnl:.2f} | Raison: {exit_reason or signal.reason}")
                
                # Mise à jour courbe equity
                bot_state["equity_curve"].append({
                    "timestamp": now,
                    "equity": bot_state["balance"]
                })
                
                position = None
                bot_state["current_position"] = None
                bot_state["execution"].record_trade(-0.1)
                
                # Broadcast update
                await broadcast_update({"type": "trade", "data": trade})
        
        # Ouverture de nouvelle position si signal et pas de position en cours
        if not position and signal.direction and signal.confidence > 0:
            # Calcul de la taille de position (5% du capital)
            notional = bot_state["balance"] * 0.05
            quantity = notional / current_price
            
            # Vérification des garde-fous
            check = bot_state["execution"].pre_trade_checks(
                spread=0.0005,
                desired_exposure=0.05,
                available_balance=bot_state["balance"],
                order_notional=notional
            )
            
            if check.allowed:
                now = datetime.now().isoformat()
                position = {
                    "side": signal.direction,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "entry_time": now
                }
                bot_state["current_position"] = position
                bot_state["execution"].record_trade(0.05)
                
                trade = {
                    "timestamp": now,
                    "symbol": symbol,
                    "side": signal.direction,
                    "price": current_price,
                    "quantity": quantity,
                    "pnl": None,
                    "reason": signal.reason
                }
                bot_state["trades"].append(trade)
                
                log.info(f"🟢 Position ouverte: {signal.direction} @ ${current_price:.2f} | Conf: {signal.confidence:.2%} | {signal.reason}")
                
                # Broadcast update
                await broadcast_update({"type": "trade", "data": trade})
        
        # Throttle pour simulation temps réel (optionnel)
        await asyncio.sleep(0.05)  # 50ms par bougie = plus visible
    
    log.info("🏁 Boucle de trading terminée")
    bot_state["running"] = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
