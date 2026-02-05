"""
Stratégie Hybride V2 : Regime Detection -> Technical Signal -> ML Validation.

Architecture à 3 étages :
1. Détection de Régime (ADX) : TREND vs RANGE
2. Signal Technique : EMA Cross (Trend) ou Bollinger Rebound (Range)
3. Validation ML : Le Transformer agit comme un "veto" ou boost de confiance

Mode "ML Pure" disponible: Le Transformer décide seul.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import pandas as pd

from src.ai.inference import InferenceEngine
from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: Optional[str]  # "long", "short", None
    confidence: float
    reason: str
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    regime: str = "NONE"  # "TREND" ou "RANGE"
    ml_probability: float = 0.5  # Nouvelle: probabilité ML


class HybridStrategy:
    """
    Stratégie Hybride V2: Regime Detection -> Technical Signal -> ML Validation.
    
    Deux modes:
    1. HYBRID: Le ML valide/veto les signaux techniques (défaut)
    2. ML_PURE: Le Transformer décide seul (activé par settings.ML_PURE_MODE)
    
    Le TransformerPro binaire prédit P(UP) = probabilité de hausse.
    """

    def __init__(self, inference: Optional[InferenceEngine] = None, max_candles: int = 500) -> None:
        self.inference = inference or InferenceEngine()
        self.buffer: Deque[dict] = deque(maxlen=max_candles)
        
        # Mode ML Pure (optionnel)
        self.ml_pure_mode = getattr(settings, 'ML_PURE_MODE', False)
        
        # Seuils ML (probabilité de hausse)
        self.ml_long_threshold = getattr(settings, 'ML_LONG_THRESHOLD', 0.55)
        self.ml_short_threshold = getattr(settings, 'ML_SHORT_THRESHOLD', 0.45)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul vectorisé des indicateurs nécessaires."""
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # EMA
        df["ema_fast"] = close.ewm(span=settings.EMA_FAST, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=settings.EMA_SLOW, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/settings.RSI_WINDOW, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/settings.RSI_WINDOW, adjust=False).mean()
        rs = gain / loss.replace(0, np.inf)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR (Volatilité)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # Bollinger Bands
        sma = close.rolling(window=settings.BB_WINDOW).mean()
        std = close.rolling(window=settings.BB_WINDOW).std()
        df["bb_upper"] = sma + (std * settings.BB_STD)
        df["bb_lower"] = sma - (std * settings.BB_STD)
        df["bb_mid"] = sma

        # ADX (Force de tendance)
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm_calc = low.diff().abs()
        minus_dm_calc = minus_dm_calc.where((minus_dm_calc > plus_dm) & (low.diff() < 0), 0.0)
        
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
        tr_smooth = tr_smooth.replace(0, np.inf)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm_calc.ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.inf)
        dx = (abs(plus_di - minus_di) / di_sum) * 100
        df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # Nettoyer les valeurs infinies
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        return df

    def update(self, candle: dict) -> Signal:
        """Processus de décision principal."""
        self.buffer.append(candle)
        self.inference.update_buffer(candle)

        # Besoin de suffisamment de data pour les indicateurs
        min_candles = max(50, settings.EMA_SLOW + 10)
        if len(self.buffer) < min_candles:
            return Signal(
                direction=None, 
                confidence=0.0, 
                reason=f"Warmup ({len(self.buffer)}/{min_candles})"
            )

        df = pd.DataFrame(self.buffer)
        df = self._calculate_indicators(df)
        
        # Snapshot actuel
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = float(curr["close"])
        atr = float(curr["atr"]) if curr["atr"] > 0 else price * 0.01  # Fallback 1%
        
        # 0. Mode ML Pure - Le Transformer décide seul
        if self.ml_pure_mode and settings.ML_ENABLED:
            return self._ml_pure_decision(df, price, atr)
        
        # 1. Détection de Régime
        adx_value = float(curr["adx"]) if not np.isnan(curr["adx"]) else 20.0
        regime = "TREND" if adx_value > settings.ADX_THRESHOLD else "RANGE"
        
        direction = None
        confidence = 0.0
        reason = ""

        # 2. Logique Technique
        if regime == "TREND":
            # Golden Cross EMA
            ema_fast_curr = float(curr["ema_fast"])
            ema_slow_curr = float(curr["ema_slow"])
            ema_fast_prev = float(prev["ema_fast"])
            ema_slow_prev = float(prev["ema_slow"])
            
            if ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev:
                direction = "long"
                reason = f"EMA Cross Up | ADX={adx_value:.1f}"
                confidence = 0.70
            # Death Cross EMA
            elif ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev:
                direction = "short"
                reason = f"EMA Cross Down | ADX={adx_value:.1f}"
                confidence = 0.70
                
        else:  # RANGE
            rsi = float(curr["rsi"])
            bb_lower = float(curr["bb_lower"])
            bb_upper = float(curr["bb_upper"])
            
            # Rebond Bollinger Bas + RSI survendu
            if price < bb_lower and rsi < settings.RSI_OVERSOLD:
                direction = "long"
                reason = f"BB Rebound + RSI={rsi:.0f}"
                confidence = 0.60
            # Rebond Bollinger Haut + RSI suracheté
            elif price > bb_upper and rsi > settings.RSI_OVERBOUGHT:
                direction = "short"
                reason = f"BB Rejection + RSI={rsi:.0f}"
                confidence = 0.60

        # Pas de signal technique ? On sort.
        if not direction:
            rsi = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50
            return Signal(
                direction=None, 
                confidence=0.0, 
                reason=f"[{regime}] ADX={adx_value:.1f} RSI={rsi:.0f} | Waiting...",
                price=price,
                regime=regime
            )

        # 3. Validation ML V2 (TransformerPro binaire)
        ml_prob = 0.5  # Probabilité par défaut
        if settings.ML_ENABLED:
            try:
                ml_prob, ml_ready = self.inference.predict()
                
                if ml_ready:
                    # ML prédit P(UP) = probabilité de hausse
                    # Si direction=LONG et P(UP) < seuil_short => VETO
                    # Si direction=SHORT et P(UP) > seuil_long => VETO
                    
                    if direction == "long" and ml_prob < self.ml_short_threshold:
                        return Signal(
                            direction=None, 
                            confidence=0.0, 
                            reason=f"VETO ML: P(UP)={ml_prob:.1%} vs LONG",
                            price=price,
                            regime=regime,
                            ml_probability=ml_prob
                        )
                    elif direction == "short" and ml_prob > self.ml_long_threshold:
                        return Signal(
                            direction=None, 
                            confidence=0.0, 
                            reason=f"VETO ML: P(UP)={ml_prob:.1%} vs SHORT",
                            price=price,
                            regime=regime,
                            ml_probability=ml_prob
                        )
                    
                    # Boost de confiance si ML confirme fortement
                    if direction == "long" and ml_prob > 0.6:
                        confidence += 0.15
                        reason += f" | ML P(UP)={ml_prob:.0%}✓"
                    elif direction == "short" and ml_prob < 0.4:
                        confidence += 0.15
                        reason += f" | ML P(DOWN)={1-ml_prob:.0%}✓"
                    else:
                        reason += f" | ML P(UP)={ml_prob:.0%}"
                        
            except Exception as e:
                log.warning(f"ML inference error: {e}")

        # 4. Calcul Dynamique des Objectifs (ATR)
        if direction == "long":
            sl_price = price - (atr * settings.ATR_MULTIPLIER_SL)
            risk = price - sl_price
            tp_price = price + (risk * settings.RISK_REWARD_RATIO)
        else:
            sl_price = price + (atr * settings.ATR_MULTIPLIER_SL)
            risk = sl_price - price
            tp_price = price - (risk * settings.RISK_REWARD_RATIO)

        log.info(f"📊 Signal: {direction.upper()} @ ${price:,.2f} | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f} | {reason}")

        return Signal(
            direction=direction, 
            confidence=min(confidence, 0.95),
            reason=reason, 
            price=price, 
            stop_loss=sl_price, 
            take_profit=tp_price, 
            regime=regime,
            ml_probability=ml_prob
        )

    def _ml_pure_decision(self, df: pd.DataFrame, price: float, atr: float) -> Signal:
        """
        Mode ML Pure: Le Transformer décide seul, sans analyse technique.
        
        Logique:
        - P(UP) >= 0.55 → LONG
        - P(UP) <= 0.45 → SHORT
        - Sinon → NEUTRAL
        """
        try:
            ml_prob, ml_ready = self.inference.predict()
            
            if not ml_ready:
                return Signal(
                    direction=None,
                    confidence=0.0,
                    reason="ML Warmup...",
                    price=price,
                    regime="ML_PURE"
                )
            
            # Décision basée sur probabilité
            if ml_prob >= self.ml_long_threshold:
                direction = "long"
                confidence = (ml_prob - 0.5) * 2  # Scale 0.5-1.0 → 0-1
                reason = f"ML PURE: P(UP)={ml_prob:.1%}"
            elif ml_prob <= self.ml_short_threshold:
                direction = "short"
                confidence = (0.5 - ml_prob) * 2  # Scale 0-0.5 → 0-1
                reason = f"ML PURE: P(DOWN)={1-ml_prob:.1%}"
            else:
                return Signal(
                    direction=None,
                    confidence=0.0,
                    reason=f"ML NEUTRAL: P(UP)={ml_prob:.1%}",
                    price=price,
                    regime="ML_PURE",
                    ml_probability=ml_prob
                )
            
            # Calcul des objectifs ATR
            if direction == "long":
                sl_price = price - (atr * settings.ATR_MULTIPLIER_SL)
                risk = price - sl_price
                tp_price = price + (risk * settings.RISK_REWARD_RATIO)
            else:
                sl_price = price + (atr * settings.ATR_MULTIPLIER_SL)
                risk = sl_price - price
                tp_price = price - (risk * settings.RISK_REWARD_RATIO)
            
            log.info(f"🤖 ML PURE: {direction.upper()} @ ${price:,.2f} | P(UP)={ml_prob:.1%}")
            
            return Signal(
                direction=direction,
                confidence=min(confidence, 0.95),
                reason=reason,
                price=price,
                stop_loss=sl_price,
                take_profit=tp_price,
                regime="ML_PURE",
                ml_probability=ml_prob
            )
            
        except Exception as e:
            log.error(f"ML Pure decision error: {e}")
            return Signal(
                direction=None,
                confidence=0.0,
                reason=f"ML Error: {e}",
                price=price,
                regime="ML_PURE"
            )
