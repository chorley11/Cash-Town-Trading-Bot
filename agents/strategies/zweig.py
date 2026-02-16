"""
Zweig Super Model Strategy Agent - FIXED VERSION
Based on Martin Zweig's market timing principles.

DIAGNOSIS OF ORIGINAL PROBLEMS:
1. Bad R:R ratio (2% SL / 3% TP = 1.5:1) - needs >40% WR to break even
2. No trend confirmation - traded against the trend
3. Too frequent signals - scored on every bar at extreme
4. Missing the essence of Zweig - it's about THRUST (transition), not static scores

FIXES IMPLEMENTED:
1. ATR-based stops with 2:1 R:R minimum (like trend-following)
2. ADX trend confirmation required (ADX > 25)
3. Signal only on SCORE TRANSITIONS (cross from neutral to extreme)
4. Long-bias (Zweig Breadth Thrust is bullish indicator)
5. Stricter confidence thresholds
6. Volume confirmation required

Core Logic:
- Multi-factor scoring system (0-10)
- Only signal on score TRANSITIONS (thrust), not static extremes
- Trend confirmation via ADX
- Long-biased (Zweig is fundamentally bullish)
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide, Position

logger = logging.getLogger(__name__)

class ZweigAgent(BaseStrategyAgent):
    """
    Zweig Super Model Strategy - FIXED
    
    Key Changes from Original:
    - Signals on TRANSITIONS only (score crossing threshold)
    - Requires ADX > 25 (trend confirmation)
    - ATR-based stops (2:1 R:R)
    - Long-biased (shorts require higher threshold)
    - Volume confirmation for all signals
    """
    
    DEFAULT_CONFIG = {
        # MA settings
        'sma_short': 10,
        'sma_medium': 20,
        'sma_long': 50,
        
        # Momentum
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        
        # Risk
        'atr_period': 14,
        'stop_loss_atr': 2.0,      # FIXED: ATR-based stops
        'take_profit_atr': 4.0,    # FIXED: 2:1 R:R
        
        # Volume
        'volume_period': 20,
        'volume_threshold': 1.2,   # NEW: Require 1.2x avg volume
        
        # Thresholds - ADJUSTED
        'short_threshold': 2,      # STRICTER: Was 3, now 2 (harder to short)
        'long_threshold': 7,       # Keep same
        'neutral_low': 4,          # NEW: Below this = bearish zone
        'neutral_high': 6,         # NEW: Above this = bullish zone
        
        # Trend confirmation - NEW
        'adx_period': 14,
        'adx_threshold': 25,       # NEW: Require strong trend
        
        # Signal filtering
        'min_confidence': 0.60,    # STRICTER: Was implicit 0.3
        'persistence_bars': 3,     # STRICTER: Was 2
        
        # Long bias
        'long_bias': True,         # NEW: Prefer longs (Zweig is bullish)
        'short_penalty': 0.15,     # NEW: Reduce short confidence
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='zweig',
            name='Zweig Super Model',
            symbols=symbols,
            config=merged_config
        )
        # Track scores for TRANSITION detection (not just persistence)
        self._score_history: Dict[str, List[float]] = {}
        self._last_signal_zone: Dict[str, str] = {}  # 'neutral', 'bullish', 'bearish'
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['sma_short']}",
            f"sma_{self.config['sma_medium']}",
            f"sma_{self.config['sma_long']}",
            f"rsi_{self.config['rsi_period']}",
            'macd',
            f"atr_{self.config['atr_period']}",
            f"adx_{self.config['adx_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate Zweig Super Model signals"""
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 100:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Analyze symbol and calculate Super Model score with TRANSITION detection"""
        
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate components
        trend_score = self._calculate_trend_score(closes)
        momentum_score = self._calculate_momentum_score(closes)
        health_score = self._calculate_health_score(closes, volumes)
        
        # Total base score (0-8 range before persistence)
        base_score = trend_score + momentum_score + health_score
        
        # Track score history
        if symbol not in self._score_history:
            self._score_history[symbol] = []
        self._score_history[symbol].append(base_score)
        if len(self._score_history[symbol]) > 20:
            self._score_history[symbol] = self._score_history[symbol][-20:]
        
        # Determine current zone
        current_zone = self._get_zone(base_score)
        prev_zone = self._last_signal_zone.get(symbol, 'neutral')
        
        # KEY FIX: Only signal on ZONE TRANSITIONS (the "thrust")
        # Not on staying in an extreme zone
        is_transition = current_zone != prev_zone and current_zone != 'neutral'
        
        if not is_transition:
            return None
        
        # Check persistence - score must stay extreme for N bars
        if not self._check_persistence(symbol, current_zone):
            return None
        
        # NEW: Calculate ADX for trend confirmation
        adx = self._adx(highs, lows, closes, self.config['adx_period'])
        if adx is None or len(adx) == 0:
            return None
        current_adx = adx[-1]
        
        # NEW: Require strong trend (ADX > threshold)
        if current_adx < self.config['adx_threshold']:
            logger.debug(f"{symbol}: Zweig signal rejected - ADX {current_adx:.1f} < {self.config['adx_threshold']}")
            return None
        
        # NEW: Volume confirmation
        volume_sma = self._sma(volumes, self.config['volume_period'])
        if volume_sma is not None and len(volume_sma) > 0:
            volume_ratio = volumes[-1] / volume_sma[-1]
            if volume_ratio < self.config['volume_threshold']:
                logger.debug(f"{symbol}: Zweig signal rejected - volume {volume_ratio:.2f}x < {self.config['volume_threshold']}x")
                return None
        else:
            volume_ratio = 1.0
        
        # Calculate ATR for stops
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        current_atr = atr[-1] if atr is not None and len(atr) > 0 else current_price * 0.02
        
        # For confidence, check how strong the transition was
        persistence_bonus = self._get_persistence_bonus(symbol, current_zone)
        
        signal = None
        
        # LONG signal: Transition to bullish zone
        if current_zone == 'bullish':
            # Base confidence from score strength
            score_strength = (base_score - self.config['long_threshold']) / 3  # 0-1 range
            confidence = 0.55 + (score_strength * 0.2) + (persistence_bonus * 0.1)
            
            # ADX bonus
            if current_adx >= 35:
                confidence += 0.1
            elif current_adx >= 30:
                confidence += 0.05
            
            # Volume bonus
            if volume_ratio > 1.5:
                confidence += 0.05
            
            confidence = min(max(confidence, 0.5), 0.90)
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr'])
                take_profit = current_price + (current_atr * self.config['take_profit_atr'])
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"Zweig LONG thrust: Score {base_score:.1f}/10, ADX={current_adx:.1f}, Vol={volume_ratio:.1f}x",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'trend_score': trend_score,
                        'momentum_score': momentum_score,
                        'health_score': health_score,
                        'total_score': base_score,
                        'adx': current_adx,
                        'volume_ratio': volume_ratio,
                        'zone_transition': f"{prev_zone} -> {current_zone}"
                    }
                )
                
        # SHORT signal: Transition to bearish zone (stricter requirements)
        elif current_zone == 'bearish':
            # Apply long bias penalty
            if self.config['long_bias']:
                # Require even lower score for shorts
                if base_score > self.config['short_threshold']:
                    return None
            
            score_strength = (self.config['short_threshold'] - base_score) / 3
            confidence = 0.50 + (score_strength * 0.2) + (persistence_bonus * 0.1)
            
            # ADX bonus (need strong trend for shorts too)
            if current_adx >= 35:
                confidence += 0.1
            elif current_adx >= 30:
                confidence += 0.05
            
            # Apply short penalty
            confidence -= self.config['short_penalty']
            
            confidence = min(max(confidence, 0.4), 0.85)
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr'])
                take_profit = current_price - (current_atr * self.config['take_profit_atr'])
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"Zweig SHORT thrust: Score {base_score:.1f}/10, ADX={current_adx:.1f}, Vol={volume_ratio:.1f}x",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'trend_score': trend_score,
                        'momentum_score': momentum_score,
                        'health_score': health_score,
                        'total_score': base_score,
                        'adx': current_adx,
                        'volume_ratio': volume_ratio,
                        'zone_transition': f"{prev_zone} -> {current_zone}"
                    }
                )
        
        # Update zone tracking
        if signal:
            self._last_signal_zone[symbol] = current_zone
        
        return signal
    
    def _get_zone(self, score: float) -> str:
        """Determine which zone a score is in"""
        if score >= self.config['long_threshold']:
            return 'bullish'
        elif score <= self.config['short_threshold']:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_trend_score(self, closes: np.ndarray) -> float:
        """Calculate trend score (0-3)"""
        score = 0
        
        sma_short = self._sma(closes, self.config['sma_short'])
        sma_medium = self._sma(closes, self.config['sma_medium'])
        sma_long = self._sma(closes, self.config['sma_long'])
        
        if sma_short is None or sma_medium is None or sma_long is None:
            return 1.5  # Neutral
        
        current_price = closes[-1]
        
        # Price above/below MAs
        if current_price > sma_short[-1]:
            score += 1
        if current_price > sma_medium[-1]:
            score += 1
        if current_price > sma_long[-1]:
            score += 1
        
        return score
    
    def _calculate_momentum_score(self, closes: np.ndarray) -> float:
        """Calculate momentum score (0-3)"""
        score = 0
        
        # RSI component
        rsi = self._rsi(closes, self.config['rsi_period'])
        if rsi is not None and len(rsi) > 0:
            current_rsi = rsi[-1]
            if current_rsi > 50:
                score += 1
            if current_rsi > 60:
                score += 0.5
            if current_rsi < 40:
                score -= 0.5
        
        # MACD component
        macd, signal_line, hist = self._macd(closes)
        if macd is not None:
            if macd[-1] > signal_line[-1]:
                score += 1
            if hist[-1] > 0 and len(hist) > 1 and hist[-1] > hist[-2]:
                score += 0.5
        
        return max(0, min(3, score))
    
    def _calculate_health_score(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate market health score (0-2)"""
        score = 1  # Start neutral
        
        # Volume health
        volume_sma = self._sma(volumes, self.config['volume_period'])
        if volume_sma is not None and len(volume_sma) > 0:
            if volumes[-1] > volume_sma[-1]:
                score += 0.5
            else:
                score -= 0.5
        
        # Volatility check
        if len(closes) >= 20:
            recent_std = np.std(closes[-10:]) / np.mean(closes[-10:])
            older_std = np.std(closes[-20:-10]) / np.mean(closes[-20:-10])
            
            if recent_std < older_std:
                score += 0.5
        
        return max(0, min(2, score))
    
    def _check_persistence(self, symbol: str, current_zone: str) -> bool:
        """Check if signal zone has persisted for required bars"""
        history = self._score_history.get(symbol, [])
        if len(history) < self.config['persistence_bars']:
            return False
        
        recent_scores = history[-self.config['persistence_bars']:]
        
        if current_zone == 'bullish':
            # All recent scores must be in bullish territory
            return all(s >= self.config['neutral_high'] for s in recent_scores)
        elif current_zone == 'bearish':
            # All recent scores must be in bearish territory
            return all(s <= self.config['neutral_low'] for s in recent_scores)
        
        return False
    
    def _get_persistence_bonus(self, symbol: str, current_zone: str) -> float:
        """Calculate persistence bonus (how long signal has held)"""
        history = self._score_history.get(symbol, [])
        if len(history) < 3:
            return 0
        
        consecutive = 0
        threshold_high = self.config['neutral_high']
        threshold_low = self.config['neutral_low']
        
        for score in reversed(history):
            if current_zone == 'bullish' and score >= threshold_high:
                consecutive += 1
            elif current_zone == 'bearish' and score <= threshold_low:
                consecutive += 1
            else:
                break
        
        # Return bonus: 0.1 for 3 bars, up to 0.3 for 6+ bars
        return min(0.3, (consecutive - 2) * 0.1) if consecutive >= 3 else 0
    
    # ===== Indicator Calculations =====
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices: np.ndarray):
        """Calculate MACD"""
        if len(prices) < self.config['macd_slow'] + self.config['macd_signal']:
            return None, None, None
        
        ema_fast = self._ema(prices, self.config['macd_fast'])
        ema_slow = self._ema(prices, self.config['macd_slow'])
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        min_len = min(len(ema_fast), len(ema_slow))
        macd = ema_fast[-min_len:] - ema_slow[-min_len:]
        
        signal_line = self._ema(macd, self.config['macd_signal'])
        if signal_line is None:
            return None, None, None
        
        min_len = min(len(macd), len(signal_line))
        hist = macd[-min_len:] - signal_line[-min_len:]
        
        return macd[-min_len:], signal_line[-min_len:], hist
    
    def _ema(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.convolve(tr, np.ones(period)/period, mode='valid')
    
    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        """Calculate ADX (Average Directional Index)"""
        if len(high) < period * 2:
            return None
        
        # Calculate +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate TR
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Smooth with EMA
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, period) / (atr + 1e-10)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)
        
        return adx
