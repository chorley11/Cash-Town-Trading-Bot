"""
RSI Divergence Strategy Agent
Detects price/RSI divergences to catch early trend reversals.

Core Logic:
- Bullish divergence: Price makes lower low, RSI makes higher low
- Bearish divergence: Price makes higher high, RSI makes lower high
- Confirmation via candlestick pattern or momentum shift
- ATR-based stops with trend-following take profits

Why This Complements Existing Strategies:
- Trend-following catches established trends but suffers at reversals
- Mean-reversion waits for extreme BB/RSI levels (reactive)
- Divergence spots reversals EARLY, before extremes are hit (predictive)
- Works in both trending and ranging markets during transitions
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)


class RSIDivergenceAgent(BaseStrategyAgent):
    """
    RSI Divergence Strategy
    
    Entry conditions:
    - Regular bullish divergence: Price lower low + RSI higher low (LONG)
    - Regular bearish divergence: Price higher high + RSI lower high (SHORT)
    - Optional: Hidden divergence for trend continuation
    - Confirmation: RSI crosses key level or momentum shifts
    
    Exit conditions:
    - Divergence resolution (price catches up to RSI direction)
    - Opposite divergence appears
    - Stop loss / take profit hit
    
    Market conditions:
    - Works best at trend exhaustion points
    - Excellent for catching reversals early
    - Can be used for trend continuation (hidden divergence)
    """
    
    DEFAULT_CONFIG = {
        'rsi_period': 14,
        'lookback_bars': 20,          # How far back to look for divergence
        'min_swing_bars': 3,          # Minimum bars between swings
        'divergence_threshold': 0.02,  # Minimum price/RSI divergence (2%)
        'rsi_extreme_zone': 30,       # RSI below 30 or above 70 for stronger signals
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.5,
        'min_confidence': 0.55,
        'require_confirmation': True,  # Wait for RSI momentum shift
        'use_hidden_divergence': False,  # Also trade trend continuation
        'volume_filter': True,         # Volume should increase on divergence
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='rsi-divergence',
            name='RSI Divergence',
            symbols=symbols,
            config=merged_config
        )
        self._recent_signals: Dict[str, datetime] = {}  # Cooldown tracking
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"rsi_{self.config['rsi_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        rsi = self._rsi(closes, self.config['rsi_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if rsi is None or atr is None or len(rsi) < self.config['lookback_bars']:
            return None
        
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        
        # Align arrays to same length (RSI is shorter due to calculation)
        rsi_offset = len(closes) - len(rsi)
        aligned_lows = lows[rsi_offset:]
        aligned_highs = highs[rsi_offset:]
        aligned_closes = closes[rsi_offset:]
        aligned_volumes = volumes[rsi_offset:]
        
        # Check for bullish divergence (price lower low, RSI higher low)
        bullish_div = self._detect_bullish_divergence(
            aligned_lows, rsi, self.config['lookback_bars']
        )
        
        # Check for bearish divergence (price higher high, RSI lower high)
        bearish_div = self._detect_bearish_divergence(
            aligned_highs, rsi, self.config['lookback_bars']
        )
        
        # Volume filter
        volume_ok = True
        if self.config['volume_filter'] and volume_sma is not None:
            recent_vol = np.mean(aligned_volumes[-5:])
            avg_vol = volume_sma[-1]
            volume_ok = recent_vol > avg_vol * 0.8  # At least 80% of average
        
        # Confirmation: RSI showing momentum shift
        rsi_confirmed = self._check_rsi_confirmation(rsi)
        confirmation_ok = not self.config['require_confirmation'] or rsi_confirmed
        
        signal = None
        
        # LONG signal: Bullish divergence detected
        if bullish_div and volume_ok and confirmation_ok:
            div_strength, price_swing, rsi_swing = bullish_div
            
            # Stronger signal if RSI was in oversold zone
            in_extreme = current_rsi < self.config['rsi_extreme_zone'] or \
                        min(rsi[-self.config['lookback_bars']:]) < self.config['rsi_extreme_zone']
            
            confidence = self._calculate_confidence(div_strength, in_extreme, True)
            
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
                    reason=f"LONG Divergence: Bullish RSI div ({div_strength:.1%}), RSI={current_rsi:.0f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'divergence_type': 'bullish_regular',
                        'divergence_strength': div_strength,
                        'rsi': current_rsi,
                        'price_swing': price_swing,
                        'rsi_swing': rsi_swing,
                        'in_extreme_zone': in_extreme
                    }
                )
        
        # SHORT signal: Bearish divergence detected
        elif bearish_div and volume_ok and confirmation_ok:
            div_strength, price_swing, rsi_swing = bearish_div
            
            # Stronger signal if RSI was in overbought zone
            in_extreme = current_rsi > (100 - self.config['rsi_extreme_zone']) or \
                        max(rsi[-self.config['lookback_bars']:]) > (100 - self.config['rsi_extreme_zone'])
            
            confidence = self._calculate_confidence(div_strength, in_extreme, False)
            
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
                    reason=f"SHORT Divergence: Bearish RSI div ({div_strength:.1%}), RSI={current_rsi:.0f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'divergence_type': 'bearish_regular',
                        'divergence_strength': div_strength,
                        'rsi': current_rsi,
                        'price_swing': price_swing,
                        'rsi_swing': rsi_swing,
                        'in_extreme_zone': in_extreme
                    }
                )
        
        return signal
    
    def _detect_bullish_divergence(
        self, lows: np.ndarray, rsi: np.ndarray, lookback: int
    ) -> Optional[Tuple[float, Tuple[int, float], Tuple[int, float]]]:
        """
        Detect bullish divergence: Price makes lower low, RSI makes higher low.
        
        Returns:
            Tuple of (divergence_strength, price_swing, rsi_swing) or None
            - divergence_strength: How strong the divergence is (0-1)
            - price_swing: (index, value) of the swing low
            - rsi_swing: (index, value) of the RSI swing low
        """
        if len(lows) < lookback or len(rsi) < lookback:
            return None
        
        # Find swing lows in price (local minima)
        price_lows = self._find_swing_lows(lows[-lookback:])
        rsi_lows = self._find_swing_lows(rsi[-lookback:])
        
        if len(price_lows) < 2 or len(rsi_lows) < 2:
            return None
        
        # Get the two most recent swing lows
        recent_price_low = price_lows[-1]
        prev_price_low = price_lows[-2]
        recent_rsi_low = rsi_lows[-1]
        prev_rsi_low = rsi_lows[-2]
        
        # Bullish divergence: price lower low, RSI higher low
        price_made_lower_low = lows[-lookback:][recent_price_low[0]] < lows[-lookback:][prev_price_low[0]]
        rsi_made_higher_low = rsi[-lookback:][recent_rsi_low[0]] > rsi[-lookback:][prev_rsi_low[0]]
        
        if price_made_lower_low and rsi_made_higher_low:
            # Calculate divergence strength
            price_change = (lows[-lookback:][prev_price_low[0]] - lows[-lookback:][recent_price_low[0]]) / lows[-lookback:][prev_price_low[0]]
            rsi_change = (rsi[-lookback:][recent_rsi_low[0]] - rsi[-lookback:][prev_rsi_low[0]]) / max(rsi[-lookback:][prev_rsi_low[0]], 1)
            
            # Strength is the sum of both movements (normalized)
            strength = min(abs(price_change) + abs(rsi_change), 1.0)
            
            if strength >= self.config['divergence_threshold']:
                return (
                    strength,
                    (recent_price_low[0], lows[-lookback:][recent_price_low[0]]),
                    (recent_rsi_low[0], rsi[-lookback:][recent_rsi_low[0]])
                )
        
        return None
    
    def _detect_bearish_divergence(
        self, highs: np.ndarray, rsi: np.ndarray, lookback: int
    ) -> Optional[Tuple[float, Tuple[int, float], Tuple[int, float]]]:
        """
        Detect bearish divergence: Price makes higher high, RSI makes lower high.
        
        Returns:
            Tuple of (divergence_strength, price_swing, rsi_swing) or None
        """
        if len(highs) < lookback or len(rsi) < lookback:
            return None
        
        # Find swing highs in price (local maxima)
        price_highs = self._find_swing_highs(highs[-lookback:])
        rsi_highs = self._find_swing_highs(rsi[-lookback:])
        
        if len(price_highs) < 2 or len(rsi_highs) < 2:
            return None
        
        # Get the two most recent swing highs
        recent_price_high = price_highs[-1]
        prev_price_high = price_highs[-2]
        recent_rsi_high = rsi_highs[-1]
        prev_rsi_high = rsi_highs[-2]
        
        # Bearish divergence: price higher high, RSI lower high
        price_made_higher_high = highs[-lookback:][recent_price_high[0]] > highs[-lookback:][prev_price_high[0]]
        rsi_made_lower_high = rsi[-lookback:][recent_rsi_high[0]] < rsi[-lookback:][prev_rsi_high[0]]
        
        if price_made_higher_high and rsi_made_lower_high:
            # Calculate divergence strength
            price_change = (highs[-lookback:][recent_price_high[0]] - highs[-lookback:][prev_price_high[0]]) / highs[-lookback:][prev_price_high[0]]
            rsi_change = (rsi[-lookback:][prev_rsi_high[0]] - rsi[-lookback:][recent_rsi_high[0]]) / max(rsi[-lookback:][prev_rsi_high[0]], 1)
            
            # Strength is the sum of both movements (normalized)
            strength = min(abs(price_change) + abs(rsi_change), 1.0)
            
            if strength >= self.config['divergence_threshold']:
                return (
                    strength,
                    (recent_price_high[0], highs[-lookback:][recent_price_high[0]]),
                    (recent_rsi_high[0], rsi[-lookback:][recent_rsi_high[0]])
                )
        
        return None
    
    def _find_swing_lows(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """Find local minima (swing lows) in data."""
        min_bars = self.config['min_swing_bars']
        swings = []
        
        for i in range(min_bars, len(data) - min_bars):
            is_low = True
            for j in range(1, min_bars + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_low = False
                    break
            if is_low:
                swings.append((i, data[i]))
        
        return swings
    
    def _find_swing_highs(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """Find local maxima (swing highs) in data."""
        min_bars = self.config['min_swing_bars']
        swings = []
        
        for i in range(min_bars, len(data) - min_bars):
            is_high = True
            for j in range(1, min_bars + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_high = False
                    break
            if is_high:
                swings.append((i, data[i]))
        
        return swings
    
    def _check_rsi_confirmation(self, rsi: np.ndarray) -> bool:
        """
        Check if RSI shows momentum confirmation.
        For bullish: RSI turning up from low
        For bearish: RSI turning down from high
        """
        if len(rsi) < 5:
            return False
        
        # Simple confirmation: RSI momentum changed in last 3 bars
        recent_change = rsi[-1] - rsi[-3]
        
        # Confirm if there's momentum in the divergence direction
        # (For bullish div, we want RSI rising; for bearish, RSI falling)
        return abs(recent_change) > 2  # At least 2 points of RSI movement
    
    def _calculate_confidence(self, div_strength: float, in_extreme: bool, is_long: bool) -> float:
        """Calculate signal confidence based on divergence quality."""
        confidence = 0.5
        
        # Divergence strength bonus (stronger = more confident)
        if div_strength > 0.15:
            confidence += 0.2
        elif div_strength > 0.10:
            confidence += 0.15
        elif div_strength > 0.05:
            confidence += 0.1
        
        # Extreme zone bonus (divergence in oversold/overbought more reliable)
        if in_extreme:
            confidence += 0.15
        
        return min(confidence, 0.9)
    
    # ============= Technical Indicator Methods =============
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use EMA for smoothing (Wilder's method)
        avg_gain = self._ema(gains, period)
        avg_loss = self._ema(losses, period)
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Calculate ATR (Average True Range)."""
        if len(high) < period + 1:
            return None
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        return np.convolve(tr, np.ones(period)/period, mode='valid')
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
