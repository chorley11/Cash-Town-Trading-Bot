"""
Trend Following Strategy Agent
Classic trend-following using moving average crossovers and momentum.

Core Logic:
- MA crossover for trend direction (fast crosses slow)
- ADX for trend strength confirmation
- ATR-based position sizing and stops
- Only trade when trend is strong (ADX > 25)
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class TrendFollowingAgent(BaseStrategyAgent):
    """
    Trend Following Strategy
    
    Entry conditions:
    - Fast MA crosses above/below slow MA
    - ADX > 25 (strong trend)
    - Price above/below both MAs
    
    Exit conditions:
    - Opposite MA crossover
    - ADX drops below 20
    - Stop loss / Take profit hit
    """
    
    DEFAULT_CONFIG = {
        'ma_fast': 10,
        'ma_slow': 30,
        'adx_period': 14,
        'adx_threshold': 25,
        'adx_exit_threshold': 20,
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 4.0,
        'min_confidence': 0.55,
        'require_volume_confirmation': True,
        'volume_threshold': 1.1,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='trend-following',
            name='Trend Following',
            symbols=symbols,
            config=merged_config
        )
        self._prev_ma_state: Dict[str, str] = {}  # Track previous MA relationship
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['ma_fast']}",
            f"sma_{self.config['ma_slow']}",
            f"adx_{self.config['adx_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        analyzed = 0
        skipped_no_data = 0
        skipped_conditions = 0
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    skipped_no_data += 1
                    continue
                
                analyzed += 1
                signal, reason = self._analyze_symbol_debug(symbol, data)
                if signal:
                    signals.append(signal)
                else:
                    skipped_conditions += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        if analyzed > 0:
            logger.debug(f"[trend-following] Analyzed {analyzed}, skipped {skipped_no_data} (no data), {skipped_conditions} (conditions not met)")
        
        return signals
    
    def _analyze_symbol_debug(self, symbol: str, data: Dict):
        """Wrapper to add debug info"""
        signal = self._analyze_symbol(symbol, data)
        if signal:
            return signal, "signal_generated"
        return None, "conditions_not_met"
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        ma_fast = self._sma(closes, self.config['ma_fast'])
        ma_slow = self._sma(closes, self.config['ma_slow'])
        adx = self._adx(highs, lows, closes, self.config['adx_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if ma_fast is None or ma_slow is None or adx is None or atr is None:
            return None
        
        current_ma_fast = ma_fast[-1]
        current_ma_slow = ma_slow[-1]
        prev_ma_fast = ma_fast[-2] if len(ma_fast) > 1 else current_ma_fast
        prev_ma_slow = ma_slow[-2] if len(ma_slow) > 1 else current_ma_slow
        current_adx = adx[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1] if volume_sma is not None else current_volume
        
        # Determine current and previous MA state
        current_state = 'bullish' if current_ma_fast > current_ma_slow else 'bearish'
        prev_state = 'bullish' if prev_ma_fast > prev_ma_slow else 'bearish'
        
        # Detect crossover
        bullish_cross = prev_state == 'bearish' and current_state == 'bullish'
        bearish_cross = prev_state == 'bullish' and current_state == 'bearish'
        
        # Check ADX for trend strength
        strong_trend = current_adx >= self.config['adx_threshold']
        
        # Volume confirmation
        volume_ok = not self.config['require_volume_confirmation'] or \
                   (current_volume > avg_volume * self.config['volume_threshold'])
        
        signal = None
        
        # LONG signal: Bullish crossover + strong trend
        if bullish_cross and strong_trend and volume_ok and current_price > current_ma_fast:
            confidence = self._calculate_confidence(current_adx, current_volume / avg_volume, True)
            
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
                    reason=f"LONG Trend: Bullish MA cross, ADX={current_adx:.1f}, volume {current_volume/avg_volume:.1f}x",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'ma_fast': current_ma_fast,
                        'ma_slow': current_ma_slow,
                        'adx': current_adx,
                        'atr': current_atr
                    }
                )
        
        # SHORT signal: Bearish crossover + strong trend
        elif bearish_cross and strong_trend and volume_ok and current_price < current_ma_fast:
            confidence = self._calculate_confidence(current_adx, current_volume / avg_volume, False)
            
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
                    reason=f"SHORT Trend: Bearish MA cross, ADX={current_adx:.1f}, volume {current_volume/avg_volume:.1f}x",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'ma_fast': current_ma_fast,
                        'ma_slow': current_ma_slow,
                        'adx': current_adx,
                        'atr': current_atr
                    }
                )
        
        # Update state tracking
        self._prev_ma_state[symbol] = current_state
        
        return signal
    
    def _calculate_confidence(self, adx: float, volume_ratio: float, is_long: bool) -> float:
        confidence = 0.5
        
        # ADX strength bonus
        if adx >= 40:
            confidence += 0.2
        elif adx >= 30:
            confidence += 0.15
        elif adx >= 25:
            confidence += 0.1
        
        # Volume bonus
        if volume_ratio > 1.5:
            confidence += 0.15
        elif volume_ratio > 1.2:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
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
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
