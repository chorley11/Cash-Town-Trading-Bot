"""
Volatility Breakout Strategy Agent (Bollinger Squeeze)

Core Logic:
- Detect low volatility compression (tight Bollinger Bands)
- Trade breakouts when volatility expands
- Use Keltner Channels for squeeze confirmation
- ATR for stops and position sizing

Why it works:
- Volatility is mean reverting - compression precedes expansion
- Breakouts from compression often have strong momentum
- Squeeze patterns are one of the most reliable setup patterns
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class VolatilityBreakoutAgent(BaseStrategyAgent):
    """
    Volatility Breakout Strategy (Bollinger Squeeze)
    
    Setup conditions (squeeze):
    - Bollinger Bands inside Keltner Channels
    - ATR at low relative to recent history
    - Price consolidating
    
    Entry conditions:
    - Break outside Bollinger Bands
    - Momentum confirmation
    - Volume expansion
    
    Exit conditions:
    - Opposite band touch
    - Momentum fades
    - Stop loss / Take profit hit
    """
    
    DEFAULT_CONFIG = {
        # Bollinger Bands
        'bb_period': 20,
        'bb_std': 2.0,
        
        # Keltner Channels
        'kc_period': 20,
        'kc_atr_mult': 1.5,
        
        # Squeeze detection
        'squeeze_lookback': 6,  # Candles Bollinger must be inside Keltner
        'atr_lookback': 50,
        'atr_percentile': 25,  # ATR must be below this percentile
        
        # Breakout confirmation
        'breakout_threshold': 0.5,  # % beyond band to confirm
        'volume_threshold': 1.5,  # Volume vs avg
        'momentum_period': 5,
        'momentum_threshold': 0.8,  # % momentum to confirm
        
        # Risk management
        'atr_period': 14,
        'stop_loss_atr': 1.5,
        'take_profit_atr': 3.0,
        
        # Confidence
        'min_confidence': 0.60,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='volatility-breakout',
            name='Volatility Breakout',
            symbols=symbols,
            config=merged_config
        )
        # Track squeeze state
        self._squeeze_count: Dict[str, int] = {}  # symbol -> candles in squeeze
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"bbands_{self.config['bb_period']}_{self.config['bb_std']}",
            f"keltner_{self.config['kc_period']}_{self.config['kc_atr_mult']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20',
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < self.config['atr_lookback']:
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
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(
            closes, self.config['bb_period'], self.config['bb_std']
        )
        
        # Calculate Keltner Channels
        kc_upper, kc_middle, kc_lower = self._keltner_channels(
            highs, lows, closes,
            self.config['kc_period'], self.config['kc_atr_mult']
        )
        
        # Calculate ATR
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if bb_upper is None or kc_upper is None or atr is None or volume_sma is None:
            return None
        
        # Get current values
        curr_bb_upper = bb_upper[-1]
        curr_bb_lower = bb_lower[-1]
        curr_kc_upper = kc_upper[-1]
        curr_kc_lower = kc_lower[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Check if in squeeze (Bollinger inside Keltner)
        in_squeeze = curr_bb_lower > curr_kc_lower and curr_bb_upper < curr_kc_upper
        
        # Track squeeze duration
        if symbol not in self._squeeze_count:
            self._squeeze_count[symbol] = 0
        
        if in_squeeze:
            self._squeeze_count[symbol] += 1
        else:
            # Check for breakout from squeeze
            was_in_squeeze = self._squeeze_count[symbol] >= self.config['squeeze_lookback']
            self._squeeze_count[symbol] = 0
            
            if was_in_squeeze:
                # Potential breakout - analyze direction
                return self._analyze_breakout(
                    symbol, closes, highs, lows, volumes,
                    curr_bb_upper, curr_bb_lower, current_atr,
                    current_volume, avg_volume
                )
        
        return None
    
    def _analyze_breakout(self, symbol: str, closes: np.ndarray,
                         highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray,
                         bb_upper: float, bb_lower: float, current_atr: float,
                         current_volume: float, avg_volume: float) -> Optional[Signal]:
        """Analyze potential breakout from squeeze"""
        current_price = closes[-1]
        prev_price = closes[-2]
        
        # Calculate momentum
        momentum_period = self.config['momentum_period']
        momentum = (current_price - closes[-momentum_period]) / closes[-momentum_period] * 100
        
        # Check volume expansion
        volume_ratio = current_volume / avg_volume
        volume_ok = volume_ratio >= self.config['volume_threshold']
        
        # ATR percentile check (should be expanding from low)
        atr_history = self._atr(highs, lows, closes, self.config['atr_period'])
        if atr_history is not None and len(atr_history) > self.config['atr_lookback']:
            recent_atr = atr_history[-self.config['atr_lookback']:]
            atr_percentile = np.percentile(recent_atr, self.config['atr_percentile'])
            atr_was_low = atr_history[-2] <= atr_percentile  # Previous was low
        else:
            atr_was_low = True  # Assume OK if not enough data
        
        signal = None
        
        # BULLISH BREAKOUT: Price closes above upper band
        breakout_threshold = bb_upper * (1 + self.config['breakout_threshold'] / 100)
        if current_price > bb_upper and prev_price <= bb_upper:
            if momentum > self.config['momentum_threshold'] and volume_ok:
                confidence = self._calculate_confidence(
                    momentum, volume_ratio, atr_was_low, True
                )
                
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
                        reason=f"LONG Breakout: Squeeze release upward, momentum +{momentum:.1f}%, vol {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'bb_upper': bb_upper,
                            'momentum': momentum,
                            'volume_ratio': volume_ratio,
                            'squeeze_candles': self.config['squeeze_lookback']
                        }
                    )
        
        # BEARISH BREAKOUT: Price closes below lower band
        elif current_price < bb_lower and prev_price >= bb_lower:
            if momentum < -self.config['momentum_threshold'] and volume_ok:
                confidence = self._calculate_confidence(
                    abs(momentum), volume_ratio, atr_was_low, False
                )
                
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
                        reason=f"SHORT Breakout: Squeeze release downward, momentum {momentum:.1f}%, vol {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'bb_lower': bb_lower,
                            'momentum': momentum,
                            'volume_ratio': volume_ratio,
                            'squeeze_candles': self.config['squeeze_lookback']
                        }
                    )
        
        return signal
    
    def _calculate_confidence(self, momentum: float, volume_ratio: float,
                             atr_was_low: bool, is_long: bool) -> float:
        confidence = 0.5
        
        # Momentum strength
        if momentum > 2:
            confidence += 0.15
        elif momentum > 1.5:
            confidence += 0.1
        elif momentum > 1:
            confidence += 0.05
        
        # Volume confirmation
        if volume_ratio > 2.5:
            confidence += 0.15
        elif volume_ratio > 2:
            confidence += 0.1
        elif volume_ratio > 1.5:
            confidence += 0.05
        
        # ATR was compressed (good squeeze)
        if atr_was_low:
            confidence += 0.1
        
        return min(confidence, 0.90)
    
    def _bollinger_bands(self, data: np.ndarray, period: int, std_mult: float):
        """Calculate Bollinger Bands"""
        if len(data) < period:
            return None, None, None
        
        middle = self._sma(data, period)
        if middle is None:
            return None, None, None
        
        # Calculate rolling std
        rolling_std = np.zeros(len(data) - period + 1)
        for i in range(len(rolling_std)):
            rolling_std[i] = np.std(data[i:i+period])
        
        upper = middle + (rolling_std * std_mult)
        lower = middle - (rolling_std * std_mult)
        
        return upper, middle, lower
    
    def _keltner_channels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         period: int, atr_mult: float):
        """Calculate Keltner Channels"""
        middle = self._ema(close, period)
        atr = self._atr(high, low, close, period)
        
        if middle is None or atr is None:
            return None, None, None
        
        # Align lengths
        min_len = min(len(middle), len(atr))
        middle = middle[-min_len:]
        atr = atr[-min_len:]
        
        upper = middle + (atr * atr_mult)
        lower = middle - (atr * atr_mult)
        
        return upper, middle, lower
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
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
