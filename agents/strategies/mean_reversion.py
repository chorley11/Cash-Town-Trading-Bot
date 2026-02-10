"""
Mean Reversion Strategy Agent
Trades pullbacks to the mean using Bollinger Bands and RSI.

Core Logic:
- Enter when price touches Bollinger Band extremes
- RSI confirms oversold/overbought
- Exit at middle band or opposite extreme
- Works best in ranging markets
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class MeanReversionAgent(BaseStrategyAgent):
    """
    Mean Reversion Strategy
    
    Entry conditions:
    - Price at/beyond Bollinger Band (2 std dev)
    - RSI confirms extreme (< 30 or > 70)
    - Not in strong trend (ADX < 25)
    
    Exit conditions:
    - Price returns to middle band (SMA)
    - Opposite band touched
    - Stop loss hit
    """
    
    DEFAULT_CONFIG = {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'adx_period': 14,
        'adx_max': 25,  # Only trade when ADX below this (ranging market)
        'atr_period': 14,
        'stop_loss_atr': 1.5,
        'take_profit_target': 'middle_band',  # or 'opposite_band'
        'min_confidence': 0.55,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='mean-reversion',
            name='Mean Reversion',
            symbols=symbols,
            config=merged_config
        )
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"bb_{self.config['bb_period']}_{self.config['bb_std']}",
            f"rsi_{self.config['rsi_period']}",
            f"adx_{self.config['adx_period']}",
            f"atr_{self.config['atr_period']}"
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
        
        current_price = closes[-1]
        
        # Calculate indicators
        bb_middle, bb_upper, bb_lower = self._bollinger_bands(
            closes, 
            self.config['bb_period'], 
            self.config['bb_std']
        )
        rsi = self._rsi(closes, self.config['rsi_period'])
        adx = self._adx(highs, lows, closes, self.config['adx_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        
        if bb_middle is None or rsi is None or atr is None:
            return None
        
        current_bb_middle = bb_middle[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_lower = bb_lower[-1]
        current_rsi = rsi[-1]
        current_adx = adx[-1] if adx is not None else 20
        current_atr = atr[-1]
        
        # Check if market is ranging (not trending)
        is_ranging = current_adx < self.config['adx_max']
        
        if not is_ranging:
            return None  # Skip trending markets
        
        # Calculate band position
        band_width = current_bb_upper - current_bb_lower
        price_position = (current_price - current_bb_lower) / band_width  # 0 = lower, 1 = upper
        
        signal = None
        
        # LONG signal: Price at lower band + RSI oversold
        if current_price <= current_bb_lower and current_rsi < self.config['rsi_oversold']:
            confidence = self._calculate_confidence(current_rsi, price_position, current_adx, True)
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr'])
                take_profit = current_bb_middle  # Target middle band
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG MeanRev: At lower BB, RSI={current_rsi:.0f}, ADX={current_adx:.0f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'bb_middle': current_bb_middle,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'rsi': current_rsi,
                        'adx': current_adx,
                        'band_position': price_position
                    }
                )
        
        # SHORT signal: Price at upper band + RSI overbought
        elif current_price >= current_bb_upper and current_rsi > self.config['rsi_overbought']:
            confidence = self._calculate_confidence(current_rsi, price_position, current_adx, False)
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr'])
                take_profit = current_bb_middle  # Target middle band
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT MeanRev: At upper BB, RSI={current_rsi:.0f}, ADX={current_adx:.0f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'bb_middle': current_bb_middle,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'rsi': current_rsi,
                        'adx': current_adx,
                        'band_position': price_position
                    }
                )
        
        return signal
    
    def _calculate_confidence(self, rsi: float, band_position: float, adx: float, is_long: bool) -> float:
        confidence = 0.5
        
        # RSI extremity bonus
        if is_long:
            if rsi < 20:
                confidence += 0.2
            elif rsi < 25:
                confidence += 0.15
            elif rsi < 30:
                confidence += 0.1
        else:
            if rsi > 80:
                confidence += 0.2
            elif rsi > 75:
                confidence += 0.15
            elif rsi > 70:
                confidence += 0.1
        
        # Band extremity bonus
        if is_long and band_position < 0:  # Below lower band
            confidence += 0.1
        elif not is_long and band_position > 1:  # Above upper band
            confidence += 0.1
        
        # Ranging market bonus (lower ADX = better for mean reversion)
        if adx < 15:
            confidence += 0.1
        elif adx < 20:
            confidence += 0.05
        
        return min(confidence, 0.9)
    
    def _bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float):
        if len(prices) < period:
            return None, None, None
        
        middle = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # Calculate rolling std
        stds = []
        for i in range(len(prices) - period + 1):
            stds.append(np.std(prices[i:i+period]))
        stds = np.array(stds)
        
        upper = middle + (stds * std_dev)
        lower = middle - (stds * std_dev)
        
        return middle, upper, lower
    
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
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.convolve(tr, np.ones(period)/period, mode='valid')
    
    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        if len(high) < period * 2:
            return None
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = ema(tr, period)
        plus_di = 100 * ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * ema(minus_dm, period) / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return ema(dx, period)
