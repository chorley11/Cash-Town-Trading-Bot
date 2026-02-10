"""
Turtle Trading Strategy Agent
Based on the legendary Turtle Trading system by Richard Dennis.

Core Logic:
- Enter on 20-day breakout (price exceeds 20-day high/low)
- Add to winning positions (pyramiding)
- Exit on 10-day breakout in opposite direction
- ATR-based position sizing and stops
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class TurtleAgent(BaseStrategyAgent):
    """
    Turtle Trading Strategy
    
    Entry (System 1):
    - Long: Price breaks above 20-day high
    - Short: Price breaks below 20-day low
    
    Exit:
    - Long: Price breaks below 10-day low
    - Short: Price breaks above 10-day high
    
    Position sizing:
    - 1 unit = 1% of account per 1 ATR move
    - Max 4 units per market
    """
    
    DEFAULT_CONFIG = {
        'entry_period': 20,      # Days for entry breakout
        'exit_period': 10,       # Days for exit breakout
        'atr_period': 20,
        'stop_loss_atr': 2.0,    # 2N stop
        'pyramid_threshold': 0.5, # Add position every 0.5 ATR
        'max_units': 4,
        'unit_risk_pct': 1.0,    # Risk 1% per unit
        'min_confidence': 0.5,
        'require_volume': True,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='turtle',
            name='Turtle Trading',
            symbols=symbols,
            config=merged_config
        )
        self._last_breakout: Dict[str, dict] = {}  # Track last breakout per symbol
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"high_{self.config['entry_period']}",
            f"low_{self.config['entry_period']}",
            f"high_{self.config['exit_period']}",
            f"low_{self.config['exit_period']}",
            f"atr_{self.config['atr_period']}"
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 30:
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
        
        # Calculate Donchian Channels
        entry_high = self._rolling_max(highs, self.config['entry_period'])
        entry_low = self._rolling_min(lows, self.config['entry_period'])
        exit_high = self._rolling_max(highs, self.config['exit_period'])
        exit_low = self._rolling_min(lows, self.config['exit_period'])
        
        # ATR for position sizing
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        
        if entry_high is None or atr is None:
            return None
        
        # Get previous values (excluding current bar)
        prev_entry_high = entry_high[-2]
        prev_entry_low = entry_low[-2]
        current_atr = atr[-1]
        
        # Check for breakouts
        long_breakout = current_price > prev_entry_high
        short_breakout = current_price < prev_entry_low
        
        # Volume confirmation
        volume_ok = True
        if self.config['require_volume']:
            volume_sma = self._sma(volumes, 20)
            if volume_sma is not None:
                volume_ok = volumes[-1] > volume_sma[-1]
        
        signal = None
        
        # LONG signal: 20-day high breakout
        if long_breakout and volume_ok:
            confidence = self._calculate_confidence(
                current_price, prev_entry_high, current_atr, True
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr'])
                # Turtle doesn't use fixed take profit - trails with exit breakout
                take_profit = current_price + (current_atr * 4)  # Approximate
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG Turtle: {self.config['entry_period']}-day high breakout at {prev_entry_high:.4f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'breakout_level': prev_entry_high,
                        'atr': current_atr,
                        'exit_low': exit_low[-1] if exit_low is not None else None
                    }
                )
                
                self._last_breakout[symbol] = {
                    'side': 'long',
                    'price': current_price,
                    'time': datetime.utcnow()
                }
        
        # SHORT signal: 20-day low breakout
        elif short_breakout and volume_ok:
            confidence = self._calculate_confidence(
                current_price, prev_entry_low, current_atr, False
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr'])
                take_profit = current_price - (current_atr * 4)
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT Turtle: {self.config['entry_period']}-day low breakout at {prev_entry_low:.4f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'breakout_level': prev_entry_low,
                        'atr': current_atr,
                        'exit_high': exit_high[-1] if exit_high is not None else None
                    }
                )
                
                self._last_breakout[symbol] = {
                    'side': 'short',
                    'price': current_price,
                    'time': datetime.utcnow()
                }
        
        return signal
    
    def _calculate_confidence(self, price: float, breakout_level: float, atr: float, is_long: bool) -> float:
        confidence = 0.55  # Base confidence for breakouts
        
        # Strength of breakout (how far past the level)
        breakout_distance = abs(price - breakout_level) / atr
        if breakout_distance > 0.5:
            confidence += 0.15
        elif breakout_distance > 0.25:
            confidence += 0.1
        
        return min(confidence, 0.85)
    
    def _rolling_max(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        result = np.zeros(len(data) - period + 1)
        for i in range(len(result)):
            result[i] = np.max(data[i:i+period])
        return result
    
    def _rolling_min(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        result = np.zeros(len(data) - period + 1)
        for i in range(len(result)):
            result[i] = np.min(data[i:i+period])
        return result
    
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
