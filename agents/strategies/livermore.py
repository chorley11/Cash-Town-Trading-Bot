"""
Livermore Trading Strategy Agent
Based on Jesse Livermore's trading principles from "Reminiscences of a Stock Operator"

Core Logic:
- Trade with the primary trend
- Wait for pivotal points (key price levels)
- Add to winning positions, cut losers quickly
- Track "line of least resistance"
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class LivermoreAgent(BaseStrategyAgent):
    """
    Livermore Trading Strategy
    
    Key concepts:
    - Pivotal Points: Key price levels where direction changes
    - Line of Least Resistance: The direction price wants to go
    - Pyramiding: Add to winners, never average down
    - Time: Wait for the right moment, don't force trades
    """
    
    DEFAULT_CONFIG = {
        'pivot_lookback': 20,       # Periods to identify pivots
        'trend_period': 50,         # For primary trend
        'momentum_period': 14,
        'atr_period': 14,
        'breakout_threshold': 0.02, # 2% beyond pivot
        'stop_loss_atr': 1.5,
        'take_profit_atr': 4.0,
        'min_confidence': 0.55,
        'volume_confirmation': True,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='livermore',
            name='Livermore Method',
            symbols=symbols,
            config=merged_config
        )
        self._pivots: Dict[str, dict] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['trend_period']}",
            f"rsi_{self.config['momentum_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 60:
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
        
        # Primary trend
        trend_ma = self._sma(closes, self.config['trend_period'])
        if trend_ma is None:
            return None
        
        primary_trend = 'bullish' if current_price > trend_ma[-1] else 'bearish'
        
        # Find pivotal points
        pivot_high, pivot_low = self._find_pivots(highs, lows, self.config['pivot_lookback'])
        
        # Momentum
        rsi = self._rsi(closes, self.config['momentum_period'])
        current_rsi = rsi[-1] if rsi is not None else 50
        
        # ATR for stops
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        current_atr = atr[-1] if atr is not None else current_price * 0.02
        
        # Volume
        volume_sma = self._sma(volumes, 20)
        volume_ratio = volumes[-1] / volume_sma[-1] if volume_sma is not None else 1.0
        
        # Determine line of least resistance
        breakout_threshold = pivot_high * (1 + self.config['breakout_threshold'])
        breakdown_threshold = pivot_low * (1 - self.config['breakout_threshold'])
        
        signal = None
        
        # LONG: Bullish trend + breaking above pivotal high
        if primary_trend == 'bullish' and current_price > breakout_threshold:
            if not self.config['volume_confirmation'] or volume_ratio > 1.2:
                confidence = self._calculate_confidence(
                    current_price, pivot_high, current_rsi, volume_ratio, True
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
                        reason=f"LONG Livermore: Pivotal breakout ${pivot_high:.4f}, trend bullish, volume {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'pivot_high': pivot_high,
                            'pivot_low': pivot_low,
                            'primary_trend': primary_trend,
                            'rsi': current_rsi,
                            'volume_ratio': volume_ratio
                        }
                    )
        
        # SHORT: Bearish trend + breaking below pivotal low
        elif primary_trend == 'bearish' and current_price < breakdown_threshold:
            if not self.config['volume_confirmation'] or volume_ratio > 1.2:
                confidence = self._calculate_confidence(
                    current_price, pivot_low, current_rsi, volume_ratio, False
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
                        reason=f"SHORT Livermore: Pivotal breakdown ${pivot_low:.4f}, trend bearish, volume {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'pivot_high': pivot_high,
                            'pivot_low': pivot_low,
                            'primary_trend': primary_trend,
                            'rsi': current_rsi,
                            'volume_ratio': volume_ratio
                        }
                    )
        
        # Store pivots
        self._pivots[symbol] = {'high': pivot_high, 'low': pivot_low}
        
        return signal
    
    def _find_pivots(self, highs: np.ndarray, lows: np.ndarray, lookback: int):
        """Find recent pivot high and low"""
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        pivot_high = np.max(recent_highs)
        pivot_low = np.min(recent_lows)
        
        return pivot_high, pivot_low
    
    def _calculate_confidence(self, price: float, pivot: float, rsi: float, 
                             volume_ratio: float, is_long: bool) -> float:
        confidence = 0.5
        
        # Breakout strength
        breakout_pct = abs(price - pivot) / pivot * 100
        if breakout_pct > 5:
            confidence += 0.15
        elif breakout_pct > 3:
            confidence += 0.1
        elif breakout_pct > 2:
            confidence += 0.05
        
        # RSI confirmation
        if is_long and 50 < rsi < 70:
            confidence += 0.1
        elif not is_long and 30 < rsi < 50:
            confidence += 0.1
        
        # Volume
        if volume_ratio > 2.0:
            confidence += 0.15
        elif volume_ratio > 1.5:
            confidence += 0.1
        
        return min(confidence, 0.85)
    
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
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.convolve(tr, np.ones(period)/period, mode='valid')
