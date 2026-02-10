"""
BTS Lynch Strategy Agent
Based on Peter Lynch's "Buy The Strong" principles adapted for crypto.

Core Logic:
- Identify strong downtrends (for shorts) or uptrends (for longs)
- Enter on support/resistance breakdowns/breakouts
- Volume confirmation required
- Trend-following with momentum
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide, Position

logger = logging.getLogger(__name__)

class BTSLynchAgent(BaseStrategyAgent):
    """
    BTS Lynch Strategy - Buy The Strong, Sell The Weak
    
    Generates signals based on:
    - Trend direction (20/50 SMA)
    - Support/Resistance breaks
    - Volume confirmation
    - Momentum (RSI)
    """
    
    DEFAULT_CONFIG = {
        'sma_fast': 20,
        'sma_slow': 50,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volume_threshold': 1.2,  # 20% above average
        'atr_period': 14,
        'stop_loss_atr_mult': 2.0,
        'take_profit_atr_mult': 3.0,
        'min_confidence': 0.6,
        'lookback_periods': 100
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='bts-lynch',
            name='BTS Lynch Strategy',
            symbols=symbols,
            config=merged_config
        )
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['sma_fast']}",
            f"sma_{self.config['sma_slow']}",
            f"rsi_{self.config['rsi_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate BTS Lynch signals for all symbols.
        
        Args:
            market_data: Dict with symbol -> OHLCV + indicators data
        """
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < self.config['lookback_periods']:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Analyze single symbol for BTS Lynch setup"""
        
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        sma_fast = self._sma(closes, self.config['sma_fast'])
        sma_slow = self._sma(closes, self.config['sma_slow'])
        rsi = self._rsi(closes, self.config['rsi_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if sma_fast is None or sma_slow is None or rsi is None or atr is None:
            return None
        
        # Current values
        current_sma_fast = sma_fast[-1]
        current_sma_slow = sma_slow[-1]
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Determine trend
        is_downtrend = current_sma_fast < current_sma_slow and current_price < current_sma_fast
        is_uptrend = current_sma_fast > current_sma_slow and current_price > current_sma_fast
        
        # Volume confirmation
        volume_spike = current_volume > avg_volume * self.config['volume_threshold']
        
        # Support/Resistance analysis
        recent_lows = lows[-20:]
        recent_highs = highs[-20:]
        support = np.min(recent_lows)
        resistance = np.max(recent_highs)
        
        # Check for breakdown/breakout
        support_breakdown = current_price < support and closes[-2] >= support
        resistance_breakout = current_price > resistance and closes[-2] <= resistance
        
        signal = None
        
        # SHORT signal: Downtrend + Support breakdown + Volume
        if is_downtrend and support_breakdown and volume_spike:
            confidence = self._calculate_confidence(
                trend_aligned=True,
                volume_confirmed=True,
                rsi_value=current_rsi,
                is_short=True
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr_mult'])
                take_profit = current_price - (current_atr * self.config['take_profit_atr_mult'])
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT BTS Lynch: Downtrend + support breakdown + volume spike ({current_volume/avg_volume:.1f}x)",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'sma_fast': current_sma_fast,
                        'sma_slow': current_sma_slow,
                        'rsi': current_rsi,
                        'atr': current_atr,
                        'volume_ratio': current_volume / avg_volume
                    }
                )
        
        # LONG signal: Uptrend + Resistance breakout + Volume
        elif is_uptrend and resistance_breakout and volume_spike:
            confidence = self._calculate_confidence(
                trend_aligned=True,
                volume_confirmed=True,
                rsi_value=current_rsi,
                is_short=False
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr_mult'])
                take_profit = current_price + (current_atr * self.config['take_profit_atr_mult'])
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG BTS Lynch: Uptrend + resistance breakout + volume spike ({current_volume/avg_volume:.1f}x)",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'sma_fast': current_sma_fast,
                        'sma_slow': current_sma_slow,
                        'rsi': current_rsi,
                        'atr': current_atr,
                        'volume_ratio': current_volume / avg_volume
                    }
                )
        
        return signal
    
    def _calculate_confidence(
        self,
        trend_aligned: bool,
        volume_confirmed: bool,
        rsi_value: float,
        is_short: bool
    ) -> float:
        """Calculate signal confidence score"""
        confidence = 0.5  # Base
        
        if trend_aligned:
            confidence += 0.15
        
        if volume_confirmed:
            confidence += 0.15
        
        # RSI bonus
        if is_short and rsi_value > 50:  # Shorting when RSI still high = better
            confidence += 0.1
        elif not is_short and rsi_value < 50:  # Longing when RSI still low = better
            confidence += 0.1
        
        # Extreme RSI = higher confidence
        if is_short and rsi_value > self.config['rsi_overbought']:
            confidence += 0.1
        elif not is_short and rsi_value < self.config['rsi_oversold']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    # Technical indicator calculations
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
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        
        return atr
