"""
Ronin Momentum Breakout Strategy Agent

Ported from Ronin Trader (github.com/ronin-trader)

Core Logic:
- Detect price breaking above/below recent range
- Require volume confirmation (1.5x average)
- Target range expansion (measured move)

Why it works:
- Breakouts with volume indicate real conviction
- Range expansion targets give clear profit objectives
- Works well on low/mid cap futures with sudden moves
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)


class RoninMomentumBreakoutAgent(BaseStrategyAgent):
    """
    Ronin Momentum Breakout Strategy
    
    Entry conditions:
    - Price breaks above recent range high (long) or below range low (short)
    - Volume confirmation: current volume > 1.5x average
    
    Exit conditions:
    - Stop loss: just below/above breakout level
    - Take profit: range expansion target
    """
    
    DEFAULT_CONFIG = {
        # Breakout detection
        'lookback_period': 20,  # Periods to calculate range
        'breakout_buffer': 2,   # Ignore last N candles for range calc
        
        # Volume confirmation
        'volume_multiplier': 1.5,  # Require 1.5x avg volume
        
        # Risk management
        'stop_buffer_pct': 0.01,  # 1% buffer below breakout level
        
        # Confidence
        'base_confidence': 0.4,
        'volume_confirmed_confidence': 0.6,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='ronin-momentum-breakout',
            name='Ronin Momentum Breakout',
            symbols=symbols,
            config=merged_config
        )
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"highest_{self.config['lookback_period']}",
            f"lowest_{self.config['lookback_period']}",
            f"volume_sma_{self.config['lookback_period']}",
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                lookback = self.config['lookback_period']
                buffer = self.config['breakout_buffer']
                
                if not data or len(data.get('close', [])) < lookback + buffer + 5:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"[ronin-momentum-breakout] Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        lookback = self.config['lookback_period']
        buffer = self.config['breakout_buffer']
        
        # Get recent range (excluding last buffer candles) - Ronin logic
        range_slice_high = highs[-(lookback + buffer):-buffer]
        range_slice_low = lows[-(lookback + buffer):-buffer]
        
        range_high = np.max(range_slice_high)
        range_low = np.min(range_slice_low)
        range_size = range_high - range_low
        
        if range_size == 0:
            return None
        
        current_price = closes[-1]
        current_volume = volumes[-1]
        
        # Calculate average volume from the range period
        avg_volume = np.mean(volumes[-(lookback + buffer):-buffer])
        
        # Volume confirmation check
        volume_confirmed = current_volume > avg_volume * self.config['volume_multiplier']
        
        signal = None
        
        # Bullish breakout
        if current_price > range_high and volume_confirmed:
            # Stop just below breakout level
            stop_loss = range_high * (1 - self.config['stop_buffer_pct'])
            # Target: range expansion (measured move)
            take_profit = current_price + range_size
            
            confidence = self.config['volume_confirmed_confidence'] if volume_confirmed else self.config['base_confidence']
            
            signal = Signal(
                strategy_id=self.agent_id,
                symbol=symbol,
                side=SignalSide.LONG,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.utcnow(),
                reason=f"Bullish breakout above {range_high:.4f} with {current_volume/avg_volume:.1f}x volume",
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=1.0,
                metadata={
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_size': range_size,
                    'volume_ratio': current_volume / avg_volume,
                    'volume_confirmed': volume_confirmed,
                    'strategy_origin': 'ronin-trader'
                }
            )
        
        # Bearish breakout
        elif current_price < range_low and volume_confirmed:
            # Stop just above breakout level
            stop_loss = range_low * (1 + self.config['stop_buffer_pct'])
            # Target: range expansion (measured move)
            take_profit = current_price - range_size
            
            confidence = self.config['volume_confirmed_confidence'] if volume_confirmed else self.config['base_confidence']
            
            signal = Signal(
                strategy_id=self.agent_id,
                symbol=symbol,
                side=SignalSide.SHORT,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.utcnow(),
                reason=f"Bearish breakout below {range_low:.4f} with {current_volume/avg_volume:.1f}x volume",
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=1.0,
                metadata={
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_size': range_size,
                    'volume_ratio': current_volume / avg_volume,
                    'volume_confirmed': volume_confirmed,
                    'strategy_origin': 'ronin-trader'
                }
            )
        
        return signal
