"""
Ronin Volume Spike Strategy Agent

Ported from Ronin Trader (github.com/ronin-trader)

Core Logic:
- Detect when current volume exceeds average by threshold multiplier
- Trade in the direction of the accompanying price move
- Volume spikes indicate institutional interest or news events

Why it works:
- Unusual volume = something happening
- Following the volume direction captures momentum
- Works best for catching early moves in low/mid cap futures
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)


class RoninVolumeSpikeAgent(BaseStrategyAgent):
    """
    Ronin Volume Spike Strategy
    
    Entry conditions:
    - Current volume >= threshold * average volume (20-period)
    - Trade direction follows price direction
    
    Exit conditions:
    - Stop loss: 1% (configurable)
    - Take profit: 3% (configurable, 3:1 R:R)
    - Time-based exit after max_hold_candles
    """
    
    DEFAULT_CONFIG = {
        # Volume spike detection
        'volume_threshold': 3.0,  # Volume must be 3x average
        'lookback_period': 20,    # Periods for average volume calculation
        
        # Risk management (Ronin defaults)
        'stop_loss_pct': 0.01,    # 1% stop
        'take_profit_pct': 0.03,  # 3% TP (3:1 R:R)
        
        # Confidence scaling
        'base_confidence': 0.5,
        'confidence_per_ratio': 0.1,  # Add per 1x above threshold
        'max_confidence': 0.9,
        
        # Optional filters
        'min_price_change_pct': 0.001,  # Minimum price move to confirm direction
        'max_hold_candles': 24,  # Maximum hold time in candles
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='ronin-volume-spike',
            name='Ronin Volume Spike',
            symbols=symbols,
            config=merged_config
        )
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"volume_sma_{self.config['lookback_period']}",
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < self.config['lookback_period'] + 5:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"[ronin-volume-spike] Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        volumes = np.array(data['volume'])
        
        lookback = self.config['lookback_period']
        threshold = self.config['volume_threshold']
        
        # Calculate average volume (excluding current candle)
        avg_volume = np.mean(volumes[-(lookback+1):-1])
        current_volume = volumes[-1]
        
        if avg_volume == 0:
            return None
        
        volume_ratio = current_volume / avg_volume
        
        # Check for volume spike
        if volume_ratio < threshold:
            return None
        
        # Determine direction from price change
        current_close = closes[-1]
        prev_close = closes[-2]
        price_change = (current_close - prev_close) / prev_close
        
        # Skip if price move too small to determine direction
        if abs(price_change) < self.config['min_price_change_pct']:
            logger.debug(f"[ronin-volume-spike] {symbol}: Volume spike but unclear direction")
            return None
        
        # Go with momentum direction (Ronin logic)
        side = SignalSide.LONG if price_change > 0 else SignalSide.SHORT
        
        # Calculate confidence based on volume ratio
        confidence = self.config['base_confidence'] + \
                    (volume_ratio - threshold) * self.config['confidence_per_ratio']
        confidence = min(confidence, self.config['max_confidence'])
        
        # Calculate stop/take profit (Ronin style)
        stop_pct = self.config['stop_loss_pct']
        tp_pct = self.config['take_profit_pct']
        
        if side == SignalSide.LONG:
            stop_loss = current_close * (1 - stop_pct)
            take_profit = current_close * (1 + tp_pct)
        else:
            stop_loss = current_close * (1 + stop_pct)
            take_profit = current_close * (1 - tp_pct)
        
        return Signal(
            strategy_id=self.agent_id,
            symbol=symbol,
            side=side,
            confidence=confidence,
            price=current_close,
            timestamp=datetime.utcnow(),
            reason=f"Volume spike {volume_ratio:.1f}x avg, price {'+' if price_change > 0 else ''}{price_change*100:.2f}%",
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=1.0,
            metadata={
                'volume_ratio': volume_ratio,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'price_change_pct': price_change * 100,
                'strategy_origin': 'ronin-trader'
            }
        )
