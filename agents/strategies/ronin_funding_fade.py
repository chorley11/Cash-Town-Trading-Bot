"""
Ronin Funding Fade Strategy Agent

Ported from Ronin Trader (github.com/ronin-trader)

Core Logic:
- When funding is extremely positive (longs paying), SHORT
- When funding is extremely negative (shorts paying), LONG  
- Simple mean reversion on crowded positioning

Why it works:
- Extreme funding = crowded trade likely to unwind
- Collecting funding while fading the crowd
- Simpler than Cash Town's FundingFadeAgent - no ADX filter

Note: Cash Town already has a more sophisticated FundingFadeAgent.
This version preserves Ronin's simpler approach for comparison/backtesting.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)


class RoninFundingFadeAgent(BaseStrategyAgent):
    """
    Ronin Funding Rate Fade Strategy
    
    Entry conditions:
    - Funding rate exceeds threshold
    - Positive funding → SHORT (fade longs)
    - Negative funding → LONG (fade shorts)
    
    Exit conditions:
    - Stop loss: 1% (configurable)
    - Take profit: 3% (configurable)
    - Funding normalizes
    """
    
    DEFAULT_CONFIG = {
        # Funding thresholds (Ronin defaults)
        'funding_threshold': 0.0005,  # 0.05% absolute value
        
        # Risk management (Ronin defaults)
        'stop_loss_pct': 0.01,    # 1% stop
        'take_profit_pct': 0.03,  # 3% TP
        
        # Confidence scaling
        'base_confidence': 0.4,
        'confidence_per_bps': 100,  # Confidence boost per 1bp of funding
        'max_confidence': 0.8,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='ronin-funding-fade',
            name='Ronin Funding Fade',
            symbols=symbols,
            config=merged_config
        )
        self._funding_data: Dict[str, float] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            'funding_rate',  # Requires external futures data feed
        ]
    
    def set_funding_data(self, funding_data: Dict[str, float]):
        """Update funding rates from external data feed"""
        self._funding_data = funding_data
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 10:
                    continue
                
                # Get funding rate for this symbol
                funding_rate = self._funding_data.get(symbol, 0)
                
                signal = self._analyze_symbol(symbol, data, funding_rate)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"[ronin-funding-fade] Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict, funding_rate: float) -> Optional[Signal]:
        threshold = self.config['funding_threshold']
        
        # Check if funding is extreme enough
        if abs(funding_rate) < threshold:
            return None
        
        closes = np.array(data['close'])
        current_price = closes[-1]
        
        if current_price == 0:
            return None
        
        # Ronin logic: fade the funding
        # Positive funding = longs paying = too many longs = SHORT
        # Negative funding = shorts paying = too many shorts = LONG
        side = SignalSide.SHORT if funding_rate > 0 else SignalSide.LONG
        
        # Calculate confidence based on funding extremeness
        confidence = self.config['base_confidence'] + \
                    abs(funding_rate) * self.config['confidence_per_bps']
        confidence = min(confidence, self.config['max_confidence'])
        
        # Calculate stop/take profit
        stop_pct = self.config['stop_loss_pct']
        tp_pct = self.config['take_profit_pct']
        
        if side == SignalSide.LONG:
            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + tp_pct)
        else:
            stop_loss = current_price * (1 + stop_pct)
            take_profit = current_price * (1 - tp_pct)
        
        return Signal(
            strategy_id=self.agent_id,
            symbol=symbol,
            side=side,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.utcnow(),
            reason=f"Extreme funding {funding_rate*100:.3f}% - fading crowd",
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=1.0,
            metadata={
                'funding_rate': funding_rate,
                'funding_pct': funding_rate * 100,
                'strategy_origin': 'ronin-trader'
            }
        )
    
    def should_exit(self, position: 'Position', current_price: float,
                   current_funding: float = None) -> Optional[str]:
        """Check for exit including funding normalization"""
        # Standard stop/take profit from base
        base_exit = super().should_exit(position, current_price)
        if base_exit:
            return base_exit
        
        # Ronin: exit if funding normalizes
        if current_funding is not None:
            if abs(current_funding) < self.config['funding_threshold'] / 2:
                return 'funding_normalized'
        
        return None
