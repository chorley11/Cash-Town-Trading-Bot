"""
Trailing Stop Manager - Automatically adjusts stop losses as positions profit

Logic:
1. When position is +5% profitable: Move stop to breakeven
2. When position is +10% profitable: Start trailing at 1% behind current price
3. Continue trailing as price moves in our favor

This lets winners run to +10% before locking in profits with tight trailing stops.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """Track trailing stop state for a position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    current_stop: float
    highest_profit_pct: float = 0.0
    last_updated: datetime = None
    trail_distance_pct: float = 1.0  # Trail 1% behind


class TrailingStopManager:
    """
    Manages trailing stops for all positions.
    Call update() periodically with current prices.
    """
    
    def __init__(self, executor=None, trail_distance_pct: float = 1.0):
        """
        Args:
            executor: KuCoinFuturesExecutor for placing stop orders
            trail_distance_pct: How far behind to trail (default 1%)
        """
        self.executor = executor
        self.trail_distance_pct = trail_distance_pct
        self.positions: Dict[str, TrailingStopState] = {}
        
        # Thresholds for moving stops
        self.breakeven_threshold = 5.0  # Move to breakeven at +5%
        self.trail_start_threshold = 10.0  # Start trailing at +10%
        
        logger.info(f"TrailingStopManager initialized: trail={trail_distance_pct}%")
    
    def register_position(self, symbol: str, side: str, entry_price: float, 
                         initial_stop: float):
        """Register a new position for trailing stop management"""
        self.positions[symbol] = TrailingStopState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_profit_pct=0.0,
            last_updated=datetime.utcnow(),
            trail_distance_pct=self.trail_distance_pct
        )
        logger.info(f"📍 Registered trailing stop: {symbol} {side} @ {entry_price}, stop={initial_stop}")
    
    def unregister_position(self, symbol: str):
        """Remove a position (closed or stopped out)"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"📍 Removed trailing stop: {symbol}")
    
    def update(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Update all trailing stops based on current prices.
        
        Args:
            prices: Dict of symbol -> current price
            
        Returns:
            List of stop adjustments made
        """
        adjustments = []
        
        for symbol, state in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            adjustment = self._check_and_adjust(state, current_price)
            if adjustment:
                adjustments.append(adjustment)
        
        return adjustments
    
    def _check_and_adjust(self, state: TrailingStopState, current_price: float) -> Optional[Dict]:
        """Check if stop needs adjustment and adjust if so"""
        
        # Calculate current profit %
        if state.side == 'long':
            profit_pct = (current_price - state.entry_price) / state.entry_price * 100
        else:
            profit_pct = (state.entry_price - current_price) / state.entry_price * 100
        
        # Track highest profit seen
        if profit_pct > state.highest_profit_pct:
            state.highest_profit_pct = profit_pct
        
        # Calculate new stop price
        new_stop = None
        reason = None
        
        if profit_pct >= self.trail_start_threshold:
            # Trail at X% behind current price
            if state.side == 'long':
                new_stop = current_price * (1 - self.trail_distance_pct / 100)
            else:
                new_stop = current_price * (1 + self.trail_distance_pct / 100)
            reason = f"trailing at {self.trail_distance_pct}% behind"
            
        elif profit_pct >= self.breakeven_threshold:
            # Move to breakeven (plus small buffer)
            if state.side == 'long':
                new_stop = state.entry_price * 1.001  # Tiny profit buffer
            else:
                new_stop = state.entry_price * 0.999
            reason = "moved to breakeven"
        
        # Only adjust if new stop is better than current
        if new_stop:
            should_update = False
            if state.side == 'long' and new_stop > state.current_stop:
                should_update = True
            elif state.side == 'short' and new_stop < state.current_stop:
                should_update = True
            
            if should_update:
                old_stop = state.current_stop
                state.current_stop = new_stop
                state.last_updated = datetime.utcnow()
                
                # Update stop order on exchange
                if self.executor:
                    try:
                        # Cancel old stop and place new one
                        self.executor.cancel_all_orders(state.symbol)
                        stop_side = 'sell' if state.side == 'long' else 'buy'
                        # Get position size
                        position = self.executor.get_position(state.symbol)
                        if position:
                            self.executor.place_stop_order(
                                state.symbol, stop_side, 
                                abs(position.size), new_stop
                            )
                    except Exception as e:
                        logger.error(f"Failed to update stop order: {e}")
                
                logger.info(f"📈 TRAILING STOP: {state.symbol} {reason}")
                logger.info(f"   Stop moved: ${old_stop:.2f} -> ${new_stop:.2f} (profit: {profit_pct:.1f}%)")
                
                return {
                    'symbol': state.symbol,
                    'side': state.side,
                    'old_stop': old_stop,
                    'new_stop': new_stop,
                    'profit_pct': profit_pct,
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return None
    
    def get_status(self) -> Dict:
        """Get status of all trailing stops"""
        return {
            'trail_distance_pct': self.trail_distance_pct,
            'breakeven_threshold': self.breakeven_threshold,
            'trail_start_threshold': self.trail_start_threshold,
            'positions': {
                symbol: {
                    'side': state.side,
                    'entry_price': state.entry_price,
                    'current_stop': state.current_stop,
                    'highest_profit_pct': state.highest_profit_pct,
                    'last_updated': state.last_updated.isoformat() if state.last_updated else None
                }
                for symbol, state in self.positions.items()
            }
        }


# Global instance for use across modules
_trailing_manager: Optional[TrailingStopManager] = None

def get_trailing_manager() -> TrailingStopManager:
    """Get or create global trailing stop manager"""
    global _trailing_manager
    if _trailing_manager is None:
        _trailing_manager = TrailingStopManager()
    return _trailing_manager

def set_trailing_manager(manager: TrailingStopManager):
    """Set global trailing stop manager"""
    global _trailing_manager
    _trailing_manager = manager
