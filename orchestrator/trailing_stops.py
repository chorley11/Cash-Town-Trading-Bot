"""
Trailing Stop Manager - Margin-Based Dynamic Stops

Logic:
1. Wait until position is +10% on MARGIN (not price)
2. Calculate optimal stop distance based on volatility (ATR)
3. Trail dynamically - tighter as profit grows
4. Lock in gains while letting winners run

Key difference from price-based: A +10% margin profit at 5x leverage 
is only a +2% price move, but that's real money in your pocket.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """Track trailing stop state for a position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    margin_used: float  # Margin locked for this position
    current_stop: float
    leverage: float = 1.0
    highest_margin_profit_pct: float = 0.0
    last_updated: datetime = None
    is_trailing: bool = False  # Whether trailing has activated
    atr: float = 0.0  # Average True Range for this symbol
    trail_distance_pct: float = 2.0  # Dynamic - adjusted based on conditions


@dataclass 
class VolatilityData:
    """Volatility metrics for a symbol"""
    symbol: str
    atr: float  # Average True Range (absolute)
    atr_pct: float  # ATR as % of price
    recent_high: float
    recent_low: float
    last_updated: datetime = None


class TrailingStopManager:
    """
    Manages trailing stops for all positions using margin-based profit.
    """
    
    def __init__(self, executor=None):
        """
        Args:
            executor: KuCoinFuturesExecutor for placing stop orders
        """
        self.executor = executor
        self.positions: Dict[str, TrailingStopState] = {}
        self.volatility: Dict[str, VolatilityData] = {}
        
        # Activation threshold - position must be up 10% on margin
        self.activation_threshold_pct = 10.0
        
        # Stop distance bounds (as multiple of ATR)
        self.min_atr_multiple = 1.0  # Minimum 1x ATR
        self.max_atr_multiple = 3.0  # Maximum 3x ATR
        
        # Tightening schedule - as profit grows, stop tightens
        # profit_pct: atr_multiple
        self.tightening_schedule = {
            10: 3.0,   # At 10% profit, trail at 3x ATR
            20: 2.5,   # At 20% profit, trail at 2.5x ATR
            30: 2.0,   # At 30% profit, trail at 2x ATR
            50: 1.5,   # At 50% profit, trail at 1.5x ATR
            100: 1.0,  # At 100%+ profit, trail at 1x ATR (tight)
        }
        
        logger.info(f"TrailingStopManager initialized: margin-based, activation={self.activation_threshold_pct}%")
    
    def update_volatility(self, symbol: str, prices: List[float], highs: List[float] = None, 
                         lows: List[float] = None):
        """
        Update volatility metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            prices: Recent close prices
            highs: Recent high prices (optional)
            lows: Recent low prices (optional)
        """
        if len(prices) < 14:
            return
        
        # Calculate ATR if we have high/low data
        if highs and lows and len(highs) >= 14:
            tr_list = []
            for i in range(1, len(prices)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]),
                    abs(lows[i] - prices[i-1])
                )
                tr_list.append(tr)
            atr = np.mean(tr_list[-14:])
        else:
            # Fallback: use price range as proxy
            atr = np.std(prices[-14:]) * 2
        
        current_price = prices[-1]
        atr_pct = (atr / current_price) * 100
        
        self.volatility[symbol] = VolatilityData(
            symbol=symbol,
            atr=atr,
            atr_pct=atr_pct,
            recent_high=max(prices[-14:]),
            recent_low=min(prices[-14:]),
            last_updated=datetime.utcnow()
        )
    
    def register_position(self, symbol: str, side: str, entry_price: float, 
                         margin_used: float, leverage: float, initial_stop: float = None):
        """Register a new position for trailing stop management"""
        
        # Calculate initial stop if not provided (2x ATR or 5%)
        if initial_stop is None:
            vol = self.volatility.get(symbol)
            if vol:
                atr_distance = vol.atr * 2
            else:
                atr_distance = entry_price * 0.05  # Default 5%
            
            if side == 'long':
                initial_stop = entry_price - atr_distance
            else:
                initial_stop = entry_price + atr_distance
        
        self.positions[symbol] = TrailingStopState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            margin_used=margin_used,
            leverage=leverage,
            current_stop=initial_stop,
            highest_margin_profit_pct=0.0,
            last_updated=datetime.utcnow(),
            is_trailing=False
        )
        
        logger.info(f"📍 Registered position: {symbol} {side}")
        logger.info(f"   Entry: ${entry_price:.4f}, Margin: ${margin_used:.2f}, Leverage: {leverage}x")
        logger.info(f"   Initial stop: ${initial_stop:.4f}")
    
    def unregister_position(self, symbol: str):
        """Remove a position (closed or stopped out)"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"📍 Removed trailing stop: {symbol}")
    
    def update(self, prices: Dict[str, float], positions_data: Dict[str, Dict] = None) -> List[Dict]:
        """
        Update all trailing stops based on current prices and P&L.
        
        Args:
            prices: Dict of symbol -> current price
            positions_data: Dict of symbol -> {unrealized_pnl, margin, etc}
            
        Returns:
            List of stop adjustments made
        """
        adjustments = []
        
        for symbol, state in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            # Get unrealized PnL from positions data if available
            unrealized_pnl = None
            if positions_data and symbol in positions_data:
                unrealized_pnl = positions_data[symbol].get('unrealized_pnl')
            
            adjustment = self._check_and_adjust(state, current_price, unrealized_pnl)
            if adjustment:
                adjustments.append(adjustment)
        
        return adjustments
    
    def _calculate_margin_profit_pct(self, state: TrailingStopState, current_price: float,
                                     unrealized_pnl: float = None) -> float:
        """Calculate profit as percentage of margin used"""
        
        if unrealized_pnl is not None:
            # Use actual unrealized PnL if available
            margin_profit_pct = (unrealized_pnl / state.margin_used) * 100
        else:
            # Calculate from price move and leverage
            if state.side == 'long':
                price_profit_pct = (current_price - state.entry_price) / state.entry_price * 100
            else:
                price_profit_pct = (state.entry_price - current_price) / state.entry_price * 100
            
            # Margin profit = price profit * leverage
            margin_profit_pct = price_profit_pct * state.leverage
        
        return margin_profit_pct
    
    def _calculate_optimal_stop_distance(self, state: TrailingStopState, 
                                         margin_profit_pct: float) -> float:
        """
        Calculate optimal stop distance based on volatility and profit level.
        
        Returns distance in price terms (not %).
        """
        # Get ATR for this symbol
        vol = self.volatility.get(state.symbol)
        if vol:
            atr = vol.atr
        else:
            # Fallback: estimate ATR as 2% of entry price
            atr = state.entry_price * 0.02
        
        # Determine ATR multiple based on profit level
        atr_multiple = self.max_atr_multiple  # Start with widest
        
        for threshold, multiple in sorted(self.tightening_schedule.items()):
            if margin_profit_pct >= threshold:
                atr_multiple = multiple
        
        # Calculate stop distance
        stop_distance = atr * atr_multiple
        
        # Store for reference
        state.trail_distance_pct = (stop_distance / state.entry_price) * 100
        state.atr = atr
        
        return stop_distance
    
    def _check_and_adjust(self, state: TrailingStopState, current_price: float,
                         unrealized_pnl: float = None) -> Optional[Dict]:
        """Check if stop needs adjustment and adjust if so"""
        
        # Calculate margin-based profit
        margin_profit_pct = self._calculate_margin_profit_pct(state, current_price, unrealized_pnl)
        
        # Track highest margin profit seen
        if margin_profit_pct > state.highest_margin_profit_pct:
            state.highest_margin_profit_pct = margin_profit_pct
        
        # Check if we should activate trailing
        if not state.is_trailing:
            if margin_profit_pct >= self.activation_threshold_pct:
                state.is_trailing = True
                logger.info(f"🎯 TRAILING ACTIVATED: {state.symbol}")
                logger.info(f"   Margin profit: {margin_profit_pct:.1f}% (threshold: {self.activation_threshold_pct}%)")
            else:
                # Not yet at activation threshold
                return None
        
        # Calculate optimal stop distance
        stop_distance = self._calculate_optimal_stop_distance(state, margin_profit_pct)
        
        # Calculate new stop price
        if state.side == 'long':
            new_stop = current_price - stop_distance
        else:
            new_stop = current_price + stop_distance
        
        # Only adjust if new stop is better (tighter) than current
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
                    self.executor.cancel_all_orders(state.symbol)
                    stop_side = 'sell' if state.side == 'long' else 'buy'
                    position = self.executor.get_position(state.symbol)
                    if position:
                        self.executor.place_stop_order(
                            state.symbol, stop_side, 
                            abs(position.size), new_stop
                        )
                except Exception as e:
                    logger.error(f"Failed to update stop order: {e}")
            
            vol = self.volatility.get(state.symbol)
            atr_mult = stop_distance / vol.atr if vol and vol.atr > 0 else 0
            
            logger.info(f"📈 TRAILING STOP ADJUSTED: {state.symbol}")
            logger.info(f"   Stop: ${old_stop:.4f} -> ${new_stop:.4f}")
            logger.info(f"   Margin profit: {margin_profit_pct:.1f}%, ATR mult: {atr_mult:.1f}x")
            
            return {
                'symbol': state.symbol,
                'side': state.side,
                'old_stop': old_stop,
                'new_stop': new_stop,
                'margin_profit_pct': margin_profit_pct,
                'atr_multiple': atr_mult,
                'stop_distance': stop_distance,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return None
    
    def get_status(self) -> Dict:
        """Get status of all trailing stops"""
        return {
            'activation_threshold_pct': self.activation_threshold_pct,
            'tightening_schedule': self.tightening_schedule,
            'positions': {
                symbol: {
                    'side': state.side,
                    'entry_price': state.entry_price,
                    'margin_used': state.margin_used,
                    'leverage': state.leverage,
                    'current_stop': state.current_stop,
                    'is_trailing': state.is_trailing,
                    'highest_margin_profit_pct': state.highest_margin_profit_pct,
                    'trail_distance_pct': state.trail_distance_pct,
                    'last_updated': state.last_updated.isoformat() if state.last_updated else None
                }
                for symbol, state in self.positions.items()
            },
            'volatility': {
                symbol: {
                    'atr': vol.atr,
                    'atr_pct': vol.atr_pct
                }
                for symbol, vol in self.volatility.items()
            }
        }


# Global instance
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
