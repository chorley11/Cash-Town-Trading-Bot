"""
Position Manager - Active position monitoring and rotation

Closes positions that:
1. Haven't started winning after grace period ("stuck")
2. Were winning but fell back to negative ("gave back gains")
3. Hit time-based stops
4. Violate drawdown limits

Then rotates capital to fresh, higher-confidence signals.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class PositionState(Enum):
    NEW = "new"              # Just opened
    WINNING = "winning"      # Currently in profit
    LOSING = "losing"        # Currently in loss
    STUCK = "stuck"          # Never reached profit threshold
    FALLEN = "fallen"        # Was winning, now losing
    STALE = "stale"          # Too old, not performing

@dataclass
class TrackedPosition:
    """Position being tracked for rotation decisions"""
    agent_id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float
    
    # Tracking state
    current_price: float = 0.0
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    peak_pnl: float = 0.0
    peak_pnl_pct: float = 0.0
    peak_time: datetime = None
    
    state: PositionState = PositionState.NEW
    state_changed_at: datetime = None
    
    # Metadata
    signal_confidence: float = 0.0
    stop_loss: float = None
    take_profit: float = None
    
    def update(self, current_price: float):
        """Update position with new price"""
        self.current_price = current_price
        
        if self.side == 'long':
            self.current_pnl = (current_price - self.entry_price) * self.size
            self.current_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            self.current_pnl = (self.entry_price - current_price) * self.size
            self.current_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
        
        # Track peak PnL
        if self.current_pnl > self.peak_pnl:
            self.peak_pnl = self.current_pnl
            self.peak_pnl_pct = self.current_pnl_pct
            self.peak_time = datetime.utcnow()

@dataclass
class RotationConfig:
    """Configuration for position rotation"""
    # Grace period - how long before judging a position
    grace_period_minutes: int = 30
    
    # Stuck detection - never reached this profit %
    stuck_threshold_pct: float = 0.5
    stuck_max_minutes: int = 120  # Close if stuck after 2 hours
    
    # Fallen detection - was winning, now losing
    fallen_peak_threshold_pct: float = 1.0  # Must have been up at least 1%
    fallen_giveback_pct: float = 80  # Close if gave back 80% of gains
    fallen_negative_close: bool = True  # Close if fallen to negative
    
    # Stale detection - too old
    max_hold_hours: int = 48
    
    # Minimum confidence for replacement signal
    min_replacement_confidence: float = 0.6
    
    # Cooldown after closing (prevent immediate re-entry)
    cooldown_minutes: int = 15

@dataclass
class RotationDecision:
    """Decision to close and potentially replace a position"""
    position: TrackedPosition
    reason: str
    urgency: str  # 'immediate', 'soon', 'optional'
    replacement_signal: Optional[Any] = None

class PositionManager:
    """
    Manages active positions across all agents.
    Identifies underperforming positions and rotates to better opportunities.
    """
    
    def __init__(self, config: RotationConfig = None):
        self.config = config or RotationConfig()
        self.positions: Dict[str, TrackedPosition] = {}  # key: agent_id:symbol
        self.closed_recently: Dict[str, datetime] = {}  # For cooldown tracking
        self.pending_signals: List[Any] = []  # Signals waiting for capital
    
    def track_position(self, position: TrackedPosition):
        """Add or update a tracked position"""
        key = f"{position.agent_id}:{position.symbol}"
        
        if key in self.positions:
            # Update existing
            old = self.positions[key]
            position.peak_pnl = max(old.peak_pnl, position.current_pnl)
            position.peak_pnl_pct = max(old.peak_pnl_pct, position.current_pnl_pct)
            position.peak_time = old.peak_time or position.entry_time
        
        self.positions[key] = position
        logger.debug(f"Tracking position: {key}")
    
    def remove_position(self, agent_id: str, symbol: str):
        """Remove a position from tracking"""
        key = f"{agent_id}:{symbol}"
        if key in self.positions:
            del self.positions[key]
            self.closed_recently[key] = datetime.utcnow()
            logger.info(f"Removed position from tracking: {key}")
    
    def update_price(self, symbol: str, price: float):
        """Update all positions for a symbol with new price"""
        for key, position in self.positions.items():
            if position.symbol == symbol:
                position.update(price)
                self._update_state(position)
    
    def _update_state(self, position: TrackedPosition):
        """Update position state based on performance"""
        now = datetime.utcnow()
        age_minutes = (now - position.entry_time).total_seconds() / 60
        old_state = position.state
        
        # Still in grace period
        if age_minutes < self.config.grace_period_minutes:
            position.state = PositionState.NEW
        
        # Check for WINNING
        elif position.current_pnl_pct >= self.config.stuck_threshold_pct:
            position.state = PositionState.WINNING
        
        # Check for FALLEN (was winning, now losing)
        elif (position.peak_pnl_pct >= self.config.fallen_peak_threshold_pct and 
              position.current_pnl_pct <= 0):
            position.state = PositionState.FALLEN
        
        # Check for STUCK (never reached profit threshold)
        elif (age_minutes > self.config.stuck_max_minutes and
              position.peak_pnl_pct < self.config.stuck_threshold_pct):
            position.state = PositionState.STUCK
        
        # Check for STALE (too old)
        elif age_minutes > self.config.max_hold_hours * 60:
            position.state = PositionState.STALE
        
        # Otherwise LOSING
        elif position.current_pnl_pct < 0:
            position.state = PositionState.LOSING
        
        # Track state change time
        if position.state != old_state:
            position.state_changed_at = now
            logger.info(f"Position {position.symbol} state: {old_state.value} -> {position.state.value}")
    
    def evaluate_rotations(self, available_signals: List[Any] = None) -> List[RotationDecision]:
        """
        Evaluate all positions and return rotation decisions.
        
        Args:
            available_signals: New signals that could replace closed positions
        
        Returns:
            List of RotationDecision objects
        """
        decisions = []
        
        for key, position in list(self.positions.items()):
            decision = self._evaluate_position(position, available_signals)
            if decision:
                decisions.append(decision)
        
        # Sort by urgency
        urgency_order = {'immediate': 0, 'soon': 1, 'optional': 2}
        decisions.sort(key=lambda d: urgency_order.get(d.urgency, 3))
        
        return decisions
    
    def _evaluate_position(self, position: TrackedPosition, 
                          signals: List[Any] = None) -> Optional[RotationDecision]:
        """Evaluate a single position for rotation"""
        now = datetime.utcnow()
        age_minutes = (now - position.entry_time).total_seconds() / 60
        
        # FALLEN - Was winning, now negative (URGENT)
        if position.state == PositionState.FALLEN:
            if self.config.fallen_negative_close and position.current_pnl_pct < 0:
                return RotationDecision(
                    position=position,
                    reason=f"Gave back gains: was +{position.peak_pnl_pct:.1f}%, now {position.current_pnl_pct:.1f}%",
                    urgency='immediate',
                    replacement_signal=self._find_replacement(position, signals)
                )
            
            # Check if gave back too much of gains
            if position.peak_pnl_pct > 0:
                giveback_pct = (position.peak_pnl_pct - position.current_pnl_pct) / position.peak_pnl_pct * 100
                if giveback_pct >= self.config.fallen_giveback_pct:
                    return RotationDecision(
                        position=position,
                        reason=f"Gave back {giveback_pct:.0f}% of gains",
                        urgency='immediate',
                        replacement_signal=self._find_replacement(position, signals)
                    )
        
        # STUCK - Never got going
        if position.state == PositionState.STUCK:
            return RotationDecision(
                position=position,
                reason=f"Stuck {age_minutes:.0f}min, never reached +{self.config.stuck_threshold_pct}%",
                urgency='soon',
                replacement_signal=self._find_replacement(position, signals)
            )
        
        # STALE - Too old
        if position.state == PositionState.STALE:
            return RotationDecision(
                position=position,
                reason=f"Position too old ({age_minutes/60:.1f}h)",
                urgency='soon',
                replacement_signal=self._find_replacement(position, signals)
            )
        
        # LOSING for too long after grace period
        if position.state == PositionState.LOSING:
            if age_minutes > self.config.grace_period_minutes * 2:
                # Optional rotation if we have better signal
                replacement = self._find_replacement(position, signals)
                if replacement and replacement.confidence > position.signal_confidence + 0.1:
                    return RotationDecision(
                        position=position,
                        reason=f"Losing {position.current_pnl_pct:.1f}%, better signal available",
                        urgency='optional',
                        replacement_signal=replacement
                    )
        
        return None
    
    def _find_replacement(self, position: TrackedPosition, 
                         signals: List[Any] = None) -> Optional[Any]:
        """Find a suitable replacement signal for a position being closed"""
        if not signals:
            return None
        
        # Check cooldown
        key = f"{position.agent_id}:{position.symbol}"
        if key in self.closed_recently:
            cooldown_end = self.closed_recently[key] + timedelta(minutes=self.config.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                # Filter out same symbol during cooldown
                signals = [s for s in signals if s.symbol != position.symbol]
        
        # Find best signal above minimum confidence
        valid_signals = [s for s in signals if s.confidence >= self.config.min_replacement_confidence]
        
        if not valid_signals:
            return None
        
        # Return highest confidence signal
        return max(valid_signals, key=lambda s: s.confidence)
    
    def is_in_cooldown(self, agent_id: str, symbol: str) -> bool:
        """Check if a symbol is in cooldown for an agent"""
        key = f"{agent_id}:{symbol}"
        if key not in self.closed_recently:
            return False
        
        cooldown_end = self.closed_recently[key] + timedelta(minutes=self.config.cooldown_minutes)
        return datetime.utcnow() < cooldown_end
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tracked positions"""
        by_state = {}
        for position in self.positions.values():
            state = position.state.value
            if state not in by_state:
                by_state[state] = []
            by_state[state].append({
                'agent': position.agent_id,
                'symbol': position.symbol,
                'side': position.side,
                'pnl_pct': position.current_pnl_pct,
                'peak_pnl_pct': position.peak_pnl_pct,
                'age_min': (datetime.utcnow() - position.entry_time).total_seconds() / 60
            })
        
        return {
            'total_positions': len(self.positions),
            'by_state': by_state,
            'cooldowns_active': len([k for k, v in self.closed_recently.items() 
                                    if datetime.utcnow() < v + timedelta(minutes=self.config.cooldown_minutes)])
        }
    
    def summary(self) -> str:
        """Human-readable summary"""
        status = self.get_status()
        lines = [
            f"📊 Position Manager Status",
            f"Total: {status['total_positions']} positions",
        ]
        
        for state, positions in status['by_state'].items():
            icon = {
                'new': '🆕', 'winning': '✅', 'losing': '📉',
                'stuck': '🔒', 'fallen': '📉⬇️', 'stale': '⏰'
            }.get(state, '❓')
            lines.append(f"{icon} {state.upper()}: {len(positions)}")
            for p in positions[:3]:  # Show first 3
                lines.append(f"   • {p['symbol']} {p['side']} {p['pnl_pct']:+.1f}% (peak {p['peak_pnl_pct']:+.1f}%)")
        
        return "\n".join(lines)
