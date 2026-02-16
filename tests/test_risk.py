"""
Tests for risk management components.

Tests:
- Position sizing
- Risk limits enforcement
- Position state tracking
- Rotation decisions
- Drawdown detection
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agents.base import Signal, SignalSide
from orchestrator.position_manager import (
    PositionManager, 
    RotationConfig, 
    TrackedPosition, 
    PositionState,
    RotationDecision
)


class TestTrackedPosition:
    """Tests for TrackedPosition data class"""
    
    def test_update_long_position_profit(self, tracked_position):
        """Long position with higher price should show profit"""
        tracked_position.side = 'long'
        tracked_position.entry_price = 50000.0
        tracked_position.size = 0.1
        
        tracked_position.update(51000.0)
        
        assert tracked_position.current_price == 51000.0
        assert tracked_position.current_pnl == 100.0  # (51000 - 50000) * 0.1
        assert tracked_position.current_pnl_pct == 2.0  # 2%
    
    def test_update_long_position_loss(self, tracked_position):
        """Long position with lower price should show loss"""
        tracked_position.side = 'long'
        tracked_position.entry_price = 50000.0
        tracked_position.size = 0.1
        
        tracked_position.update(49000.0)
        
        assert tracked_position.current_pnl == -100.0
        assert tracked_position.current_pnl_pct == -2.0
    
    def test_update_short_position_profit(self, tracked_position):
        """Short position with lower price should show profit"""
        tracked_position.side = 'short'
        tracked_position.entry_price = 50000.0
        tracked_position.size = 0.1
        
        tracked_position.update(49000.0)
        
        assert tracked_position.current_pnl == 100.0  # (50000 - 49000) * 0.1
        assert tracked_position.current_pnl_pct == 2.0
    
    def test_update_short_position_loss(self, tracked_position):
        """Short position with higher price should show loss"""
        tracked_position.side = 'short'
        tracked_position.entry_price = 50000.0
        tracked_position.size = 0.1
        
        tracked_position.update(51000.0)
        
        assert tracked_position.current_pnl == -100.0
        assert tracked_position.current_pnl_pct == -2.0
    
    def test_update_tracks_peak_pnl(self, tracked_position):
        """Should track peak PnL"""
        tracked_position.side = 'long'
        tracked_position.entry_price = 50000.0
        tracked_position.size = 0.1
        tracked_position.peak_pnl = 0
        tracked_position.peak_pnl_pct = 0
        
        # Price goes up
        tracked_position.update(52000.0)
        assert tracked_position.peak_pnl == 200.0
        assert tracked_position.peak_pnl_pct == 4.0
        
        # Price falls but peak should remain
        tracked_position.update(51000.0)
        assert tracked_position.peak_pnl == 200.0  # Still 200
        assert tracked_position.current_pnl == 100.0  # Current is 100


class TestPositionManager:
    """Tests for PositionManager class"""
    
    def test_track_position(self, position_manager, tracked_position):
        """Should add position to tracking"""
        position_manager.track_position(tracked_position)
        
        key = f"{tracked_position.agent_id}:{tracked_position.symbol}"
        assert key in position_manager.positions
    
    def test_remove_position(self, position_manager, tracked_position):
        """Should remove position from tracking"""
        position_manager.track_position(tracked_position)
        position_manager.remove_position(tracked_position.agent_id, tracked_position.symbol)
        
        key = f"{tracked_position.agent_id}:{tracked_position.symbol}"
        assert key not in position_manager.positions
        assert key in position_manager.closed_recently
    
    def test_update_price_updates_all_matching(self, position_manager):
        """Update price should update all positions for symbol"""
        pos1 = TrackedPosition(
            agent_id='strat-1',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            size=0.1
        )
        pos2 = TrackedPosition(
            agent_id='strat-2',
            symbol='BTC-USDT',
            side='short',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            size=0.1
        )
        
        position_manager.track_position(pos1)
        position_manager.track_position(pos2)
        
        position_manager.update_price('BTC-USDT', 51000.0)
        
        assert pos1.current_price == 51000.0
        assert pos2.current_price == 51000.0
    
    def test_cooldown_after_close(self, position_manager, tracked_position):
        """Should be in cooldown after closing"""
        position_manager.track_position(tracked_position)
        position_manager.remove_position(tracked_position.agent_id, tracked_position.symbol)
        
        assert position_manager.is_in_cooldown(tracked_position.agent_id, tracked_position.symbol)
    
    def test_get_status(self, position_manager, tracked_position):
        """Should return status summary"""
        position_manager.track_position(tracked_position)
        
        status = position_manager.get_status()
        
        assert 'total_positions' in status
        assert 'by_state' in status
        assert status['total_positions'] == 1


class TestPositionStates:
    """Tests for position state transitions"""
    
    def test_new_state_in_grace_period(self, position_manager, rotation_config):
        """Position should be NEW during grace period"""
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=10),  # Within 30min grace
            size=0.1
        )
        position_manager.track_position(pos)
        position_manager.update_price('BTC-USDT', 50500.0)
        
        assert pos.state == PositionState.NEW
    
    def test_winning_state(self, position_manager, rotation_config):
        """Position should be WINNING when in profit above threshold"""
        rotation_config.grace_period_minutes = 5
        rotation_config.stuck_threshold_pct = 0.5
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=30),
            size=0.1
        )
        pm.track_position(pos)
        pm.update_price('BTC-USDT', 51000.0)  # +2%
        
        assert pos.state == PositionState.WINNING
    
    def test_losing_state(self, position_manager, rotation_config):
        """Position should be LOSING when in loss"""
        rotation_config.grace_period_minutes = 5
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=30),
            size=0.1
        )
        pm.track_position(pos)
        pm.update_price('BTC-USDT', 49500.0)  # -1%
        
        assert pos.state == PositionState.LOSING
    
    def test_stuck_state(self, rotation_config):
        """Position should be STUCK if never reached profit threshold"""
        rotation_config.grace_period_minutes = 5
        rotation_config.stuck_max_minutes = 60
        rotation_config.stuck_threshold_pct = 1.0
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=120),  # Past stuck_max_minutes
            size=0.1,
            peak_pnl_pct=0.3  # Never reached 1% threshold
        )
        pm.track_position(pos)
        pm.update_price('BTC-USDT', 50100.0)  # Only +0.2%
        
        assert pos.state == PositionState.STUCK
    
    def test_fallen_state(self, rotation_config):
        """Position should be FALLEN if was winning but now losing"""
        rotation_config.grace_period_minutes = 5
        rotation_config.fallen_peak_threshold_pct = 1.0
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=60),
            size=0.1,
            peak_pnl_pct=2.0,  # Was up 2%
            peak_pnl=100.0
        )
        pm.track_position(pos)
        pm.update_price('BTC-USDT', 49500.0)  # Now -1%
        
        assert pos.state == PositionState.FALLEN
    
    def test_stale_state(self, rotation_config):
        """Position should be STALE if held too long"""
        rotation_config.grace_period_minutes = 5
        rotation_config.max_hold_hours = 24
        rotation_config.stuck_max_minutes = 60
        rotation_config.stuck_threshold_pct = 2.0  # High threshold to avoid WINNING state
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(hours=48),  # 2 days old
            size=0.1,
            peak_pnl_pct=3.0,  # Was profitable enough to not be stuck
            peak_pnl=150.0  # Corresponding dollar value (entry * size * pct/100)
        )
        pm.track_position(pos)
        # Price gives 1% profit - above 0 (not losing) but below 2% stuck_threshold (not winning)
        # Also peak_pnl_pct (3.0) > stuck_threshold (2.0) so not STUCK
        pm.update_price('BTC-USDT', 50500.0)  
        
        assert pos.state == PositionState.STALE


class TestRotationDecisions:
    """Tests for rotation decision logic"""
    
    def test_fallen_position_immediate_close(self, rotation_config):
        """FALLEN positions should trigger immediate close"""
        rotation_config.grace_period_minutes = 5
        rotation_config.fallen_peak_threshold_pct = 1.0
        rotation_config.fallen_negative_close = True
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=60),
            size=0.1,
            peak_pnl_pct=2.0,
            peak_pnl=100.0,
            state=PositionState.FALLEN
        )
        pos.current_pnl_pct = -0.5  # Now negative
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 1
        assert decisions[0].urgency == 'immediate'
        assert 'Gave back gains' in decisions[0].reason
    
    def test_stuck_position_soon_close(self, rotation_config):
        """STUCK positions should trigger 'soon' close"""
        rotation_config.grace_period_minutes = 5
        rotation_config.stuck_max_minutes = 60
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=120),
            size=0.1,
            peak_pnl_pct=0.1,
            state=PositionState.STUCK
        )
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 1
        assert decisions[0].urgency == 'soon'
        assert 'Stuck' in decisions[0].reason
    
    def test_stale_position_soon_close(self, rotation_config):
        """STALE positions should trigger 'soon' close"""
        rotation_config.max_hold_hours = 24
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(hours=48),
            size=0.1,
            state=PositionState.STALE
        )
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 1
        assert decisions[0].urgency == 'soon'
        assert 'too old' in decisions[0].reason
    
    def test_winning_position_no_rotation(self, rotation_config):
        """WINNING positions should not trigger rotation"""
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=60),
            size=0.1,
            current_pnl_pct=3.0,
            peak_pnl_pct=3.0,
            state=PositionState.WINNING
        )
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 0
    
    def test_rotation_finds_replacement_signal(self, rotation_config, sample_long_signal):
        """Rotation should find replacement signal when available"""
        rotation_config.min_replacement_confidence = 0.55
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=120),
            size=0.1,
            state=PositionState.STUCK,
            peak_pnl_pct=0.1
        )
        
        # Different symbol for replacement
        replacement = Signal(
            strategy_id='trend-following',
            symbol='ETH-USDT',
            side=SignalSide.LONG,
            confidence=0.75,
            price=3000.0,
            timestamp=datetime.utcnow(),
            reason='Test'
        )
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations(available_signals=[replacement])
        
        assert len(decisions) == 1
        assert decisions[0].replacement_signal is not None
        assert decisions[0].replacement_signal.symbol == 'ETH-USDT'
    
    def test_rotation_respects_cooldown_for_replacement(self, rotation_config):
        """Replacement should not be same symbol if in cooldown"""
        rotation_config.cooldown_minutes = 15
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=120),
            size=0.1,
            state=PositionState.STUCK,
            peak_pnl_pct=0.1
        )
        
        # Mark BTC-USDT as recently closed
        pm.closed_recently['test:BTC-USDT'] = datetime.utcnow()
        
        # Only available signal is same symbol
        same_symbol_signal = Signal(
            strategy_id='trend-following',
            symbol='BTC-USDT',
            side=SignalSide.LONG,
            confidence=0.75,
            price=50500.0,
            timestamp=datetime.utcnow(),
            reason='Test'
        )
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations(available_signals=[same_symbol_signal])
        
        # Should not use same symbol during cooldown
        assert decisions[0].replacement_signal is None
    
    def test_decisions_sorted_by_urgency(self, rotation_config):
        """Decisions should be sorted immediate > soon > optional"""
        rotation_config.grace_period_minutes = 5
        pm = PositionManager(rotation_config)
        
        pos_fallen = TrackedPosition(
            agent_id='test-1',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=60),
            size=0.1,
            peak_pnl_pct=2.0,
            state=PositionState.FALLEN
        )
        pos_fallen.current_pnl_pct = -0.5
        
        pos_stuck = TrackedPosition(
            agent_id='test-2',
            symbol='ETH-USDT',
            side='long',
            entry_price=3000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=180),
            size=1.0,
            peak_pnl_pct=0.1,
            state=PositionState.STUCK
        )
        
        pm.track_position(pos_fallen)
        pm.track_position(pos_stuck)
        
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 2
        assert decisions[0].urgency == 'immediate'
        assert decisions[1].urgency == 'soon'


class TestRiskLimits:
    """Tests for risk limit enforcement"""
    
    def test_max_signals_per_cycle_enforced(self):
        """Should not exceed max signals per cycle"""
        from orchestrator.signal_aggregator import SignalAggregator, AggregatorConfig
        
        config = AggregatorConfig(max_signals_per_cycle=2)
        aggregator = SignalAggregator(config)
        
        now = datetime.utcnow()
        signals = {
            f'strat-{i}': [
                Signal(
                    strategy_id=f'strat-{i}',
                    symbol=f'COIN{i}-USDT',
                    side=SignalSide.LONG,
                    confidence=0.70,
                    price=100.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
            for i in range(5)
        }
        
        result = aggregator.aggregate(signals)
        
        assert len(result) <= 2
    
    def test_position_exposure_limit(self, signal_aggregator):
        """Should not signal on symbols with existing positions"""
        signal_aggregator.set_positions({
            'BTC-USDT': 'long',
            'ETH-USDT': 'short'
        })
        
        now = datetime.utcnow()
        signals = {
            'test': [
                Signal(
                    strategy_id='test',
                    symbol='BTC-USDT',  # Already have position
                    side=SignalSide.LONG,
                    confidence=0.80,
                    price=50000.0,
                    timestamp=now,
                    reason='Test'
                ),
                Signal(
                    strategy_id='test',
                    symbol='SOL-USDT',  # No position
                    side=SignalSide.LONG,
                    confidence=0.70,
                    price=100.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
        }
        
        result = signal_aggregator.aggregate(signals)
        
        # Only SOL signal should be selected
        assert len(result) == 1
        assert result[0].symbol == 'SOL-USDT'


class TestGiveback:
    """Tests for giveback detection in fallen positions"""
    
    def test_giveback_percentage_calculation(self, rotation_config):
        """Should correctly calculate giveback percentage"""
        rotation_config.fallen_giveback_pct = 80
        rotation_config.fallen_peak_threshold_pct = 1.0
        rotation_config.fallen_negative_close = False  # Test giveback, not negative
        pm = PositionManager(rotation_config)
        
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(minutes=60),
            size=0.1,
            peak_pnl_pct=5.0,  # Was up 5%
            peak_pnl=250.0,
            state=PositionState.FALLEN
        )
        pos.current_pnl_pct = 0.5  # Now only up 0.5% (gave back 90%)
        pos.current_pnl = 25.0
        
        pm.track_position(pos)
        decisions = pm.evaluate_rotations()
        
        assert len(decisions) == 1
        assert 'Gave back' in decisions[0].reason


class TestPositionManagerSummary:
    """Tests for position manager summary output"""
    
    def test_summary_output(self, position_manager):
        """Summary should be human-readable"""
        pos = TrackedPosition(
            agent_id='test',
            symbol='BTC-USDT',
            side='long',
            entry_price=50000.0,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            size=0.1,
            current_pnl_pct=2.5,
            peak_pnl_pct=3.0,
            state=PositionState.WINNING
        )
        
        position_manager.track_position(pos)
        summary = position_manager.summary()
        
        assert 'Position Manager' in summary
        assert 'BTC-USDT' in summary
        assert 'WINNING' in summary
