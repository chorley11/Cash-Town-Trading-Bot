"""
Tests for the Signal Aggregator and Smart Orchestrator.

Tests signal aggregation logic:
- Ranking signals by confidence
- Consensus detection
- Conflict handling
- Cooldown management
- Position filtering
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agents.base import Signal, SignalSide
from orchestrator.signal_aggregator import SignalAggregator, AggregatorConfig, AggregatedSignal
from orchestrator.smart_orchestrator import SmartOrchestrator


class TestSignalAggregator:
    """Tests for SignalAggregator class"""
    
    def test_aggregate_empty_signals(self, signal_aggregator):
        """Aggregator should return empty list for no signals"""
        result = signal_aggregator.aggregate({})
        assert result == []
    
    def test_aggregate_single_signal(self, signal_aggregator, sample_long_signal):
        """Single high-confidence signal should be selected"""
        signals = {'trend-following': [sample_long_signal]}
        result = signal_aggregator.aggregate(signals)
        
        assert len(result) == 1
        assert result[0].signal.symbol == 'BTC-USDT'
        assert result[0].signal.side == SignalSide.LONG
        assert result[0].rank == 1
    
    def test_aggregate_filters_low_confidence(self, signal_aggregator, low_confidence_signal):
        """Signals below min_confidence should be filtered"""
        signals = {'zweig': [low_confidence_signal]}
        result = signal_aggregator.aggregate(signals)
        
        assert len(result) == 0
    
    def test_aggregate_ranks_by_confidence(self, signal_aggregator):
        """Signals should be ranked by adjusted confidence (highest first)"""
        now = datetime.utcnow()
        signals = {
            'strategy-a': [
                Signal(
                    strategy_id='strategy-a',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.60,
                    price=50000.0,
                    timestamp=now,
                    reason='Test'
                )
            ],
            'strategy-b': [
                Signal(
                    strategy_id='strategy-b',
                    symbol='ETH-USDT',
                    side=SignalSide.LONG,
                    confidence=0.80,
                    price=3000.0,
                    timestamp=now,
                    reason='Test'
                )
            ],
            'strategy-c': [
                Signal(
                    strategy_id='strategy-c',
                    symbol='SOL-USDT',
                    side=SignalSide.LONG,
                    confidence=0.70,
                    price=100.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
        }
        
        result = signal_aggregator.aggregate(signals)
        
        assert len(result) == 3
        # Should be sorted by confidence: ETH > SOL > BTC
        assert result[0].symbol == 'ETH-USDT'
        assert result[1].symbol == 'SOL-USDT'
        assert result[2].symbol == 'BTC-USDT'
        
        # Check ranks are assigned correctly
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[2].rank == 3
    
    def test_aggregate_consensus_detection(self, signal_aggregator, consensus_signals):
        """Multiple strategies agreeing should boost consensus score"""
        result = signal_aggregator.aggregate(consensus_signals)
        
        assert len(result) == 1
        assert result[0].symbol == 'BTC-USDT'
        assert result[0].consensus_score == 1.0  # All agree
        assert len(result[0].sources) == 3  # 3 strategies agree
        assert len(result[0].conflicts) == 0  # No conflicts
    
    def test_aggregate_conflict_handling(self, signal_aggregator, conflicting_signals):
        """Conflicting signals should be detected and penalized"""
        result = signal_aggregator.aggregate(conflicting_signals)
        
        assert len(result) == 1
        # Majority direction wins (LONG has higher confidence)
        assert result[0].signal.side == SignalSide.LONG
        assert len(result[0].conflicts) == 1  # One strategy disagrees
        assert 'mean-reversion' in result[0].conflicts
    
    def test_aggregate_consensus_bonus(self, aggregator_config, consensus_signals):
        """Consensus should boost adjusted confidence"""
        aggregator_config.consensus_bonus = 0.05
        aggregator = SignalAggregator(aggregator_config)
        
        result = aggregator.aggregate(consensus_signals)
        
        # Base confidence 0.75 + 2 * 0.05 consensus bonus = 0.85
        assert result[0].adjusted_confidence > 0.75
    
    def test_aggregate_conflict_penalty(self, aggregator_config, conflicting_signals):
        """Conflicts should reduce adjusted confidence"""
        aggregator_config.conflict_penalty = 0.1
        aggregator = SignalAggregator(aggregator_config)
        
        result = aggregator.aggregate(conflicting_signals)
        
        # Base confidence 0.70 - 0.10 penalty = 0.60
        assert result[0].adjusted_confidence < 0.70
    
    def test_aggregate_respects_max_signals(self, aggregator_config):
        """Should limit signals to max_signals_per_cycle"""
        aggregator_config.max_signals_per_cycle = 2
        aggregator = SignalAggregator(aggregator_config)
        
        now = datetime.utcnow()
        signals = {
            f'strategy-{i}': [
                Signal(
                    strategy_id=f'strategy-{i}',
                    symbol=f'COIN{i}-USDT',
                    side=SignalSide.LONG,
                    confidence=0.60 + i * 0.05,
                    price=100.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
            for i in range(5)
        }
        
        result = aggregator.aggregate(signals)
        
        assert len(result) == 2
    
    def test_aggregate_filters_blacklisted_symbols(self, aggregator_config, sample_long_signal):
        """Blacklisted symbols should be filtered"""
        aggregator_config.blacklist = {'BTC-USDT'}
        aggregator = SignalAggregator(aggregator_config)
        
        signals = {'trend-following': [sample_long_signal]}
        result = aggregator.aggregate(signals)
        
        assert len(result) == 0
    
    def test_cooldown_blocks_repeat_signals(self, signal_aggregator, sample_long_signal):
        """Recently signaled symbols should be in cooldown"""
        signals = {'trend-following': [sample_long_signal]}
        
        # First aggregation
        result1 = signal_aggregator.aggregate(signals)
        assert len(result1) == 1
        
        # Mark as signaled
        signal_aggregator.mark_signaled('BTC-USDT')
        
        # Second aggregation should skip due to cooldown
        result2 = signal_aggregator.aggregate(signals)
        assert len(result2) == 0
    
    def test_current_position_blocks_signal(self, signal_aggregator, sample_long_signal):
        """Should not signal on symbols with existing positions"""
        signal_aggregator.set_positions({'BTC-USDT': 'long'})
        
        signals = {'trend-following': [sample_long_signal]}
        result = signal_aggregator.aggregate(signals)
        
        assert len(result) == 0
    
    def test_strategy_performance_weighting(self, signal_aggregator):
        """High-performing strategies should get a boost"""
        signal_aggregator.set_strategy_performance({
            'winning-strat': 0.8,
            'losing-strat': -0.5
        })
        
        now = datetime.utcnow()
        signals = {
            'winning-strat': [
                Signal(
                    strategy_id='winning-strat',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.60,
                    price=50000.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
        }
        
        result = signal_aggregator.aggregate(signals)
        
        # Should get performance boost
        assert result[0].adjusted_confidence > 0.60
    
    def test_get_status(self, signal_aggregator):
        """Status should return configuration and state"""
        signal_aggregator.mark_signaled('BTC-USDT')
        
        status = signal_aggregator.get_status()
        
        assert 'config' in status
        assert status['config']['min_confidence'] == 0.55
        assert 'active_cooldowns' in status


class TestSmartOrchestrator:
    """Tests for SmartOrchestrator class"""
    
    def test_receive_signal(self, smart_orchestrator):
        """Should properly receive and store signals"""
        signal_data = {
            'symbol': 'BTC-USDT',
            'side': 'long',
            'confidence': 0.75,
            'price': 50000.0,
            'reason': 'Test signal',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        smart_orchestrator.receive_signal('trend-following', signal_data)
        
        assert 'trend-following' in smart_orchestrator.raw_signals
        assert len(smart_orchestrator.raw_signals['trend-following']) == 1
    
    def test_get_actionable_signals_empty(self, smart_orchestrator):
        """Should return empty list when no signals"""
        result = smart_orchestrator.get_actionable_signals()
        assert result == []
    
    def test_get_actionable_signals_processes_and_clears(self, smart_orchestrator):
        """Should process signals and clear raw signals after"""
        signal_data = {
            'symbol': 'BTC-USDT',
            'side': 'long',
            'confidence': 0.75,
            'price': 50000.0,
            'reason': 'Test signal'
        }
        smart_orchestrator.receive_signal('trend-following', signal_data)
        
        result = smart_orchestrator.get_actionable_signals()
        
        assert len(result) == 1
        assert smart_orchestrator.raw_signals == {}  # Cleared
    
    def test_record_trade_result_updates_performance(self, smart_orchestrator):
        """Recording trade results should update strategy performance"""
        smart_orchestrator.record_trade_result(
            symbol='BTC-USDT',
            side='long',
            pnl=100.0,
            pnl_pct=5.0,
            strategy_id='trend-following',
            reason='take_profit'
        )
        
        assert 'trend-following' in smart_orchestrator.strategy_performance
        perf = smart_orchestrator.strategy_performance['trend-following']
        assert perf['trades'] == 1
        assert perf['wins'] == 1
        assert perf['total_pnl_pct'] == 5.0
    
    def test_record_trade_result_tracks_rescued(self, smart_orchestrator):
        """Should track rescued signal performance separately"""
        smart_orchestrator.record_trade_result(
            symbol='BTC-USDT',
            side='long',
            pnl=100.0,
            pnl_pct=5.0,
            strategy_id='trend-following',
            reason='take_profit',
            was_rescued=True
        )
        
        assert smart_orchestrator.rescue_stats['rescued_wins'] == 1
    
    def test_get_strategy_multipliers_defaults(self, smart_orchestrator):
        """Should return default multipliers initially"""
        multipliers = smart_orchestrator.get_strategy_multipliers()
        
        assert 'trend-following' in multipliers
        # Default or learned multiplier should exist
        assert multipliers['trend-following'] >= 1.0  # At least default
        assert multipliers.get('zweig', 0.7) <= 1.0  # Probationary or worse
    
    def test_update_strategy_performance_calculates_multiplier(self, smart_orchestrator):
        """Should calculate multiplier based on performance"""
        # Simulate 10+ trades with good win rate
        for i in range(12):
            smart_orchestrator._update_strategy_performance(
                'test-strat',
                won=i < 7,  # 7 wins, 5 losses = 58% WR
                pnl_pct=2.0 if i < 7 else -1.5
            )
        
        perf = smart_orchestrator.strategy_performance['test-strat']
        assert perf['trades'] == 12
        # Profitable strategy should get multiplier > 1
        assert perf['multiplier'] >= 1.0
    
    def test_update_strategy_performance_disables_losers(self, smart_orchestrator):
        """Should disable strategies with terrible performance"""
        # Simulate 10+ trades with bad win rate (<35%)
        for i in range(15):
            smart_orchestrator._update_strategy_performance(
                'terrible-strat',
                won=i < 4,  # 4 wins, 11 losses = 27% WR
                pnl_pct=2.0 if i < 4 else -3.0
            )
        
        perf = smart_orchestrator.strategy_performance['terrible-strat']
        assert perf['multiplier'] == 0.0  # Disabled
    
    def test_get_learning_summary(self, smart_orchestrator):
        """Should return comprehensive learning summary"""
        smart_orchestrator.record_trade_result(
            symbol='BTC-USDT',
            side='long',
            pnl=100.0,
            pnl_pct=5.0,
            strategy_id='trend-following',
            reason='take_profit'
        )
        
        summary = smart_orchestrator.get_learning_summary()
        
        assert 'strategy_performance' in summary
        assert 'second_chance' in summary
        assert 'data_files' in summary


class TestAggregatedSignal:
    """Tests for AggregatedSignal data class"""
    
    def test_is_unanimous_true(self, sample_long_signal):
        """is_unanimous should be True when no conflicts"""
        agg = AggregatedSignal(
            signal=sample_long_signal,
            rank=1,
            sources=['trend-following', 'turtle'],
            conflicts=[],
            consensus_score=1.0,
            adjusted_confidence=0.80
        )
        assert agg.is_unanimous is True
    
    def test_is_unanimous_false(self, sample_long_signal):
        """is_unanimous should be False when conflicts exist"""
        agg = AggregatedSignal(
            signal=sample_long_signal,
            rank=1,
            sources=['trend-following'],
            conflicts=['mean-reversion'],
            consensus_score=0.5,
            adjusted_confidence=0.60
        )
        assert agg.is_unanimous is False
    
    def test_properties_delegate_to_signal(self, sample_long_signal):
        """Properties should delegate to underlying signal"""
        agg = AggregatedSignal(
            signal=sample_long_signal,
            rank=1,
            sources=['trend-following'],
            conflicts=[],
            consensus_score=1.0,
            adjusted_confidence=0.75
        )
        
        assert agg.symbol == 'BTC-USDT'
        assert agg.side == SignalSide.LONG


class TestOrchestratorEdgeCases:
    """Edge case tests for orchestrator components"""
    
    def test_malformed_signal_data(self, smart_orchestrator):
        """Should handle malformed signal data gracefully"""
        bad_data = {
            'symbol': 'BTC-USDT',
            # Missing required fields
        }
        
        # Should not raise
        smart_orchestrator.receive_signal('test', bad_data)
    
    def test_equal_confidence_tiebreaker(self, signal_aggregator):
        """Should handle tie in confidence gracefully"""
        now = datetime.utcnow()
        signals = {
            'strategy-a': [
                Signal(
                    strategy_id='strategy-a',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.70,
                    price=50000.0,
                    timestamp=now,
                    reason='Test'
                )
            ],
            'strategy-b': [
                Signal(
                    strategy_id='strategy-b',
                    symbol='ETH-USDT',
                    side=SignalSide.LONG,
                    confidence=0.70,
                    price=3000.0,
                    timestamp=now,
                    reason='Test'
                )
            ]
        }
        
        result = signal_aggregator.aggregate(signals)
        
        # Both should be selected
        assert len(result) == 2
    
    def test_signal_on_same_symbol_multiple_strategies_different_confidence(self, signal_aggregator):
        """Same symbol from multiple strategies should use best signal"""
        now = datetime.utcnow()
        signals = {
            'strategy-a': [
                Signal(
                    strategy_id='strategy-a',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.60,
                    price=50000.0,
                    timestamp=now,
                    reason='Lower confidence'
                )
            ],
            'strategy-b': [
                Signal(
                    strategy_id='strategy-b',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.80,
                    price=50000.0,
                    timestamp=now,
                    reason='Higher confidence'
                )
            ]
        }
        
        result = signal_aggregator.aggregate(signals)
        
        assert len(result) == 1
        # Should use the higher confidence signal
        assert result[0].signal.confidence == 0.80
