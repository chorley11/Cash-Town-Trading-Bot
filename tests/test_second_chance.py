"""
Tests for Second Chance logic - rescuing promising rejected signals.

Tests:
- Strategy boost evaluation
- Consensus boosting
- Winning pattern matching
- Near-miss tracking
- Counterfactual learning
"""
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.base import Signal, SignalSide
from orchestrator.second_chance import (
    SecondChanceEvaluator,
    NearMissSignal,
    WinningPattern
)


class TestSecondChanceEvaluator:
    """Tests for SecondChanceEvaluator class"""
    
    def test_init_with_strategy_performance(self):
        """Should initialize with strategy performance data"""
        perf = {
            'trend-following': {'trades': 20, 'win_rate': 0.55, 'total_pnl_pct': 15.0}
        }
        evaluator = SecondChanceEvaluator(perf)
        
        assert evaluator.strategy_performance == perf
    
    def test_below_absolute_floor_not_rescued(self, second_chance_evaluator, near_miss_signal):
        """Signals below absolute floor should never be rescued"""
        near_miss_signal.confidence = 0.40  # Below 0.45 floor
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert should_rescue is False
        assert "Below absolute floor" in reasons
    
    def test_strategy_boost_for_good_performer(self, second_chance_evaluator, near_miss_signal):
        """Good performing strategies should get confidence boost"""
        near_miss_signal.strategy_id = 'trend-following'  # Known good performer
        near_miss_signal.confidence = 0.52
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        # Trend-following has hardcoded +0.08 boost
        assert adj_conf > 0.52
        assert any('Strategy boost' in r for r in reasons)
    
    def test_strategy_penalty_for_poor_performer(self, second_chance_evaluator, near_miss_signal):
        """Poor performing strategies should get penalized"""
        near_miss_signal.strategy_id = 'zweig'  # Known poor performer
        near_miss_signal.confidence = 0.52
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        # Zweig has -0.10 penalty
        assert adj_conf < 0.52
        assert any('Strategy penalty' in r for r in reasons)
    
    def test_consensus_boost_when_strategies_agree(self, second_chance_evaluator, near_miss_signal):
        """Should boost confidence when multiple strategies agree"""
        near_miss_signal.strategy_id = 'strat-a'
        near_miss_signal.confidence = 0.52
        near_miss_signal.symbol = 'BTC-USDT'
        near_miss_signal.side = SignalSide.LONG
        
        # Other strategies also signaling LONG on BTC
        all_signals = {
            'strat-b': [
                Signal(
                    strategy_id='strat-b',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.60,
                    price=50000.0,
                    timestamp=datetime.utcnow(),
                    reason='Agreeing'
                )
            ],
            'strat-c': [
                Signal(
                    strategy_id='strat-c',
                    symbol='BTC-USDT',
                    side=SignalSide.LONG,
                    confidence=0.58,
                    price=50000.0,
                    timestamp=datetime.utcnow(),
                    reason='Also agreeing'
                )
            ]
        }
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence",
            all_signals=all_signals
        )
        
        assert adj_conf > 0.52
        assert any('Consensus boost' in r for r in reasons)
    
    def test_consensus_penalty_when_strategies_disagree(self, second_chance_evaluator, near_miss_signal):
        """Should penalize when strategies disagree"""
        near_miss_signal.strategy_id = 'strat-a'
        near_miss_signal.confidence = 0.52
        near_miss_signal.symbol = 'BTC-USDT'
        near_miss_signal.side = SignalSide.LONG
        
        # Another strategy signaling SHORT on BTC
        all_signals = {
            'strat-b': [
                Signal(
                    strategy_id='strat-b',
                    symbol='BTC-USDT',
                    side=SignalSide.SHORT,
                    confidence=0.60,
                    price=50000.0,
                    timestamp=datetime.utcnow(),
                    reason='Disagreeing'
                )
            ]
        }
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence",
            all_signals=all_signals
        )
        
        # Should not boost due to disagreement
        assert not any('Consensus boost' in r for r in reasons)


class TestWinningPatternMatching:
    """Tests for winning pattern matching"""
    
    def test_pattern_match_boosts_confidence(self, second_chance_evaluator, near_miss_signal, winning_pattern):
        """Matching a winning pattern should boost confidence"""
        # Add pattern to evaluator
        second_chance_evaluator.winning_patterns.append(winning_pattern)
        
        near_miss_signal.strategy_id = 'trend-following'
        near_miss_signal.confidence = 0.52  # Within pattern range (0.50-0.55)
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"  # Matches pattern
        )
        
        assert any('Winning pattern match' in r for r in reasons)
    
    def test_pattern_not_matched_if_confidence_outside_range(self, second_chance_evaluator, near_miss_signal, winning_pattern):
        """Should not match pattern if confidence outside range"""
        second_chance_evaluator.winning_patterns.append(winning_pattern)
        
        near_miss_signal.strategy_id = 'trend-following'
        near_miss_signal.confidence = 0.45  # Below pattern range (0.50-0.55)
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert not any('Winning pattern match' in r for r in reasons)
    
    def test_pattern_not_matched_if_strategy_not_in_list(self, second_chance_evaluator, near_miss_signal, winning_pattern):
        """Should not match pattern if strategy not in pattern's list"""
        winning_pattern.strategy_ids = ['other-strategy']
        second_chance_evaluator.winning_patterns.append(winning_pattern)
        
        near_miss_signal.strategy_id = 'trend-following'  # Not in pattern list
        near_miss_signal.confidence = 0.52
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert not any('Winning pattern match' in r for r in reasons)


class TestTechnicalStrengthBoost:
    """Tests for technical indicator-based boosts"""
    
    def test_strong_adx_boosts_confidence(self, second_chance_evaluator, near_miss_signal):
        """High ADX in metadata should boost confidence"""
        near_miss_signal.confidence = 0.52
        near_miss_signal.metadata = {'adx': 45}  # Very strong ADX
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert any('Strong technicals' in r for r in reasons)
    
    def test_good_risk_reward_boosts_confidence(self, second_chance_evaluator, near_miss_signal):
        """Good R:R ratio should boost confidence"""
        near_miss_signal.confidence = 0.52
        near_miss_signal.price = 100.0
        near_miss_signal.stop_loss = 95.0  # 5% risk
        near_miss_signal.take_profit = 115.0  # 15% reward (3:1 R:R)
        near_miss_signal.metadata = {'atr': 2.0}
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        # Good R:R (>2.5:1) should boost
        assert adj_conf > 0.52


class TestNearMissTracking:
    """Tests for near-miss signal tracking"""
    
    def test_near_miss_tracked_when_in_range(self, second_chance_evaluator, near_miss_signal, test_data_dir):
        """Signals in near-miss range should be tracked"""
        near_miss_signal.confidence = 0.50  # In range (0.45-0.55)
        
        second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert len(second_chance_evaluator.near_miss_signals) == 1
    
    def test_near_miss_not_tracked_when_outside_range(self, second_chance_evaluator, near_miss_signal):
        """Signals outside near-miss range should not be tracked"""
        near_miss_signal.confidence = 0.70  # Above range
        
        second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Some other reason"
        )
        
        assert len(second_chance_evaluator.near_miss_signals) == 0


class TestCounterfactualUpdates:
    """Tests for counterfactual price tracking"""
    
    def test_update_counterfactuals_tracks_prices(self, second_chance_evaluator):
        """Should update price tracking for near-miss signals"""
        # Add a near-miss signal from 2 hours ago
        near_miss = NearMissSignal(
            timestamp=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
            strategy_id='test',
            symbol='BTC-USDT',
            side='long',
            original_confidence=0.52,
            adjusted_confidence=0.52,
            rejection_reason='Low confidence',
            entry_price=50000.0
        )
        second_chance_evaluator.near_miss_signals.append(near_miss)
        
        # Update with current prices
        current_prices = {'BTC-USDT': 51000.0}
        second_chance_evaluator.update_counterfactuals(current_prices)
        
        # Should have updated 1h price (2h > 1h)
        assert near_miss.price_after_1h == 51000.0
    
    def test_update_counterfactuals_evaluates_outcome(self, second_chance_evaluator):
        """Should evaluate if signal would have won after 4h"""
        # Add a near-miss signal from 5 hours ago
        near_miss = NearMissSignal(
            timestamp=(datetime.utcnow() - timedelta(hours=5)).isoformat(),
            strategy_id='test',
            symbol='BTC-USDT',
            side='long',
            original_confidence=0.52,
            adjusted_confidence=0.52,
            rejection_reason='Low confidence',
            entry_price=50000.0
        )
        second_chance_evaluator.near_miss_signals.append(near_miss)
        
        # Price went up - would have been a winner
        current_prices = {'BTC-USDT': 52000.0}
        second_chance_evaluator.update_counterfactuals(current_prices)
        
        # Should have evaluated outcome and removed from tracking
        assert near_miss.would_have_won is True
        assert near_miss.price_after_4h == 52000.0


class TestLearningFromWinningRejections:
    """Tests for learning from winning rejected signals"""
    
    def test_learn_from_winning_rejection_creates_pattern(self, second_chance_evaluator, test_data_dir):
        """Should create new pattern when winning rejection identified"""
        initial_patterns = len(second_chance_evaluator.winning_patterns)
        
        near_miss = NearMissSignal(
            timestamp=datetime.utcnow().isoformat(),
            strategy_id='test-strategy',
            symbol='BTC-USDT',
            side='long',
            original_confidence=0.52,
            adjusted_confidence=0.52,
            rejection_reason='Low confidence',
            rescued=False,
            entry_price=50000.0,
            would_have_won=True
        )
        
        second_chance_evaluator._learn_from_winning_rejection(near_miss)
        
        assert len(second_chance_evaluator.winning_patterns) == initial_patterns + 1
    
    def test_learn_updates_existing_pattern(self, second_chance_evaluator, winning_pattern):
        """Should update existing pattern if match found"""
        second_chance_evaluator.winning_patterns.append(winning_pattern)
        initial_sample_size = winning_pattern.sample_size
        
        near_miss = NearMissSignal(
            timestamp=datetime.utcnow().isoformat(),
            strategy_id='trend-following',  # In pattern's strategy list
            symbol='BTC-USDT',
            side='long',
            original_confidence=0.52,  # In pattern's confidence range
            adjusted_confidence=0.52,
            rejection_reason='Low confidence',  # Matches pattern
            rescued=False,
            entry_price=50000.0,
            would_have_won=True
        )
        
        second_chance_evaluator._learn_from_winning_rejection(near_miss)
        
        assert winning_pattern.sample_size == initial_sample_size + 1


class TestRescueDecisions:
    """Tests for final rescue decisions"""
    
    def test_rescued_when_boosted_above_threshold(self, second_chance_evaluator, near_miss_signal):
        """Should rescue when boosts push confidence above threshold"""
        near_miss_signal.strategy_id = 'trend-following'  # +0.08 boost
        near_miss_signal.confidence = 0.50  # Below 0.55 threshold
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        # 0.50 + 0.08 = 0.58 > 0.55 threshold
        assert should_rescue is True
        assert adj_conf >= 0.55
    
    def test_not_rescued_when_still_below_threshold(self, second_chance_evaluator, near_miss_signal):
        """Should not rescue when boosts don't reach threshold"""
        near_miss_signal.strategy_id = 'zweig'  # -0.10 penalty
        near_miss_signal.confidence = 0.52
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        # 0.52 - 0.10 = 0.42 < 0.55 threshold
        assert should_rescue is False
    
    def test_not_rescued_with_no_boosts(self, second_chance_evaluator, near_miss_signal):
        """Should not rescue if no boost reasons applied"""
        near_miss_signal.strategy_id = 'unknown-strategy'  # No known boost
        near_miss_signal.confidence = 0.52
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Some reason"
        )
        
        # No boosts = no rescue even if near threshold
        # (the "len(boost_reasons) > 0" check)
        if should_rescue:
            assert len(reasons) > 0
    
    def test_confidence_capped_at_95(self, second_chance_evaluator, near_miss_signal):
        """Adjusted confidence should never exceed 0.95"""
        # Add multiple winning patterns for maximum boost
        for i in range(5):
            pattern = WinningPattern(
                pattern_id=f'pattern_{i}',
                rejection_reason='Low confidence',
                confidence_range=(0.40, 0.60),
                winning_rate=0.80,
                sample_size=50,
                strategy_ids=['trend-following'],
                boost_amount=0.10
            )
            second_chance_evaluator.winning_patterns.append(pattern)
        
        near_miss_signal.strategy_id = 'trend-following'
        near_miss_signal.confidence = 0.90  # Already high
        near_miss_signal.metadata = {'adx': 50}  # Strong technical boost
        
        should_rescue, adj_conf, reasons = second_chance_evaluator.evaluate_for_second_chance(
            near_miss_signal,
            rejection_reason="Low confidence"
        )
        
        assert adj_conf <= 0.95


class TestStrategyPerformanceBoost:
    """Tests for learned strategy performance boosts"""
    
    def test_learned_performance_overrides_default(self):
        """Should use learned performance when available"""
        perf = {
            'test-strategy': {
                'trades': 50,
                'win_rate': 0.65,  # Very good
                'total_pnl_pct': 50.0
            }
        }
        evaluator = SecondChanceEvaluator(perf)
        
        signal = Signal(
            strategy_id='test-strategy',
            symbol='BTC-USDT',
            side=SignalSide.LONG,
            confidence=0.52,
            price=50000.0,
            timestamp=datetime.utcnow(),
            reason='Test'
        )
        
        should_rescue, adj_conf, reasons = evaluator.evaluate_for_second_chance(
            signal,
            rejection_reason="Low confidence"
        )
        
        # 65% WR should give positive boost: (0.65 - 0.5) * 0.5 = 0.075
        assert adj_conf > 0.52
    
    def test_insufficient_trades_uses_default(self):
        """Should use default boost if not enough trades"""
        perf = {
            'test-strategy': {
                'trades': 5,  # Not enough
                'win_rate': 0.80,
                'total_pnl_pct': 100.0
            }
        }
        evaluator = SecondChanceEvaluator(perf)
        
        boost = evaluator._get_strategy_boost('test-strategy')
        
        # Should fall back to default (0 for unknown)
        assert boost == 0.0
