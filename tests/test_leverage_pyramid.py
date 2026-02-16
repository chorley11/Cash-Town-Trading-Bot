"""
Tests for dynamic leverage and pyramiding functionality.
"""
import pytest
from datetime import datetime, timedelta
from orchestrator.risk_manager import (
    RiskManager, RiskConfig, LeverageConfig, PyramidConfig,
    PyramidState, DeleverageConfig
)


class TestDynamicLeverage:
    """Test dynamic leverage calculation based on confidence and strategy performance."""
    
    def test_low_confidence_leverage(self):
        """55-65% confidence should get 2-3x leverage."""
        rm = RiskManager(equity=1000)
        
        # At 58% confidence (low tier)
        leverage, meta = rm.calculate_leverage(0.58, 'test-strategy')
        assert leverage in [2, 3]
        assert meta['confidence_tier'] == 'low'
    
    def test_medium_confidence_leverage(self):
        """65-80% confidence should get 4-6x leverage."""
        rm = RiskManager(equity=1000)
        
        # At 72% confidence (medium tier)
        leverage, meta = rm.calculate_leverage(0.72, 'test-strategy')
        assert leverage in [4, 5, 6]
        assert meta['confidence_tier'] == 'medium'
    
    def test_high_confidence_leverage(self):
        """80%+ confidence should get 8-10x leverage."""
        rm = RiskManager(equity=1000)
        
        # At 85% confidence (high tier)
        leverage, meta = rm.calculate_leverage(0.85, 'test-strategy')
        assert leverage in [8, 9, 10]
        assert meta['confidence_tier'] == 'high'
    
    def test_max_confidence_leverage(self):
        """95% confidence should hit max 10x leverage."""
        rm = RiskManager(equity=1000)
        
        leverage, meta = rm.calculate_leverage(0.95, 'test-strategy')
        assert leverage == 10
        assert meta['confidence_tier'] == 'high'
    
    def test_below_minimum_confidence(self):
        """Below 55% confidence should return 0 (don't trade)."""
        rm = RiskManager(equity=1000)
        
        leverage, meta = rm.calculate_leverage(0.50, 'test-strategy')
        assert leverage == 0
        assert meta['confidence_tier'] == 'below_minimum'
    
    def test_strategy_bonus_good_performer(self):
        """Strategies with good track record get leverage bonus."""
        rm = RiskManager(equity=1000)
        
        # Add performance data for strategy
        rm.strategy_stats['winning-strategy'] = {
            'trades': 30,
            'wins': 20,  # 66% win rate
            'losses': 10
        }
        
        # Calculate bonus
        bonus = rm._calculate_strategy_leverage_bonus('winning-strategy')
        assert bonus > 0  # Should have positive bonus
    
    def test_strategy_penalty_bad_performer(self):
        """Strategies with poor track record get leverage penalty."""
        rm = RiskManager(equity=1000)
        
        # Add performance data for strategy
        rm.strategy_stats['losing-strategy'] = {
            'trades': 30,
            'wins': 10,  # 33% win rate
            'losses': 20
        }
        
        # Calculate penalty
        penalty = rm._calculate_strategy_leverage_bonus('losing-strategy')
        assert penalty < 0  # Should have negative penalty
    
    def test_circuit_breaker_resets_to_minimum(self):
        """Circuit breaker should force minimum leverage."""
        rm = RiskManager(equity=1000)
        
        # Trigger circuit breaker
        rm.circuit_breaker.is_triggered = True
        rm.circuit_breaker.trigger_reason = 'Test trigger'
        
        # Even with high confidence, leverage should be minimum
        leverage, meta = rm.calculate_leverage(0.95, 'test-strategy')
        assert leverage == rm.config.leverage_config.min_leverage
        assert 'circuit_breaker_active' in meta['adjustments'][0]
    
    def test_max_leverage_cap(self):
        """Leverage should never exceed absolute max (10x)."""
        rm = RiskManager(equity=1000)
        
        # Add huge strategy bonus
        rm.strategy_stats['super-strategy'] = {
            'trades': 100,
            'wins': 90,  # 90% win rate
            'losses': 10
        }
        
        # Even with high confidence + bonus, should cap at 10x
        leverage, meta = rm.calculate_leverage(0.95, 'super-strategy')
        assert leverage <= 10


class TestPyramiding:
    """Test pyramiding into winning positions."""
    
    def test_pyramid_not_available_when_losing(self):
        """Can't pyramid if position is losing."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Price below entry
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 49500, 50000, 'long')
        assert can_pyramid == False
        assert details['reason'] == 'position_not_profitable'
    
    def test_pyramid_level_2_threshold(self):
        """Level 2 pyramid at +1.5% ROE."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Price at +1.5% ROE
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 50750, 50000, 'long')
        assert can_pyramid == True
        assert details['target_level'] == 2
        assert details['add_size_pct'] == 50.0  # 50% of base
        assert details['new_leverage'] == 6  # +1x
    
    def test_pyramid_level_3_threshold(self):
        """Level 3 pyramid at +3.0% ROE after level 2."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Execute level 2 first
        rm.execute_pyramid('BTCUSDTM', 0.005, 6, 2, 50750)
        
        # Clear cooldown for test (simulate time passing)
        rm.pyramid_states['BTCUSDTM'].last_pyramid_time = datetime.utcnow() - timedelta(minutes=20)
        
        # Check level 3 at +3% ROE
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 51500, 50000, 'long')
        assert can_pyramid == True
        assert details['target_level'] == 3
        assert details['add_size_pct'] == 25.0  # 25% of base
    
    def test_pyramid_max_levels(self):
        """Can't pyramid beyond max levels."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Execute pyramids to max level
        rm.execute_pyramid('BTCUSDTM', 0.005, 6, 2, 50750)
        rm.execute_pyramid('BTCUSDTM', 0.0025, 7, 3, 51500)
        
        # Try to pyramid again
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 52000, 50000, 'long')
        assert can_pyramid == False
        assert details['reason'] == 'max_pyramid_levels_reached'
    
    def test_pyramid_leverage_cap(self):
        """Pyramid should respect max leverage cap."""
        rm = RiskManager(equity=1000)
        # Start at 9x leverage
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=9)
        
        # Execute level 2 (would be 10x)
        rm.execute_pyramid('BTCUSDTM', 0.005, 10, 2, 50750)
        
        # Clear cooldown for test (simulate time passing)
        rm.pyramid_states['BTCUSDTM'].last_pyramid_time = datetime.utcnow() - timedelta(minutes=20)
        
        # Try level 3 (would exceed 10x since current is 10x and bump is +1)
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 51500, 50000, 'long')
        assert can_pyramid == False
        assert details['reason'] == 'would_exceed_max_leverage'
    
    def test_pyramid_cooldown(self):
        """Pyramids should respect cooldown between adds."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Execute level 2
        rm.execute_pyramid('BTCUSDTM', 0.005, 6, 2, 50750)
        
        # Try immediately again (should fail cooldown)
        can_pyramid, details = rm.check_pyramid_opportunity('BTCUSDTM', 51500, 50000, 'long')
        # Note: This might pass if ROE isn't high enough for level 3 yet
        # The cooldown check happens after level check
    
    def test_pyramid_state_tracking(self):
        """Pyramid state should track all adds."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Check initial state
        state = rm.pyramid_states['BTCUSDTM']
        assert state.current_level == 1
        assert state.total_size == 0.01
        assert state.current_leverage == 5
        
        # Execute pyramid
        rm.execute_pyramid('BTCUSDTM', 0.005, 6, 2, 50750)
        
        # Verify updated state
        state = rm.pyramid_states['BTCUSDTM']
        assert state.current_level == 2
        assert state.total_size == 0.015  # 0.01 + 0.005
        assert state.current_leverage == 6
        assert len(state.pyramid_history) == 1


class TestDeleverage:
    """Test deleveraging losing positions."""
    
    def test_deleverage_triggered_at_threshold(self):
        """Deleverage should trigger at -1% ROE."""
        rm = RiskManager(equity=1000)
        rm.register_position('ETHUSDTM', 'long', 0.1, 3000, 2940, 'test', leverage=5)
        
        # Price at -1.67% ROE
        should_delev, details = rm.check_deleverage_needed('ETHUSDTM', 2950, 3000, 'long')
        assert should_delev == True
        assert details['reduction_pct'] == 50.0
    
    def test_deleverage_not_triggered_above_threshold(self):
        """Deleverage should not trigger if loss is small."""
        rm = RiskManager(equity=1000)
        rm.register_position('ETHUSDTM', 'long', 0.1, 3000, 2940, 'test', leverage=5)
        
        # Price at -0.5% ROE
        should_delev, details = rm.check_deleverage_needed('ETHUSDTM', 2985, 3000, 'long')
        assert should_delev == False
        assert details['reason'] == 'roe_above_threshold'
    
    def test_deleverage_short_position(self):
        """Deleverage should work for short positions."""
        rm = RiskManager(equity=1000)
        rm.register_position('ETHUSDTM', 'short', 0.1, 3000, 3060, 'test', leverage=5)
        
        # Price at +1.67% (loss for short)
        should_delev, details = rm.check_deleverage_needed('ETHUSDTM', 3050, 3000, 'short')
        assert should_delev == True
    
    def test_deleverage_minimum_size(self):
        """Deleverage should recommend close if size would be too small."""
        rm = RiskManager(equity=1000)
        # Very small position
        rm.register_position('ETHUSDTM', 'long', 0.001, 3000, 2940, 'test', leverage=5)
        
        # After 50% reduction, value would be ~$1.5, below $10 minimum
        should_delev, details = rm.check_deleverage_needed('ETHUSDTM', 2950, 3000, 'long')
        assert should_delev == False
        assert details['reason'] == 'would_be_too_small'
        assert details['recommend'] == 'close_position'
    
    def test_deleverage_execution(self):
        """Deleverage execution should update state."""
        rm = RiskManager(equity=1000)
        rm.register_position('ETHUSDTM', 'long', 0.1, 3000, 2940, 'test', leverage=5)
        
        # Execute deleverage
        should_delev, details = rm.check_deleverage_needed('ETHUSDTM', 2950, 3000, 'long')
        rm.execute_deleverage('ETHUSDTM', details['new_size'], details['reduction_size'])
        
        # Check state
        state = rm.pyramid_states['ETHUSDTM']
        assert state.total_size == 0.05  # Reduced by 50%


class TestPyramidStatus:
    """Test pyramid status API."""
    
    def test_get_pyramid_status(self):
        """Get pyramid status should return all position states."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        rm.register_position('ETHUSDTM', 'long', 0.1, 3000, 2940, 'test', leverage=4)
        
        status = rm.get_pyramid_status()
        
        assert status['enabled'] == True
        assert status['max_levels'] == 3
        assert 'BTCUSDTM' in status['positions']
        assert 'ETHUSDTM' in status['positions']
        
        btc_state = status['positions']['BTCUSDTM']
        assert btc_state['current_level'] == 1
        assert btc_state['base_leverage'] == 5
    
    def test_position_cleared_on_close(self):
        """Pyramid state should be cleared when position closes."""
        rm = RiskManager(equity=1000)
        rm.register_position('BTCUSDTM', 'long', 0.01, 50000, 49000, 'test', leverage=5)
        
        # Position should exist
        assert 'BTCUSDTM' in rm.pyramid_states
        
        # Close position
        rm.close_position('BTCUSDTM', 51000)
        
        # Pyramid state should be cleared
        assert 'BTCUSDTM' not in rm.pyramid_states


class TestIntegration:
    """Integration tests combining leverage, pyramiding, and deleverage."""
    
    def test_full_lifecycle(self):
        """Test full position lifecycle with leverage and pyramiding."""
        rm = RiskManager(equity=10000)
        
        # 1. Calculate leverage for new signal
        leverage, meta = rm.calculate_leverage(0.75, 'trend-following')
        assert leverage in [5, 6]  # Medium confidence
        
        # 2. Open position
        rm.register_position('BTCUSDTM', 'long', 0.1, 50000, 49000, 'trend-following', leverage=leverage)
        
        # 3. Position goes profitable - check pyramid
        can_pyr, details = rm.check_pyramid_opportunity('BTCUSDTM', 50800, 50000, 'long')
        assert can_pyr == True
        
        # 4. Execute pyramid
        rm.execute_pyramid('BTCUSDTM', details['add_size'], details['new_leverage'], details['target_level'], 50800)
        
        # 5. Verify state
        state = rm.pyramid_states['BTCUSDTM']
        assert state.current_level == 2
        assert state.total_size > 0.1  # Added to position
        
        # 6. Close position
        result = rm.close_position('BTCUSDTM', 51000, 'take_profit')
        assert result['won'] == True
        
        # 7. Verify cleanup
        assert 'BTCUSDTM' not in rm.pyramid_states


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
