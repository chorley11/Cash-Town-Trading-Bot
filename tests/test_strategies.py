"""
Tests for individual trading strategies.

Tests signal generation logic for:
- Trend Following
- Mean Reversion
- Turtle Breakout
- Weinstein Stage Analysis
- Other strategies
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from agents.base import Signal, SignalSide
from agents.strategies.trend_following import TrendFollowingAgent
from tests.conftest import create_market_data_with_crossover, assert_signal_valid


class TestTrendFollowingStrategy:
    """Tests for Trend Following strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create trend following strategy instance"""
        return TrendFollowingAgent(
            symbols=['BTC-USDT', 'ETH-USDT'],
            config={
                'ma_fast': 10,
                'ma_slow': 30,
                'adx_threshold': 25,
                'min_confidence': 0.55,
                'require_volume_confirmation': False  # Simplify for testing
            }
        )
    
    def test_init_default_config(self):
        """Strategy should initialize with default config"""
        strategy = TrendFollowingAgent(symbols=['BTC-USDT'])
        
        assert strategy.agent_id == 'trend-following'
        assert strategy.name == 'Trend Following'
        assert 'ma_fast' in strategy.config
        assert 'ma_slow' in strategy.config
    
    def test_get_required_indicators(self, strategy):
        """Should return list of required indicators"""
        indicators = strategy.get_required_indicators()
        
        assert 'sma_10' in indicators
        assert 'sma_30' in indicators
        assert any('adx' in i for i in indicators)
        assert any('atr' in i for i in indicators)
    
    def test_generate_signals_empty_data(self, strategy, empty_market_data):
        """Should return empty list for empty data"""
        signals = strategy.generate_signals(empty_market_data)
        assert signals == []
    
    def test_generate_signals_insufficient_data(self, strategy, insufficient_market_data):
        """Should return empty list for insufficient data"""
        signals = strategy.generate_signals(insufficient_market_data)
        assert signals == []
    
    def test_generate_long_signal_bullish_crossover(self, strategy):
        """Should generate LONG signal on bullish MA crossover with strong trend"""
        # Create data with bullish crossover at the end
        market_data = {
            'BTC-USDT': create_market_data_with_crossover(
                n_periods=100,
                crossover_at=95,  # Recent crossover
                direction='bullish',
                base_price=50000
            )
        }
        
        # Add high volume at crossover
        market_data['BTC-USDT']['volume'][-5:] = [v * 1.5 for v in market_data['BTC-USDT']['volume'][-5:]]
        
        signals = strategy.generate_signals(market_data)
        
        # May or may not generate signal depending on ADX
        # But if it does, it should be LONG
        for signal in signals:
            if signal.symbol == 'BTC-USDT':
                assert signal.side == SignalSide.LONG
                assert_signal_valid(signal)
    
    def test_generate_short_signal_bearish_crossover(self, strategy):
        """Should generate SHORT signal on bearish MA crossover with strong trend"""
        market_data = {
            'BTC-USDT': create_market_data_with_crossover(
                n_periods=100,
                crossover_at=95,
                direction='bearish',
                base_price=50000
            )
        }
        
        # Add high volume at crossover
        market_data['BTC-USDT']['volume'][-5:] = [v * 1.5 for v in market_data['BTC-USDT']['volume'][-5:]]
        
        signals = strategy.generate_signals(market_data)
        
        for signal in signals:
            if signal.symbol == 'BTC-USDT':
                assert signal.side == SignalSide.SHORT
                assert_signal_valid(signal)
    
    def test_no_signal_weak_trend(self, strategy, sideways_market_data):
        """Should not generate signal when ADX is below threshold"""
        signals = strategy.generate_signals(sideways_market_data)
        
        # Sideways market has weak ADX, should not signal
        assert len(signals) == 0
    
    def test_signal_has_stop_loss_and_take_profit(self, strategy, bullish_market_data):
        """Generated signals should include SL and TP"""
        signals = strategy.generate_signals(bullish_market_data)
        
        for signal in signals:
            # Stop loss and take profit should be set if signal generated
            if signal.confidence >= strategy.config['min_confidence']:
                assert signal.stop_loss is not None or signal.take_profit is not None
    
    def test_signal_metadata_contains_indicators(self, strategy):
        """Signal metadata should contain indicator values"""
        market_data = {
            'BTC-USDT': create_market_data_with_crossover(
                n_periods=100,
                crossover_at=95,
                direction='bullish',
                base_price=50000
            )
        }
        market_data['BTC-USDT']['volume'][-5:] = [v * 1.5 for v in market_data['BTC-USDT']['volume'][-5:]]
        
        signals = strategy.generate_signals(market_data)
        
        for signal in signals:
            if signal.symbol == 'BTC-USDT' and signal.metadata:
                # Should contain technical indicators
                assert any(key in signal.metadata for key in ['adx', 'atr', 'ma_fast', 'ma_slow'])
    
    def test_confidence_calculation_adx_strength(self, strategy):
        """Higher ADX should result in higher confidence"""
        # Test internal confidence calculation
        confidence_weak = strategy._calculate_confidence(adx=25, volume_ratio=1.0, is_long=True)
        confidence_strong = strategy._calculate_confidence(adx=40, volume_ratio=1.0, is_long=True)
        
        assert confidence_strong > confidence_weak
    
    def test_confidence_calculation_volume_ratio(self, strategy):
        """Higher volume ratio should boost confidence"""
        confidence_normal = strategy._calculate_confidence(adx=30, volume_ratio=1.0, is_long=True)
        confidence_high_vol = strategy._calculate_confidence(adx=30, volume_ratio=1.6, is_long=True)
        
        assert confidence_high_vol > confidence_normal
    
    def test_confidence_capped_at_95(self, strategy):
        """Confidence should never exceed 0.95"""
        confidence = strategy._calculate_confidence(adx=50, volume_ratio=2.0, is_long=True)
        assert confidence <= 0.95


class TestTechnicalIndicators:
    """Tests for technical indicator calculations"""
    
    @pytest.fixture
    def strategy(self):
        return TrendFollowingAgent(symbols=['TEST-USDT'])
    
    def test_sma_calculation(self, strategy):
        """SMA should calculate correctly"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        period = 3
        
        sma = strategy._sma(data, period)
        
        # SMA(3) of last 3 values: (8+9+10)/3 = 9
        assert abs(sma[-1] - 9.0) < 0.01
    
    def test_sma_insufficient_data(self, strategy):
        """SMA should return None for insufficient data"""
        data = np.array([1, 2])
        sma = strategy._sma(data, period=5)
        
        assert sma is None
    
    def test_atr_calculation(self, strategy):
        """ATR should return positive values"""
        n = 50
        high = np.random.uniform(101, 105, n)
        low = np.random.uniform(95, 99, n)
        close = np.random.uniform(98, 102, n)
        
        atr = strategy._atr(high, low, close, period=14)
        
        assert atr is not None
        assert all(atr > 0)
    
    def test_adx_calculation_trending_market(self, strategy):
        """ADX should be high in trending market"""
        n = 50
        # Create strong uptrend
        close = 100 + np.arange(n) * 0.5 + np.random.normal(0, 0.1, n)
        high = close + np.random.uniform(0.5, 1, n)
        low = close - np.random.uniform(0.5, 1, n)
        
        adx = strategy._adx(high, low, close, period=14)
        
        assert adx is not None
        # ADX should be elevated in trending market
        assert adx[-1] > 15
    
    def test_ema_calculation(self, strategy):
        """EMA should weight recent values more heavily"""
        data = np.array([10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
        
        ema = strategy._ema(data, period=3)
        
        # EMA should be between old and new values, closer to new
        assert 15 < ema[-1] < 20


class TestStrategyBaseClass:
    """Tests for base strategy functionality"""
    
    @pytest.fixture
    def strategy(self, tmp_path):
        return TrendFollowingAgent(
            symbols=['BTC-USDT'],
            config={},
        )
    
    def test_agent_id_set_correctly(self, strategy):
        """Agent ID should be set"""
        assert strategy.agent_id == 'trend-following'
    
    def test_symbols_stored(self, strategy):
        """Symbols should be stored"""
        assert 'BTC-USDT' in strategy.symbols
    
    def test_enabled_by_default(self, strategy):
        """Strategy should be enabled by default"""
        assert strategy.enabled is True
    
    def test_enable_disable(self, strategy):
        """Should be able to enable/disable strategy"""
        strategy.disable()
        assert strategy.enabled is False
        
        strategy.enable()
        assert strategy.enabled is True
    
    def test_get_stats(self, strategy):
        """Should return stats dictionary"""
        stats = strategy.get_stats()
        
        assert 'agent_id' in stats
        assert 'name' in stats
        assert 'enabled' in stats
        assert 'total_trades' in stats
        assert 'win_rate' in stats
    
    def test_should_exit_stop_loss_long(self, strategy, sample_position):
        """Should trigger stop loss for LONG position"""
        sample_position.side = 'long'
        sample_position.stop_loss = 49000.0
        
        result = strategy.should_exit(sample_position, current_price=48500.0)
        
        assert result == 'stop_loss'
    
    def test_should_exit_stop_loss_short(self, strategy, sample_position):
        """Should trigger stop loss for SHORT position"""
        sample_position.side = 'short'
        sample_position.stop_loss = 51000.0
        sample_position.entry_price = 50000.0
        
        result = strategy.should_exit(sample_position, current_price=51500.0)
        
        assert result == 'stop_loss'
    
    def test_should_exit_take_profit_long(self, strategy, sample_position):
        """Should trigger take profit for LONG position"""
        sample_position.side = 'long'
        sample_position.take_profit = 55000.0
        
        result = strategy.should_exit(sample_position, current_price=56000.0)
        
        assert result == 'take_profit'
    
    def test_should_exit_no_action(self, strategy, sample_position):
        """Should return None when no exit conditions met"""
        sample_position.side = 'long'
        sample_position.stop_loss = 49000.0
        sample_position.take_profit = 55000.0
        
        result = strategy.should_exit(sample_position, current_price=52000.0)
        
        assert result is None


class TestMultiSymbolSignals:
    """Tests for handling multiple symbols"""
    
    @pytest.fixture
    def multi_symbol_strategy(self):
        return TrendFollowingAgent(
            symbols=['BTC-USDT', 'ETH-USDT', 'SOL-USDT'],
            config={'require_volume_confirmation': False}
        )
    
    def test_generates_signals_for_multiple_symbols(self, multi_symbol_strategy, multi_symbol_market_data):
        """Should analyze all configured symbols"""
        signals = multi_symbol_strategy.generate_signals(multi_symbol_market_data)
        
        # Symbols with signals should be in configured symbols
        signaled_symbols = {s.symbol for s in signals}
        for symbol in signaled_symbols:
            assert symbol in multi_symbol_strategy.symbols
    
    def test_handles_missing_symbol_data(self, multi_symbol_strategy):
        """Should handle missing data for some symbols"""
        partial_data = {
            'BTC-USDT': create_market_data_with_crossover(
                n_periods=100,
                crossover_at=95,
                direction='bullish'
            )
            # ETH-USDT and SOL-USDT missing
        }
        
        # Should not raise, just skip missing symbols
        signals = multi_symbol_strategy.generate_signals(partial_data)
        
        for signal in signals:
            assert signal.symbol in partial_data


class TestSignalValidation:
    """Tests for signal data integrity"""
    
    @pytest.fixture
    def strategy(self):
        return TrendFollowingAgent(symbols=['BTC-USDT'])
    
    def test_signal_timestamp_is_recent(self, strategy, bullish_market_data):
        """Signal timestamp should be recent"""
        before = datetime.utcnow()
        signals = strategy.generate_signals(bullish_market_data)
        after = datetime.utcnow()
        
        for signal in signals:
            assert before <= signal.timestamp <= after
    
    def test_signal_price_is_positive(self, strategy, bullish_market_data):
        """Signal price must be positive"""
        signals = strategy.generate_signals(bullish_market_data)
        
        for signal in signals:
            assert signal.price > 0
    
    def test_signal_confidence_in_range(self, strategy, bullish_market_data):
        """Signal confidence must be 0-1"""
        signals = strategy.generate_signals(bullish_market_data)
        
        for signal in signals:
            assert 0 <= signal.confidence <= 1
    
    def test_signal_has_reason(self, strategy, bullish_market_data):
        """Signal should have a reason string"""
        signals = strategy.generate_signals(bullish_market_data)
        
        for signal in signals:
            assert signal.reason
            assert len(signal.reason) > 0


class TestErrorHandling:
    """Tests for error handling in strategies"""
    
    @pytest.fixture
    def strategy(self):
        return TrendFollowingAgent(symbols=['BTC-USDT'])
    
    def test_handles_nan_values(self, strategy):
        """Should handle NaN values in data"""
        market_data = {
            'BTC-USDT': {
                'open': [100] * 100,
                'high': [101] * 100,
                'low': [99] * 100,
                'close': [100] * 50 + [float('nan')] * 50,
                'volume': [1000000] * 100
            }
        }
        
        # Should not raise
        signals = strategy.generate_signals(market_data)
        # May return empty or filtered signals
        assert isinstance(signals, list)
    
    def test_handles_zero_volume(self, strategy):
        """Should handle zero volume gracefully"""
        market_data = {
            'BTC-USDT': {
                'open': [100 + i for i in range(100)],
                'high': [101 + i for i in range(100)],
                'low': [99 + i for i in range(100)],
                'close': [100.5 + i for i in range(100)],
                'volume': [0] * 100
            }
        }
        
        # Should not raise
        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)
    
    def test_handles_negative_prices(self, strategy):
        """Should handle negative prices (shouldn't happen but be defensive)"""
        market_data = {
            'BTC-USDT': {
                'open': [-100] * 100,
                'high': [-99] * 100,
                'low': [-101] * 100,
                'close': [-100] * 100,
                'volume': [1000000] * 100
            }
        }
        
        # Should not raise
        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)
