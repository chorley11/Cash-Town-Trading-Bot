"""
Tests for Ronin Trader strategies (ported Feb 2026)

Tests:
1. Ronin Volume Spike
2. Ronin Funding Fade
3. Ronin Momentum Breakout
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from agents.strategies.ronin_volume_spike import RoninVolumeSpikeAgent
from agents.strategies.ronin_funding_fade import RoninFundingFadeAgent
from agents.strategies.ronin_momentum_breakout import RoninMomentumBreakoutAgent
from agents.base import SignalSide


# ============== Fixtures ==============

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data with normal volume"""
    np.random.seed(42)
    n = 100
    
    returns = np.random.normal(0, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = (close + np.roll(close, 1)) / 2
    volume = np.random.uniform(1000, 3000, n)  # Normal volume
    
    return {
        'open': open_price.tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': volume.tolist(),
    }


@pytest.fixture
def volume_spike_data():
    """Generate data with a volume spike on last candle"""
    np.random.seed(42)
    n = 50
    
    close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    high = close + np.abs(np.random.normal(0, 0.3, n))
    low = close - np.abs(np.random.normal(0, 0.3, n))
    open_price = close + np.random.normal(0, 0.2, n)
    
    # Normal volume then spike
    volume = np.random.uniform(1000, 2000, n)
    volume[-1] = 8000  # 4x spike on last candle
    
    # Make last candle bullish (price up)
    close[-1] = close[-2] * 1.02  # +2% move
    high[-1] = close[-1] * 1.005
    
    return {
        'open': open_price.tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': volume.tolist(),
    }


@pytest.fixture
def breakout_data():
    """Generate data with a breakout pattern"""
    np.random.seed(123)
    n = 50
    
    # Range-bound for first 45 candles
    base = 100
    noise = np.random.normal(0, 1, 45)
    close_range = base + noise
    
    # Then breakout upwards
    close_breakout = base + 5 + np.array([0, 1, 2, 3, 4])  # Breaking above range
    close = np.concatenate([close_range, close_breakout])
    
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_price = close + np.random.normal(0, 0.3, n)
    
    # Volume spike on breakout
    volume = np.random.uniform(1000, 2000, n)
    volume[-1] = 4000  # 2x volume on breakout
    
    return {
        'open': open_price.tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': volume.tolist(),
    }


@pytest.fixture
def ranging_market():
    """Generate ranging/sideways market data"""
    np.random.seed(456)
    n = 50
    
    # Oscillate around 100
    t = np.linspace(0, 4 * np.pi, n)
    close = 100 + 3 * np.sin(t) + np.random.normal(0, 0.5, n)
    high = close + np.abs(np.random.normal(0, 0.3, n))
    low = close - np.abs(np.random.normal(0, 0.3, n))
    open_price = close + np.random.normal(0, 0.2, n)
    volume = np.random.uniform(1000, 2000, n)
    
    return {
        'open': open_price.tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': volume.tolist(),
    }


# ============== Ronin Volume Spike Tests ==============

class TestRoninVolumeSpike:
    """Tests for RoninVolumeSpikeAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT', 'ETHUSDT'])
        
        assert agent.agent_id == 'ronin-volume-spike'
        assert agent.name == 'Ronin Volume Spike'
        assert len(agent.symbols) == 2
        assert agent.config['volume_threshold'] == 3.0
    
    def test_custom_config(self):
        """Test agent accepts custom config"""
        config = {
            'volume_threshold': 4.0,
            'stop_loss_pct': 0.02,
        }
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT'], config=config)
        
        assert agent.config['volume_threshold'] == 4.0
        assert agent.config['stop_loss_pct'] == 0.02
        assert agent.config['take_profit_pct'] == 0.03  # Default preserved
    
    def test_required_indicators(self):
        """Test required indicators list"""
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT'])
        indicators = agent.get_required_indicators()
        
        assert 'volume_sma_20' in indicators
    
    def test_no_signal_normal_volume(self, sample_ohlcv):
        """Test no signal when volume is normal"""
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': sample_ohlcv}
        
        signals = agent.generate_signals(market_data)
        assert len(signals) == 0
    
    def test_signal_on_volume_spike(self, volume_spike_data):
        """Test signal generated on volume spike with price move"""
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': volume_spike_data}
        
        signals = agent.generate_signals(market_data)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.strategy_id == 'ronin-volume-spike'
        assert signal.symbol == 'BTCUSDT'
        assert signal.side == SignalSide.LONG  # Price went up
        assert signal.confidence >= 0.5
        assert 'volume spike' in signal.reason.lower()
        assert signal.metadata['strategy_origin'] == 'ronin-trader'
    
    def test_stop_loss_take_profit_calculation(self, volume_spike_data):
        """Test SL/TP are calculated correctly"""
        agent = RoninVolumeSpikeAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': volume_spike_data}
        
        signals = agent.generate_signals(market_data)
        signal = signals[0]
        
        price = signal.price
        expected_sl = price * 0.99  # 1% stop
        expected_tp = price * 1.03  # 3% take profit
        
        assert abs(signal.stop_loss - expected_sl) < 0.01
        assert abs(signal.take_profit - expected_tp) < 0.01


# ============== Ronin Funding Fade Tests ==============

class TestRoninFundingFade:
    """Tests for RoninFundingFadeAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        
        assert agent.agent_id == 'ronin-funding-fade'
        assert agent.name == 'Ronin Funding Fade'
        assert agent.config['funding_threshold'] == 0.0005
    
    def test_required_indicators(self):
        """Test required indicators include funding rate"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        indicators = agent.get_required_indicators()
        
        assert 'funding_rate' in indicators
    
    def test_no_signal_neutral_funding(self, sample_ohlcv):
        """Test no signal when funding is near zero"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        agent.set_funding_data({'BTCUSDT': 0.0001})  # Below threshold
        
        market_data = {'BTCUSDT': sample_ohlcv}
        signals = agent.generate_signals(market_data)
        
        assert len(signals) == 0
    
    def test_short_on_high_positive_funding(self, sample_ohlcv):
        """Test SHORT signal when funding is high positive"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        agent.set_funding_data({'BTCUSDT': 0.001})  # 0.1% - very high
        
        market_data = {'BTCUSDT': sample_ohlcv}
        signals = agent.generate_signals(market_data)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == SignalSide.SHORT
        assert 'fading crowd' in signal.reason.lower()
    
    def test_long_on_high_negative_funding(self, sample_ohlcv):
        """Test LONG signal when funding is high negative"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        agent.set_funding_data({'BTCUSDT': -0.001})  # -0.1% - very negative
        
        market_data = {'BTCUSDT': sample_ohlcv}
        signals = agent.generate_signals(market_data)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == SignalSide.LONG
        assert signal.metadata['strategy_origin'] == 'ronin-trader'
    
    def test_confidence_scales_with_funding(self, sample_ohlcv):
        """Test confidence increases with more extreme funding"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        
        # Moderate funding
        agent.set_funding_data({'BTCUSDT': 0.0006})
        signals1 = agent.generate_signals({'BTCUSDT': sample_ohlcv})
        
        # Extreme funding
        agent.set_funding_data({'BTCUSDT': 0.002})
        signals2 = agent.generate_signals({'BTCUSDT': sample_ohlcv})
        
        assert signals2[0].confidence > signals1[0].confidence
    
    def test_exit_on_funding_normalization(self, sample_ohlcv):
        """Test exit signal when funding normalizes"""
        agent = RoninFundingFadeAgent(symbols=['BTCUSDT'])
        
        # Mock position
        class MockPosition:
            side = 'short'
            stop_loss = 110
            take_profit = 90
        
        position = MockPosition()
        
        # Funding still elevated
        exit_reason = agent.should_exit(position, 100, current_funding=0.0008)
        assert exit_reason is None
        
        # Funding normalized
        exit_reason = agent.should_exit(position, 100, current_funding=0.0001)
        assert exit_reason == 'funding_normalized'


# ============== Ronin Momentum Breakout Tests ==============

class TestRoninMomentumBreakout:
    """Tests for RoninMomentumBreakoutAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        
        assert agent.agent_id == 'ronin-momentum-breakout'
        assert agent.name == 'Ronin Momentum Breakout'
        assert agent.config['lookback_period'] == 20
        assert agent.config['volume_multiplier'] == 1.5
    
    def test_required_indicators(self):
        """Test required indicators list"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        indicators = agent.get_required_indicators()
        
        assert any('highest' in ind for ind in indicators)
        assert any('lowest' in ind for ind in indicators)
        assert any('volume_sma' in ind for ind in indicators)
    
    def test_no_signal_in_range(self, ranging_market):
        """Test no signal when price is range-bound"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': ranging_market}
        
        signals = agent.generate_signals(market_data)
        assert len(signals) == 0
    
    def test_long_on_bullish_breakout(self, breakout_data):
        """Test LONG signal on bullish breakout with volume"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': breakout_data}
        
        signals = agent.generate_signals(market_data)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == SignalSide.LONG
        assert 'bullish breakout' in signal.reason.lower()
        assert signal.metadata['volume_confirmed'] == True
        assert signal.metadata['strategy_origin'] == 'ronin-trader'
    
    def test_stop_loss_below_breakout_level(self, breakout_data):
        """Test stop loss is placed below breakout level"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': breakout_data}
        
        signals = agent.generate_signals(market_data)
        signal = signals[0]
        
        # Stop should be below the range high (breakout level)
        range_high = signal.metadata['range_high']
        assert signal.stop_loss < range_high
    
    def test_take_profit_range_expansion(self, breakout_data):
        """Test take profit targets range expansion"""
        agent = RoninMomentumBreakoutAgent(symbols=['BTCUSDT'])
        market_data = {'BTCUSDT': breakout_data}
        
        signals = agent.generate_signals(market_data)
        signal = signals[0]
        
        # TP should be approximately price + range_size
        range_size = signal.metadata['range_size']
        expected_tp = signal.price + range_size
        assert abs(signal.take_profit - expected_tp) < 0.1


# ============== Integration Tests ==============

class TestRoninStrategiesIntegration:
    """Integration tests for Ronin strategies"""
    
    def test_all_strategies_import(self):
        """Test all Ronin strategies can be imported from registry"""
        from agents.strategies import STRATEGY_REGISTRY
        
        assert 'ronin-volume-spike' in STRATEGY_REGISTRY
        assert 'ronin-funding-fade' in STRATEGY_REGISTRY
        assert 'ronin-momentum-breakout' in STRATEGY_REGISTRY
    
    def test_strategy_registry_instantiation(self):
        """Test strategies can be instantiated from registry"""
        from agents.strategies import STRATEGY_REGISTRY
        
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        for strategy_id in ['ronin-volume-spike', 'ronin-funding-fade', 'ronin-momentum-breakout']:
            agent_class = STRATEGY_REGISTRY[strategy_id]
            agent = agent_class(symbols=symbols)
            
            assert agent.agent_id == strategy_id
            assert len(agent.symbols) == 2
    
    def test_signals_have_consistent_structure(self, volume_spike_data, sample_ohlcv):
        """Test all signals have consistent structure"""
        from agents.strategies import (
            RoninVolumeSpikeAgent,
            RoninFundingFadeAgent,
            RoninMomentumBreakoutAgent
        )
        
        agents = [
            RoninVolumeSpikeAgent(symbols=['BTCUSDT']),
            RoninFundingFadeAgent(symbols=['BTCUSDT']),
            RoninMomentumBreakoutAgent(symbols=['BTCUSDT']),
        ]
        
        # Set funding data for funding fade
        agents[1].set_funding_data({'BTCUSDT': 0.001})
        
        for agent in agents:
            market_data = {'BTCUSDT': volume_spike_data}
            signals = agent.generate_signals(market_data)
            
            for signal in signals:
                # Verify signal structure
                assert hasattr(signal, 'strategy_id')
                assert hasattr(signal, 'symbol')
                assert hasattr(signal, 'side')
                assert hasattr(signal, 'confidence')
                assert hasattr(signal, 'price')
                assert hasattr(signal, 'stop_loss')
                assert hasattr(signal, 'take_profit')
                assert hasattr(signal, 'reason')
                assert hasattr(signal, 'metadata')
                
                # Verify ronin origin marker
                assert signal.metadata.get('strategy_origin') == 'ronin-trader'
