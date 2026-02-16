"""
Pytest fixtures and shared test utilities for Cash Town tests.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch
import tempfile
import os

# Set up test data directory before imports
TEST_DATA_DIR = tempfile.mkdtemp()
os.environ['DATA_DIR'] = TEST_DATA_DIR

from agents.base import Signal, SignalSide, Position
from orchestrator.signal_aggregator import SignalAggregator, AggregatorConfig, AggregatedSignal
from orchestrator.second_chance import SecondChanceEvaluator, WinningPattern
from orchestrator.position_manager import PositionManager, RotationConfig, TrackedPosition, PositionState
from orchestrator.smart_orchestrator import SmartOrchestrator


# ====================
# MARKET DATA FIXTURES
# ====================

@pytest.fixture
def bullish_market_data() -> Dict[str, Any]:
    """Generate bullish trending market data (uptrend)"""
    n = 100
    # Create uptrending data
    base_price = 100
    trend = np.linspace(0, 30, n)  # +30% trend
    noise = np.random.normal(0, 1, n)
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_price = close - np.random.normal(0.5, 0.3, n)
    volume = np.random.uniform(1000000, 2000000, n)
    
    return {
        'BTC-USDT': {
            'open': open_price.tolist(),
            'high': high.tolist(),
            'low': low.tolist(),
            'close': close.tolist(),
            'volume': volume.tolist(),
        }
    }


@pytest.fixture
def bearish_market_data() -> Dict[str, Any]:
    """Generate bearish trending market data (downtrend)"""
    n = 100
    base_price = 130
    trend = np.linspace(0, -30, n)  # -30% trend
    noise = np.random.normal(0, 1, n)
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_price = close + np.random.normal(0.5, 0.3, n)
    volume = np.random.uniform(1000000, 2000000, n)
    
    return {
        'BTC-USDT': {
            'open': open_price.tolist(),
            'high': high.tolist(),
            'low': low.tolist(),
            'close': close.tolist(),
            'volume': volume.tolist(),
        }
    }


@pytest.fixture
def sideways_market_data() -> Dict[str, Any]:
    """Generate sideways/ranging market data"""
    n = 100
    base_price = 100
    noise = np.random.normal(0, 3, n)  # Just noise, no trend
    
    close = base_price + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_price = close + np.random.normal(0, 0.5, n)
    volume = np.random.uniform(500000, 1000000, n)  # Lower volume in ranging
    
    return {
        'BTC-USDT': {
            'open': open_price.tolist(),
            'high': high.tolist(),
            'low': low.tolist(),
            'close': close.tolist(),
            'volume': volume.tolist(),
        }
    }


@pytest.fixture
def multi_symbol_market_data(bullish_market_data, bearish_market_data) -> Dict[str, Any]:
    """Market data for multiple symbols"""
    return {
        'BTC-USDT': bullish_market_data['BTC-USDT'],
        'ETH-USDT': bearish_market_data['BTC-USDT'],  # Reuse bearish data
        'SOL-USDT': bullish_market_data['BTC-USDT'],  # Reuse bullish data
    }


@pytest.fixture
def empty_market_data() -> Dict[str, Any]:
    """Empty market data for edge case testing"""
    return {}


@pytest.fixture
def insufficient_market_data() -> Dict[str, Any]:
    """Market data with insufficient history"""
    return {
        'BTC-USDT': {
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000],
        }
    }


# ====================
# SIGNAL FIXTURES
# ====================

@pytest.fixture
def sample_long_signal() -> Signal:
    """Sample LONG signal"""
    return Signal(
        strategy_id='trend-following',
        symbol='BTC-USDT',
        side=SignalSide.LONG,
        confidence=0.75,
        price=50000.0,
        timestamp=datetime.utcnow(),
        reason='Bullish MA crossover',
        stop_loss=49000.0,
        take_profit=52000.0,
        metadata={'adx': 35, 'atr': 500}
    )


@pytest.fixture
def sample_short_signal() -> Signal:
    """Sample SHORT signal"""
    return Signal(
        strategy_id='mean-reversion',
        symbol='ETH-USDT',
        side=SignalSide.SHORT,
        confidence=0.68,
        price=3000.0,
        timestamp=datetime.utcnow(),
        reason='RSI overbought',
        stop_loss=3100.0,
        take_profit=2800.0,
        metadata={'rsi': 78}
    )


@pytest.fixture
def low_confidence_signal() -> Signal:
    """Signal below typical confidence threshold"""
    return Signal(
        strategy_id='zweig',
        symbol='SOL-USDT',
        side=SignalSide.LONG,
        confidence=0.48,
        price=100.0,
        timestamp=datetime.utcnow(),
        reason='Weak thrust signal',
        metadata={}
    )


@pytest.fixture
def near_miss_signal() -> Signal:
    """Signal just below threshold (for second chance testing)"""
    return Signal(
        strategy_id='trend-following',
        symbol='AVAX-USDT',
        side=SignalSide.LONG,
        confidence=0.52,
        price=35.0,
        timestamp=datetime.utcnow(),
        reason='Near-miss bullish setup',
        stop_loss=33.0,
        take_profit=40.0,
        metadata={'adx': 28, 'atr': 2.0}
    )


@pytest.fixture
def conflicting_signals() -> Dict[str, List[Signal]]:
    """Multiple signals on same symbol with opposite directions"""
    now = datetime.utcnow()
    return {
        'trend-following': [
            Signal(
                strategy_id='trend-following',
                symbol='BTC-USDT',
                side=SignalSide.LONG,
                confidence=0.70,
                price=50000.0,
                timestamp=now,
                reason='Trend up'
            )
        ],
        'mean-reversion': [
            Signal(
                strategy_id='mean-reversion',
                symbol='BTC-USDT',
                side=SignalSide.SHORT,
                confidence=0.65,
                price=50000.0,
                timestamp=now,
                reason='Overbought'
            )
        ]
    }


@pytest.fixture
def consensus_signals() -> Dict[str, List[Signal]]:
    """Multiple signals agreeing on direction"""
    now = datetime.utcnow()
    return {
        'trend-following': [
            Signal(
                strategy_id='trend-following',
                symbol='BTC-USDT',
                side=SignalSide.LONG,
                confidence=0.75,
                price=50000.0,
                timestamp=now,
                reason='Trend up'
            )
        ],
        'turtle': [
            Signal(
                strategy_id='turtle',
                symbol='BTC-USDT',
                side=SignalSide.LONG,
                confidence=0.68,
                price=50000.0,
                timestamp=now,
                reason='Breakout'
            )
        ],
        'weinstein': [
            Signal(
                strategy_id='weinstein',
                symbol='BTC-USDT',
                side=SignalSide.LONG,
                confidence=0.62,
                price=50000.0,
                timestamp=now,
                reason='Stage 2'
            )
        ]
    }


# ====================
# ORCHESTRATOR FIXTURES
# ====================

@pytest.fixture
def aggregator_config() -> AggregatorConfig:
    """Default aggregator config for testing"""
    return AggregatorConfig(
        min_confidence=0.55,
        min_consensus=1,
        max_signals_per_cycle=3,
        conflict_penalty=0.1,
        consensus_bonus=0.05,
        cooldown_minutes=15
    )


@pytest.fixture
def signal_aggregator(aggregator_config) -> SignalAggregator:
    """Pre-configured signal aggregator"""
    return SignalAggregator(aggregator_config)


@pytest.fixture
def smart_orchestrator() -> SmartOrchestrator:
    """Pre-configured smart orchestrator"""
    config = AggregatorConfig(
        min_confidence=0.55,
        min_consensus=1,
        max_signals_per_cycle=10,
        cooldown_minutes=0  # No cooldown for testing
    )
    return SmartOrchestrator(config)


# ====================
# POSITION FIXTURES
# ====================

@pytest.fixture
def sample_position() -> Position:
    """Sample open position"""
    return Position(
        id='pos-001',
        symbol='BTC-USDT',
        side='long',
        entry_price=50000.0,
        current_price=51000.0,
        size=0.1,
        value=5000.0,
        unrealized_pnl=100.0,
        stop_loss=49000.0,
        take_profit=55000.0,
        opened_at=datetime.utcnow() - timedelta(hours=1),
        strategy_id='trend-following'
    )


@pytest.fixture
def tracked_position() -> TrackedPosition:
    """Sample tracked position for position manager"""
    return TrackedPosition(
        agent_id='trend-following',
        symbol='BTC-USDT',
        side='long',
        entry_price=50000.0,
        entry_time=datetime.utcnow() - timedelta(hours=1),
        size=0.1,
        current_price=51000.0,
        current_pnl=100.0,
        current_pnl_pct=2.0,
        peak_pnl=150.0,
        peak_pnl_pct=3.0,
        signal_confidence=0.75
    )


@pytest.fixture
def rotation_config() -> RotationConfig:
    """Configuration for position rotation testing"""
    return RotationConfig(
        grace_period_minutes=30,
        stuck_threshold_pct=0.5,
        stuck_max_minutes=120,
        fallen_peak_threshold_pct=1.0,
        fallen_giveback_pct=80,
        max_hold_hours=48
    )


@pytest.fixture
def position_manager(rotation_config) -> PositionManager:
    """Pre-configured position manager"""
    return PositionManager(rotation_config)


# ====================
# SECOND CHANCE FIXTURES
# ====================

@pytest.fixture
def second_chance_evaluator() -> SecondChanceEvaluator:
    """Pre-configured second chance evaluator"""
    strategy_perf = {
        'trend-following': {'trades': 20, 'win_rate': 0.55, 'total_pnl_pct': 15.0},
        'zweig': {'trades': 15, 'win_rate': 0.20, 'total_pnl_pct': -25.0},
    }
    return SecondChanceEvaluator(strategy_perf)


@pytest.fixture
def winning_pattern() -> WinningPattern:
    """Sample winning pattern for second chance"""
    return WinningPattern(
        pattern_id='pattern_1',
        rejection_reason='Low confidence',
        confidence_range=(0.50, 0.55),
        winning_rate=0.65,
        sample_size=20,
        strategy_ids=['trend-following'],
        boost_amount=0.08
    )


# ====================
# UTILITY FIXTURES
# ====================

@pytest.fixture
def test_data_dir(tmp_path):
    """Temporary data directory for tests"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture(autouse=True)
def reset_env(test_data_dir):
    """Reset environment for each test"""
    os.environ['DATA_DIR'] = str(test_data_dir)
    yield
    # Cleanup happens automatically with tmp_path


# ====================
# HELPER FUNCTIONS
# ====================

def create_market_data_with_crossover(
    n_periods: int = 100,
    crossover_at: int = 80,
    direction: str = 'bullish',
    base_price: float = 100.0
) -> Dict[str, Any]:
    """
    Create market data with a specific MA crossover point.
    
    Args:
        n_periods: Total number of candles
        crossover_at: Index where crossover occurs
        direction: 'bullish' or 'bearish'
        base_price: Starting price
    
    Returns:
        Market data dict for a single symbol
    """
    prices = np.zeros(n_periods)
    
    if direction == 'bullish':
        # Downtrend then uptrend
        prices[:crossover_at] = base_price - np.linspace(0, 10, crossover_at)
        prices[crossover_at:] = base_price - 10 + np.linspace(0, 20, n_periods - crossover_at)
    else:
        # Uptrend then downtrend
        prices[:crossover_at] = base_price + np.linspace(0, 10, crossover_at)
        prices[crossover_at:] = base_price + 10 - np.linspace(0, 20, n_periods - crossover_at)
    
    noise = np.random.normal(0, 0.5, n_periods)
    close = prices + noise
    high = close + np.abs(np.random.normal(0.5, 0.2, n_periods))
    low = close - np.abs(np.random.normal(0.5, 0.2, n_periods))
    
    return {
        'open': (close - np.random.normal(0, 0.2, n_periods)).tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': (np.random.uniform(1000000, 2000000, n_periods)).tolist()
    }


def assert_signal_valid(signal: Signal):
    """Assert that a signal has all required fields properly set"""
    assert signal.strategy_id, "Signal must have strategy_id"
    assert signal.symbol, "Signal must have symbol"
    assert signal.side in [SignalSide.LONG, SignalSide.SHORT, SignalSide.NEUTRAL]
    assert 0 <= signal.confidence <= 1, f"Confidence must be 0-1, got {signal.confidence}"
    assert signal.price > 0, "Price must be positive"
    assert signal.timestamp is not None, "Signal must have timestamp"
