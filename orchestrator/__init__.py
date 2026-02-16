"""
Cash Town Orchestrator Package

The brain of the trading system - signal aggregation, risk management,
position rotation, and profit optimization.
"""
from .signal_aggregator import SignalAggregator, AggregatorConfig, AggregatedSignal
from .smart_orchestrator import SmartOrchestrator
from .position_manager import PositionManager, RotationConfig
from .risk_manager import RiskManager, RiskConfig, create_risk_manager
from .profit_watchdog import ProfitWatchdog, run_watchdog_cycle
from .server import Orchestrator, get_orchestrator, run_server

__all__ = [
    'SignalAggregator',
    'AggregatorConfig',
    'AggregatedSignal',
    'SmartOrchestrator',
    'PositionManager',
    'RotationConfig',
    'RiskManager',
    'RiskConfig',
    'create_risk_manager',
    'ProfitWatchdog',
    'run_watchdog_cycle',
    'Orchestrator',
    'get_orchestrator',
    'run_server',
]
