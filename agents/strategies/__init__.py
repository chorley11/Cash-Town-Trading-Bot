"""
Cash Town Strategy Agents
Each agent implements one trading strategy independently.
"""
from .bts_lynch import BTSLynchAgent
from .zweig import ZweigAgent
from .trend_following import TrendFollowingAgent
from .mean_reversion import MeanReversionAgent
from .turtle import TurtleAgent
from .weinstein import WeinsteinAgent
from .livermore import LivermoreAgent
from .stat_arb import StatArbAgent
from .rsi_divergence import RSIDivergenceAgent

# New strategies (Feb 2026)
from .funding_fade import FundingFadeAgent
from .oi_divergence import OIDivergenceAgent
from .liquidation_hunter import LiquidationHunterAgent
from .volatility_breakout import VolatilityBreakoutAgent
from .correlation_pairs import CorrelationPairsAgent

__all__ = [
    # Original strategies
    'BTSLynchAgent',
    'ZweigAgent',
    'TrendFollowingAgent',
    'MeanReversionAgent',
    'TurtleAgent',
    'WeinsteinAgent',
    'LivermoreAgent',
    'StatArbAgent',
    'RSIDivergenceAgent',
    # New strategies
    'FundingFadeAgent',
    'OIDivergenceAgent',
    'LiquidationHunterAgent',
    'VolatilityBreakoutAgent',
    'CorrelationPairsAgent',
    'STRATEGY_REGISTRY',
]

# Strategy registry - maps strategy ID to agent class
STRATEGY_REGISTRY = {
    # Original strategies
    'bts-lynch': BTSLynchAgent,
    'zweig': ZweigAgent,
    'trend-following': TrendFollowingAgent,
    'mean-reversion': MeanReversionAgent,
    'turtle': TurtleAgent,
    'weinstein': WeinsteinAgent,
    'livermore': LivermoreAgent,
    'stat-arb': StatArbAgent,
    'rsi-divergence': RSIDivergenceAgent,
    # New strategies (Feb 2026) - Futures-specific
    'funding-fade': FundingFadeAgent,
    'oi-divergence': OIDivergenceAgent,
    'liquidation-hunter': LiquidationHunterAgent,
    'volatility-breakout': VolatilityBreakoutAgent,
    'correlation-pairs': CorrelationPairsAgent,
}
