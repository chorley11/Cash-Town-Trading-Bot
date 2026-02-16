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

__all__ = [
    'BTSLynchAgent',
    'ZweigAgent',
    'TrendFollowingAgent',
    'MeanReversionAgent',
    'TurtleAgent',
    'WeinsteinAgent',
    'LivermoreAgent',
    'StatArbAgent',
    'RSIDivergenceAgent',
    'STRATEGY_REGISTRY',
]

# Strategy registry - maps strategy ID to agent class
STRATEGY_REGISTRY = {
    'bts-lynch': BTSLynchAgent,
    'zweig': ZweigAgent,
    'trend-following': TrendFollowingAgent,
    'mean-reversion': MeanReversionAgent,
    'turtle': TurtleAgent,
    'weinstein': WeinsteinAgent,
    'livermore': LivermoreAgent,
    'stat-arb': StatArbAgent,
    'rsi-divergence': RSIDivergenceAgent,
}
