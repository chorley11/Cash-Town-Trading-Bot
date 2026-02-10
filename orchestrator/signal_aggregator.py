"""
Signal Aggregator - Collects and ranks signals from all strategy agents

Responsibilities:
1. Collect signals from all active strategy agents
2. Filter by minimum confidence
3. Detect conflicts (same symbol, opposite directions)
4. Rank by confidence and strategy performance
5. Apply position limits and exposure rules
6. Return actionable signals for execution
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from collections import defaultdict

from agents.base import Signal, SignalSide

logger = logging.getLogger(__name__)

@dataclass
class AggregatedSignal:
    """A signal ready for execution with additional metadata"""
    signal: Signal
    rank: int
    sources: List[str]  # Strategy IDs that agree
    conflicts: List[str]  # Strategy IDs that disagree
    consensus_score: float  # How many strategies agree
    adjusted_confidence: float  # Confidence after adjustments
    
    @property
    def symbol(self) -> str:
        return self.signal.symbol
    
    @property
    def side(self) -> SignalSide:
        return self.signal.side
    
    @property
    def is_unanimous(self) -> bool:
        return len(self.conflicts) == 0

@dataclass
class AggregatorConfig:
    """Configuration for signal aggregation"""
    min_confidence: float = 0.55
    min_consensus: int = 1  # Minimum strategies agreeing
    max_signals_per_cycle: int = 3
    conflict_penalty: float = 0.1  # Reduce confidence when strategies disagree
    consensus_bonus: float = 0.05  # Per additional strategy agreeing
    cooldown_minutes: int = 15  # Don't re-signal same symbol within this period
    max_exposure_per_symbol: int = 1  # Max positions per symbol
    blacklist: Set[str] = field(default_factory=set)  # Symbols to never trade

class SignalAggregator:
    """
    Aggregates signals from multiple strategy agents and produces
    a ranked list of actionable signals.
    """
    
    def __init__(self, config: AggregatorConfig = None):
        self.config = config or AggregatorConfig()
        self.recent_signals: Dict[str, datetime] = {}  # symbol -> last signal time
        self.strategy_performance: Dict[str, float] = {}  # strategy_id -> performance score
        self.current_positions: Dict[str, str] = {}  # symbol -> side
    
    def set_positions(self, positions: Dict[str, str]):
        """Update current positions (symbol -> 'long'/'short')"""
        self.current_positions = positions
    
    def set_strategy_performance(self, performance: Dict[str, float]):
        """Update strategy performance scores for weighting"""
        self.strategy_performance = performance
    
    def aggregate(self, all_signals: Dict[str, List[Signal]]) -> List[AggregatedSignal]:
        """
        Aggregate signals from all strategies.
        
        Args:
            all_signals: Dict of strategy_id -> list of signals
        
        Returns:
            Ranked list of AggregatedSignal ready for execution
        """
        if not all_signals:
            return []
        
        # Group signals by symbol
        by_symbol: Dict[str, List[Signal]] = defaultdict(list)
        for strategy_id, signals in all_signals.items():
            for signal in signals:
                if signal.confidence >= self.config.min_confidence:
                    by_symbol[signal.symbol].append(signal)
        
        aggregated = []
        
        for symbol, signals in by_symbol.items():
            # Skip blacklisted symbols
            if symbol in self.config.blacklist:
                logger.debug(f"Skipping blacklisted symbol: {symbol}")
                continue
            
            # Skip if in cooldown
            if self._in_cooldown(symbol):
                logger.debug(f"Skipping {symbol} - in cooldown")
                continue
            
            # Skip if already have position
            if symbol in self.current_positions:
                logger.debug(f"Skipping {symbol} - already have {self.current_positions[symbol]} position")
                continue
            
            # Analyze consensus
            agg_signal = self._analyze_symbol_signals(symbol, signals)
            if agg_signal:
                aggregated.append(agg_signal)
        
        # Sort by adjusted confidence (highest first)
        aggregated.sort(key=lambda s: s.adjusted_confidence, reverse=True)
        
        # Limit signals per cycle
        aggregated = aggregated[:self.config.max_signals_per_cycle]
        
        # Assign ranks
        for i, sig in enumerate(aggregated):
            sig.rank = i + 1
        
        logger.info(f"Aggregated {len(aggregated)} actionable signals from {len(all_signals)} strategies")
        
        return aggregated
    
    def _analyze_symbol_signals(self, symbol: str, signals: List[Signal]) -> Optional[AggregatedSignal]:
        """Analyze all signals for a single symbol"""
        if not signals:
            return None
        
        # Count long vs short signals
        longs = [s for s in signals if s.side == SignalSide.LONG]
        shorts = [s for s in signals if s.side == SignalSide.SHORT]
        
        # Determine majority direction
        if len(longs) > len(shorts):
            majority_signals = longs
            minority_signals = shorts
            direction = SignalSide.LONG
        elif len(shorts) > len(longs):
            majority_signals = shorts
            minority_signals = longs
            direction = SignalSide.SHORT
        else:
            # Equal - use highest confidence signal's direction
            best = max(signals, key=lambda s: s.confidence)
            if best.side == SignalSide.LONG:
                majority_signals = longs
                minority_signals = shorts
                direction = SignalSide.LONG
            else:
                majority_signals = shorts
                minority_signals = longs
                direction = SignalSide.SHORT
        
        # Check minimum consensus
        if len(majority_signals) < self.config.min_consensus:
            logger.debug(f"{symbol}: Not enough consensus ({len(majority_signals)} < {self.config.min_consensus})")
            return None
        
        # Use the highest confidence signal as the base
        best_signal = max(majority_signals, key=lambda s: s.confidence)
        
        # Calculate adjusted confidence
        base_confidence = best_signal.confidence
        
        # Bonus for consensus
        consensus_bonus = (len(majority_signals) - 1) * self.config.consensus_bonus
        
        # Penalty for conflicts
        conflict_penalty = len(minority_signals) * self.config.conflict_penalty
        
        # Strategy performance weighting
        perf_bonus = self.strategy_performance.get(best_signal.strategy_id, 0) * 0.1
        
        adjusted_confidence = min(0.95, max(0.1, 
            base_confidence + consensus_bonus - conflict_penalty + perf_bonus
        ))
        
        # Build aggregated signal
        sources = [s.strategy_id for s in majority_signals]
        conflicts = [s.strategy_id for s in minority_signals]
        consensus_score = len(majority_signals) / len(signals)
        
        return AggregatedSignal(
            signal=best_signal,
            rank=0,  # Will be set later
            sources=sources,
            conflicts=conflicts,
            consensus_score=consensus_score,
            adjusted_confidence=adjusted_confidence
        )
    
    def _in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.recent_signals:
            return False
        
        cooldown_end = self.recent_signals[symbol] + timedelta(minutes=self.config.cooldown_minutes)
        return datetime.utcnow() < cooldown_end
    
    def mark_signaled(self, symbol: str):
        """Mark a symbol as recently signaled (for cooldown)"""
        self.recent_signals[symbol] = datetime.utcnow()
    
    def get_status(self) -> Dict:
        """Get aggregator status"""
        now = datetime.utcnow()
        active_cooldowns = {
            symbol: (self.recent_signals[symbol] + timedelta(minutes=self.config.cooldown_minutes) - now).seconds
            for symbol in self.recent_signals
            if self._in_cooldown(symbol)
        }
        
        return {
            'config': {
                'min_confidence': self.config.min_confidence,
                'min_consensus': self.config.min_consensus,
                'max_signals_per_cycle': self.config.max_signals_per_cycle,
                'cooldown_minutes': self.config.cooldown_minutes
            },
            'active_cooldowns': active_cooldowns,
            'current_positions': len(self.current_positions),
            'blacklisted_symbols': list(self.config.blacklist)
        }
