"""
Second Chance Logic - Rescue promising signals that were initially rejected

This module implements counterfactual learning to reduce false negatives:
1. Tracks "near-miss" signals (just below threshold)
2. Uses strategy track record to boost confidence
3. Identifies winning rejection patterns from historical data
4. Implements pattern matching for second-chance signals

The goal: Catch winners we would have passed on.
"""
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

from agents.base import Signal, SignalSide

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
SECOND_CHANCE_LOG = DATA_DIR / 'second_chance.jsonl'
PATTERN_CACHE = DATA_DIR / 'winning_patterns.json'


@dataclass
class NearMissSignal:
    """Signal that was close to being selected"""
    timestamp: str
    strategy_id: str
    symbol: str
    side: str
    original_confidence: float
    adjusted_confidence: float  # After second-chance boost
    rejection_reason: str
    boost_reasons: List[str] = field(default_factory=list)
    rescued: bool = False
    
    # Counterfactual tracking
    entry_price: float = 0.0
    price_after_1h: Optional[float] = None
    price_after_4h: Optional[float] = None
    would_have_won: Optional[bool] = None


@dataclass
class WinningPattern:
    """Pattern identified from counterfactual analysis of winners"""
    pattern_id: str
    rejection_reason: str
    confidence_range: Tuple[float, float]  # (min, max)
    winning_rate: float  # % of times this pattern would have won
    sample_size: int
    strategy_ids: List[str]  # Strategies this pattern applies to
    boost_amount: float  # How much to boost confidence


class SecondChanceEvaluator:
    """
    Evaluates rejected signals for potential "second chance".
    
    Key insights that drive this logic:
    1. A 0.54 confidence from trend-following (51% WR) is better than
       a 0.60 from zweig (14% WR) - strategy track record matters
    2. Signals just below threshold might be winners - near-misses deserve scrutiny
    3. Historical counterfactual data reveals winning rejection patterns
    4. Multiple weak signals agreeing might form a strong signal
    """
    
    # Confidence floor - anything below this is truly too weak
    ABSOLUTE_FLOOR = 0.45
    
    # Near-miss threshold - signals in this range get second-chance evaluation
    NEAR_MISS_RANGE = (0.45, 0.55)
    
    # Strategy performance boost based on historical win rates
    # Strategies with proven track records get a confidence boost
    STRATEGY_BOOST = {
        'trend-following': 0.08,   # STAR: 51% WR, +$208 - deserves benefit of doubt
        'mean-reversion': 0.03,
        'turtle': 0.02,
        'weinstein': 0.01,
        'livermore': 0.01,
        'bts-lynch': 0.00,         # Neutral - need more data
        'zweig': -0.10,            # PENALTY: 14% WR - actively avoid
    }
    
    def __init__(self, strategy_performance: Dict[str, Dict] = None):
        self.strategy_performance = strategy_performance or {}
        self.winning_patterns: List[WinningPattern] = []
        self.near_miss_signals: List[NearMissSignal] = []
        
        # Load winning patterns from historical analysis
        self._load_winning_patterns()
    
    def _load_winning_patterns(self):
        """Load identified winning patterns from file"""
        if PATTERN_CACHE.exists():
            try:
                with open(PATTERN_CACHE, 'r') as f:
                    data = json.load(f)
                    for p in data.get('patterns', []):
                        self.winning_patterns.append(WinningPattern(**p))
                logger.info(f"Loaded {len(self.winning_patterns)} winning patterns")
            except Exception as e:
                logger.warning(f"Failed to load winning patterns: {e}")
    
    def _save_winning_patterns(self):
        """Save identified patterns to file"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(PATTERN_CACHE, 'w') as f:
                json.dump({
                    'patterns': [asdict(p) for p in self.winning_patterns],
                    'updated': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save winning patterns: {e}")
    
    def evaluate_for_second_chance(
        self, 
        signal: Signal, 
        rejection_reason: str,
        all_signals: Dict[str, List[Signal]] = None
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate a rejected signal for second chance.
        
        Returns:
            (should_rescue, adjusted_confidence, boost_reasons)
        """
        boost_reasons = []
        confidence = signal.confidence
        
        # Check if below absolute floor
        if confidence < self.ABSOLUTE_FLOOR:
            return False, confidence, ["Below absolute floor"]
        
        # === BOOST 1: Strategy Track Record ===
        strategy_boost = self._get_strategy_boost(signal.strategy_id)
        if strategy_boost > 0:
            boost_reasons.append(f"Strategy boost: +{strategy_boost:.0%} ({signal.strategy_id})")
            confidence += strategy_boost
        elif strategy_boost < 0:
            boost_reasons.append(f"Strategy penalty: {strategy_boost:.0%} ({signal.strategy_id})")
            confidence += strategy_boost
        
        # === BOOST 2: Consensus from other strategies ===
        if all_signals:
            consensus_boost = self._evaluate_consensus(signal, all_signals)
            if consensus_boost > 0:
                boost_reasons.append(f"Consensus boost: +{consensus_boost:.0%}")
                confidence += consensus_boost
        
        # === BOOST 3: Winning Pattern Match ===
        pattern_boost = self._match_winning_pattern(signal, rejection_reason)
        if pattern_boost > 0:
            boost_reasons.append(f"Winning pattern match: +{pattern_boost:.0%}")
            confidence += pattern_boost
        
        # === BOOST 4: Strong Technical Setup ===
        if signal.metadata:
            tech_boost = self._evaluate_technical_strength(signal)
            if tech_boost > 0:
                boost_reasons.append(f"Strong technicals: +{tech_boost:.0%}")
                confidence += tech_boost
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        # Rescue if boosted above threshold
        should_rescue = confidence >= 0.55 and len(boost_reasons) > 0
        
        # Track for counterfactual analysis
        if self.NEAR_MISS_RANGE[0] <= signal.confidence <= self.NEAR_MISS_RANGE[1]:
            self._track_near_miss(signal, rejection_reason, confidence, boost_reasons, should_rescue)
        
        return should_rescue, confidence, boost_reasons
    
    def _get_strategy_boost(self, strategy_id: str) -> float:
        """Get confidence boost based on strategy track record"""
        # First check if we have learned performance data
        if strategy_id in self.strategy_performance:
            perf = self.strategy_performance[strategy_id]
            trades = perf.get('trades', 0)
            
            if trades >= 10:  # Need enough data to trust
                win_rate = perf.get('win_rate', 0.5)
                total_pnl = perf.get('total_pnl_pct', 0)
                
                # Calculate boost based on actual performance
                # Base: 0 at 50% WR, +0.10 at 60% WR, -0.10 at 40% WR
                win_rate_boost = (win_rate - 0.5) * 0.5
                
                # Additional boost if actually profitable
                pnl_boost = min(0.05, max(-0.05, total_pnl / 1000))
                
                return win_rate_boost + pnl_boost
        
        # Fall back to hardcoded defaults
        return self.STRATEGY_BOOST.get(strategy_id, 0.0)
    
    def _evaluate_consensus(self, signal: Signal, all_signals: Dict[str, List[Signal]]) -> float:
        """Boost if multiple strategies agree on direction"""
        agreeing = 0
        disagreeing = 0
        
        for strategy_id, signals in all_signals.items():
            if strategy_id == signal.strategy_id:
                continue
            
            for other in signals:
                if other.symbol == signal.symbol:
                    if other.side == signal.side:
                        agreeing += 1
                    else:
                        disagreeing += 1
        
        # Net agreement boost
        if agreeing > disagreeing:
            return min(0.10, agreeing * 0.03)
        elif disagreeing > agreeing:
            return -min(0.05, disagreeing * 0.02)
        
        return 0.0
    
    def _match_winning_pattern(self, signal: Signal, rejection_reason: str) -> float:
        """Check if signal matches a known winning rejection pattern"""
        for pattern in self.winning_patterns:
            # Check rejection reason match
            if pattern.rejection_reason not in rejection_reason:
                continue
            
            # Check confidence range
            conf_min, conf_max = pattern.confidence_range
            if not (conf_min <= signal.confidence <= conf_max):
                continue
            
            # Check strategy match
            if pattern.strategy_ids and signal.strategy_id not in pattern.strategy_ids:
                continue
            
            # Pattern matches! Return boost proportional to win rate
            if pattern.winning_rate >= 0.60:  # Only boost if pattern wins >60%
                return pattern.boost_amount
        
        return 0.0
    
    def _evaluate_technical_strength(self, signal: Signal) -> float:
        """Boost based on strong technical indicators in metadata"""
        boost = 0.0
        meta = signal.metadata or {}
        
        # Strong ADX (trend strength)
        if 'adx' in meta:
            adx = meta['adx']
            if adx >= 40:
                boost += 0.05  # Very strong trend
            elif adx >= 35:
                boost += 0.03
        
        # Good risk/reward implied by ATR-based stops
        if 'atr' in meta and signal.stop_loss and signal.take_profit:
            entry = signal.price
            sl_dist = abs(entry - signal.stop_loss)
            tp_dist = abs(signal.take_profit - entry)
            
            if tp_dist > 0 and sl_dist > 0:
                rr_ratio = tp_dist / sl_dist
                if rr_ratio >= 2.5:
                    boost += 0.03  # Excellent R:R
                elif rr_ratio >= 2.0:
                    boost += 0.02
        
        return boost
    
    def _track_near_miss(
        self, 
        signal: Signal, 
        rejection_reason: str,
        adjusted_confidence: float,
        boost_reasons: List[str],
        rescued: bool
    ):
        """Track near-miss signal for counterfactual analysis"""
        near_miss = NearMissSignal(
            timestamp=datetime.utcnow().isoformat(),
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side.value,
            original_confidence=signal.confidence,
            adjusted_confidence=adjusted_confidence,
            rejection_reason=rejection_reason,
            boost_reasons=boost_reasons,
            rescued=rescued,
            entry_price=signal.price
        )
        
        self.near_miss_signals.append(near_miss)
        
        # Log for future analysis
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(SECOND_CHANCE_LOG, 'a') as f:
                f.write(json.dumps(asdict(near_miss)) + '\n')
        except Exception as e:
            logger.error(f"Failed to log near-miss: {e}")
    
    def update_counterfactuals(self, current_prices: Dict[str, float]):
        """Update price tracking for near-miss signals"""
        now = datetime.utcnow()
        
        for near_miss in self.near_miss_signals[:]:  # Copy for safe removal
            signal_time = datetime.fromisoformat(near_miss.timestamp)
            age_hours = (now - signal_time).total_seconds() / 3600
            
            if near_miss.symbol not in current_prices:
                continue
            
            current_price = current_prices[near_miss.symbol]
            
            # Update price tracking
            if age_hours >= 1 and near_miss.price_after_1h is None:
                near_miss.price_after_1h = current_price
            
            if age_hours >= 4 and near_miss.price_after_4h is None:
                near_miss.price_after_4h = current_price
                
                # Evaluate if it would have won
                if near_miss.side == 'long':
                    pnl_pct = (current_price - near_miss.entry_price) / near_miss.entry_price * 100
                else:
                    pnl_pct = (near_miss.entry_price - current_price) / near_miss.entry_price * 100
                
                near_miss.would_have_won = pnl_pct > 0
                
                # Remove from active tracking
                self.near_miss_signals.remove(near_miss)
                
                # If this was a winning rejection that we didn't rescue, learn from it
                if near_miss.would_have_won and not near_miss.rescued:
                    self._learn_from_winning_rejection(near_miss)
    
    def _learn_from_winning_rejection(self, near_miss: NearMissSignal):
        """Learn from a winning signal we rejected"""
        logger.info(f"📊 LEARNING: Rejected {near_miss.strategy_id} {near_miss.side} "
                   f"{near_miss.symbol} would have WON! Conf: {near_miss.original_confidence:.0%}")
        
        # Find or create matching pattern
        pattern_found = False
        for pattern in self.winning_patterns:
            if (pattern.rejection_reason in near_miss.rejection_reason and
                pattern.confidence_range[0] <= near_miss.original_confidence <= pattern.confidence_range[1]):
                # Update existing pattern
                pattern.sample_size += 1
                if near_miss.strategy_id not in pattern.strategy_ids:
                    pattern.strategy_ids.append(near_miss.strategy_id)
                # Recalculate winning rate with decay toward new data
                pattern.winning_rate = 0.9 * pattern.winning_rate + 0.1 * 1.0
                pattern_found = True
                break
        
        if not pattern_found:
            # Create new winning pattern
            new_pattern = WinningPattern(
                pattern_id=f"pattern_{len(self.winning_patterns)+1}",
                rejection_reason=near_miss.rejection_reason,
                confidence_range=(max(0.45, near_miss.original_confidence - 0.03),
                                 min(0.55, near_miss.original_confidence + 0.03)),
                winning_rate=1.0,  # Start optimistic, will decay if losers appear
                sample_size=1,
                strategy_ids=[near_miss.strategy_id],
                boost_amount=0.06  # Start with moderate boost
            )
            self.winning_patterns.append(new_pattern)
            logger.info(f"📈 NEW PATTERN: {new_pattern.pattern_id} for {near_miss.rejection_reason}")
        
        # Save updated patterns
        self._save_winning_patterns()
    
    def analyze_historical_rejections(self, counterfactual_file: Path) -> Dict:
        """
        Analyze historical counterfactual data to identify winning rejection patterns.
        Call this periodically to update the pattern database.
        """
        if not counterfactual_file.exists():
            return {'status': 'no_data', 'patterns': []}
        
        # Group by rejection reason and confidence range
        patterns_data = {}
        
        try:
            with open(counterfactual_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('was_selected') or record.get('would_have_won') is None:
                            continue
                        
                        # Create pattern key
                        reason = record.get('selection_reason', 'unknown')
                        conf = record.get('confidence', 0)
                        conf_bucket = round(conf * 20) / 20  # 5% buckets
                        key = f"{reason}|{conf_bucket}"
                        
                        if key not in patterns_data:
                            patterns_data[key] = {
                                'reason': reason,
                                'conf_bucket': conf_bucket,
                                'wins': 0,
                                'total': 0,
                                'strategies': set()
                            }
                        
                        patterns_data[key]['total'] += 1
                        if record.get('would_have_won'):
                            patterns_data[key]['wins'] += 1
                        patterns_data[key]['strategies'].add(record.get('strategy_id', 'unknown'))
                    except:
                        continue
            
            # Convert to winning patterns (>55% win rate and sample size >= 5)
            new_patterns = []
            for key, data in patterns_data.items():
                if data['total'] >= 5:
                    win_rate = data['wins'] / data['total']
                    if win_rate >= 0.55:
                        pattern = WinningPattern(
                            pattern_id=f"hist_{len(new_patterns)+1}",
                            rejection_reason=data['reason'],
                            confidence_range=(data['conf_bucket'] - 0.025, data['conf_bucket'] + 0.025),
                            winning_rate=win_rate,
                            sample_size=data['total'],
                            strategy_ids=list(data['strategies']),
                            boost_amount=min(0.12, (win_rate - 0.5) * 0.4)  # Scale boost with win rate
                        )
                        new_patterns.append(pattern)
            
            # Update patterns
            self.winning_patterns = new_patterns
            self._save_winning_patterns()
            
            return {
                'status': 'analyzed',
                'total_rejections': sum(p['total'] for p in patterns_data.values()),
                'winning_patterns': len(new_patterns),
                'patterns': [asdict(p) for p in new_patterns]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical rejections: {e}")
            return {'status': 'error', 'error': str(e)}


def integrate_second_chance(orchestrator) -> None:
    """
    Integrate second-chance logic into the SmartOrchestrator.
    Call this to add second-chance evaluation to the signal pipeline.
    """
    evaluator = SecondChanceEvaluator(orchestrator.strategy_performance)
    
    # Store original method
    original_log_all_signals = orchestrator._log_all_signals
    
    def enhanced_log_all_signals(selected):
        """Enhanced logging that includes second-chance evaluation"""
        selected_symbols = {s.symbol for s in selected}
        rescued_signals = []
        
        # Check rejected signals for second chance
        for strategy_id, signals in orchestrator.raw_signals.items():
            for signal in signals:
                if signal.symbol not in selected_symbols:
                    rejection_reason = orchestrator._get_rejection_reason(signal)
                    
                    # Evaluate for second chance
                    should_rescue, adj_conf, boost_reasons = evaluator.evaluate_for_second_chance(
                        signal, rejection_reason, orchestrator.raw_signals
                    )
                    
                    if should_rescue:
                        logger.info(f"🎯 SECOND CHANCE: Rescuing {strategy_id} {signal.side.value} "
                                   f"{signal.symbol} ({signal.confidence:.0%} -> {adj_conf:.0%})")
                        for reason in boost_reasons:
                            logger.info(f"   ↳ {reason}")
                        
                        # Boost confidence and add to rescued
                        signal.confidence = adj_conf
                        rescued_signals.append(signal)
        
        # Add rescued signals to raw_signals for processing
        if rescued_signals:
            for signal in rescued_signals:
                if signal.strategy_id not in orchestrator.raw_signals:
                    orchestrator.raw_signals[signal.strategy_id] = []
                # Mark as rescued to avoid double-processing
                signal.metadata['rescued'] = True
        
        # Call original
        original_log_all_signals(selected)
    
    # Monkey-patch
    orchestrator._log_all_signals = enhanced_log_all_signals
    orchestrator.second_chance_evaluator = evaluator
    
    logger.info("✅ Second-chance logic integrated into orchestrator")
