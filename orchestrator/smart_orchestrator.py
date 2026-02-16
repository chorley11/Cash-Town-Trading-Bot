"""
Smart Orchestrator - Intelligent signal selection with learning

Key improvements:
1. Uses SignalAggregator to rank and filter signals
2. Stores ALL signals for learning (even rejected ones)
3. Clears signals after processing
4. Integrates reflection for self-improvement
5. Tracks what would have happened with rejected signals
6. SECOND CHANCE LOGIC: Rescues promising signals that were initially rejected
7. RISK MANAGER: All signals pass through centralized risk controls
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .signal_aggregator import SignalAggregator, AggregatorConfig, AggregatedSignal
from .second_chance import SecondChanceEvaluator
from .risk_manager import RiskManager, RiskConfig, create_risk_manager
from .profit_watchdog import ProfitWatchdog
from agents.base import Signal, SignalSide

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
SIGNALS_LOG = DATA_DIR / 'signals_history.jsonl'
TRADES_LOG = DATA_DIR / 'trades_history.jsonl'
COUNTERFACTUAL_LOG = DATA_DIR / 'counterfactual.jsonl'

@dataclass
class SignalRecord:
    """Record of a signal for learning"""
    timestamp: str
    strategy_id: str
    symbol: str
    side: str
    confidence: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str
    
    # Decision outcome
    was_selected: bool
    selection_reason: str
    aggregated_rank: Optional[int]
    consensus_score: Optional[float]
    
    # Counterfactual tracking (filled in later)
    price_1h_later: Optional[float] = None
    price_4h_later: Optional[float] = None
    price_24h_later: Optional[float] = None
    would_have_won: Optional[bool] = None
    potential_pnl: Optional[float] = None

class SmartOrchestrator:
    """
    Intelligent orchestrator that:
    1. Selects the best signals using aggregation
    2. Learns from past decisions
    3. Tracks counterfactuals for rejected signals
    4. ALL signals pass through RiskManager before execution
    """
    
    def __init__(self, config: AggregatorConfig = None, risk_config: RiskConfig = None):
        # Learning-first: no arbitrary limits. Let the bot learn what works.
        # Limits emerge from data, not assumptions.
        self.aggregator = SignalAggregator(config or AggregatorConfig(
            min_confidence=0.55,
            min_consensus=1,
            max_signals_per_cycle=99,   # Effectively unlimited - learn from results
            cooldown_minutes=0,          # No cooldown - learn when re-entry is good/bad
        ))
        
        # Raw signals from strategies (cleared after processing)
        self.raw_signals: Dict[str, List[Signal]] = {}
        
        # Signals waiting for counterfactual check
        self.pending_counterfactual: List[SignalRecord] = []
        
        # Current positions (symbol -> side)
        self.positions: Dict[str, str] = {}
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict] = {}
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load strategy performance
        self._load_strategy_performance()
        
        # SECOND CHANCE: Rescue promising signals that were initially rejected
        self.second_chance = SecondChanceEvaluator(self.strategy_performance)
        
        # Track rescued signals stats
        self.rescue_stats = {'total_rescued': 0, 'rescued_wins': 0, 'rescued_losses': 0}
        
        # RISK MANAGER: Centralized risk control
        self.risk_manager = create_risk_manager(equity=0.0)
        if risk_config:
            self.risk_manager.config = risk_config
        logger.info("🛡️ RiskManager integrated with SmartOrchestrator")
        
        # PROFIT WATCHDOG: The self-improving feedback loop
        self.watchdog = ProfitWatchdog(orchestrator=self)
        logger.info("🐕 ProfitWatchdog integrated with SmartOrchestrator")
    
    def receive_signal(self, strategy_id: str, signal_data: dict):
        """Receive a signal from a strategy agent"""
        try:
            signal = Signal(
                strategy_id=strategy_id,
                symbol=signal_data['symbol'],
                side=SignalSide(signal_data['side']),
                confidence=signal_data['confidence'],
                price=signal_data.get('price', 0),
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit'),
                reason=signal_data.get('reason', ''),
                timestamp=datetime.fromisoformat(signal_data['timestamp']) if signal_data.get('timestamp') else datetime.utcnow()
            )
            
            if strategy_id not in self.raw_signals:
                self.raw_signals[strategy_id] = []
            
            self.raw_signals[strategy_id].append(signal)
            
            logger.debug(f"Received signal: {strategy_id} -> {signal.side.value} {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error receiving signal: {e}")
    
    def get_actionable_signals(self) -> List[AggregatedSignal]:
        """
        Process raw signals and return only the best ones for execution.
        Also logs all signals for learning and applies SECOND CHANCE logic.
        ALL SIGNALS PASS THROUGH RISK MANAGER CHECKS.
        """
        if not self.raw_signals:
            return []
        
        # Update aggregator with current positions
        self.aggregator.set_positions(self.positions)
        self.aggregator.set_strategy_performance({
            k: v.get('score', 0) for k, v in self.strategy_performance.items()
        })
        
        # Sync risk manager positions
        self.risk_manager.positions = {
            symbol: self.risk_manager.positions.get(symbol)
            for symbol in self.positions
            if symbol in self.risk_manager.positions
        }
        
        # Aggregate signals (first pass)
        aggregated = self.aggregator.aggregate(self.raw_signals)
        
        # RISK FILTER: Pass all aggregated signals through risk manager
        risk_approved = []
        for sig in aggregated:
            can_trade, reason = self.risk_manager.can_open_position(
                sig.symbol, sig.side.value, sig.signal.strategy_id
            )
            if can_trade:
                # Calculate optimal position size
                stop_loss = sig.signal.stop_loss or sig.signal.price * 0.98
                size, meta = self.risk_manager.calculate_position_size(
                    symbol=sig.symbol,
                    price=sig.signal.price,
                    side=sig.side.value,
                    stop_loss=stop_loss,
                    strategy_id=sig.signal.strategy_id,
                    base_confidence=sig.adjusted_confidence
                )
                if size > 0:
                    sig.signal.metadata['risk_approved'] = True
                    sig.signal.metadata['risk_position_size'] = size
                    sig.signal.metadata['risk_meta'] = meta
                    risk_approved.append(sig)
                else:
                    logger.info(f"🛡️ Risk blocked {sig.symbol}: {meta.get('reason', 'size too small')}")
            else:
                logger.info(f"🛡️ Risk blocked {sig.symbol}: {reason}")
        
        aggregated = risk_approved
        selected_symbols = {s.symbol for s in aggregated}
        
        # ==========================================
        # SECOND CHANCE LOGIC: Rescue promising rejects
        # ==========================================
        rescued_signals = []
        for strategy_id, signals in self.raw_signals.items():
            for signal in signals:
                if signal.symbol not in selected_symbols:
                    rejection_reason = self._get_rejection_reason(signal)
                    
                    # Skip if already in position (can't rescue)
                    if signal.symbol in self.positions:
                        continue
                    
                    # Evaluate for second chance
                    should_rescue, adj_conf, boost_reasons = self.second_chance.evaluate_for_second_chance(
                        signal, rejection_reason, self.raw_signals
                    )
                    
                    if should_rescue:
                        logger.info(f"🎯 SECOND CHANCE: Rescuing {strategy_id} {signal.side.value} "
                                   f"{signal.symbol} ({signal.confidence:.0%} -> {adj_conf:.0%})")
                        for reason in boost_reasons:
                            logger.debug(f"   ↳ {reason}")
                        
                        # Create aggregated signal from rescued signal
                        signal.confidence = adj_conf  # Boost confidence
                        signal.metadata['rescued'] = True
                        signal.metadata['original_confidence'] = signal.confidence
                        signal.metadata['boost_reasons'] = boost_reasons
                        
                        rescued_agg = AggregatedSignal(
                            signal=signal,
                            rank=len(aggregated) + len(rescued_signals) + 1,
                            sources=[strategy_id],
                            conflicts=[],
                            consensus_score=1.0,
                            adjusted_confidence=adj_conf
                        )
                        rescued_signals.append(rescued_agg)
                        selected_symbols.add(signal.symbol)
                        self.rescue_stats['total_rescued'] += 1
        
        # Add rescued signals to aggregated list
        if rescued_signals:
            logger.info(f"🎯 Rescued {len(rescued_signals)} signals via second-chance logic")
            aggregated.extend(rescued_signals)
            # Re-sort by adjusted confidence
            aggregated.sort(key=lambda s: s.adjusted_confidence, reverse=True)
            # Re-assign ranks
            for i, sig in enumerate(aggregated):
                sig.rank = i + 1
        
        # Log ALL signals (selected, rescued, and rejected) for learning
        self._log_all_signals(aggregated)
        
        # Mark selected symbols for cooldown
        for sig in aggregated:
            self.aggregator.mark_signaled(sig.symbol)
        
        # Clear raw signals (they've been processed)
        self.raw_signals.clear()
        
        logger.info(f"Selected {len(aggregated)} signals (incl. {len(rescued_signals)} rescued)")
        
        return aggregated
    
    def _log_all_signals(self, selected: List[AggregatedSignal]):
        """Log all signals (selected and rejected) for learning + WATCHDOG tracking"""
        selected_symbols = {s.symbol for s in selected}
        
        # Log selected signals
        for agg in selected:
            record = SignalRecord(
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=agg.signal.strategy_id,
                symbol=agg.symbol,
                side=agg.side.value,
                confidence=agg.signal.confidence,
                price=agg.signal.price,
                stop_loss=agg.signal.stop_loss,
                take_profit=agg.signal.take_profit,
                reason=agg.signal.reason,
                was_selected=True,
                selection_reason=f"Rank {agg.rank}, consensus {agg.consensus_score:.0%}",
                aggregated_rank=agg.rank,
                consensus_score=agg.consensus_score
            )
            self._save_signal_record(record)
            
            # WATCHDOG: Record accepted decision
            self.watchdog.record_decision(
                signal_data={
                    'strategy_id': agg.signal.strategy_id,
                    'symbol': agg.symbol,
                    'side': agg.side.value,
                    'confidence': agg.signal.confidence,
                    'price': agg.signal.price,
                    'stop_loss': agg.signal.stop_loss,
                    'take_profit': agg.signal.take_profit,
                },
                accepted=True
            )
        
        # Log rejected signals for counterfactual analysis
        for strategy_id, signals in self.raw_signals.items():
            for signal in signals:
                if signal.symbol not in selected_symbols:
                    rejection_reason = self._get_rejection_reason(signal)
                    record = SignalRecord(
                        timestamp=datetime.utcnow().isoformat(),
                        strategy_id=strategy_id,
                        symbol=signal.symbol,
                        side=signal.side.value,
                        confidence=signal.confidence,
                        price=signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reason=signal.reason,
                        was_selected=False,
                        selection_reason=rejection_reason,
                        aggregated_rank=None,
                        consensus_score=None
                    )
                    self._save_signal_record(record)
                    
                    # Add to counterfactual tracking
                    self.pending_counterfactual.append(record)
                    
                    # WATCHDOG: Record rejected decision
                    self.watchdog.record_decision(
                        signal_data={
                            'strategy_id': strategy_id,
                            'symbol': signal.symbol,
                            'side': signal.side.value,
                            'confidence': signal.confidence,
                            'price': signal.price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                        },
                        accepted=False,
                        rejection_reason=rejection_reason
                    )
    
    def _get_rejection_reason(self, signal: Signal) -> str:
        """Determine why a signal was rejected"""
        if signal.confidence < self.aggregator.config.min_confidence:
            return f"Low confidence ({signal.confidence:.0%} < {self.aggregator.config.min_confidence:.0%})"
        if signal.symbol in self.positions:
            return f"Already in {self.positions[signal.symbol]} position"
        if self.aggregator._in_cooldown(signal.symbol):
            return "Symbol in cooldown"
        return "Lower rank than selected signals"
    
    def _save_signal_record(self, record: SignalRecord):
        """Save signal record to log file"""
        try:
            with open(SIGNALS_LOG, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            logger.error(f"Error saving signal record: {e}")
    
    def record_trade_result(self, symbol: str, side: str, pnl: float, 
                           pnl_pct: float, strategy_id: str, reason: str,
                           was_rescued: bool = False, exit_price: float = 0.0):
        """Record a completed trade for learning"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy_id': strategy_id,
            'exit_reason': reason,
            'won': pnl > 0,
            'was_rescued': was_rescued
        }
        
        try:
            with open(TRADES_LOG, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error saving trade record: {e}")
        
        # Track rescued signal performance
        if was_rescued:
            if pnl > 0:
                self.rescue_stats['rescued_wins'] += 1
                logger.info(f"🎯 RESCUED WIN: {symbol} +{pnl_pct:.1f}%")
            else:
                self.rescue_stats['rescued_losses'] += 1
                logger.info(f"🎯 RESCUED LOSS: {symbol} {pnl_pct:.1f}%")
        
        # Update strategy performance
        self._update_strategy_performance(strategy_id, pnl > 0, pnl_pct)
        
        # Update risk manager with trade result
        if exit_price > 0:
            self.risk_manager.close_position(symbol, exit_price, reason)
    
    def update_equity(self, equity: float):
        """Update account equity in risk manager"""
        self.risk_manager.update_equity(equity)
    
    def register_position(self, symbol: str, side: str, size: float, 
                         entry_price: float, stop_loss: float, strategy_id: str):
        """Register a new position with risk manager"""
        self.risk_manager.register_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            strategy_id=strategy_id
        )
        self.positions[symbol] = side
    
    def update_volatility(self, symbol: str, prices: List[float]):
        """Update volatility data for risk manager"""
        self.risk_manager.update_volatility(symbol, prices)
    
    def get_risk_status(self) -> Dict:
        """Get risk manager status"""
        return self.risk_manager.get_status()
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed (circuit breaker check)"""
        return self.risk_manager.circuit_breaker.can_trade()
    
    def _update_strategy_performance(self, strategy_id: str, won: bool, pnl_pct: float):
        """Update strategy performance tracking with dynamic multiplier calculation"""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'trades': 0,
                'wins': 0,
                'total_pnl_pct': 0,
                'score': 0,
                'multiplier': 1.0  # Dynamic position sizing multiplier
            }
        
        perf = self.strategy_performance[strategy_id]
        perf['trades'] += 1
        perf['wins'] += 1 if won else 0
        perf['total_pnl_pct'] += pnl_pct
        
        # Calculate score (win rate * avg return, normalized)
        win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
        avg_return = perf['total_pnl_pct'] / perf['trades'] if perf['trades'] > 0 else 0
        perf['score'] = win_rate * (1 + avg_return / 10)  # Normalized score 0-2
        perf['win_rate'] = win_rate
        
        # DYNAMIC MULTIPLIER: Adjust position sizing based on track record
        # Need minimum 10 trades to start adjusting
        if perf['trades'] >= 10:
            if win_rate < 0.35:
                # Terrible: disable this strategy
                perf['multiplier'] = 0.0
                logger.warning(f"🚫 Strategy {strategy_id} disabled: {win_rate:.0%} WR")
            elif win_rate < 0.45:
                # Poor: heavily reduce
                perf['multiplier'] = 0.5
            elif perf['total_pnl_pct'] < 0:
                # Negative P&L despite OK WR: bad R:R
                perf['multiplier'] = 0.7
            elif win_rate >= 0.50 and perf['total_pnl_pct'] > 0:
                # Winner: boost proportionally to P&L
                perf['multiplier'] = min(2.0, 1.0 + perf['total_pnl_pct'] / 500)
            else:
                perf['multiplier'] = 1.0
        
        self._save_strategy_performance()
    
    def _load_strategy_performance(self):
        """Load strategy performance from file"""
        perf_file = DATA_DIR / 'strategy_performance.json'
        if perf_file.exists():
            try:
                with open(perf_file) as f:
                    self.strategy_performance = json.load(f)
            except:
                pass
    
    def _save_strategy_performance(self):
        """Save strategy performance to file"""
        perf_file = DATA_DIR / 'strategy_performance.json'
        try:
            with open(perf_file, 'w') as f:
                json.dump(self.strategy_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving strategy performance: {e}")
    
    def update_counterfactuals(self, current_prices: Dict[str, float]):
        """
        Update counterfactual tracking for rejected signals.
        Call this periodically to see what would have happened.
        Also updates second-chance near-miss tracking.
        """
        now = datetime.utcnow()
        completed = []
        
        for record in self.pending_counterfactual:
            signal_time = datetime.fromisoformat(record.timestamp)
            age_hours = (now - signal_time).total_seconds() / 3600
            
            if record.symbol not in current_prices:
                continue
            
            current_price = current_prices[record.symbol]
            entry_price = record.price
            
            # Update price tracking
            if age_hours >= 1 and record.price_1h_later is None:
                record.price_1h_later = current_price
            if age_hours >= 4 and record.price_4h_later is None:
                record.price_4h_later = current_price
            if age_hours >= 24 and record.price_24h_later is None:
                record.price_24h_later = current_price
                
                # Calculate if it would have won
                if record.side == 'long':
                    potential_pnl = (current_price - entry_price) / entry_price * 100
                else:
                    potential_pnl = (entry_price - current_price) / entry_price * 100
                
                record.potential_pnl = potential_pnl
                record.would_have_won = potential_pnl > 0
                
                # Save counterfactual result
                self._save_counterfactual(record)
                completed.append(record)
        
        # Remove completed records
        for record in completed:
            self.pending_counterfactual.remove(record)
        
        # Also update second-chance near-miss tracking
        self.second_chance.update_counterfactuals(current_prices)
    
    def _save_counterfactual(self, record: SignalRecord):
        """Save counterfactual analysis result"""
        try:
            with open(COUNTERFACTUAL_LOG, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            logger.error(f"Error saving counterfactual: {e}")
    
    def get_learning_summary(self) -> dict:
        """Get summary of learning data for reflection"""
        return {
            'strategy_performance': self.strategy_performance,
            'strategy_multipliers': self.get_strategy_multipliers(),
            'pending_counterfactuals': len(self.pending_counterfactual),
            'second_chance': {
                'total_rescued': self.rescue_stats['total_rescued'],
                'rescued_wins': self.rescue_stats['rescued_wins'],
                'rescued_losses': self.rescue_stats['rescued_losses'],
                'winning_patterns': len(self.second_chance.winning_patterns),
                'pending_near_misses': len(self.second_chance.near_miss_signals),
            },
            'risk_manager': self.risk_manager.get_status(),
            'data_files': {
                'signals': str(SIGNALS_LOG),
                'trades': str(TRADES_LOG),
                'counterfactual': str(COUNTERFACTUAL_LOG)
            }
        }
    
    def get_strategy_multipliers(self) -> Dict[str, float]:
        """
        Get current strategy position sizing multipliers.
        Used by execution engine for dynamic position sizing.
        """
        multipliers = {
            # Default multipliers (before we have data)
            'trend-following': 1.5,  # Known performer: +$208, 51% WR
            'mean-reversion': 1.0,
            'turtle': 1.0,
            'weinstein': 1.0,
            'livermore': 1.0,
            'bts-lynch': 0.8,
            'zweig': 0.7,  # FIXED v2: Probationary - thrust detection + trend confirmation
        }
        
        # Override with learned multipliers where available
        for strategy_id, perf in self.strategy_performance.items():
            if 'multiplier' in perf:
                multipliers[strategy_id] = perf['multiplier']
        
        return multipliers
    
    def analyze_counterfactuals(self) -> Dict:
        """
        Analyze historical counterfactual data to identify winning rejection patterns.
        Call this periodically (e.g., daily) to learn from past decisions.
        
        Returns summary of patterns found and potential improvements.
        """
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'counterfactual_analysis': {},
            'pattern_analysis': {},
            'recommendations': []
        }
        
        # Analyze main counterfactual log
        if COUNTERFACTUAL_LOG.exists():
            try:
                winners = []
                losers = []
                by_reason = {}
                
                with open(COUNTERFACTUAL_LOG, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get('would_have_won') is None:
                                continue
                            
                            reason = record.get('selection_reason', 'unknown')
                            if reason not in by_reason:
                                by_reason[reason] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                            
                            if record.get('would_have_won'):
                                winners.append(record)
                                by_reason[reason]['wins'] += 1
                            else:
                                losers.append(record)
                                by_reason[reason]['losses'] += 1
                            
                            by_reason[reason]['total_pnl'] += record.get('potential_pnl', 0)
                        except:
                            continue
                
                total = len(winners) + len(losers)
                result['counterfactual_analysis'] = {
                    'total_rejections_tracked': total,
                    'would_have_won': len(winners),
                    'would_have_lost': len(losers),
                    'hypothetical_win_rate': len(winners) / total if total > 0 else 0,
                    'by_rejection_reason': by_reason
                }
                
                # Generate recommendations based on analysis
                for reason, stats in by_reason.items():
                    total_for_reason = stats['wins'] + stats['losses']
                    if total_for_reason >= 5:
                        win_rate = stats['wins'] / total_for_reason
                        if win_rate >= 0.60:
                            result['recommendations'].append({
                                'type': 'REDUCE_FALSE_NEGATIVES',
                                'reason': reason,
                                'win_rate': f"{win_rate:.0%}",
                                'sample_size': total_for_reason,
                                'suggestion': f"Signals rejected for '{reason}' win {win_rate:.0%} of the time. Consider relaxing this filter."
                            })
                        elif win_rate <= 0.35:
                            result['recommendations'].append({
                                'type': 'GOOD_REJECTION',
                                'reason': reason,
                                'win_rate': f"{win_rate:.0%}",
                                'sample_size': total_for_reason,
                                'suggestion': f"Rejecting '{reason}' is correct - only {win_rate:.0%} would have won."
                            })
            except Exception as e:
                result['counterfactual_analysis'] = {'error': str(e)}
        else:
            result['counterfactual_analysis'] = {'status': 'no_data_yet'}
        
        # Analyze second-chance patterns
        result['pattern_analysis'] = self.second_chance.analyze_historical_rejections(COUNTERFACTUAL_LOG)
        
        return result
