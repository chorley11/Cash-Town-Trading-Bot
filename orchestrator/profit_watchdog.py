"""
PROFIT WATCHDOG - The Self-Improving Feedback Loop

This is the conscience of the trading system. It watches everything,
tracks what actually makes money, and auto-tunes for maximum profit.

Key responsibilities:
1. Monitor every orchestrator decision (accept/reject)
2. Track actual P&L vs predicted outcomes
3. Detect strategy performance drift from historical
4. Generate alerts for underperformance
5. Recommend parameter changes based on real data

The watchdog runs AFTER each orchestrator cycle, learning from results.
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
WATCHDOG_LOG = DATA_DIR / 'watchdog_history.jsonl'
ALERTS_LOG = DATA_DIR / 'watchdog_alerts.jsonl'
TUNING_LOG = DATA_DIR / 'parameter_tuning.jsonl'


@dataclass
class DecisionRecord:
    """Record of a single orchestrator decision"""
    timestamp: str
    signal_id: str  # strategy_id:symbol
    strategy_id: str
    symbol: str
    side: str
    confidence: float
    price_at_decision: float
    
    # Decision
    accepted: bool
    rejection_reason: Optional[str]
    
    # Predicted outcome (from signal)
    predicted_direction: str  # 'up' or 'down'
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    # Actual outcome (filled later)
    price_1h: Optional[float] = None
    price_4h: Optional[float] = None
    price_24h: Optional[float] = None
    actual_pnl: Optional[float] = None
    actual_pnl_pct: Optional[float] = None
    was_winner: Optional[bool] = None
    outcome_filled: bool = False


@dataclass
class StrategyDrift:
    """Tracks strategy performance drift"""
    strategy_id: str
    historical_win_rate: float
    recent_win_rate: float  # Last 20 trades
    drift_pct: float  # % change from historical
    is_underperforming: bool
    sample_size: int
    confidence_in_drift: str  # 'low', 'medium', 'high'


@dataclass
class ParameterRecommendation:
    """Recommended parameter change"""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    expected_impact: str
    confidence: str  # 'low', 'medium', 'high'
    evidence: Dict


@dataclass
class WatchdogAlert:
    """Alert from the watchdog"""
    timestamp: str
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'underperformance', 'drift', 'missed_opportunity', 'parameter'
    message: str
    details: Dict
    recommended_action: str


class ProfitWatchdog:
    """
    The profit-obsessed guardian of the trading system.
    
    Runs after each orchestrator cycle to:
    - Log all decisions
    - Track outcomes
    - Detect problems
    - Recommend improvements
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        
        # Decision tracking
        self.pending_outcomes: List[DecisionRecord] = []
        self.decision_history: List[DecisionRecord] = []
        
        # Strategy performance tracking
        self.strategy_baselines: Dict[str, Dict] = {}
        self.recent_performance: Dict[str, List[bool]] = defaultdict(list)  # Last N outcomes
        
        # Alert tracking
        self.active_alerts: List[WatchdogAlert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Parameter tracking
        self.parameter_experiments: List[Dict] = []
        
        # Stats
        self.stats = {
            'total_decisions_tracked': 0,
            'total_accepts': 0,
            'total_rejects': 0,
            'accept_winners': 0,
            'accept_losers': 0,
            'reject_would_have_won': 0,
            'reject_would_have_lost': 0,
            'alerts_generated': 0,
            'recommendations_made': 0,
        }
        
        # Ensure data directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load watchdog state from disk"""
        state_file = DATA_DIR / 'watchdog_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.stats = data.get('stats', self.stats)
                    self.strategy_baselines = data.get('strategy_baselines', {})
                    
                    # Load pending outcomes
                    for record_data in data.get('pending_outcomes', []):
                        self.pending_outcomes.append(DecisionRecord(**record_data))
            except Exception as e:
                logger.warning(f"Could not load watchdog state: {e}")
    
    def _save_state(self):
        """Save watchdog state to disk"""
        state_file = DATA_DIR / 'watchdog_state.json'
        try:
            data = {
                'stats': self.stats,
                'strategy_baselines': self.strategy_baselines,
                'pending_outcomes': [asdict(r) for r in self.pending_outcomes[-500:]],
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save watchdog state: {e}")
    
    # ==========================================
    # CORE: Record decisions
    # ==========================================
    
    def record_decision(self, signal_data: Dict, accepted: bool, 
                       rejection_reason: Optional[str] = None):
        """
        Record an orchestrator decision.
        Call this for EVERY signal processed, accepted or rejected.
        """
        record = DecisionRecord(
            timestamp=datetime.utcnow().isoformat(),
            signal_id=f"{signal_data.get('strategy_id', 'unknown')}:{signal_data.get('symbol', 'unknown')}",
            strategy_id=signal_data.get('strategy_id', 'unknown'),
            symbol=signal_data.get('symbol', 'unknown'),
            side=signal_data.get('side', 'unknown'),
            confidence=signal_data.get('confidence', 0),
            price_at_decision=signal_data.get('price', 0),
            accepted=accepted,
            rejection_reason=rejection_reason,
            predicted_direction='up' if signal_data.get('side') == 'long' else 'down',
            stop_loss=signal_data.get('stop_loss'),
            take_profit=signal_data.get('take_profit'),
        )
        
        # Log to file
        self._log_decision(record)
        
        # Track for outcome filling
        self.pending_outcomes.append(record)
        
        # Update stats
        self.stats['total_decisions_tracked'] += 1
        if accepted:
            self.stats['total_accepts'] += 1
        else:
            self.stats['total_rejects'] += 1
        
        logger.debug(f"Watchdog recorded: {'✅' if accepted else '❌'} {record.signal_id}")
    
    def _log_decision(self, record: DecisionRecord):
        """Log decision to file"""
        try:
            with open(WATCHDOG_LOG, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            logger.error(f"Could not log decision: {e}")
    
    # ==========================================
    # CORE: Update outcomes with market data
    # ==========================================
    
    def update_outcomes(self, current_prices: Dict[str, float]):
        """
        Update pending decisions with actual market outcomes.
        Call this periodically with current prices.
        """
        now = datetime.utcnow()
        completed = []
        
        for record in self.pending_outcomes:
            if record.outcome_filled:
                completed.append(record)
                continue
            
            if record.symbol not in current_prices:
                continue
            
            signal_time = datetime.fromisoformat(record.timestamp)
            age_hours = (now - signal_time).total_seconds() / 3600
            current_price = current_prices[record.symbol]
            entry_price = record.price_at_decision
            
            if entry_price == 0:
                continue
            
            # Fill price snapshots
            if age_hours >= 1 and record.price_1h is None:
                record.price_1h = current_price
            if age_hours >= 4 and record.price_4h is None:
                record.price_4h = current_price
            if age_hours >= 24 and record.price_24h is None:
                record.price_24h = current_price
                
                # Calculate outcome at 24h mark
                if record.side == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                record.actual_pnl_pct = pnl_pct
                record.was_winner = pnl_pct > 0
                record.outcome_filled = True
                
                # Update strategy tracking
                self._record_outcome(record)
                completed.append(record)
        
        # Remove completed
        for record in completed:
            if record in self.pending_outcomes:
                self.pending_outcomes.remove(record)
        
        self._save_state()
    
    def _record_outcome(self, record: DecisionRecord):
        """Record outcome for learning"""
        # Update stats
        if record.accepted:
            if record.was_winner:
                self.stats['accept_winners'] += 1
            else:
                self.stats['accept_losers'] += 1
        else:
            if record.was_winner:
                self.stats['reject_would_have_won'] += 1
            else:
                self.stats['reject_would_have_lost'] += 1
        
        # Track recent performance per strategy
        self.recent_performance[record.strategy_id].append(record.was_winner)
        # Keep last 30
        self.recent_performance[record.strategy_id] = self.recent_performance[record.strategy_id][-30:]
        
        logger.info(f"Watchdog outcome: {record.signal_id} {'✅ WIN' if record.was_winner else '❌ LOSS'} "
                   f"({record.actual_pnl_pct:.1f}%) [{'accepted' if record.accepted else 'rejected'}]")
    
    # ==========================================
    # ANALYSIS: Detect strategy drift
    # ==========================================
    
    def detect_strategy_drift(self) -> List[StrategyDrift]:
        """
        Detect strategies whose recent performance deviates from historical.
        Returns list of strategies with significant drift.
        """
        drifts = []
        
        for strategy_id, recent_outcomes in self.recent_performance.items():
            if len(recent_outcomes) < 10:
                continue  # Not enough data
            
            # Get historical baseline
            baseline = self.strategy_baselines.get(strategy_id, {})
            historical_wr = baseline.get('win_rate', 0.5)
            
            # Calculate recent win rate
            recent_wr = sum(recent_outcomes) / len(recent_outcomes)
            
            # Calculate drift
            if historical_wr > 0:
                drift_pct = (recent_wr - historical_wr) / historical_wr * 100
            else:
                drift_pct = 0
            
            # Determine confidence in drift measurement
            if len(recent_outcomes) >= 30:
                confidence = 'high'
            elif len(recent_outcomes) >= 20:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Is it underperforming?
            is_underperforming = recent_wr < historical_wr * 0.75  # 25% worse than historical
            
            drift = StrategyDrift(
                strategy_id=strategy_id,
                historical_win_rate=historical_wr,
                recent_win_rate=recent_wr,
                drift_pct=drift_pct,
                is_underperforming=is_underperforming,
                sample_size=len(recent_outcomes),
                confidence_in_drift=confidence
            )
            
            if abs(drift_pct) >= 15:  # Significant drift
                drifts.append(drift)
        
        return drifts
    
    def update_baseline(self, strategy_id: str, win_rate: float, trades: int, pnl: float):
        """Update the baseline performance for a strategy"""
        self.strategy_baselines[strategy_id] = {
            'win_rate': win_rate,
            'trades': trades,
            'total_pnl': pnl,
            'updated': datetime.utcnow().isoformat()
        }
        self._save_state()
    
    # ==========================================
    # ANALYSIS: Generate parameter recommendations
    # ==========================================
    
    def analyze_and_recommend(self) -> List[ParameterRecommendation]:
        """
        Analyze historical data and generate parameter recommendations.
        """
        recommendations = []
        
        # 1. Analyze confidence threshold effectiveness
        conf_rec = self._analyze_confidence_threshold()
        if conf_rec:
            recommendations.append(conf_rec)
        
        # 2. Analyze strategy multipliers
        mult_recs = self._analyze_strategy_multipliers()
        recommendations.extend(mult_recs)
        
        # 3. Analyze rejection patterns
        reject_recs = self._analyze_rejections()
        recommendations.extend(reject_recs)
        
        self.stats['recommendations_made'] += len(recommendations)
        
        return recommendations
    
    def _analyze_confidence_threshold(self) -> Optional[ParameterRecommendation]:
        """Analyze if confidence threshold should be adjusted"""
        # Load recent decisions
        decisions_by_confidence = defaultdict(lambda: {'wins': 0, 'losses': 0})
        
        if not WATCHDOG_LOG.exists():
            return None
        
        try:
            with open(WATCHDOG_LOG, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('was_winner') is None:
                            continue
                        
                        # Bucket by confidence (0.5-0.6, 0.6-0.7, etc.)
                        conf = record.get('confidence', 0)
                        bucket = int(conf * 10) / 10  # Round to 0.1
                        
                        if record.get('was_winner'):
                            decisions_by_confidence[bucket]['wins'] += 1
                        else:
                            decisions_by_confidence[bucket]['losses'] += 1
                    except:
                        continue
        except:
            return None
        
        # Analyze win rates by confidence bucket
        current_threshold = 0.55
        if self.orchestrator:
            current_threshold = getattr(
                getattr(self.orchestrator, 'aggregator', None), 
                'config', 
                type('', (), {'min_confidence': 0.55})()
            ).min_confidence
        
        # Find optimal threshold
        best_threshold = current_threshold
        best_edge = 0
        
        for bucket, stats in sorted(decisions_by_confidence.items()):
            total = stats['wins'] + stats['losses']
            if total < 10:
                continue
            
            wr = stats['wins'] / total
            edge = wr - 0.5  # Edge over random
            
            if edge > best_edge and bucket >= 0.5:
                best_edge = edge
                best_threshold = bucket
        
        # Only recommend if significant improvement
        if abs(best_threshold - current_threshold) >= 0.05:
            return ParameterRecommendation(
                parameter='min_confidence',
                current_value=current_threshold,
                recommended_value=best_threshold,
                reason=f"Signals with confidence >={best_threshold:.0%} have {best_edge+0.5:.0%} win rate",
                expected_impact=f"+{best_edge*10:.1f}% edge improvement",
                confidence='medium' if sum(s['wins']+s['losses'] for s in decisions_by_confidence.values()) > 50 else 'low',
                evidence={'win_rates_by_confidence': {k: v for k, v in decisions_by_confidence.items()}}
            )
        
        return None
    
    def _analyze_strategy_multipliers(self) -> List[ParameterRecommendation]:
        """Analyze if strategy multipliers should be adjusted"""
        recommendations = []
        
        for strategy_id, outcomes in self.recent_performance.items():
            if len(outcomes) < 15:
                continue
            
            win_rate = sum(outcomes) / len(outcomes)
            baseline = self.strategy_baselines.get(strategy_id, {}).get('win_rate', 0.5)
            
            # Get current multiplier
            current_mult = 1.0
            if self.orchestrator:
                mults = getattr(self.orchestrator, 'get_strategy_multipliers', lambda: {})()
                current_mult = mults.get(strategy_id, 1.0)
            
            # Calculate recommended multiplier
            if win_rate < 0.35:
                recommended = 0.0  # Disable
                reason = f"Strategy has only {win_rate:.0%} win rate - disable"
            elif win_rate < 0.45:
                recommended = 0.5
                reason = f"Reduce exposure - {win_rate:.0%} win rate is below threshold"
            elif win_rate >= 0.55:
                recommended = min(2.0, 1.0 + (win_rate - 0.5) * 4)
                reason = f"Strong performer - {win_rate:.0%} win rate deserves more capital"
            else:
                continue  # No change needed
            
            if abs(recommended - current_mult) >= 0.2:
                recommendations.append(ParameterRecommendation(
                    parameter=f'strategy_multiplier:{strategy_id}',
                    current_value=current_mult,
                    recommended_value=recommended,
                    reason=reason,
                    expected_impact="Better capital allocation",
                    confidence='high' if len(outcomes) >= 30 else 'medium',
                    evidence={'win_rate': win_rate, 'sample_size': len(outcomes)}
                ))
        
        return recommendations
    
    def _analyze_rejections(self) -> List[ParameterRecommendation]:
        """Analyze if we're rejecting too many winners"""
        reject_wins = self.stats.get('reject_would_have_won', 0)
        reject_losses = self.stats.get('reject_would_have_lost', 0)
        total_rejects = reject_wins + reject_losses
        
        if total_rejects < 20:
            return []
        
        reject_wr = reject_wins / total_rejects
        
        recommendations = []
        
        if reject_wr >= 0.55:
            # We're rejecting too many good signals
            recommendations.append(ParameterRecommendation(
                parameter='signal_filtering',
                current_value='current filters',
                recommended_value='relaxed filters',
                reason=f"Rejected signals have {reject_wr:.0%} win rate - we're being too conservative",
                expected_impact=f"Capture {reject_wins} more winning trades",
                confidence='medium',
                evidence={
                    'rejected_would_have_won': reject_wins,
                    'rejected_would_have_lost': reject_losses,
                    'rejection_win_rate': reject_wr
                }
            ))
        elif reject_wr <= 0.40:
            # Our rejection is working well
            recommendations.append(ParameterRecommendation(
                parameter='signal_filtering',
                current_value='current filters',
                recommended_value='maintain or tighten',
                reason=f"Good rejection - only {reject_wr:.0%} of rejects would have won",
                expected_impact="Filters are working correctly",
                confidence='high',
                evidence={
                    'rejected_would_have_won': reject_wins,
                    'rejected_would_have_lost': reject_losses,
                    'rejection_win_rate': reject_wr
                }
            ))
        
        return recommendations
    
    # ==========================================
    # ALERTS: Generate and manage alerts
    # ==========================================
    
    def generate_alerts(self) -> List[WatchdogAlert]:
        """Generate alerts for issues that need attention"""
        alerts = []
        
        # 1. Strategy drift alerts
        drifts = self.detect_strategy_drift()
        for drift in drifts:
            if drift.is_underperforming:
                alert = WatchdogAlert(
                    timestamp=datetime.utcnow().isoformat(),
                    severity='warning' if drift.confidence_in_drift != 'high' else 'critical',
                    category='drift',
                    message=f"Strategy {drift.strategy_id} underperforming: "
                            f"{drift.recent_win_rate:.0%} vs historical {drift.historical_win_rate:.0%}",
                    details=asdict(drift),
                    recommended_action=f"Review and potentially reduce multiplier for {drift.strategy_id}"
                )
                alerts.append(alert)
        
        # 2. Overall system performance
        accept_wins = self.stats.get('accept_winners', 0)
        accept_losses = self.stats.get('accept_losers', 0)
        total_accepted = accept_wins + accept_losses
        
        if total_accepted >= 20:
            system_wr = accept_wins / total_accepted
            if system_wr < 0.45:
                alert = WatchdogAlert(
                    timestamp=datetime.utcnow().isoformat(),
                    severity='critical',
                    category='underperformance',
                    message=f"System win rate critically low: {system_wr:.0%}",
                    details={
                        'accept_winners': accept_wins,
                        'accept_losers': accept_losses,
                        'system_win_rate': system_wr
                    },
                    recommended_action="Review all strategy multipliers and confidence thresholds"
                )
                alerts.append(alert)
        
        # 3. Missed opportunity alert
        reject_wins = self.stats.get('reject_would_have_won', 0)
        reject_losses = self.stats.get('reject_would_have_lost', 0)
        if reject_wins > 20 and reject_wins > reject_losses:
            alert = WatchdogAlert(
                timestamp=datetime.utcnow().isoformat(),
                severity='warning',
                category='missed_opportunity',
                message=f"Rejecting too many winners: {reject_wins} good signals rejected",
                details={
                    'rejected_would_have_won': reject_wins,
                    'rejected_would_have_lost': reject_losses
                },
                recommended_action="Consider relaxing confidence threshold or consensus requirements"
            )
            alerts.append(alert)
        
        # Log alerts
        for alert in alerts:
            self._log_alert(alert)
            self.stats['alerts_generated'] += 1
        
        self.active_alerts = alerts
        return alerts
    
    def _log_alert(self, alert: WatchdogAlert):
        """Log alert to file"""
        try:
            with open(ALERTS_LOG, 'a') as f:
                f.write(json.dumps(asdict(alert)) + '\n')
        except Exception as e:
            logger.error(f"Could not log alert: {e}")
    
    # ==========================================
    # API: Get watchdog status
    # ==========================================
    
    def get_status(self) -> Dict:
        """Get full watchdog status for API"""
        # Calculate derived stats
        total_accepted = self.stats['accept_winners'] + self.stats['accept_losers']
        total_rejected = self.stats['reject_would_have_won'] + self.stats['reject_would_have_lost']
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_decisions_tracked': self.stats['total_decisions_tracked'],
                'pending_outcomes': len(self.pending_outcomes),
                'active_alerts': len(self.active_alerts),
            },
            'accept_performance': {
                'total': total_accepted,
                'winners': self.stats['accept_winners'],
                'losers': self.stats['accept_losers'],
                'win_rate': self.stats['accept_winners'] / total_accepted if total_accepted > 0 else 0
            },
            'reject_counterfactual': {
                'total': total_rejected,
                'would_have_won': self.stats['reject_would_have_won'],
                'would_have_lost': self.stats['reject_would_have_lost'],
                'rejection_accuracy': self.stats['reject_would_have_lost'] / total_rejected if total_rejected > 0 else 0
            },
            'strategy_drift': [asdict(d) for d in self.detect_strategy_drift()],
            'recommendations': [asdict(r) for r in self.analyze_and_recommend()],
            'alerts': [asdict(a) for a in self.active_alerts],
            'strategy_baselines': self.strategy_baselines,
            'recent_performance': {k: {'wins': sum(v), 'total': len(v), 'win_rate': sum(v)/len(v) if v else 0} 
                                   for k, v in self.recent_performance.items()}
        }
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Get recent decisions with outcomes"""
        decisions = []
        
        if not WATCHDOG_LOG.exists():
            return decisions
        
        try:
            with open(WATCHDOG_LOG, 'r') as f:
                lines = f.readlines()[-limit:]
                for line in reversed(lines):
                    try:
                        decisions.append(json.loads(line))
                    except:
                        continue
        except:
            pass
        
        return decisions
    
    # ==========================================
    # AUTO-TUNE: Apply recommendations
    # ==========================================
    
    def auto_tune(self, dry_run: bool = True) -> List[Dict]:
        """
        Apply parameter recommendations automatically.
        
        Args:
            dry_run: If True, only log what would change
        
        Returns:
            List of changes made (or would be made)
        """
        recommendations = self.analyze_and_recommend()
        changes = []
        
        for rec in recommendations:
            if rec.confidence in ('low',):
                continue  # Skip low confidence recommendations
            
            change = {
                'parameter': rec.parameter,
                'old_value': rec.current_value,
                'new_value': rec.recommended_value,
                'reason': rec.reason,
                'applied': not dry_run
            }
            
            if not dry_run and self.orchestrator:
                # Apply the change
                try:
                    if rec.parameter == 'min_confidence':
                        self.orchestrator.aggregator.config.min_confidence = rec.recommended_value
                        logger.info(f"🔧 Auto-tuned min_confidence: {rec.current_value} -> {rec.recommended_value}")
                    elif rec.parameter.startswith('strategy_multiplier:'):
                        strategy_id = rec.parameter.split(':')[1]
                        if strategy_id in self.orchestrator.strategy_performance:
                            self.orchestrator.strategy_performance[strategy_id]['multiplier'] = rec.recommended_value
                            logger.info(f"🔧 Auto-tuned {strategy_id} multiplier: {rec.current_value} -> {rec.recommended_value}")
                    
                    change['applied'] = True
                except Exception as e:
                    change['error'] = str(e)
                    change['applied'] = False
            
            changes.append(change)
            
            # Log the tuning action
            self._log_tuning(change)
        
        return changes
    
    def _log_tuning(self, change: Dict):
        """Log parameter tuning"""
        try:
            change['timestamp'] = datetime.utcnow().isoformat()
            with open(TUNING_LOG, 'a') as f:
                f.write(json.dumps(change) + '\n')
        except Exception as e:
            logger.error(f"Could not log tuning: {e}")


# ==========================================
# WATCHDOG CYCLE: Run after each orchestrator cycle
# ==========================================

def run_watchdog_cycle(watchdog: ProfitWatchdog, current_prices: Dict[str, float]):
    """
    Run a complete watchdog cycle.
    Call this after each orchestrator processing cycle.
    """
    logger.info("🐕 Watchdog cycle starting...")
    
    # 1. Update outcomes with current prices
    watchdog.update_outcomes(current_prices)
    
    # 2. Generate alerts
    alerts = watchdog.generate_alerts()
    if alerts:
        for alert in alerts:
            level = logging.WARNING if alert.severity == 'warning' else logging.ERROR if alert.severity == 'critical' else logging.INFO
            logger.log(level, f"🚨 WATCHDOG ALERT [{alert.severity}]: {alert.message}")
    
    # 3. Check for auto-tune opportunities (dry run by default)
    changes = watchdog.auto_tune(dry_run=True)
    if changes:
        logger.info(f"🔧 Watchdog has {len(changes)} parameter recommendations")
    
    logger.info("🐕 Watchdog cycle complete")
    
    return watchdog.get_status()
