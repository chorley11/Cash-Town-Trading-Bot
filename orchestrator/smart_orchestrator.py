"""
Smart Orchestrator - Intelligent signal selection with learning

Key improvements:
1. Uses SignalAggregator to rank and filter signals
2. Stores ALL signals for learning (even rejected ones)
3. Clears signals after processing
4. Integrates reflection for self-improvement
5. Tracks what would have happened with rejected signals
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .signal_aggregator import SignalAggregator, AggregatorConfig, AggregatedSignal
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
    """
    
    def __init__(self, config: AggregatorConfig = None):
        self.aggregator = SignalAggregator(config or AggregatorConfig(
            min_confidence=0.55,
            min_consensus=1,
            max_signals_per_cycle=3,
            cooldown_minutes=30,  # Increased from 15
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
        Also logs all signals for learning.
        """
        if not self.raw_signals:
            return []
        
        # Update aggregator with current positions
        self.aggregator.set_positions(self.positions)
        self.aggregator.set_strategy_performance({
            k: v.get('score', 0) for k, v in self.strategy_performance.items()
        })
        
        # Aggregate signals
        aggregated = self.aggregator.aggregate(self.raw_signals)
        
        # Log ALL signals (selected and rejected) for learning
        self._log_all_signals(aggregated)
        
        # Mark selected symbols for cooldown
        for sig in aggregated:
            self.aggregator.mark_signaled(sig.symbol)
        
        # Clear raw signals (they've been processed)
        self.raw_signals.clear()
        
        logger.info(f"Selected {len(aggregated)} signals from pool")
        
        return aggregated
    
    def _log_all_signals(self, selected: List[AggregatedSignal]):
        """Log all signals (selected and rejected) for learning"""
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
        
        # Log rejected signals for counterfactual analysis
        for strategy_id, signals in self.raw_signals.items():
            for signal in signals:
                if signal.symbol not in selected_symbols:
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
                        selection_reason=self._get_rejection_reason(signal),
                        aggregated_rank=None,
                        consensus_score=None
                    )
                    self._save_signal_record(record)
                    
                    # Add to counterfactual tracking
                    self.pending_counterfactual.append(record)
    
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
                           pnl_pct: float, strategy_id: str, reason: str):
        """Record a completed trade for learning"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy_id': strategy_id,
            'exit_reason': reason,
            'won': pnl > 0
        }
        
        try:
            with open(TRADES_LOG, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error saving trade record: {e}")
        
        # Update strategy performance
        self._update_strategy_performance(strategy_id, pnl > 0, pnl_pct)
    
    def _update_strategy_performance(self, strategy_id: str, won: bool, pnl_pct: float):
        """Update strategy performance tracking"""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'trades': 0,
                'wins': 0,
                'total_pnl_pct': 0,
                'score': 0
            }
        
        perf = self.strategy_performance[strategy_id]
        perf['trades'] += 1
        perf['wins'] += 1 if won else 0
        perf['total_pnl_pct'] += pnl_pct
        
        # Calculate score (win rate * avg return, normalized)
        win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
        avg_return = perf['total_pnl_pct'] / perf['trades'] if perf['trades'] > 0 else 0
        perf['score'] = win_rate * (1 + avg_return / 10)  # Normalized score 0-2
        
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
            'pending_counterfactuals': len(self.pending_counterfactual),
            'data_files': {
                'signals': str(SIGNALS_LOG),
                'trades': str(TRADES_LOG),
                'counterfactual': str(COUNTERFACTUAL_LOG)
            }
        }
