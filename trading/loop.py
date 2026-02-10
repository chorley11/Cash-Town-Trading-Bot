"""
Trading Loop - Main coordination loop for Cash Town

This is the brain that ties everything together:
1. Fetches market data
2. Runs all strategy agents
3. Aggregates signals
4. Executes trades
5. Manages positions
"""
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from data.feed import DataFeedManager
from agents.strategies import STRATEGY_REGISTRY
from agents.base import BaseStrategyAgent, Signal
from orchestrator.signal_aggregator import SignalAggregator, AggregatorConfig
from orchestrator.position_manager import PositionManager, RotationConfig, TrackedPosition
from execution.engine import ExecutionEngine, ExecutionMode, RiskConfig
from execution.kucoin import KuCoinFuturesExecutor

logger = logging.getLogger(__name__)

@dataclass
class LoopConfig:
    """Configuration for the trading loop"""
    # Timing
    data_refresh_seconds: int = 60
    signal_check_seconds: int = 300  # 5 minutes
    position_check_seconds: int = 30
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: [
        'XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'AVAXUSDTM', 'LINKUSDTM'
    ])
    
    # Strategies to run
    enabled_strategies: List[str] = field(default_factory=lambda: [
        'bts-lynch', 'zweig', 'trend-following', 'mean-reversion',
        'turtle', 'weinstein', 'livermore'
    ])
    
    # Execution mode
    execution_mode: ExecutionMode = ExecutionMode.PAPER

@dataclass
class LoopStats:
    """Statistics for the trading loop"""
    started_at: datetime = None
    cycles: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    last_signal_time: datetime = None
    last_trade_time: datetime = None
    errors: int = 0

class TradingLoop:
    """
    Main trading loop for Cash Town.
    
    Coordinates data feed, strategy agents, signal aggregation,
    and execution into a continuous trading system.
    """
    
    def __init__(self, config: LoopConfig = None):
        self.config = config or LoopConfig()
        self.stats = LoopStats()
        self.running = False
        
        # Initialize components
        self._init_components()
        
        # Threads
        self._data_thread: Optional[threading.Thread] = None
        self._signal_thread: Optional[threading.Thread] = None
        self._position_thread: Optional[threading.Thread] = None
    
    def _init_components(self):
        """Initialize all components"""
        logger.info("Initializing Cash Town components...")
        
        # Data feed
        self.data_feed = DataFeedManager(
            symbols=self.config.symbols,
            interval='15min'
        )
        
        # Strategy agents
        self.agents: Dict[str, BaseStrategyAgent] = {}
        for strategy_id in self.config.enabled_strategies:
            if strategy_id in STRATEGY_REGISTRY:
                agent_class = STRATEGY_REGISTRY[strategy_id]
                self.agents[strategy_id] = agent_class(symbols=self.config.symbols)
                logger.info(f"  ✓ Loaded strategy: {strategy_id}")
            else:
                logger.warning(f"  ✗ Unknown strategy: {strategy_id}")
        
        # Signal aggregator
        self.aggregator = SignalAggregator(AggregatorConfig(
            min_confidence=0.55,
            min_consensus=1,
            max_signals_per_cycle=3,
            cooldown_minutes=15
        ))
        
        # Position manager
        self.position_manager = PositionManager(RotationConfig(
            grace_period_minutes=30,
            stuck_max_minutes=120,
            fallen_negative_close=True,
            max_hold_hours=48
        ))
        
        # Execution engine
        self.executor = KuCoinFuturesExecutor()
        self.execution = ExecutionEngine(
            executor=self.executor,
            risk_config=RiskConfig(
                max_position_pct=2.0,
                max_total_exposure_pct=20.0,
                max_positions=5,
                max_daily_loss_pct=5.0,
                default_leverage=5
            ),
            mode=self.config.execution_mode
        )
        
        logger.info(f"Components initialized: {len(self.agents)} strategies, mode={self.config.execution_mode.value}")
    
    def start(self):
        """Start the trading loop"""
        if self.running:
            logger.warning("Trading loop already running")
            return
        
        self.running = True
        self.stats.started_at = datetime.utcnow()
        
        logger.info("🚀 Starting Cash Town Trading Loop")
        logger.info(f"   Mode: {self.config.execution_mode.value}")
        logger.info(f"   Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"   Strategies: {', '.join(self.config.enabled_strategies)}")
        
        # Initial data fetch
        logger.info("Fetching initial market data...")
        self.data_feed.start(poll_interval=self.config.data_refresh_seconds)
        time.sleep(2)  # Wait for initial data
        
        # Refresh execution state
        self.execution.refresh_state()
        
        # Start signal loop
        self._signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        self._signal_thread.start()
        
        # Start position monitoring loop
        self._position_thread = threading.Thread(target=self._position_loop, daemon=True)
        self._position_thread.start()
        
        logger.info("✅ Cash Town Trading Loop started")
    
    def stop(self):
        """Stop the trading loop"""
        logger.info("Stopping Cash Town Trading Loop...")
        self.running = False
        
        # Stop data feed
        self.data_feed.stop()
        
        # Wait for threads
        if self._signal_thread:
            self._signal_thread.join(timeout=5)
        if self._position_thread:
            self._position_thread.join(timeout=5)
        
        logger.info("✅ Cash Town Trading Loop stopped")
    
    def _signal_loop(self):
        """Main signal generation and execution loop"""
        while self.running:
            try:
                self._process_signals()
                self.stats.cycles += 1
            except Exception as e:
                logger.error(f"Signal loop error: {e}")
                self.stats.errors += 1
            
            # Sleep until next cycle
            self._sleep(self.config.signal_check_seconds)
    
    def _position_loop(self):
        """Position monitoring and rotation loop"""
        while self.running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error(f"Position loop error: {e}")
                self.stats.errors += 1
            
            self._sleep(self.config.position_check_seconds)
    
    def _sleep(self, seconds: int):
        """Sleep with running check"""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)
    
    def _process_signals(self):
        """Generate and process signals from all agents"""
        logger.debug("Processing signals...")
        
        # Get market data
        market_data = self.data_feed.get_data()
        if not market_data:
            logger.warning("No market data available")
            return
        
        # Generate signals from all agents
        all_signals: Dict[str, List[Signal]] = {}
        
        for strategy_id, agent in self.agents.items():
            try:
                signals = agent.generate_signals(market_data)
                if signals:
                    all_signals[strategy_id] = signals
                    self.stats.signals_generated += len(signals)
                    logger.debug(f"  {strategy_id}: {len(signals)} signals")
            except Exception as e:
                logger.error(f"Error generating signals from {strategy_id}: {e}")
        
        if not all_signals:
            logger.debug("No signals generated this cycle")
            return
        
        # Update aggregator with current positions
        current_positions = {p.symbol: p.side for p in self.execution.positions.values()}
        self.aggregator.set_positions(current_positions)
        
        # Aggregate signals
        aggregated = self.aggregator.aggregate(all_signals)
        
        if not aggregated:
            logger.debug("No actionable signals after aggregation")
            return
        
        # Log signals
        for agg in aggregated:
            logger.info(f"📊 Signal: {agg.signal.side.value.upper()} {agg.symbol}")
            logger.info(f"   Confidence: {agg.adjusted_confidence:.0%} (base: {agg.signal.confidence:.0%})")
            logger.info(f"   Sources: {agg.sources}")
            if agg.conflicts:
                logger.info(f"   Conflicts: {agg.conflicts}")
        
        self.stats.last_signal_time = datetime.utcnow()
        
        # Execute signals
        for agg in aggregated:
            result = self.execution.execute_signal(agg)
            if result.success:
                self.stats.trades_executed += 1
                self.stats.last_trade_time = datetime.utcnow()
                self.aggregator.mark_signaled(agg.symbol)
                logger.info(f"✅ Trade executed: {result.side} {result.size} {result.symbol}")
            else:
                logger.warning(f"❌ Trade failed: {result.message}")
    
    def _check_positions(self):
        """Check positions for rotation"""
        # Refresh state
        self.execution.refresh_state()
        
        # Update position manager with current positions
        for symbol, position in self.execution.positions.items():
            tracked = TrackedPosition(
                agent_id='cash-town',
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                entry_time=datetime.utcnow(),  # Would need actual entry time
                size=position.size,
                current_price=position.entry_price,  # Would need current price
                current_pnl=position.unrealized_pnl,
                current_pnl_pct=position.unrealized_pnl / (position.margin) * 100 if position.margin > 0 else 0
            )
            self.position_manager.track_position(tracked)
        
        # Evaluate rotations
        decisions = self.position_manager.evaluate_rotations()
        
        for decision in decisions:
            if decision.urgency == 'immediate':
                logger.warning(f"🔄 Closing position: {decision.position.symbol} - {decision.reason}")
                result = self.execution.close_position(decision.position.symbol, decision.reason)
                if result.success:
                    self.position_manager.remove_position('cash-town', decision.position.symbol)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            'running': self.running,
            'mode': self.config.execution_mode.value,
            'stats': {
                'started_at': self.stats.started_at.isoformat() if self.stats.started_at else None,
                'cycles': self.stats.cycles,
                'signals_generated': self.stats.signals_generated,
                'trades_executed': self.stats.trades_executed,
                'errors': self.stats.errors,
                'last_signal': self.stats.last_signal_time.isoformat() if self.stats.last_signal_time else None,
                'last_trade': self.stats.last_trade_time.isoformat() if self.stats.last_trade_time else None
            },
            'components': {
                'data_feed': {
                    'symbols': len(self.config.symbols),
                    'last_refresh': self.data_feed.last_refresh.isoformat() if self.data_feed.last_refresh else None
                },
                'agents': {
                    'total': len(self.agents),
                    'enabled': list(self.agents.keys())
                },
                'aggregator': self.aggregator.get_status(),
                'execution': self.execution.get_status(),
                'positions': self.position_manager.get_status()
            }
        }


def run_trading_loop(mode: str = 'paper'):
    """Convenience function to run the trading loop"""
    exec_mode = ExecutionMode.PAPER if mode == 'paper' else ExecutionMode.LIVE
    
    config = LoopConfig(execution_mode=exec_mode)
    loop = TradingLoop(config)
    
    try:
        loop.start()
        
        # Keep running
        while loop.running:
            time.sleep(60)
            status = loop.get_status()
            logger.info(f"Loop status: {status['stats']['cycles']} cycles, "
                       f"{status['stats']['signals_generated']} signals, "
                       f"{status['stats']['trades_executed']} trades")
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        loop.stop()


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'paper'
    run_trading_loop(mode)
