#!/usr/bin/env python3
"""
Cash Town Executor - Processes aggregated signals and executes trades

The executor:
1. Polls orchestrator for pending signals
2. Aggregates signals (consensus check)
3. Applies risk checks
4. Executes trades on KuCoin

Usage:
    python run_executor.py paper   # Paper trading
    python run_executor.py live    # Live trading
"""
import sys
import os
import time
import logging
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.kucoin import KuCoinFuturesExecutor
from execution.engine import ExecutionEngine, ExecutionMode, RiskConfig
from agents.base import SignalSide

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/cash-town-executor.log')
    ]
)
logger = logging.getLogger('executor')

ORCHESTRATOR_URL = "http://localhost:8888"

@dataclass
class AggregatedSignal:
    symbol: str
    side: str
    confidence: float
    sources: List[str]
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

class SignalProcessor:
    """Processes and aggregates signals from multiple agents"""
    
    def __init__(self, min_confidence: float = 0.55, min_consensus: int = 1):
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.processed_signals: set = set()  # Track processed to avoid duplicates
    
    def aggregate(self, signals: List[Dict]) -> List[AggregatedSignal]:
        """Aggregate signals by symbol, apply consensus logic"""
        if not signals:
            return []
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for sig in signals:
            # Skip if already processed (by timestamp + symbol + strategy)
            sig_key = f"{sig['timestamp']}_{sig['symbol']}_{sig['strategy_id']}"
            if sig_key in self.processed_signals:
                continue
            self.processed_signals.add(sig_key)
            
            if sig['confidence'] >= self.min_confidence:
                by_symbol[sig['symbol']].append(sig)
        
        # Keep processed set bounded
        if len(self.processed_signals) > 1000:
            self.processed_signals = set(list(self.processed_signals)[-500:])
        
        aggregated = []
        for symbol, sigs in by_symbol.items():
            # Count long vs short
            longs = [s for s in sigs if s['side'] == 'long']
            shorts = [s for s in sigs if s['side'] == 'short']
            
            # Determine majority
            if len(longs) > len(shorts) and len(longs) >= self.min_consensus:
                best = max(longs, key=lambda s: s['confidence'])
                aggregated.append(AggregatedSignal(
                    symbol=symbol,
                    side='long',
                    confidence=best['confidence'],
                    sources=[s['strategy_id'] for s in longs],
                    price=best['price'],
                    stop_loss=best.get('stop_loss'),
                    take_profit=best.get('take_profit')
                ))
            elif len(shorts) > len(longs) and len(shorts) >= self.min_consensus:
                best = max(shorts, key=lambda s: s['confidence'])
                aggregated.append(AggregatedSignal(
                    symbol=symbol,
                    side='short',
                    confidence=best['confidence'],
                    sources=[s['strategy_id'] for s in shorts],
                    price=best['price'],
                    stop_loss=best.get('stop_loss'),
                    take_profit=best.get('take_profit')
                ))
        
        return aggregated

class Executor:
    """Main executor that processes signals and places trades"""
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.PAPER):
        self.mode = mode
        self.processor = SignalProcessor()
        
        # Initialize execution engine
        self.kucoin = KuCoinFuturesExecutor()
        self.engine = ExecutionEngine(
            executor=self.kucoin,
            risk_config=RiskConfig(
                max_position_pct=2.0,
                max_total_exposure_pct=20.0,
                max_positions=5,
                max_daily_loss_pct=5.0,
                default_leverage=5
            ),
            mode=mode
        )
        
        self.running = False
        logger.info(f"Executor initialized in {mode.value} mode")
        logger.info(f"KuCoin configured: {self.kucoin.is_configured}")
    
    def start(self):
        """Start the executor loop"""
        self.running = True
        logger.info("🚀 Starting Cash Town Executor")
        logger.info(f"   Mode: {self.mode.value}")
        
        # Initial state refresh
        self.engine.refresh_state()
        logger.info(f"   Balance: ${self.engine.account_balance:,.2f}")
        
        while self.running:
            try:
                self._cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def stop(self):
        self.running = False
        logger.info("Executor stopped")
    
    def _cycle(self):
        """Process one execution cycle"""
        # Refresh state
        self.engine.refresh_state()
        
        if self.engine.killed:
            logger.warning(f"Kill switch active: {self.engine.kill_reason}")
            return
        
        # Get pending signals from orchestrator
        signals = self._fetch_signals()
        if not signals:
            return
        
        # Aggregate signals
        aggregated = self.processor.aggregate(signals)
        
        if not aggregated:
            return
        
        logger.info(f"Processing {len(aggregated)} aggregated signals")
        
        # Execute each signal
        for agg in aggregated:
            self._execute(agg)
    
    def _fetch_signals(self) -> List[Dict]:
        """Fetch pending signals from orchestrator"""
        try:
            resp = requests.get(f"{ORCHESTRATOR_URL}/signals", timeout=5)
            if resp.status_code == 200:
                return resp.json().get('signals', [])
        except Exception as e:
            logger.debug(f"Could not fetch signals: {e}")
        return []
    
    def _execute(self, signal: AggregatedSignal):
        """Execute an aggregated signal"""
        # Check if we can execute
        can_exec, reason = self.engine.can_execute()
        if not can_exec:
            logger.warning(f"Cannot execute {signal.symbol}: {reason}")
            return
        
        # Check if already have position
        if signal.symbol in self.engine.positions:
            logger.debug(f"Already have position in {signal.symbol}")
            return
        
        # Calculate position size
        price = signal.price
        position_value = self.engine.account_balance * self.engine.risk.max_position_pct / 100
        position_value *= signal.confidence
        contracts = int(position_value * self.engine.risk.default_leverage / price)
        
        if contracts <= 0:
            logger.warning(f"Position too small for {signal.symbol}")
            return
        
        # Execute
        side = 'buy' if signal.side == 'long' else 'sell'
        
        logger.info(f"{'🔥' if self.mode == ExecutionMode.LIVE else '📝'} "
                   f"Executing: {side.upper()} {contracts} {signal.symbol}")
        logger.info(f"   Price: ${price:,.2f}")
        logger.info(f"   Confidence: {signal.confidence:.0%}")
        logger.info(f"   Sources: {', '.join(signal.sources)}")
        
        if self.mode == ExecutionMode.LIVE:
            order_id = self.kucoin.place_market_order(
                symbol=signal.symbol,
                side=side,
                size=contracts,
                leverage=self.engine.risk.default_leverage
            )
            
            if order_id:
                logger.info(f"   ✅ Order placed: {order_id}")
                
                # Place stop loss
                if signal.stop_loss:
                    stop_side = 'sell' if side == 'buy' else 'buy'
                    self.kucoin.place_stop_order(
                        signal.symbol, stop_side, contracts, signal.stop_loss
                    )
                    logger.info(f"   🛡️ Stop loss set: ${signal.stop_loss:,.2f}")
            else:
                logger.error(f"   ❌ Order failed")
        else:
            logger.info(f"   📝 Paper trade recorded")
            self.engine.daily_stats.trades += 1

def main():
    if len(sys.argv) < 2:
        mode = 'paper'
    else:
        mode = sys.argv[1].lower()
    
    exec_mode = ExecutionMode.LIVE if mode == 'live' else ExecutionMode.PAPER
    
    if exec_mode == ExecutionMode.LIVE:
        print("🔥 LIVE TRADING MODE")
        print("⚠️  Real trades will be executed!")
        
        if os.environ.get('CASH_TOWN_AUTO_CONFIRM') != '1':
            response = input("Type 'YES' to confirm: ")
            if response != 'YES':
                print("Cancelled")
                sys.exit(0)
    
    executor = Executor(mode=exec_mode)
    
    try:
        executor.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        executor.stop()

if __name__ == '__main__':
    main()
