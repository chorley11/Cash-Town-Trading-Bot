#!/usr/bin/env python3
"""
Cash Town Cloud Runner v2 - With intelligent signal selection and learning

Key improvements over v1:
1. Uses SmartOrchestrator for intelligent signal selection
2. Stores ALL signals for learning (selected and rejected)
3. Tracks counterfactuals (what would have happened)
4. Learns from strategy performance over time
5. Clears signals after processing (no duplicate execution)

Usage:
    python run_cloud_v2.py              # Paper mode (default)
    python run_cloud_v2.py --live       # Live trading
"""
import sys
import os
import time
import logging
import threading
import signal as sig_module
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cash-town')

from orchestrator.smart_orchestrator import SmartOrchestrator
from orchestrator.signal_aggregator import AggregatorConfig
from agents.runner import AgentRunner
from agents.strategies import STRATEGY_REGISTRY

# Data directory for learning
DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Large caps only - matching Cucurbit's symbol universe
ALL_SYMBOLS = [
    'XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'XRPUSDTM', 'ADAUSDTM', 'LINKUSDTM',
    'DOTUSDTM', 'AVAXUSDTM', 'MATICUSDTM', 'ATOMUSDTM', 'UNIUSDTM', 'LTCUSDTM',
    'BCHUSDTM', 'NEARUSDTM', 'APTUSDTM', 'ARBUSDTM', 'OPUSDTM', 'FILUSDTM',
    'INJUSDTM', 'TIAUSDTM', 'RENDERUSDTM', 'SUIUSDTM', 'TONUSDTM', 'ICPUSDTM',
]

# Performance-ranked: trend-following is the star (+$208, 51% WR)
# FIXED: zweig re-enabled with major fixes (see agents/strategies/zweig.py)
AGENT_CONFIGS = [
    {'id': 'trend-following', 'symbols': ALL_SYMBOLS, 'interval': 300},  # STAR: 51% WR, +$208
    {'id': 'mean-reversion', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'turtle', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'weinstein', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'livermore', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'bts-lynch', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'zweig', 'symbols': ALL_SYMBOLS, 'interval': 300},  # FIXED: v2 with thrust detection
]

class CloudRunnerV2:
    """
    Improved cloud runner with intelligent signal selection.
    """
    
    def __init__(self, port: int = 8888, live_mode: bool = False):
        self.port = port
        self.live_mode = live_mode
        self.running = False
        
        # Smart orchestrator - learning-first, no arbitrary limits
        # The bot learns optimal behavior from P&L, not from my assumptions
        self.orchestrator = SmartOrchestrator(AggregatorConfig(
            min_confidence=0.55,
            min_consensus=1,
            max_signals_per_cycle=99,  # No limit - learn from results
            cooldown_minutes=0,         # No cooldown - learn when re-entry works
        ))
        
        self.agent_threads = []
        self.executor_thread = None
        self.http_thread = None
        
    def start(self):
        """Start all components"""
        self.running = True
        
        logger.info("=" * 60)
        logger.info("💰 CASH TOWN v2 - LEARNING-FIRST TRADING")
        logger.info("=" * 60)
        logger.info(f"Mode: {'🔴 LIVE' if self.live_mode else '📝 PAPER'}")
        logger.info(f"Port: {self.port}")
        logger.info(f"Philosophy: MAKE MONEY. Learn from P&L, not rules.")
        logger.info(f"Data dir: {DATA_DIR}")
        logger.info("=" * 60)
        
        # Start HTTP server for agents to report signals
        self._start_http_server()
        time.sleep(1)
        
        # Start strategy agents
        self._start_agents()
        
        # Start executor
        self._start_executor()
        
        # Main loop - just keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _start_http_server(self):
        """Start HTTP server for receiving signals"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        orchestrator = self.orchestrator
        
        class SignalHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
            
            def do_GET(self):
                if self.path == '/health':
                    self._respond(200, {'status': 'healthy'})
                elif self.path == '/signals':
                    # Return aggregated signals for executor
                    signals = orchestrator.get_actionable_signals()
                    self._respond(200, {
                        'count': len(signals),
                        'signals': [
                            {
                                'symbol': s.symbol,
                                'side': s.side.value,
                                'confidence': s.adjusted_confidence,
                                'price': s.signal.price,
                                'stop_loss': s.signal.stop_loss,
                                'take_profit': s.signal.take_profit,
                                'reason': s.signal.reason,
                                'strategy_id': s.signal.strategy_id,
                                'rank': s.rank,
                                'consensus': s.consensus_score,
                                'sources': s.sources
                            }
                            for s in signals
                        ]
                    })
                elif self.path == '/learning':
                    self._respond(200, orchestrator.get_learning_summary())
                elif self.path == '/multipliers':
                    # Endpoint for executor to get dynamic strategy multipliers
                    self._respond(200, orchestrator.get_strategy_multipliers())
                elif self.path == '/counterfactual':
                    # Analyze historical counterfactual data
                    self._respond(200, orchestrator.analyze_counterfactuals())
                elif self.path == '/rescue_stats':
                    # Second-chance rescue statistics
                    self._respond(200, orchestrator.rescue_stats)
                else:
                    self._respond(404, {'error': 'Not found'})
            
            def do_POST(self):
                if self.path == '/signals':
                    # Receive signal from agent
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        strategy_id = data.get('strategy_id', 'unknown')
                        orchestrator.receive_signal(strategy_id, data)
                        self._respond(200, {'status': 'received'})
                    except Exception as e:
                        self._respond(400, {'error': str(e)})
                elif self.path == '/trade_result':
                    # Record trade result for learning
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        orchestrator.record_trade_result(
                            symbol=data['symbol'],
                            side=data['side'],
                            pnl=data['pnl'],
                            pnl_pct=data['pnl_pct'],
                            strategy_id=data['strategy_id'],
                            reason=data.get('reason', '')
                        )
                        self._respond(200, {'status': 'recorded'})
                    except Exception as e:
                        self._respond(400, {'error': str(e)})
                else:
                    self._respond(404, {'error': 'Not found'})
            
            def _respond(self, code: int, data: dict):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
        
        def run_server():
            server = HTTPServer(('0.0.0.0', self.port), SignalHandler)
            logger.info(f"🌐 HTTP server started on port {self.port}")
            while self.running:
                server.handle_request()
        
        self.http_thread = threading.Thread(target=run_server, daemon=True)
        self.http_thread.start()
    
    def _start_agents(self):
        """Start strategy agents as background threads"""
        logger.info(f"🤖 Starting {len(AGENT_CONFIGS)} strategy agents...")
        
        for config in AGENT_CONFIGS:
            if config['id'] not in STRATEGY_REGISTRY:
                logger.warning(f"Strategy {config['id']} not found, skipping")
                continue
            
            # Set orchestrator URL for agents to report to
            os.environ['ORCHESTRATOR_URL'] = f"http://localhost:{self.port}"
            
            thread = threading.Thread(
                target=self._run_agent,
                args=(config,),
                daemon=True,
                name=f"agent-{config['id']}"
            )
            thread.start()
            self.agent_threads.append(thread)
            logger.info(f"  ✅ Started {config['id']}")
        
        logger.info(f"🤖 {len(self.agent_threads)} agents running")
    
    def _run_agent(self, config: dict):
        """Run a single agent"""
        try:
            runner = AgentRunner(
                strategy_id=config['id'],
                symbols=config.get('symbols'),
                interval_seconds=config.get('interval', 300)
            )
            runner.start()
        except Exception as e:
            logger.error(f"Agent {config['id']} crashed: {e}")
    
    def _start_executor(self):
        """Start the executor"""
        logger.info(f"⚡ Starting executor ({'LIVE' if self.live_mode else 'PAPER'})...")
        
        self.executor_thread = threading.Thread(
            target=self._run_executor,
            daemon=True,
            name="executor"
        )
        self.executor_thread.start()
        logger.info("  ✅ Executor running")
    
    def _run_executor(self):
        """Run the executor - processes ONLY aggregated signals"""
        try:
            from execution.kucoin import KuCoinFuturesExecutor
            from execution.engine import ExecutionEngine, ExecutionMode, RiskConfig
            
            mode = ExecutionMode.LIVE if self.live_mode else ExecutionMode.PAPER
            kucoin = KuCoinFuturesExecutor()
            
            risk_config = RiskConfig(
                max_position_pct=5.0,
                max_total_exposure_pct=30.0,
                max_daily_loss_pct=3.0,
                max_positions=5
            )
            
            engine = ExecutionEngine(
                executor=kucoin,
                mode=mode,
                risk_config=risk_config
            )
            
            logger.info(f"Executor initialized in {mode.value} mode")
            
            import requests
            ORCHESTRATOR_URL = f"http://localhost:{self.port}"
            
            last_signal_time = {}  # Track last signal time per symbol
            last_multiplier_update = 0  # Track when we last fetched multipliers
            MULTIPLIER_UPDATE_INTERVAL = 300  # Update multipliers every 5 minutes
            
            while self.running:
                try:
                    # Get AGGREGATED signals (already filtered/ranked)
                    resp = requests.get(f"{ORCHESTRATOR_URL}/signals", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        signals = data.get('signals', [])
                        
                        if signals:
                            logger.info(f"📊 Received {len(signals)} aggregated signals")
                            
                            for sig in signals:
                                symbol = sig['symbol']
                                
                                # Double-check we're not over-trading
                                now = datetime.utcnow()
                                if symbol in last_signal_time:
                                    elapsed = (now - last_signal_time[symbol]).total_seconds()
                                    if elapsed < 1800:  # 30 min minimum between same symbol
                                        logger.info(f"  ⏳ Skipping {symbol} - traded {elapsed/60:.0f}m ago")
                                        continue
                                
                                # Execute
                                self._execute_signal(engine, sig)
                                last_signal_time[symbol] = now
                    
                    # Refresh state and update positions
                    engine.refresh_state()
                    self.orchestrator.positions = {
                        p.symbol: 'long' if p.is_long else 'short'
                        for p in engine.positions.values()
                    }
                    
                    # DYNAMIC MULTIPLIERS: Periodically update strategy position multipliers
                    now_ts = time.time()
                    if now_ts - last_multiplier_update > MULTIPLIER_UPDATE_INTERVAL:
                        try:
                            mult_resp = requests.get(f"{ORCHESTRATOR_URL}/multipliers", timeout=5)
                            if mult_resp.status_code == 200:
                                multipliers = mult_resp.json()
                                engine.risk.strategy_boost_multipliers = multipliers
                                logger.info(f"📊 Updated strategy multipliers: {multipliers}")
                                last_multiplier_update = now_ts
                        except:
                            pass
                    
                except requests.exceptions.ConnectionError:
                    pass
                except Exception as e:
                    logger.error(f"Executor error: {e}")
                
                time.sleep(30)  # Check every 30 seconds (not 10)
                
        except Exception as e:
            logger.error(f"Executor crashed: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_signal(self, engine, signal: dict):
        """Execute a single aggregated signal"""
        from agents.base import Signal, SignalSide
        from orchestrator.signal_aggregator import AggregatedSignal
        
        try:
            sig = Signal(
                strategy_id=signal.get('strategy_id', 'unknown'),
                symbol=signal['symbol'],
                side=SignalSide(signal['side']),
                confidence=signal['confidence'],
                price=signal.get('price', 0),
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                reason=signal.get('reason', ''),
                timestamp=datetime.utcnow()
            )
            
            agg_sig = AggregatedSignal(
                signal=sig,
                rank=signal.get('rank', 1),
                sources=signal.get('sources', [signal.get('strategy_id', 'unknown')]),
                conflicts=[],
                consensus_score=signal.get('consensus', 1.0),
                adjusted_confidence=signal['confidence']
            )
            
            result = engine.execute_signal(agg_sig)
            if result and result.success:
                logger.info(f"✅ Executed: {sig.side.value} {sig.symbol} (rank #{signal.get('rank', '?')}, {len(signal.get('sources', []))} sources)")
            else:
                logger.info(f"⏭️ Skipped: {sig.symbol} - {result.message if result else 'no result'}")
            
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
    
    def stop(self):
        """Stop all components"""
        logger.info("Shutting down...")
        self.running = False
        logger.info("Goodbye! 👋")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cash Town v2 - Intelligent Trading')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8888)))
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    args = parser.parse_args()
    
    live_mode = args.live or os.environ.get('LIVE_MODE', '').lower() in ('true', '1', 'yes')
    
    runner = CloudRunnerV2(port=args.port, live_mode=live_mode)
    
    def shutdown(signum, frame):
        runner.stop()
        sys.exit(0)
    
    sig_module.signal(sig_module.SIGTERM, shutdown)
    sig_module.signal(sig_module.SIGINT, shutdown)
    
    runner.start()


if __name__ == '__main__':
    main()
