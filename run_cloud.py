#!/usr/bin/env python3
"""
Cash Town Cloud Runner - All-in-one deployment for Railway/cloud

Runs everything in a single process:
1. Orchestrator HTTP API (main thread)
2. Strategy agents (background threads)
3. Executor (background thread)

Usage:
    python run_cloud.py              # Paper mode (default)
    python run_cloud.py --live       # Live trading
    python run_cloud.py --port 8080  # Custom port
"""
import sys
import os
import time
import logging
import threading
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cash-town')

# Import components
from orchestrator.server import Orchestrator, run_server_with_orchestrator
from agents.runner import AgentRunner
from agents.strategies import STRATEGY_REGISTRY

# Agent configurations
# Top 175 pairs by 24h USD turnover on KuCoin Futures
ALL_SYMBOLS = [
    'XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'XRPUSDTM', 'RIVERUSDTM', 'ADAUSDTM',
    'PEPEUSDTM', 'PIPPINUSDTM', 'MUSDTM', 'LINKUSDTM', 'DOGEUSDTM', 'HYPEUSDTM',
    'ZROUSDTM', 'ZECUSDTM', 'XMRUSDTM', 'SUIUSDTM', 'BEATUSDTM', 'BCHUSDTM',
    'BNBUSDTM', 'LTCUSDTM', 'XAGUSDTM', 'ZKPUSDTM', 'WIFUSDTM', 'IDOLUSDTM',
    'TAOUSDTM', 'ASTERUSDTM', 'POWERUSDTM', 'EDENUSDTM', 'ICNTUSDTM', 'AXSUSDTM',
    'GIGGLEUSDTM', '2ZUSDTM', 'WHITEWHALEUSDTM', 'ESPORTSUSDTM', 'NOMUSDTM',
    'ZORAUSDTM', 'GWEIUSDTM', 'VANAUSDTM', 'RNBWUSDTM', 'SHIBUSDTM', 'ENSOUSDTM',
    'ONDOUSDTM', 'BERAUSDTM', 'AINUSDTM', 'UBUSDTM', 'ENAUSDTM', 'LYNUSDTM',
    'CROSSUSDTM', 'TRUTHUSDTM', 'TACUSDTM', 'PUMPUSDTM', 'NKNUSDTM', 'KAITOUSDTM',
    'VELVETUSDTM', 'FARTCOINUSDTM', 'FFUSDTM', 'XLMUSDTM', 'WLDUSDTM', 'USELESSUSDTM',
    'PTBUSDTM', 'AAVEUSDTM', 'XPINUSDTM', 'HBARUSDTM', 'KGENUSDTM', 'MYXUSDTM',
    'NEARUSDTM', 'FHEUSDTM', 'WOTAMALAILEUSDTM', 'CUSDTM', 'MONUSDTM', 'SENTUSDTM',
    'DOTUSDTM', 'ALCHUSDTM', 'XAUTUSDTM', 'LABUSDTM', 'TAKEUSDTM', 'TONUSDTM',
    'SIRENUSDTM', 'XPTUSDTM', 'JASMYUSDTM', 'XANUSDTM', 'BTRUSDTM', '4USDTM',
    'DUSKUSDTM', 'WLFIUSDTM', 'VFYUSDTM', 'ZKJUSDTM', 'PAXGUSDTM', 'XPDUSDTM',
    'ARCUSDTM', 'ICPUSDTM', 'BIRBUSDTM', 'FILUSDTM', 'WILDUSDTM', 'DYDXUSDTM',
    'ZAMAUSDTM', 'AVAXUSDTM', 'NPCUSDTM', 'ESUSDTM', 'KITEUSDTM', 'ETCUSDTM',
    'XNYUSDTM', 'FLOKIUSDTM', 'SOMIUSDTM', 'UNIUSDTM', 'JUPUSDTM', 'SKYUSDTM',
    'ZILUSDTM', 'METUSDTM', 'TRIAUSDTM', 'AVNTUSDTM', 'QUSDTM', 'UUSDTM',
    'STABLEUSDTM', 'TIAUSDTM', 'TRUMPUSDTM', 'APTUSDTM', 'STGUSDTM', 'ARBUSDTM',
    'LUNCUSDTM', 'MITOUSDTM', 'RENDERUSDTM', 'ARIAUSDTM', 'CLOUSDTM', 'YFIUSDTM',
    'ROSEUSDTM', 'YZYUSDTM', 'XPLUSDTM', 'NIGHTUSDTM', 'CLANKERUSDTM', 'BINANCELIFEUSDTM',
    'ZBTUSDTM', 'FETUSDTM', 'TAUSDTM', 'TURTLEUSDTM', 'DAMUSDTM', 'ENSUSDTM',
    'RECALLUSDTM', 'COLLECTUSDTM', '1000RATSUSDTM', 'LDOUSDTM', 'HANAUSDTM',
    'RAYUSDTM', 'SKRUSDTM', 'LIGHTUSDTM', 'SPACEUSDTM', 'TRXUSDTM', 'GPSUSDTM',
    'DOLOUSDTM', 'NXPCUSDTM', 'CRVUSDTM', 'SAPIENUSDTM', 'DASHUSDTM', 'MOODENGUSDTM',
    'CYSUSDTM', 'HMSTRUSDTM', 'FOLKSUSDTM', 'XTZUSDTM', 'APRUSDTM', 'LITUSDTM',
    'SEIUSDTM', 'AWEUSDTM', 'FIGHTUSDTM', '1000BONKUSDTM', 'ATUSDTM', 'BOBBOBUSDTM',
    'IPUSDTM', 'VIRTUALUSDTM', 'UAIUSDTM', 'ETHFIUSDTM', 'MIRAUSDTM', 'CCUSDTM',
    'ELSAUSDTM', 'ALLOUSDTM', 'CAKEUSDTM'
]

AGENT_CONFIGS = [
    {'id': 'trend-following', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'mean-reversion', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'turtle', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'weinstein', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'livermore', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'bts-lynch', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'zweig', 'symbols': ALL_SYMBOLS, 'interval': 300},
]

class CloudRunner:
    """Runs all Cash Town components in a single process"""
    
    def __init__(self, port: int = 8888, live_mode: bool = False):
        self.port = port
        self.live_mode = live_mode
        self.running = False
        self.orchestrator = None
        self.agent_threads = []
        self.executor_thread = None
        
    def start(self):
        """Start all components"""
        self.running = True
        logger.info("=" * 60)
        logger.info("💰 CASH TOWN CLOUD RUNNER")
        logger.info("=" * 60)
        logger.info(f"Mode: {'🔴 LIVE' if self.live_mode else '📝 PAPER'}")
        logger.info(f"Port: {self.port}")
        logger.info("=" * 60)
        
        # Start orchestrator (creates the HTTP server)
        self.orchestrator = Orchestrator()
        self.orchestrator.start()
        
        # Give orchestrator a moment to initialize
        time.sleep(1)
        
        # Start strategy agents in background threads
        self._start_agents()
        
        # Start executor in background thread
        self._start_executor()
        
        # Run HTTP server (blocks)
        logger.info(f"🌐 Starting HTTP server on port {self.port}")
        try:
            run_server_with_orchestrator(self.orchestrator, '0.0.0.0', self.port)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _start_agents(self):
        """Start strategy agents as background threads"""
        logger.info(f"🤖 Starting {len(AGENT_CONFIGS)} strategy agents...")
        
        for config in AGENT_CONFIGS:
            if config['id'] not in STRATEGY_REGISTRY:
                logger.warning(f"Strategy {config['id']} not found, skipping")
                continue
                
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
        """Run a single agent (called in thread)"""
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
        """Start the executor in a background thread"""
        logger.info(f"⚡ Starting executor ({'LIVE' if self.live_mode else 'PAPER'})...")
        
        self.executor_thread = threading.Thread(
            target=self._run_executor,
            daemon=True,
            name="executor"
        )
        self.executor_thread.start()
        logger.info("  ✅ Executor running")
    
    def _run_executor(self):
        """Run the executor (called in thread)"""
        try:
            from execution.kucoin import KuCoinFuturesExecutor
            from execution.engine import ExecutionEngine, ExecutionMode, RiskConfig
            
            # Initialize
            mode = ExecutionMode.LIVE if self.live_mode else ExecutionMode.PAPER
            kucoin = KuCoinFuturesExecutor()
            
            risk_config = RiskConfig(
                max_position_pct=5.0,
                max_total_exposure_pct=30.0,
                max_daily_loss_pct=3.0
            )
            
            engine = ExecutionEngine(
                executor=kucoin,
                mode=mode,
                risk_config=risk_config
            )
            
            logger.info(f"Executor initialized in {mode.value} mode")
            
            # Main executor loop
            import requests
            ORCHESTRATOR_URL = f"http://localhost:{self.port}"
            
            while self.running:
                try:
                    # Poll for signals
                    resp = requests.get(f"{ORCHESTRATOR_URL}/signals", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        signals = data.get('signals', [])
                        
                        if signals:
                            logger.info(f"📊 Processing {len(signals)} signals")
                            for sig in signals:
                                self._process_signal(engine, sig)
                    
                    # Refresh state
                    engine.refresh_state()
                    
                except requests.exceptions.ConnectionError:
                    pass  # Orchestrator not ready yet
                except Exception as e:
                    logger.error(f"Executor error: {e}")
                
                time.sleep(10)  # Poll every 10 seconds
                
        except Exception as e:
            logger.error(f"Executor crashed: {e}")
    
    def _process_signal(self, engine, signal: dict):
        """Process a single signal"""
        from agents.base import Signal, SignalSide
        
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
                timestamp=datetime.fromisoformat(signal['timestamp']) if signal.get('timestamp') else datetime.utcnow()
            )
            
            result = engine.execute_signal(sig)
            if result:
                logger.info(f"✅ Executed: {sig.side.value} {sig.symbol}")
            
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
    
    def stop(self):
        """Stop all components"""
        logger.info("Shutting down...")
        self.running = False
        if self.orchestrator:
            self.orchestrator.stop()
        logger.info("Goodbye! 👋")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cash Town Cloud Runner')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8888)))
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    args = parser.parse_args()
    
    # Check for LIVE_MODE environment variable
    live_mode = args.live or os.environ.get('LIVE_MODE', '').lower() in ('true', '1', 'yes')
    
    runner = CloudRunner(port=args.port, live_mode=live_mode)
    
    # Handle shutdown signals
    def shutdown(signum, frame):
        runner.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    runner.start()


if __name__ == '__main__':
    main()
