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

# SECURITY & PERFORMANCE: Import validation and monitoring
from utils.validation import validate_signal_data, validate_trade_result, redact_sensitive_data
from utils.monitoring import get_monitor, PerformanceMonitor

# BLOOMBERG DASHBOARD API
from api.endpoints import DashboardAPI
from api.websocket import LiveFeed

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
    # Original strategies
    {'id': 'trend-following', 'symbols': ALL_SYMBOLS, 'interval': 300},  # STAR: 51% WR, +$208
    {'id': 'mean-reversion', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'turtle', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'weinstein', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'livermore', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'bts-lynch', 'symbols': ALL_SYMBOLS, 'interval': 300},
    {'id': 'zweig', 'symbols': ALL_SYMBOLS, 'interval': 300},  # FIXED: v2 with thrust detection
    {'id': 'rsi-divergence', 'symbols': ALL_SYMBOLS, 'interval': 300},  # NEW: catches reversals early
    # New futures-specific strategies (Feb 2026)
    {'id': 'funding-fade', 'symbols': ALL_SYMBOLS, 'interval': 300, 'needs_futures_data': True},
    {'id': 'oi-divergence', 'symbols': ALL_SYMBOLS, 'interval': 300, 'needs_futures_data': True},
    {'id': 'liquidation-hunter', 'symbols': ALL_SYMBOLS, 'interval': 300, 'needs_futures_data': True},
    {'id': 'volatility-breakout', 'symbols': ALL_SYMBOLS, 'interval': 300},  # Uses OHLCV only
    {'id': 'correlation-pairs', 'symbols': ALL_SYMBOLS, 'interval': 300},  # Uses price data only
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
        
        # Bloomberg Dashboard API
        self.dashboard_api = DashboardAPI()
        self.live_feed = LiveFeed()
        
        # Execution engine reference (set in _start_executor)
        self.engine = None
        
        # Data feed reference (set in _start_data_feed)
        self.data_feed = None
        
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
        
        # Start watchdog background loop
        self._start_watchdog()
        
        # Main loop - just keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _start_watchdog(self):
        """Start the profit watchdog background loop"""
        logger.info("🐕 Starting Profit Watchdog...")
        
        def watchdog_loop():
            from orchestrator.profit_watchdog import run_watchdog_cycle
            from orchestrator.trailing_stops import get_trailing_manager
            
            trailing_manager = get_trailing_manager()
            if hasattr(self, 'engine') and self.engine:
                trailing_manager.executor = self.engine.executor
            
            while self.running:
                try:
                    # Get current prices for outcome tracking
                    prices = self._get_current_prices()
                    if prices:
                        run_watchdog_cycle(self.orchestrator.watchdog, prices)
                        
                        # Update trailing stops
                        adjustments = trailing_manager.update(prices)
                        for adj in adjustments:
                            logger.info(f"📈 Trailing stop adjusted: {adj['symbol']} -> ${adj['new_stop']:.2f}")
                except Exception as e:
                    logger.error(f"Watchdog cycle error: {e}")
                
                # Run every 5 minutes
                for _ in range(300):
                    if not self.running:
                        break
                    time.sleep(1)
        
        watchdog_thread = threading.Thread(
            target=watchdog_loop,
            daemon=True,
            name="watchdog"
        )
        watchdog_thread.start()
        logger.info("  ✅ Watchdog running")
    
    def _get_current_prices(self) -> dict:
        """Get current prices for watchdog outcome tracking"""
        try:
            from execution.kucoin import KuCoinFuturesExecutor
            executor = KuCoinFuturesExecutor()
            if not executor.is_configured:
                return {}
            
            prices = {}
            # Get prices for symbols we're tracking
            symbols = set()
            
            # From positions
            for symbol in self.orchestrator.positions:
                symbols.add(symbol)
            
            # From watchdog pending outcomes
            for record in self.orchestrator.watchdog.pending_outcomes:
                symbols.add(record.symbol)
            
            # Fetch prices
            for symbol in symbols:
                if not symbol:
                    continue
                try:
                    ticker = executor.client.get_ticker(symbol)
                    if ticker and 'price' in ticker:
                        prices[symbol] = float(ticker['price'])
                except:
                    pass
            
            return prices
        except Exception as e:
            logger.debug(f"Could not fetch prices: {e}")
            return {}
    
    def _start_http_server(self):
        """Start HTTP server for receiving signals + Bloomberg Dashboard API"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from urllib.parse import urlparse, parse_qs
        import json
        
        orchestrator = self.orchestrator
        dashboard_api = self.dashboard_api
        live_feed = self.live_feed
        runner_ref = self  # Reference for data feed
        
        class SignalHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
            
            def _parse_query(self):
                """Parse query parameters"""
                parsed = urlparse(self.path)
                return parse_qs(parsed.query)
            
            def _get_path(self):
                """Get path without query string"""
                return urlparse(self.path).path
            
            def do_GET(self):
                path = self._get_path()
                query = self._parse_query()
                
                # ===========================================
                # BLOOMBERG DASHBOARD API ENDPOINTS
                # ===========================================
                
                # GET /api/portfolio - Portfolio overview
                if path == '/api/portfolio':
                    self._respond(200, dashboard_api.get_portfolio())
                    return
                
                # GET /api/positions - All positions
                elif path == '/api/positions':
                    include_closed = query.get('include_closed', ['false'])[0].lower() == 'true'
                    self._respond(200, dashboard_api.get_positions(include_closed=include_closed))
                    return
                
                # GET /api/position/:symbol - Single position
                elif path.startswith('/api/position/'):
                    symbol = path.split('/')[-1]
                    self._respond(200, dashboard_api.get_position(symbol))
                    return
                
                # GET /api/trades - Trade history with filters
                elif path == '/api/trades':
                    self._respond(200, dashboard_api.get_trades(
                        strategy=query.get('strategy', [None])[0],
                        symbol=query.get('symbol', [None])[0],
                        side=query.get('side', [None])[0],
                        start_date=query.get('start_date', [None])[0],
                        end_date=query.get('end_date', [None])[0],
                        min_pnl=float(query.get('min_pnl', [0])[0]) if query.get('min_pnl') else None,
                        max_pnl=float(query.get('max_pnl', [0])[0]) if query.get('max_pnl') else None,
                        won_only=query.get('won_only', [None])[0] == 'true' if query.get('won_only') else None,
                        limit=int(query.get('limit', ['100'])[0]),
                        offset=int(query.get('offset', ['0'])[0])
                    ))
                    return
                
                # GET /api/trade/:id - Single trade
                elif path.startswith('/api/trade/'):
                    trade_id = path.split('/')[-1]
                    self._respond(200, dashboard_api.get_trade(trade_id))
                    return
                
                # GET /api/strategies - Strategy performance
                elif path == '/api/strategies':
                    self._respond(200, dashboard_api.get_strategies())
                    return
                
                # GET /api/strategy/:id - Single strategy detail
                elif path.startswith('/api/strategy/'):
                    strategy_id = path.split('/')[-1]
                    self._respond(200, dashboard_api.get_strategy(strategy_id))
                    return
                
                # GET /api/signals - Signal history (NOT internal /signals)
                elif path == '/api/signals':
                    accepted = None
                    if query.get('accepted'):
                        accepted = query['accepted'][0].lower() == 'true'
                    self._respond(200, dashboard_api.get_signals(
                        accepted=accepted,
                        strategy=query.get('strategy', [None])[0],
                        symbol=query.get('symbol', [None])[0],
                        limit=int(query.get('limit', ['100'])[0]),
                        offset=int(query.get('offset', ['0'])[0])
                    ))
                    return
                
                # GET /api/risk - Risk metrics
                elif path == '/api/risk':
                    self._respond(200, dashboard_api.get_risk())
                    return
                
                # GET /api/chart/:symbol - OHLCV chart data
                elif path.startswith('/api/chart/'):
                    symbol = path.split('/')[-1]
                    self._respond(200, dashboard_api.get_chart(
                        symbol=symbol,
                        interval=query.get('interval', ['15m'])[0],
                        limit=int(query.get('limit', ['200'])[0])
                    ))
                    return
                
                # GET /api/ws/events - Recent WebSocket events for catch-up
                elif path == '/api/ws/events':
                    limit = int(query.get('limit', ['50'])[0])
                    event_types = query.get('types', [None])[0]
                    types_list = event_types.split(',') if event_types else None
                    self._respond(200, {
                        'events': live_feed.get_recent_events(limit=limit, event_types=types_list),
                        'metadata': {'count': limit}
                    })
                    return
                
                # GET /api/ws/snapshot - Current state snapshot
                elif path == '/api/ws/snapshot':
                    self._respond(200, live_feed.get_snapshot())
                    return
                
                # GET /api/stream - Server-Sent Events for real-time updates
                elif path == '/api/stream':
                    self._stream_events()
                    return
                
                # GET /api/summary - Quick summary for dashboard header
                elif path == '/api/summary':
                    portfolio = dashboard_api.get_portfolio()['data']
                    self._respond(200, {
                        'equity': portfolio['equity']['total'],
                        'pnl': portfolio['pnl']['unrealized'],
                        'pnl_pct': portfolio['pnl']['unrealized_pct'],
                        'positions': portfolio['positions']['count'],
                        'exposure_pct': portfolio['exposure']['exposure_pct'],
                        'win_rate': portfolio['performance']['win_rate'],
                        'circuit_breaker': portfolio['risk']['circuit_breaker_active'],
                        'timestamp': portfolio['timestamp']
                    })
                    return
                
                # ===========================================
                # ORIGINAL INTERNAL ENDPOINTS
                # ===========================================
                
                if path == '/health':
                    self._respond(200, {'status': 'healthy', 'api_version': '2.0', 'dashboard': True})
                elif path == '/signals':
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
                                'sources': s.sources,
                                'risk_position_size': s.signal.metadata.get('risk_position_size'),
                                'risk_meta': s.signal.metadata.get('risk_meta', {})
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
                elif self.path == '/perf':
                    # PERFORMANCE: Monitoring stats endpoint
                    monitor = get_monitor()
                    self._respond(200, monitor.get_stats())
                elif self.path == '/risk':
                    # Risk manager status
                    self._respond(200, orchestrator.get_risk_status())
                elif self.path == '/can_trade':
                    # Check if trading is allowed (circuit breaker)
                    can, reason = orchestrator.can_trade()
                    self._respond(200, {'can_trade': can, 'reason': reason})
                elif self.path == '/watchdog':
                    # PROFIT WATCHDOG: Full analysis status
                    self._respond(200, orchestrator.watchdog.get_status())
                elif self.path == '/watchdog/decisions':
                    # Recent decisions with outcomes
                    self._respond(200, {'decisions': orchestrator.watchdog.get_recent_decisions(50)})
                elif self.path == '/watchdog/alerts':
                    # Active watchdog alerts
                    from dataclasses import asdict
                    alerts = orchestrator.watchdog.generate_alerts()
                    self._respond(200, {'alerts': [asdict(a) for a in alerts]})
                elif self.path == '/watchdog/recommendations':
                    # Parameter recommendations
                    from dataclasses import asdict
                    recs = orchestrator.watchdog.analyze_and_recommend()
                    self._respond(200, {'recommendations': [asdict(r) for r in recs]})
                elif self.path == '/watchdog/drift':
                    # Strategy drift analysis
                    from dataclasses import asdict
                    drifts = orchestrator.watchdog.detect_strategy_drift()
                    self._respond(200, {'strategy_drift': [asdict(d) for d in drifts]})
                else:
                    self._respond(404, {'error': 'Not found'})
            
            def do_POST(self):
                if self.path == '/signals':
                    # Receive signal from agent
                    content_length = int(self.headers.get('Content-Length', 0))
                    
                    # SECURITY: Limit request body size (prevent DoS)
                    if content_length > 50000:  # 50KB max
                        self._respond(413, {'error': 'Request too large'})
                        return
                    
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        
                        # SECURITY: Validate and sanitize input
                        is_valid, error, sanitized = validate_signal_data(data)
                        if not is_valid:
                            logger.warning(f"Invalid signal rejected: {error}")
                            self._respond(400, {'error': error})
                            return
                        
                        strategy_id = sanitized.get('strategy_id', 'unknown')
                        orchestrator.receive_signal(strategy_id, sanitized)
                        self._respond(200, {'status': 'received'})
                    except json.JSONDecodeError as e:
                        self._respond(400, {'error': f'Invalid JSON: {str(e)[:100]}'})
                    except Exception as e:
                        logger.error(f"Signal processing error: {e}")
                        self._respond(500, {'error': 'Internal error'})
                        
                elif self.path == '/trade_result':
                    # Record trade result for learning
                    content_length = int(self.headers.get('Content-Length', 0))
                    
                    # SECURITY: Limit request body size
                    if content_length > 10000:  # 10KB max
                        self._respond(413, {'error': 'Request too large'})
                        return
                    
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        
                        # SECURITY: Validate and sanitize input
                        is_valid, error, sanitized = validate_trade_result(data)
                        if not is_valid:
                            logger.warning(f"Invalid trade result rejected: {error}")
                            self._respond(400, {'error': error})
                            return
                        
                        orchestrator.record_trade_result(
                            symbol=sanitized['symbol'],
                            side=sanitized['side'],
                            pnl=sanitized['pnl'],
                            pnl_pct=sanitized['pnl_pct'],
                            strategy_id=sanitized['strategy_id'],
                            reason=sanitized.get('reason', '')
                        )
                        self._respond(200, {'status': 'recorded'})
                    except json.JSONDecodeError as e:
                        self._respond(400, {'error': f'Invalid JSON: {str(e)[:100]}'})
                    except Exception as e:
                        logger.error(f"Trade result processing error: {e}")
                        self._respond(500, {'error': 'Internal error'})
                        
                elif self.path == '/watchdog/tune':
                    # Trigger auto-tuning
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
                    try:
                        data = json.loads(body)
                        dry_run = data.get('dry_run', True)
                        changes = orchestrator.watchdog.auto_tune(dry_run=dry_run)
                        self._respond(200, {
                            'success': True,
                            'dry_run': dry_run,
                            'changes': changes
                        })
                    except Exception as e:
                        self._respond(500, {'error': str(e)})
                        
                elif self.path == '/watchdog/baseline':
                    # Update strategy baseline
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        orchestrator.watchdog.update_baseline(
                            strategy_id=data['strategy_id'],
                            win_rate=data['win_rate'],
                            trades=data['trades'],
                            pnl=data.get('pnl', 0)
                        )
                        self._respond(200, {'success': True})
                    except KeyError as e:
                        self._respond(400, {'error': f'Missing field: {e}'})
                    except Exception as e:
                        self._respond(500, {'error': str(e)})
                else:
                    self._respond(404, {'error': 'Not found'})
            
            def _respond(self, code: int, data: dict):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                # CORS headers for dashboard access
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def do_OPTIONS(self):
                """Handle CORS preflight requests"""
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
            
            def _stream_events(self):
                """Server-Sent Events endpoint for real-time updates"""
                from api.websocket import SSEClient
                import time as t
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                
                # Create SSE client and register with live feed
                client = SSEClient()
                live_feed.add_client(client)
                
                try:
                    # Send initial snapshot
                    snapshot = json.dumps(live_feed.get_snapshot())
                    self.wfile.write(f"event: snapshot\ndata: {snapshot}\n\n".encode())
                    self.wfile.flush()
                    
                    # Stream events
                    while runner_ref.running:
                        message = client.get(timeout=1.0)
                        if message:
                            self.wfile.write(f"event: update\ndata: {message}\n\n".encode())
                            self.wfile.flush()
                        else:
                            # Send keepalive
                            self.wfile.write(f": keepalive\n\n".encode())
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    client.disconnect()
                    live_feed.remove_client(client)
        
        def run_server():
            server = HTTPServer(('0.0.0.0', self.port), SignalHandler)
            logger.info(f"🌐 HTTP server started on port {self.port}")
            logger.info(f"📊 Bloomberg Dashboard API available at:")
            logger.info(f"   GET /api/portfolio    - Portfolio overview")
            logger.info(f"   GET /api/positions    - Open positions")
            logger.info(f"   GET /api/trades       - Trade history")
            logger.info(f"   GET /api/strategies   - Strategy performance")
            logger.info(f"   GET /api/signals      - Signal history")
            logger.info(f"   GET /api/risk         - Risk metrics")
            logger.info(f"   GET /api/chart/:sym   - OHLCV data")
            logger.info(f"   GET /api/summary      - Quick dashboard summary")
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
            
            # AGGRESSIVE BUT CONTROLLED - 50% max exposure, mandatory stops
            risk_config = RiskConfig(
                max_position_pct=25.0,  # Max 25% per position
                max_total_exposure_pct=50.0,  # Max 50% total exposure
                max_daily_loss_pct=10.0,  # 10% daily loss limit
                max_positions=50,  # Effectively unlimited
                default_leverage=10,
                default_stop_loss_pct=2.0,  # 2% stop loss required
                default_take_profit_pct=8.0
            )
            
            engine = ExecutionEngine(
                executor=kucoin,
                mode=mode,
                risk_config=risk_config
            )
            
            # Store engine reference for Dashboard API
            self.engine = engine
            
            # Connect Dashboard API components
            self.dashboard_api.set_orchestrator(self.orchestrator)
            self.dashboard_api.set_executor(engine)
            self.live_feed.set_orchestrator(self.orchestrator)
            self.live_feed.set_executor(engine)
            self.live_feed.start(update_interval=1.0)
            
            logger.info(f"Executor initialized in {mode.value} mode")
            logger.info(f"📊 Bloomberg Dashboard API connected")
            
            import requests
            ORCHESTRATOR_URL = f"http://localhost:{self.port}"
            
            last_signal_time = {}  # Track last signal time per symbol
            last_multiplier_update = 0  # Track when we last fetched multipliers
            MULTIPLIER_UPDATE_INTERVAL = 300  # Update multipliers every 5 minutes
            
            # PERFORMANCE: Get global monitor
            monitor = get_monitor()
            gc_interval = 0  # Counter for periodic GC
            
            while self.running:
                # PERFORMANCE: Start cycle tracking
                cycle_id = monitor.start_cycle()
                signals_executed = 0
                signals_received = 0
                
                try:
                    # PERFORMANCE: Time signal fetch stage
                    monitor.start_stage('fetch_signals')
                    resp = requests.get(f"{ORCHESTRATOR_URL}/signals", timeout=5)
                    monitor.end_stage('fetch_signals')
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        signals = data.get('signals', [])
                        signals_received = len(signals)
                        
                        if signals:
                            logger.info(f"📊 Received {len(signals)} aggregated signals")
                            
                            # PERFORMANCE: Time execution stage
                            monitor.start_stage('execute_signals')
                            for sig in signals:
                                symbol = sig['symbol']
                                
                                # Double-check we're not over-trading - REDUCED for full blooded mode
                                now = datetime.utcnow()
                                if symbol in last_signal_time:
                                    elapsed = (now - last_signal_time[symbol]).total_seconds()
                                    if elapsed < 300:  # 5 min minimum between same symbol (was 30 min)
                                        logger.info(f"  ⏳ Skipping {symbol} - traded {elapsed/60:.0f}m ago")
                                        continue
                                
                                # Execute
                                self._execute_signal(engine, sig)
                                signals_executed += 1
                                last_signal_time[symbol] = now
                            monitor.end_stage('execute_signals')
                    
                    # PERFORMANCE: Time state refresh stage
                    monitor.start_stage('refresh_state')
                    engine.refresh_state()
                    self.orchestrator.positions = {
                        p.symbol: 'long' if p.is_long else 'short'
                        for p in engine.positions.values()
                    }
                    
                    # Update risk manager with current equity
                    self.orchestrator.update_equity(engine.account_balance)
                    
                    # Check if trading is allowed (circuit breaker)
                    can_trade, reason = self.orchestrator.can_trade()
                    if not can_trade:
                        logger.warning(f"🛑 Trading halted: {reason}")
                    
                    monitor.end_stage('refresh_state')
                    
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
                    
                    # Record signal metrics
                    monitor.record_signals(generated=signals_received, executed=signals_executed)
                    
                except requests.exceptions.ConnectionError:
                    pass
                except Exception as e:
                    logger.error(f"Executor error: {e}")
                    monitor.record_error(str(e))
                
                # PERFORMANCE: End cycle and get metrics
                monitor.end_cycle()
                
                # PERFORMANCE: Periodic garbage collection (every 20 cycles)
                gc_interval += 1
                if gc_interval >= 20:
                    monitor.force_gc()
                    gc_interval = 0
                
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
                
                # Register with trailing stop manager
                from orchestrator.trailing_stops import get_trailing_manager
                trailing_manager = get_trailing_manager()
                stop_loss = sig.stop_loss or (result.price * (0.98 if sig.side == SignalSide.LONG else 1.02))
                trailing_manager.register_position(
                    symbol=sig.symbol,
                    side=sig.side.value,
                    entry_price=result.price,
                    initial_stop=stop_loss
                )
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
