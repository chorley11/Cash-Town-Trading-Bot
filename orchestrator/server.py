"""
Cash Town Orchestrator - Main server
Provides HTTP API for managing the trading swarm with active position management.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from dataclasses import asdict

from .registry import AgentRegistry, AgentConfig
from .health import HealthMonitor
from .position_manager import PositionManager, RotationConfig, TrackedPosition, PositionState
from .profit_watchdog import ProfitWatchdog, run_watchdog_cycle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Orchestrator:
    """Main orchestrator coordinating all agents with position rotation"""
    
    def __init__(self, rotation_config: RotationConfig = None):
        self.registry = AgentRegistry()
        self.health_monitor = HealthMonitor(self.registry)
        self.position_manager = PositionManager(rotation_config or RotationConfig())
        self.running = False
        self._health_thread = None
        self._rotation_thread = None
        self._watchdog_thread = None
        
        # Pending signals from agents waiting for capital
        self.pending_signals = []
        
        # PROFIT WATCHDOG: The feedback loop that maximizes profits
        self.watchdog = ProfitWatchdog(orchestrator=self)
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("Orchestrator started")
        
        # Initial health check
        self.health_monitor.check_all()
        
        # Start periodic health checks in background thread
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()
        
        # Start position rotation loop
        self._rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self._rotation_thread.start()
        
        # Start watchdog loop (runs every 5 minutes)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        if self._health_thread:
            self._health_thread.join(timeout=5)
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)
        logger.info("Orchestrator stopped")
    
    def _health_loop(self):
        """Periodic health check loop"""
        while self.running:
            try:
                self.health_monitor.check_all()
                
                # Update position tracking from agent statuses
                self._sync_positions_from_agents()
                
                # Check for critical alerts
                alerts = self.health_monitor.get_alerts()
                for alert in alerts:
                    if alert.severity == 'critical':
                        logger.critical(f"ALERT: {alert.message}")
                    elif alert.severity == 'error':
                        logger.error(f"ALERT: {alert.message}")
                    elif alert.severity == 'warning':
                        logger.warning(f"ALERT: {alert.message}")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            # Sleep for 60 seconds, but check running flag every second
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)
    
    def _rotation_loop(self):
        """Position rotation evaluation loop - runs every 30 seconds"""
        while self.running:
            try:
                self._evaluate_rotations()
            except Exception as e:
                logger.error(f"Rotation evaluation error: {e}")
            
            # Check every 30 seconds
            for _ in range(30):
                if not self.running:
                    break
                time.sleep(1)
    
    def _watchdog_loop(self):
        """Profit watchdog loop - runs every 5 minutes"""
        while self.running:
            try:
                # Get current prices for outcome tracking
                current_prices = self._get_current_prices()
                if current_prices:
                    run_watchdog_cycle(self.watchdog, current_prices)
            except Exception as e:
                logger.error(f"Watchdog cycle error: {e}")
            
            # Run every 5 minutes
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)
    
    def _get_current_prices(self) -> dict:
        """Get current prices for all tracked symbols"""
        try:
            from execution.kucoin import KuCoinFuturesExecutor
            executor = KuCoinFuturesExecutor()
            if not executor.is_configured:
                return {}
            
            prices = {}
            # Get prices for positions and pending signals
            symbols = set()
            
            # From positions
            positions = executor.get_positions()
            for pos in positions:
                symbols.add(pos.symbol)
            
            # From pending signals
            for sig in self.pending_signals:
                if isinstance(sig, dict):
                    symbols.add(sig.get('symbol', ''))
                else:
                    symbols.add(getattr(sig, 'symbol', ''))
            
            # From watchdog pending outcomes
            for record in self.watchdog.pending_outcomes:
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
    
    def _sync_positions_from_agents(self):
        """Sync position data from agent health statuses"""
        for agent_id, status in self.health_monitor.get_all_statuses().items():
            if not status.healthy or not hasattr(status, 'positions'):
                continue
            
            positions = getattr(status, 'positions', [])
            if not isinstance(positions, list):
                continue
            
            for pos_data in positions:
                try:
                    position = TrackedPosition(
                        agent_id=agent_id,
                        symbol=pos_data.get('symbol', ''),
                        side=pos_data.get('side', 'long'),
                        entry_price=float(pos_data.get('entry_price', 0)),
                        entry_time=datetime.fromisoformat(pos_data.get('entry_time', datetime.utcnow().isoformat())),
                        size=float(pos_data.get('size', 0)),
                        current_price=float(pos_data.get('current_price', pos_data.get('entry_price', 0))),
                        current_pnl=float(pos_data.get('pnl', 0)),
                        current_pnl_pct=float(pos_data.get('pnl_percent', 0)),
                        signal_confidence=float(pos_data.get('signal_confidence', 0.5))
                    )
                    self.position_manager.track_position(position)
                except Exception as e:
                    logger.debug(f"Could not parse position data: {e}")
    
    def _evaluate_rotations(self):
        """Evaluate positions and execute rotations"""
        decisions = self.position_manager.evaluate_rotations(self.pending_signals)
        
        if not decisions:
            return
        
        for decision in decisions:
            if decision.urgency == 'immediate':
                logger.warning(f"🔄 ROTATION NEEDED: {decision.position.symbol} - {decision.reason}")
                self._execute_rotation(decision)
            elif decision.urgency == 'soon':
                logger.info(f"🔄 Rotation suggested: {decision.position.symbol} - {decision.reason}")
                # For 'soon' urgency, wait for next cycle but log
            # 'optional' rotations are only logged at debug level
            else:
                logger.debug(f"Optional rotation: {decision.position.symbol} - {decision.reason}")
    
    def _execute_rotation(self, decision):
        """Execute a position rotation - close and optionally replace"""
        position = decision.position
        
        # Get the agent
        agent = self.registry.get(position.agent_id)
        if not agent:
            logger.error(f"Cannot rotate - agent not found: {position.agent_id}")
            return
        
        # Build close command for agent
        # This would send to the agent's API endpoint
        close_command = {
            'action': 'close_position',
            'symbol': position.symbol,
            'reason': f"Rotation: {decision.reason}",
            'side': position.side
        }
        
        try:
            # TODO: Actually send to agent API
            # For now, log the intent
            logger.info(f"📤 Would send to {agent.endpoint}: {json.dumps(close_command)}")
            
            # Remove from tracking
            self.position_manager.remove_position(position.agent_id, position.symbol)
            
            # If we have a replacement signal, queue it
            if decision.replacement_signal:
                logger.info(f"📥 Replacement signal queued: {decision.replacement_signal.symbol} "
                           f"({decision.replacement_signal.side}) confidence={decision.replacement_signal.confidence:.0%}")
            
        except Exception as e:
            logger.error(f"Failed to execute rotation: {e}")
    
    def add_pending_signal(self, signal):
        """Add a signal to pending queue (waiting for capital from rotation)"""
        self.pending_signals.append(signal)
        # Keep only recent signals
        self.pending_signals = self.pending_signals[-50:]
    
    def get_summary(self) -> dict:
        """Get full system summary"""
        return {
            'orchestrator': {
                'running': self.running,
                'timestamp': datetime.utcnow().isoformat()
            },
            'agents': [asdict(a) for a in self.registry.list_all()],
            'statuses': {k: asdict(v) for k, v in self.health_monitor.get_all_statuses().items()},
            'portfolio': self.health_monitor.get_portfolio_summary(),
            'positions': self.position_manager.get_status(),
            'alerts': [asdict(a) for a in self.health_monitor.get_alerts()[-10:]]
        }

# Global orchestrator instance
_orchestrator: Orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator

# Security configuration
ALLOWED_ORIGINS = [
    'https://cash-town-trading-bot.vercel.app',
    'https://cash-town-trading-bot-tal9.vercel.app',
    'http://localhost:3000',
    'http://localhost:8080',
    'http://127.0.0.1:8080',
]

# Public endpoints that don't require auth (health checks only)
PUBLIC_ENDPOINTS = ['/health', '/', '/dashboard']

class OrchestratorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for orchestrator API"""
    
    def _get_cors_origin(self) -> str:
        """Return allowed origin or empty string"""
        origin = self.headers.get('Origin', '')
        if origin in ALLOWED_ORIGINS:
            return origin
        # Allow localhost variations for development
        if origin.startswith('http://localhost:') or origin.startswith('http://127.0.0.1:'):
            return origin
        return ''
    
    def _check_auth(self) -> bool:
        """Check API key authentication"""
        # Always allow localhost/internal requests (for executor thread)
        client_ip = self.client_address[0] if self.client_address else ''
        if client_ip in ('127.0.0.1', 'localhost', '::1'):
            return True
        
        # Get expected API key from environment
        expected_key = os.environ.get('CASH_TOWN_API_KEY', '')
        if not expected_key:
            # No key configured - allow all (backwards compat)
            return True
        
        # Check Authorization header
        auth_header = self.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return token == expected_key
        
        # Also check X-API-Key header
        api_key = self.headers.get('X-API-Key', '')
        return api_key == expected_key
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        cors_origin = self._get_cors_origin()
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Access-Control-Allow-Credentials', 'true')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())
    
    def _send_error(self, message: str, status: int = 400):
        self._send_json({'error': message}, status)
    
    def _send_unauthorized(self):
        self._send_json({'error': 'Unauthorized - API key required'}, 401)
    
    def _read_body(self) -> dict:
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode())
    
    def _read_body_raw(self) -> str:
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return ''
        return self.rfile.read(content_length).decode()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        cors_origin = self._get_cors_origin()
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.end_headers()
    
    def do_GET(self):
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        # Health endpoint is public
        if path == '/health':
            self._send_json({
                'status': 'healthy' if orch.running else 'stopped',
                'timestamp': datetime.utcnow().isoformat()
            })
            return
        
        # All other endpoints require authentication
        # Dashboard paths are public (served as static HTML)
        is_public = path in PUBLIC_ENDPOINTS or path.startswith('/dashboard')
        if not is_public and not self._check_auth():
            self._send_unauthorized()
            return
        
        if path == '/mode':
            # Show trading mode (sanitized - no sensitive info)
            live_mode = os.environ.get('LIVE_MODE', '').lower() in ('true', '1', 'yes')
            self._send_json({
                'mode': 'LIVE' if live_mode else 'PAPER',
                'live_mode': live_mode,
                'kucoin_configured': bool(os.environ.get('KUCOIN_API_KEY')),
                'cucurbit_configured': bool(os.environ.get('CUCURBIT_API_KEY')),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        elif path == '/balance':
            # Check Cash Town's own KuCoin account balance
            try:
                from execution.kucoin import KuCoinFuturesExecutor
                executor = KuCoinFuturesExecutor()
                if executor.is_configured:
                    balance = executor.get_account_overview()
                    self._send_json({
                        'source': 'cash-town-kucoin',
                        'balance': balance,
                        'configured': True
                    })
                else:
                    self._send_json({'error': 'KuCoin not configured', 'configured': False})
            except Exception as e:
                self._send_json({'error': str(e)})
        
        elif path == '/summary':
            self._send_json(orch.get_summary())
        
        elif path == '/agents':
            agents = [asdict(a) for a in orch.registry.list_all()]
            self._send_json({'agents': agents})
        
        elif path.startswith('/agents/'):
            agent_id = path.split('/')[2]
            agent = orch.registry.get(agent_id)
            if agent:
                status = orch.health_monitor.get_status(agent_id)
                self._send_json({
                    'agent': asdict(agent),
                    'status': asdict(status) if status else None
                })
            else:
                self._send_error('Agent not found', 404)
        
        elif path == '/portfolio':
            self._send_json(orch.health_monitor.get_portfolio_summary())
        
        elif path == '/positions':
            self._send_json(orch.position_manager.get_status())
        
        elif path == '/positions/live':
            # Fetch positions directly from KuCoin with strategy attribution
            try:
                from execution.kucoin import KuCoinFuturesExecutor
                from execution.strategy_tracker import tracker
                executor = KuCoinFuturesExecutor()
                positions = executor.get_positions()
                account = executor.get_account_overview()
                self._send_json({
                    'positions': [
                        {
                            'symbol': p.symbol,
                            'side': p.side,
                            'size': p.size,
                            'entry_price': p.entry_price,
                            'unrealized_pnl': p.unrealized_pnl,
                            'margin': p.margin,
                            'leverage': p.leverage,
                            'strategy_id': tracker.get_strategy(p.symbol) or 'unknown'
                        } for p in positions
                    ],
                    'account': account,
                    'count': len(positions)
                })
            except Exception as e:
                self._send_json({'error': str(e), 'positions': []})
        
        elif path == '/strategies':
            # Get strategy performance stats
            try:
                from execution.strategy_tracker import tracker
                positions = tracker.get_all_positions()
                stats = tracker.get_strategy_stats()
                self._send_json({
                    'positions': {k: {
                        'symbol': v.symbol,
                        'strategy_id': v.strategy_id,
                        'side': v.side,
                        'entry_price': v.entry_price,
                        'size': v.size,
                        'entry_time': v.entry_time
                    } for k, v in positions.items()},
                    'stats': stats
                })
            except Exception as e:
                self._send_json({'error': str(e), 'positions': {}, 'stats': {}})
        
        elif path == '/pyramid':
            # Get pyramiding status - full visibility into position scaling
            try:
                from orchestrator.risk_manager import RiskManager, RiskConfig
                from execution.kucoin import KuCoinFuturesExecutor
                from execution.strategy_tracker import tracker
                
                # Get or create risk manager
                if hasattr(orch, 'risk_manager'):
                    rm = orch.risk_manager
                else:
                    executor = KuCoinFuturesExecutor()
                    balance = float(executor.get_account_overview().get('availableBalance', 0)) if executor.is_configured else 1000
                    rm = RiskManager(equity=balance)
                
                pyramid_status = rm.get_pyramid_status()
                
                # Enrich with live position data
                if executor.is_configured:
                    positions = executor.get_positions()
                    for pos in positions:
                        if pos.symbol in pyramid_status['positions']:
                            pstate = pyramid_status['positions'][pos.symbol]
                            pstate['live_pnl'] = pos.unrealized_pnl
                            pstate['live_side'] = pos.side
                            pstate['entry_price'] = pos.entry_price
                            
                            # Calculate ROE
                            if pos.entry_price > 0 and pos.size > 0:
                                roe = (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100 * pos.leverage
                                pstate['current_roe_pct'] = round(roe, 2)
                                
                                # Check pyramid opportunity
                                can_pyramid, details = rm.check_pyramid_opportunity(
                                    pos.symbol,
                                    pos.entry_price + (pos.unrealized_pnl / pos.size),  # Estimated current price
                                    pos.entry_price,
                                    pos.side
                                )
                                pstate['pyramid_ready'] = can_pyramid
                                pstate['pyramid_details'] = details
                
                self._send_json({
                    'success': True,
                    'timestamp': datetime.utcnow().isoformat(),
                    'pyramid_status': pyramid_status,
                    'config': {
                        'enabled': rm.config.pyramid_config.enabled,
                        'max_levels': rm.config.pyramid_config.max_pyramid_levels,
                        'level_2_threshold': rm.config.pyramid_config.level_2_roe_threshold,
                        'level_3_threshold': rm.config.pyramid_config.level_3_roe_threshold,
                        'leverage_bump': rm.config.pyramid_config.leverage_bump_per_level,
                        'max_leverage': rm.config.leverage_config.absolute_max_leverage
                    }
                })
            except Exception as e:
                logger.error(f"Error getting pyramid status: {e}")
                self._send_json({'error': str(e)})
        
        elif path == '/leverage':
            # Get leverage configuration and current usage
            try:
                from orchestrator.risk_manager import RiskManager, RiskConfig
                
                if hasattr(orch, 'risk_manager'):
                    rm = orch.risk_manager
                else:
                    rm = RiskManager(equity=1000)
                
                lev_config = rm.config.leverage_config
                
                # Get strategy bonuses
                strategy_bonuses = {}
                for strategy_id, stats in rm.strategy_stats.items():
                    bonus = rm._calculate_strategy_leverage_bonus(strategy_id)
                    trades = stats.get('trades', 0)
                    wins = stats.get('wins', 0)
                    strategy_bonuses[strategy_id] = {
                        'leverage_bonus': bonus,
                        'trades': trades,
                        'win_rate': (wins / trades * 100) if trades > 0 else 0
                    }
                
                # Current leverage by position
                position_leverage = {}
                for symbol, pstate in rm.pyramid_states.items():
                    position_leverage[symbol] = {
                        'base_leverage': pstate.base_leverage,
                        'current_leverage': pstate.current_leverage,
                        'pyramid_level': pstate.current_level
                    }
                
                self._send_json({
                    'success': True,
                    'timestamp': datetime.utcnow().isoformat(),
                    'config': {
                        'tiers': {
                            'low': {
                                'confidence_range': lev_config.low_confidence_range,
                                'leverage_range': lev_config.low_confidence_leverage
                            },
                            'medium': {
                                'confidence_range': lev_config.medium_confidence_range,
                                'leverage_range': lev_config.medium_confidence_leverage
                            },
                            'high': {
                                'confidence_range': lev_config.high_confidence_range,
                                'leverage_range': lev_config.high_confidence_leverage
                            }
                        },
                        'absolute_max': lev_config.absolute_max_leverage,
                        'absolute_min': lev_config.min_leverage,
                        'win_rate_bonus_threshold': lev_config.win_rate_bonus_threshold,
                        'max_strategy_bonus': lev_config.max_strategy_bonus
                    },
                    'strategy_bonuses': strategy_bonuses,
                    'position_leverage': position_leverage,
                    'circuit_breaker_active': rm.circuit_breaker.is_triggered
                })
            except Exception as e:
                logger.error(f"Error getting leverage info: {e}")
                self._send_json({'error': str(e)})
        
        elif path == '/positions/summary':
            self._send_json({'summary': orch.position_manager.summary()})
        
        elif path == '/alerts':
            alerts = [asdict(a) for a in orch.health_monitor.get_alerts()]
            self._send_json({'alerts': alerts})
        
        elif path == '/signals':
            # Get pending signals (may be dicts or objects)
            pending = getattr(orch, 'pending_signals', [])
            signals = [s if isinstance(s, dict) else s.__dict__ for s in pending]
            self._send_json({
                'signals': signals,
                'count': len(pending)
            })
        
        elif path == '/statuses':
            statuses = {k: asdict(v) for k, v in orch.health_monitor.get_all_statuses().items()}
            self._send_json({'statuses': statuses})
        
        elif path == '/watchdog':
            # Get profit watchdog status - full analysis
            self._send_json(orch.watchdog.get_status())
        
        elif path == '/watchdog/decisions':
            # Get recent decisions with outcomes
            limit = 50
            try:
                from urllib.parse import parse_qs
                query = parse_qs(parsed.query)
                limit = int(query.get('limit', [50])[0])
            except:
                pass
            self._send_json({'decisions': orch.watchdog.get_recent_decisions(limit)})
        
        elif path == '/watchdog/alerts':
            # Get active watchdog alerts
            alerts = orch.watchdog.generate_alerts()
            self._send_json({'alerts': [asdict(a) for a in alerts]})
        
        elif path == '/watchdog/recommendations':
            # Get parameter recommendations
            recs = orch.watchdog.analyze_and_recommend()
            self._send_json({'recommendations': [asdict(r) for r in recs]})
        
        elif path == '/watchdog/drift':
            # Get strategy drift analysis
            drifts = orch.watchdog.detect_strategy_drift()
            self._send_json({'strategy_drift': [asdict(d) for d in drifts]})
        
        elif path == '/trades':
            # Serve trade history from Obsidian vault
            trades = self._load_obsidian_trades()
            self._send_json(trades)
        
        elif path == '/cucurbit':
            # Proxy full Cucurbit health data
            try:
                import requests as req
                resp = req.get('https://autonomous-trading-cex-production.up.railway.app/health', timeout=10)
                if resp.status_code == 200:
                    self._send_json(resp.json())
                else:
                    self._send_json({'error': f'Cucurbit returned {resp.status_code}'})
            except Exception as e:
                self._send_json({'error': str(e)})
        
        elif path.startswith('/cucurbit/'):
            # Proxy any Cucurbit endpoint with auth
            self._proxy_cucurbit(path.replace('/cucurbit', ''), 'GET')
        
        elif path == '/vault/positions':
            # Serve positions from Obsidian vault
            positions = self._load_vault_positions()
            self._send_json(positions)
        
        elif path == '/vault/summary':
            # Serve summary from Obsidian vault
            summary = self._load_vault_summary()
            self._send_json(summary)
        
        elif path == '/vault/strategies':
            # Serve strategy performance from Obsidian vault
            strategies = self._load_vault_strategies()
            self._send_json(strategies)
        
        elif path == '/' or path == '/dashboard' or path.startswith('/dashboard/'):
            # Serve dashboard
            self._serve_dashboard(path)
        
        else:
            self._send_error('Not found', 404)
    
    def _serve_dashboard(self, path: str):
        """Serve dashboard HTML"""
        import os
        from pathlib import Path
        
        # Use path relative to this file
        this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dashboard_dir = this_dir.parent / 'dashboard'
        
        if path in ['/', '/dashboard', '/dashboard/']:
            file_path = dashboard_dir / 'index.html'
        else:
            # Strip /dashboard/ prefix
            rel_path = path.replace('/dashboard/', '').lstrip('/')
            file_path = dashboard_dir / rel_path
        
        if file_path.exists() and file_path.is_file():
            content_type = 'text/html'
            if file_path.suffix == '.css':
                content_type = 'text/css'
            elif file_path.suffix == '.js':
                content_type = 'application/javascript'
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(file_path.read_bytes())
        else:
            self._send_error('File not found', 404)
    
    def do_POST(self):
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        # All POST endpoints require authentication (except /signals from internal agents)
        if path != '/signals' and not self._check_auth():
            self._send_unauthorized()
            return
        
        if path == '/agents':
            # Register new agent
            try:
                data = self._read_body()
                agent = AgentConfig(
                    id=data['id'],
                    name=data['name'],
                    type=data['type'],
                    endpoint=data['endpoint'],
                    api_key=data.get('api_key'),
                    allocation_pct=data.get('allocation_pct', 0),
                    max_positions=data.get('max_positions', 10),
                    max_drawdown_pct=data.get('max_drawdown_pct', 5.0),
                    metadata=data.get('metadata', {})
                )
                if orch.registry.register(agent):
                    self._send_json({'success': True, 'agent': asdict(agent)}, 201)
                else:
                    self._send_error('Agent already exists')
            except KeyError as e:
                self._send_error(f'Missing required field: {e}')
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/refresh':
            # Trigger immediate health check
            orch.health_monitor.check_all()
            self._send_json({
                'success': True,
                'statuses': {k: asdict(v) for k, v in orch.health_monitor.get_all_statuses().items()}
            })
        
        elif path == '/signals':
            # Receive signal from an agent
            try:
                data = self._read_body()
                signal_data = {
                    'strategy_id': data['strategy_id'],
                    'symbol': data['symbol'],
                    'side': data['side'],
                    'confidence': data['confidence'],
                    'price': data['price'],
                    'stop_loss': data.get('stop_loss'),
                    'take_profit': data.get('take_profit'),
                    'reason': data.get('reason', ''),
                    'timestamp': data.get('timestamp', datetime.utcnow().isoformat()),
                    'metadata': data.get('metadata', {})
                }
                
                # Add to pending signals
                if not hasattr(orch, 'pending_signals'):
                    orch.pending_signals = []
                orch.pending_signals.append(signal_data)
                
                # Keep only recent signals (last 100)
                orch.pending_signals = orch.pending_signals[-100:]
                
                logger.info(f"📨 Signal received: {signal_data['side'].upper()} {signal_data['symbol']} from {signal_data['strategy_id']}")
                
                self._send_json({'success': True, 'message': 'Signal received'}, 201)
            except KeyError as e:
                self._send_error(f'Missing required field: {e}')
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/watchdog/record':
            # Record a decision manually (for testing/debugging)
            try:
                data = self._read_body()
                orch.watchdog.record_decision(
                    signal_data=data.get('signal', data),
                    accepted=data.get('accepted', False),
                    rejection_reason=data.get('rejection_reason')
                )
                self._send_json({'success': True, 'message': 'Decision recorded'}, 201)
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/watchdog/tune':
            # Trigger auto-tuning
            try:
                data = self._read_body()
                dry_run = data.get('dry_run', True)
                changes = orch.watchdog.auto_tune(dry_run=dry_run)
                self._send_json({
                    'success': True,
                    'dry_run': dry_run,
                    'changes': changes
                })
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/watchdog/baseline':
            # Update strategy baseline
            try:
                data = self._read_body()
                orch.watchdog.update_baseline(
                    strategy_id=data['strategy_id'],
                    win_rate=data['win_rate'],
                    trades=data['trades'],
                    pnl=data.get('pnl', 0)
                )
                self._send_json({'success': True, 'message': 'Baseline updated'})
            except KeyError as e:
                self._send_error(f'Missing required field: {e}')
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/watchdog/cycle':
            # Trigger a watchdog cycle manually
            try:
                current_prices = orch._get_current_prices()
                status = run_watchdog_cycle(orch.watchdog, current_prices)
                self._send_json({
                    'success': True,
                    'status': status,
                    'prices_fetched': len(current_prices)
                })
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/rotate':
            # Trigger immediate rotation evaluation
            try:
                decisions = orch.position_manager.evaluate_rotations(orch.pending_signals)
                self._send_json({
                    'success': True,
                    'rotation_decisions': len(decisions),
                    'decisions': [{
                        'symbol': d.position.symbol,
                        'agent': d.position.agent_id,
                        'reason': d.reason,
                        'urgency': d.urgency,
                        'has_replacement': d.replacement_signal is not None
                    } for d in decisions]
                })
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/positions/close':
            # Force close a specific position
            try:
                data = self._read_body()
                agent_id = data['agent_id']
                symbol = data['symbol']
                reason = data.get('reason', 'Manual close from orchestrator')
                
                # Log the close command
                logger.info(f"Manual close requested: {agent_id}:{symbol} - {reason}")
                
                # Remove from tracking
                orch.position_manager.remove_position(agent_id, symbol)
                
                self._send_json({'success': True, 'message': f'Close command sent for {symbol}'})
            except KeyError as e:
                self._send_error(f'Missing required field: {e}')
            except Exception as e:
                self._send_error(str(e))
        
        elif path.startswith('/agents/') and path.endswith('/enable'):
            agent_id = path.split('/')[2]
            if orch.registry.enable(agent_id):
                self._send_json({'success': True})
            else:
                self._send_error('Agent not found', 404)
        
        elif path.startswith('/agents/') and path.endswith('/disable'):
            agent_id = path.split('/')[2]
            if orch.registry.disable(agent_id):
                self._send_json({'success': True})
            else:
                self._send_error('Agent not found', 404)
        
        elif path.startswith('/cucurbit/'):
            # Proxy POST to Cucurbit
            body = self._read_body_raw()
            self._proxy_cucurbit(path.replace('/cucurbit', ''), 'POST', body)
        
        else:
            self._send_error('Not found', 404)
    
    def do_PATCH(self):
        # All PATCH endpoints require authentication
        if not self._check_auth():
            self._send_unauthorized()
            return
            
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/agents/'):
            agent_id = path.split('/')[2]
            try:
                data = self._read_body()
                if orch.registry.update(agent_id, **data):
                    agent = orch.registry.get(agent_id)
                    self._send_json({'success': True, 'agent': asdict(agent)})
                else:
                    self._send_error('Agent not found', 404)
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/rotation-config':
            # Update rotation config
            try:
                data = self._read_body()
                config = orch.position_manager.config
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                self._send_json({'success': True, 'config': asdict(config)})
            except Exception as e:
                self._send_error(str(e))
        
        elif path.startswith('/cucurbit/'):
            # Proxy PATCH to Cucurbit
            body = self._read_body_raw()
            self._proxy_cucurbit(path.replace('/cucurbit', ''), 'PATCH', body)
        
        else:
            self._send_error('Not found', 404)
    
    def do_DELETE(self):
        # All DELETE endpoints require authentication
        if not self._check_auth():
            self._send_unauthorized()
            return
            
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/agents/'):
            agent_id = path.split('/')[2]
            if orch.registry.unregister(agent_id):
                self._send_json({'success': True})
            else:
                self._send_error('Agent not found', 404)
        
        elif path.startswith('/cucurbit/'):
            # Proxy DELETE to Cucurbit
            self._proxy_cucurbit(path.replace('/cucurbit', ''), 'DELETE')
        
        else:
            self._send_error('Not found', 404)
    
    def _load_obsidian_trades(self) -> list:
        """Load trade history from Obsidian vault"""
        import os
        import re
        from pathlib import Path
        
        trades = []
        vault_path = Path(os.path.expanduser("~/.openclaw/workspace/vault/trading/daily"))
        
        if not vault_path.exists():
            return trades
        
        # Parse daily markdown files
        # Format: | Time | Symbol | Side | Entry | Exit | PnL | PnL% | Strategy | Reason |
        for md_file in sorted(vault_path.glob("*.md"), reverse=True)[:30]:  # Last 30 days
            try:
                content = md_file.read_text()
                date = md_file.stem  # YYYY-MM-DD
                
                lines = content.split('\n')
                
                for line in lines:
                    # Skip header/separator lines
                    if '|' not in line or '---' in line or 'Time' in line or 'Metric' in line:
                        continue
                    
                    # Parse trade rows: | Time | Symbol | Side | Entry | Exit | PnL | PnL% | Strategy | Reason |
                    if '/USDT' in line.upper() or '/USD' in line.upper():
                        parts = [p.strip() for p in line.split('|')]
                        parts = [p for p in parts if p]  # Remove empty
                        
                        if len(parts) >= 6:
                            try:
                                # Extract PnL (look for $ sign)
                                pnl_str = parts[5] if len(parts) > 5 else '0'
                                pnl = self._extract_number(pnl_str)
                                
                                trade = {
                                    'date': date,
                                    'time': parts[0],
                                    'symbol': parts[1],
                                    'side': 'long' if 'LONG' in parts[2].upper() else 'short',
                                    'entry_price': self._extract_number(parts[3]),
                                    'exit_price': self._extract_number(parts[4]),
                                    'pnl': pnl,
                                    'pnl_pct': parts[6] if len(parts) > 6 else '',
                                    'strategy': parts[7] if len(parts) > 7 else 'unknown',
                                    'reason': parts[8] if len(parts) > 8 else ''
                                }
                                trades.append(trade)
                            except Exception as e:
                                logger.debug(f"Error parsing line: {line[:50]}... - {e}")
            except Exception as e:
                logger.debug(f"Error parsing {md_file}: {e}")
        
        return trades
    
    def _extract_number(self, s: str) -> float:
        """Extract number from string like '$123.45' or '-$10.00'"""
        import re
        match = re.search(r'-?\$?([\d,]+\.?\d*)', s.replace(',', ''))
        if match:
            num = float(match.group(1))
            return -num if '-' in s else num
        return 0.0
    
    def _extract_strategy(self, parts: list) -> str:
        """Extract strategy name from parts"""
        strategies = ['trend', 'turtle', 'weinstein', 'livermore', 'zweig', 'lynch', 'mean', 'stat']
        for part in parts:
            for strat in strategies:
                if strat.lower() in part.lower():
                    return part
        return 'unknown'
    
    def _load_vault_positions(self) -> dict:
        """Load current positions from Obsidian vault"""
        import os
        import re
        from pathlib import Path
        
        vault_path = Path(os.path.expanduser("~/.openclaw/workspace/vault/trading/positions.md"))
        if not vault_path.exists():
            return {'positions': [], 'summary': {}}
        
        positions = []
        summary = {'open_positions': 0, 'total_value': 0, 'unrealized_pnl': 0}
        
        try:
            content = vault_path.read_text()
            lines = content.split('\n')
            
            for line in lines:
                if '|' not in line or '---' in line or 'Symbol' in line or 'Metric' in line:
                    continue
                
                # Parse summary row
                if 'Open Positions' in line:
                    summary['open_positions'] = int(self._extract_number(line))
                elif 'Total Value' in line:
                    summary['total_value'] = self._extract_number(line)
                elif 'Unrealized PnL' in line:
                    summary['unrealized_pnl'] = self._extract_number(line)
                # Parse position row: | Symbol | Side | Entry | Current | Size | Value | PnL | PnL% | Strategy |
                elif '/USDT' in line.upper() or '/USD' in line.upper():
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 8:
                        positions.append({
                            'symbol': parts[0],
                            'side': 'long' if 'LONG' in parts[1].upper() else 'short',
                            'entry_price': self._extract_number(parts[2]),
                            'current_price': self._extract_number(parts[3]),
                            'size': self._extract_number(parts[4]),
                            'value': self._extract_number(parts[5]),
                            'pnl': self._extract_number(parts[6]),
                            'pnl_pct': parts[7],
                            'strategy': parts[8] if len(parts) > 8 else 'unknown'
                        })
        except Exception as e:
            logger.debug(f"Error loading positions: {e}")
        
        return {'positions': positions, 'summary': summary}
    
    def _load_vault_summary(self) -> dict:
        """Load account summary from Obsidian vault"""
        import os
        from pathlib import Path
        
        vault_path = Path(os.path.expanduser("~/.openclaw/workspace/vault/trading/summary.md"))
        if not vault_path.exists():
            return {}
        
        summary = {
            'equity': 0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0
        }
        
        try:
            content = vault_path.read_text()
            for line in content.split('\n'):
                if 'Total Trades' in line:
                    summary['total_trades'] = int(self._extract_number(line))
                elif 'Win / Loss' in line:
                    parts = line.split('/')
                    if len(parts) >= 2:
                        summary['wins'] = int(self._extract_number(parts[0]))
                        summary['losses'] = int(self._extract_number(parts[1]))
                elif 'Win Rate' in line:
                    summary['win_rate'] = self._extract_number(line)
                elif 'Total PnL' in line:
                    summary['total_pnl'] = self._extract_number(line)
                elif 'Equity' in line:
                    summary['equity'] = self._extract_number(line)
        except Exception as e:
            logger.debug(f"Error loading summary: {e}")
        
        return summary
    
    def _load_vault_strategies(self) -> list:
        """Load strategy performance from Obsidian vault"""
        import os
        from pathlib import Path
        
        vault_path = Path(os.path.expanduser("~/.openclaw/workspace/vault/trading/summary.md"))
        if not vault_path.exists():
            return []
        
        strategies = []
        try:
            content = vault_path.read_text()
            in_strategy_table = False
            
            for line in content.split('\n'):
                if 'Strategy Performance' in line:
                    in_strategy_table = True
                    continue
                
                if in_strategy_table and '|' in line and '---' not in line and 'Strategy' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 5:
                        strategies.append({
                            'name': parts[0],
                            'trades': int(self._extract_number(parts[1])),
                            'wins': int(self._extract_number(parts[2])),
                            'win_rate': self._extract_number(parts[3]),
                            'pnl': self._extract_number(parts[4])
                        })
                
                # Stop at next section
                if in_strategy_table and line.startswith('## ') and 'Strategy' not in line:
                    break
        except Exception as e:
            logger.debug(f"Error loading strategies: {e}")
        
        return strategies
    
    def _proxy_cucurbit(self, endpoint: str, method: str = 'GET', body: str = None):
        """Proxy request to Cucurbit API with authentication"""
        import requests as req
        api_key = os.environ.get('CUCURBIT_API_KEY', '')
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        try:
            url = f'https://autonomous-trading-cex-production.up.railway.app{endpoint}'
            if method == 'GET':
                resp = req.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                resp = req.post(url, headers=headers, json=json.loads(body) if body else None, timeout=30)
            elif method == 'DELETE':
                resp = req.delete(url, headers=headers, timeout=30)
            elif method == 'PATCH':
                resp = req.patch(url, headers=headers, json=json.loads(body) if body else None, timeout=30)
            else:
                self._send_error('Method not allowed', 405)
                return
            
            if resp.status_code == 200 or resp.status_code == 201:
                self._send_json(resp.json())
            else:
                self._send_json({'error': f'Cucurbit returned {resp.status_code}', 'details': resp.text[:500]})
        except Exception as e:
            self._send_json({'error': str(e)})
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

def run_server(host: str = '0.0.0.0', port: int = 8888):
    """Run the orchestrator HTTP server"""
    orch = get_orchestrator()
    orch.start()
    
    # Run HTTP server
    server = HTTPServer((host, port), OrchestratorHandler)
    logger.info(f"Cash Town Orchestrator running on http://{host}:{port}")
    logger.info(f"Position rotation enabled - checking every 30s")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        orch.stop()
        server.shutdown()


def run_server_with_orchestrator(orch: Orchestrator, host: str = '0.0.0.0', port: int = 8888):
    """Run HTTP server with an external orchestrator instance"""
    global _orchestrator
    _orchestrator = orch
    
    server = HTTPServer((host, port), OrchestratorHandler)
    logger.info(f"Cash Town Orchestrator running on http://{host}:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    run_server()
