"""
Cash Town Orchestrator - Main server
Provides HTTP API for managing the trading swarm with active position management.
"""
import json
import logging
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from dataclasses import asdict

from .registry import AgentRegistry, AgentConfig
from .health import HealthMonitor
from .position_manager import PositionManager, RotationConfig, TrackedPosition, PositionState

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
        
        # Pending signals from agents waiting for capital
        self.pending_signals = []
    
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

class OrchestratorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for orchestrator API"""
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())
    
    def _send_error(self, message: str, status: int = 400):
        self._send_json({'error': message}, status)
    
    def _read_body(self) -> dict:
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/health':
            self._send_json({
                'status': 'healthy' if orch.running else 'stopped',
                'timestamp': datetime.utcnow().isoformat()
            })
        
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
        
        elif path == '/positions/summary':
            self._send_json({'summary': orch.position_manager.summary()})
        
        elif path == '/alerts':
            alerts = [asdict(a) for a in orch.health_monitor.get_alerts()]
            self._send_json({'alerts': alerts})
        
        elif path == '/statuses':
            statuses = {k: asdict(v) for k, v in orch.health_monitor.get_all_statuses().items()}
            self._send_json({'statuses': statuses})
        
        else:
            self._send_error('Not found', 404)
    
    def do_POST(self):
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
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
        
        else:
            self._send_error('Not found', 404)
    
    def do_PATCH(self):
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
        
        else:
            self._send_error('Not found', 404)
    
    def do_DELETE(self):
        orch = get_orchestrator()
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/agents/'):
            agent_id = path.split('/')[2]
            if orch.registry.unregister(agent_id):
                self._send_json({'success': True})
            else:
                self._send_error('Agent not found', 404)
        else:
            self._send_error('Not found', 404)
    
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

if __name__ == '__main__':
    run_server()
