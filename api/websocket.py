"""
WebSocket Live Feed for Bloomberg Dashboard

Provides real-time updates:
- Position changes
- Price updates
- Signal events
- Trade executions
- Risk alerts
"""
import json
import logging
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """WebSocket event types"""
    # Connection
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"
    
    # Portfolio events
    PORTFOLIO_UPDATE = "portfolio_update"
    EQUITY_CHANGE = "equity_change"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATE = "position_update"
    POSITION_PNL = "position_pnl"
    
    # Price events
    PRICE_UPDATE = "price_update"
    PRICE_ALERT = "price_alert"
    
    # Signal events
    SIGNAL_RECEIVED = "signal_received"
    SIGNAL_ACCEPTED = "signal_accepted"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Trade events
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"
    
    # Risk events
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    DRAWDOWN_WARNING = "drawdown_warning"
    
    # Strategy events
    STRATEGY_UPDATE = "strategy_update"
    STRATEGY_DISABLED = "strategy_disabled"


@dataclass
class LiveEvent:
    """WebSocket event structure"""
    type: str
    data: Any
    timestamp: str
    sequence: int
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp,
            'sequence': self.sequence
        })


class LiveFeed:
    """
    WebSocket live feed manager.
    
    Broadcasts real-time events to connected clients.
    Integrates with orchestrator and executor for event sourcing.
    """
    
    def __init__(self):
        self.clients: Set[Any] = set()  # WebSocket connections
        self.sequence = 0
        self.event_history: List[LiveEvent] = []
        self.max_history = 1000
        
        # State tracking for change detection
        self._last_positions: Dict[str, Dict] = {}
        self._last_prices: Dict[str, float] = {}
        self._last_equity: float = 0
        
        # References to other components
        self.orchestrator = None
        self.executor = None
        self.data_feed = None
        
        # Background update thread
        self._running = False
        self._update_thread = None
    
    def set_orchestrator(self, orchestrator):
        """Set orchestrator reference"""
        self.orchestrator = orchestrator
    
    def set_executor(self, executor):
        """Set executor reference"""
        self.executor = executor
    
    def set_data_feed(self, data_feed):
        """Set data feed reference"""
        self.data_feed = data_feed
    
    def start(self, update_interval: float = 1.0):
        """Start the live feed background updates"""
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True
        )
        self._update_thread.start()
        logger.info("📡 WebSocket live feed started")
    
    def stop(self):
        """Stop the live feed"""
        self._running = False
        logger.info("📡 WebSocket live feed stopped")
    
    def add_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"📡 Client connected ({len(self.clients)} total)")
        
        # Send connection event
        self._emit(EventType.CONNECTED, {
            'message': 'Connected to Cash Town live feed',
            'client_count': len(self.clients)
        })
    
    def remove_client(self, websocket):
        """Remove a WebSocket client"""
        self.clients.discard(websocket)
        logger.info(f"📡 Client disconnected ({len(self.clients)} total)")
    
    def _emit(self, event_type: EventType, data: Any):
        """Emit an event to all connected clients"""
        self.sequence += 1
        event = LiveEvent(
            type=event_type.value,
            data=data,
            timestamp=datetime.utcnow().isoformat(),
            sequence=self.sequence
        )
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Broadcast to clients
        message = event.to_json()
        dead_clients = set()
        
        for client in self.clients.copy():
            try:
                if hasattr(client, 'send'):
                    client.send(message)
                elif hasattr(client, 'write_message'):
                    client.write_message(message)
                else:
                    # For queue-based approach
                    if hasattr(client, 'put'):
                        client.put(message)
            except Exception as e:
                logger.debug(f"Error sending to client: {e}")
                dead_clients.add(client)
        
        # Remove dead clients
        for client in dead_clients:
            self.remove_client(client)
    
    def _update_loop(self, interval: float):
        """Background loop for detecting and emitting changes"""
        heartbeat_counter = 0
        
        while self._running:
            try:
                # Heartbeat every 30 updates
                heartbeat_counter += 1
                if heartbeat_counter >= 30:
                    self._emit(EventType.HEARTBEAT, {
                        'clients': len(self.clients),
                        'sequence': self.sequence
                    })
                    heartbeat_counter = 0
                
                # Check for position changes
                self._check_positions()
                
                # Check for price updates
                self._check_prices()
                
                # Check for equity changes
                self._check_equity()
                
                # Check risk status
                self._check_risk()
                
            except Exception as e:
                logger.error(f"Live feed update error: {e}")
            
            time.sleep(interval)
    
    def _check_positions(self):
        """Check for position changes"""
        if not self.executor:
            return
        
        current_positions = {}
        for symbol, pos in self.executor.positions.items():
            current_positions[symbol] = {
                'symbol': symbol,
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'margin': pos.margin
            }
        
        # Detect new positions
        for symbol, pos in current_positions.items():
            if symbol not in self._last_positions:
                self._emit(EventType.POSITION_OPENED, pos)
                logger.info(f"📡 Position opened: {symbol}")
        
        # Detect closed positions
        for symbol in self._last_positions:
            if symbol not in current_positions:
                self._emit(EventType.POSITION_CLOSED, {
                    'symbol': symbol,
                    'previous': self._last_positions[symbol]
                })
                logger.info(f"📡 Position closed: {symbol}")
        
        # Detect P&L changes
        for symbol, pos in current_positions.items():
            if symbol in self._last_positions:
                last = self._last_positions[symbol]
                if abs(pos['unrealized_pnl'] - last['unrealized_pnl']) > 0.01:
                    self._emit(EventType.POSITION_PNL, {
                        'symbol': symbol,
                        'pnl': pos['unrealized_pnl'],
                        'pnl_change': pos['unrealized_pnl'] - last['unrealized_pnl']
                    })
        
        self._last_positions = current_positions
    
    def _check_prices(self):
        """Check for significant price changes"""
        if not self.data_feed:
            return
        
        for symbol in self.data_feed.symbols if hasattr(self.data_feed, 'symbols') else []:
            try:
                data = self.data_feed.get_symbol_data(symbol)
                if data and data.get('close'):
                    current_price = data['close'][-1]
                    
                    if symbol in self._last_prices:
                        last_price = self._last_prices[symbol]
                        change_pct = abs(current_price - last_price) / last_price * 100
                        
                        # Emit on >0.5% change
                        if change_pct > 0.5:
                            self._emit(EventType.PRICE_UPDATE, {
                                'symbol': symbol,
                                'price': current_price,
                                'change': current_price - last_price,
                                'change_pct': round(change_pct, 2)
                            })
                    
                    self._last_prices[symbol] = current_price
            except:
                pass
    
    def _check_equity(self):
        """Check for equity changes"""
        if not self.executor:
            return
        
        current_equity = self.executor.account_balance + sum(
            p.unrealized_pnl for p in self.executor.positions.values()
        )
        
        if self._last_equity > 0:
            change = current_equity - self._last_equity
            change_pct = abs(change) / self._last_equity * 100
            
            # Emit on >0.1% change
            if change_pct > 0.1:
                self._emit(EventType.EQUITY_CHANGE, {
                    'equity': current_equity,
                    'change': change,
                    'change_pct': round(change_pct, 2) * (1 if change > 0 else -1)
                })
        
        self._last_equity = current_equity
    
    def _check_risk(self):
        """Check risk status for alerts"""
        if not self.orchestrator:
            return
        
        try:
            can_trade, reason = self.orchestrator.can_trade()
            
            if not can_trade:
                self._emit(EventType.CIRCUIT_BREAKER, {
                    'active': True,
                    'reason': reason
                })
            
            # Check drawdown
            if self.executor:
                status = self.executor.get_status()
                dd = status.get('drawdown_protection', {})
                drawdown_pct = dd.get('current_drawdown_pct', 0)
                
                # Warn at 5% drawdown
                if drawdown_pct > 5:
                    self._emit(EventType.DRAWDOWN_WARNING, {
                        'drawdown_pct': drawdown_pct,
                        'threshold_pct': dd.get('threshold_pct', 10)
                    })
        except:
            pass
    
    # ==========================================
    # MANUAL EVENT EMISSION (called by other components)
    # ==========================================
    
    def emit_signal_received(self, signal_data: Dict):
        """Emit when a signal is received from a strategy"""
        self._emit(EventType.SIGNAL_RECEIVED, signal_data)
    
    def emit_signal_accepted(self, signal_data: Dict, rank: int, sources: List[str]):
        """Emit when a signal is accepted for execution"""
        self._emit(EventType.SIGNAL_ACCEPTED, {
            **signal_data,
            'rank': rank,
            'sources': sources
        })
    
    def emit_signal_rejected(self, signal_data: Dict, reason: str):
        """Emit when a signal is rejected"""
        self._emit(EventType.SIGNAL_REJECTED, {
            **signal_data,
            'reason': reason
        })
    
    def emit_trade_executed(self, trade_data: Dict):
        """Emit when a trade is executed"""
        self._emit(EventType.TRADE_EXECUTED, trade_data)
    
    def emit_trade_failed(self, trade_data: Dict, reason: str):
        """Emit when a trade fails"""
        self._emit(EventType.TRADE_FAILED, {
            **trade_data,
            'failure_reason': reason
        })
    
    def emit_risk_alert(self, alert_type: str, details: Dict):
        """Emit a risk alert"""
        self._emit(EventType.RISK_ALERT, {
            'alert_type': alert_type,
            **details
        })
    
    def emit_strategy_update(self, strategy_id: str, metrics: Dict):
        """Emit strategy performance update"""
        self._emit(EventType.STRATEGY_UPDATE, {
            'strategy_id': strategy_id,
            **metrics
        })
    
    def emit_strategy_disabled(self, strategy_id: str, reason: str):
        """Emit when a strategy is disabled"""
        self._emit(EventType.STRATEGY_DISABLED, {
            'strategy_id': strategy_id,
            'reason': reason
        })
    
    # ==========================================
    # CLIENT API
    # ==========================================
    
    def get_recent_events(self, limit: int = 50, event_types: List[str] = None) -> List[Dict]:
        """Get recent events for catch-up on connect"""
        events = self.event_history[-limit:]
        
        if event_types:
            events = [e for e in events if e.type in event_types]
        
        return [json.loads(e.to_json()) for e in events]
    
    def get_snapshot(self) -> Dict:
        """Get current state snapshot for new connections"""
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'sequence': self.sequence,
            'positions': {},
            'prices': self._last_prices.copy(),
            'equity': self._last_equity
        }
        
        if self.executor:
            for symbol, pos in self.executor.positions.items():
                snapshot['positions'][symbol] = {
                    'symbol': symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
        
        return snapshot


# Simple threadsafe client queue for SSE fallback
class SSEClient:
    """Server-Sent Events client wrapper"""
    
    def __init__(self):
        import queue
        self.queue = queue.Queue()
        self.connected = True
    
    def put(self, message: str):
        """Add message to queue"""
        if self.connected:
            self.queue.put(message)
    
    def get(self, timeout: float = None):
        """Get next message from queue"""
        import queue
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def disconnect(self):
        """Mark as disconnected"""
        self.connected = False
