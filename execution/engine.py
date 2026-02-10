"""
Execution Engine - Manages order execution with risk controls

Responsibilities:
1. Receive aggregated signals
2. Calculate position sizes based on risk
3. Place orders with stop loss / take profit
4. Track fills and update positions
5. Enforce risk limits (max exposure, daily loss, etc.)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

from .kucoin import KuCoinFuturesExecutor, Position
from orchestrator.signal_aggregator import AggregatedSignal
from agents.base import SignalSide

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    LIVE = "live"
    PAPER = "paper"
    DISABLED = "disabled"

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_pct: float = 2.0  # Max 2% of account per position
    max_total_exposure_pct: float = 20.0  # Max 20% total exposure
    max_positions: int = 5
    max_daily_loss_pct: float = 5.0  # Kill switch at 5% daily loss
    default_leverage: int = 5
    default_stop_loss_pct: float = 2.0  # 2% stop loss
    default_take_profit_pct: float = 4.0  # 4% take profit
    min_order_value_usd: float = 10.0

@dataclass
class ExecutionResult:
    """Result of an execution attempt"""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    size: float
    price: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0

class ExecutionEngine:
    """
    Main execution engine for Cash Town.
    Handles order placement with risk management.
    """
    
    def __init__(self, executor: KuCoinFuturesExecutor = None, 
                 risk_config: RiskConfig = None,
                 mode: ExecutionMode = ExecutionMode.PAPER):
        self.executor = executor or KuCoinFuturesExecutor()
        self.risk = risk_config or RiskConfig()
        self.mode = mode
        
        # State tracking
        self.account_balance: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.execution_history: List[ExecutionResult] = []
        self.daily_stats: DailyStats = DailyStats(date=date.today())
        
        # Kill switch
        self.killed = False
        self.kill_reason: Optional[str] = None
    
    def refresh_state(self):
        """Refresh account state from exchange"""
        if not self.executor.is_configured:
            logger.warning("Executor not configured - running in paper mode")
            return
        
        try:
            # Get account balance
            overview = self.executor.get_account_overview()
            self.account_balance = float(overview.get('availableBalance', 0))
            
            # Get positions
            positions = self.executor.get_positions()
            self.positions = {p.symbol: p for p in positions}
            
            # Check daily loss limit
            self._check_daily_loss()
            
            logger.info(f"State refreshed: ${self.account_balance:.2f} balance, {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error refreshing state: {e}")
    
    def _check_daily_loss(self):
        """Check if daily loss limit hit - activate kill switch"""
        if self.daily_stats.date != date.today():
            # New day - reset stats
            self.daily_stats = DailyStats(date=date.today())
            self.killed = False
            self.kill_reason = None
        
        # Calculate daily P&L from positions
        daily_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.daily_stats.pnl = daily_pnl
        
        # Check loss limit
        if self.account_balance > 0:
            loss_pct = -daily_pnl / self.account_balance * 100
            if loss_pct >= self.risk.max_daily_loss_pct:
                self.killed = True
                self.kill_reason = f"Daily loss limit hit: {loss_pct:.1f}%"
                logger.critical(f"🛑 KILL SWITCH ACTIVATED: {self.kill_reason}")
    
    def can_execute(self) -> tuple[bool, str]:
        """Check if execution is allowed"""
        if self.killed:
            return False, f"Kill switch active: {self.kill_reason}"
        
        if self.mode == ExecutionMode.DISABLED:
            return False, "Execution disabled"
        
        if len(self.positions) >= self.risk.max_positions:
            return False, f"Max positions reached ({self.risk.max_positions})"
        
        # Check total exposure
        total_exposure = sum(p.margin for p in self.positions.values())
        max_exposure = self.account_balance * self.risk.max_total_exposure_pct / 100
        if total_exposure >= max_exposure:
            return False, f"Max exposure reached (${total_exposure:.2f}/${max_exposure:.2f})"
        
        return True, "OK"
    
    def calculate_position_size(self, symbol: str, price: float, signal: AggregatedSignal) -> float:
        """Calculate position size based on risk parameters"""
        if self.account_balance <= 0:
            return 0
        
        # Base position value
        position_value = self.account_balance * self.risk.max_position_pct / 100
        
        # Adjust by signal confidence
        position_value *= signal.adjusted_confidence
        
        # Adjust by consensus (more strategies agree = larger position)
        position_value *= (0.8 + 0.2 * signal.consensus_score)
        
        # Apply leverage
        leveraged_value = position_value * self.risk.default_leverage
        
        # Calculate contracts (for KuCoin, size is in contracts)
        # This is simplified - real implementation needs contract specs
        contracts = int(leveraged_value / price)
        
        # Ensure minimum order value
        if contracts * price < self.risk.min_order_value_usd:
            return 0
        
        return contracts
    
    def execute_signal(self, signal: AggregatedSignal) -> ExecutionResult:
        """
        Execute an aggregated signal.
        
        Returns:
            ExecutionResult with outcome
        """
        symbol = signal.symbol
        side = 'buy' if signal.side == SignalSide.LONG else 'sell'
        
        # Pre-flight checks
        can_exec, reason = self.can_execute()
        if not can_exec:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                size=0,
                price=0,
                message=f"Cannot execute: {reason}"
            )
        
        # Get current price
        price = signal.signal.price
        
        # Calculate size
        size = self.calculate_position_size(symbol, price, signal)
        if size <= 0:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                size=0,
                price=price,
                message="Position size too small"
            )
        
        # Calculate stop loss and take profit
        if signal.side == SignalSide.LONG:
            stop_loss = signal.signal.stop_loss or (price * (1 - self.risk.default_stop_loss_pct/100))
            take_profit = signal.signal.take_profit or (price * (1 + self.risk.default_take_profit_pct/100))
        else:
            stop_loss = signal.signal.stop_loss or (price * (1 + self.risk.default_stop_loss_pct/100))
            take_profit = signal.signal.take_profit or (price * (1 - self.risk.default_take_profit_pct/100))
        
        # Execute based on mode
        if self.mode == ExecutionMode.PAPER:
            return self._paper_execute(symbol, side, size, price, stop_loss, take_profit, signal)
        elif self.mode == ExecutionMode.LIVE:
            return self._live_execute(symbol, side, size, price, stop_loss, take_profit, signal)
        else:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                message="Execution mode disabled"
            )
    
    def _paper_execute(self, symbol: str, side: str, size: float, price: float,
                      stop_loss: float, take_profit: float, signal: AggregatedSignal) -> ExecutionResult:
        """Simulate execution for paper trading"""
        order_id = f"paper_{int(datetime.utcnow().timestamp()*1000)}"
        
        logger.info(f"📝 PAPER TRADE: {side.upper()} {size} {symbol} @ ${price:.2f}")
        logger.info(f"   SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
        logger.info(f"   Sources: {signal.sources}, Confidence: {signal.adjusted_confidence:.0%}")
        
        # Simulate position
        self.positions[symbol] = Position(
            symbol=symbol,
            side='long' if side == 'buy' else 'short',
            size=size,
            entry_price=price,
            leverage=self.risk.default_leverage,
            unrealized_pnl=0,
            margin=size * price / self.risk.default_leverage,
            liquidation_price=0
        )
        
        result = ExecutionResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            message=f"Paper trade executed"
        )
        
        self.execution_history.append(result)
        self.daily_stats.trades += 1
        
        return result
    
    def _live_execute(self, symbol: str, side: str, size: float, price: float,
                     stop_loss: float, take_profit: float, signal: AggregatedSignal) -> ExecutionResult:
        """Execute real trade on exchange"""
        if not self.executor.is_configured:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                message="Executor not configured - add API credentials"
            )
        
        try:
            # Place main order
            order_id = self.executor.place_market_order(
                symbol=symbol,
                side=side,
                size=size,
                leverage=self.risk.default_leverage
            )
            
            if not order_id:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side=side,
                    size=size,
                    price=price,
                    message="Order placement failed"
                )
            
            # Place stop loss
            stop_side = 'sell' if side == 'buy' else 'buy'
            self.executor.place_stop_order(symbol, stop_side, size, stop_loss)
            
            logger.info(f"🔥 LIVE TRADE: {side.upper()} {size} {symbol}")
            logger.info(f"   Order ID: {order_id}")
            logger.info(f"   SL: ${stop_loss:.2f}")
            
            result = ExecutionResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                message="Live trade executed"
            )
            
            self.execution_history.append(result)
            self.daily_stats.trades += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Live execution error: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                message=f"Execution error: {e}"
            )
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> ExecutionResult:
        """Close a position"""
        if symbol not in self.positions:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="close",
                size=0,
                price=0,
                message="No position to close"
            )
        
        position = self.positions[symbol]
        
        if self.mode == ExecutionMode.PAPER:
            del self.positions[symbol]
            logger.info(f"📝 PAPER CLOSE: {symbol} ({reason})")
            return ExecutionResult(
                success=True,
                order_id=f"paper_close_{int(datetime.utcnow().timestamp()*1000)}",
                symbol=symbol,
                side="close",
                size=position.size,
                price=position.entry_price,
                message=f"Paper position closed: {reason}"
            )
        
        elif self.mode == ExecutionMode.LIVE:
            success = self.executor.close_position(symbol)
            if success:
                del self.positions[symbol]
            return ExecutionResult(
                success=success,
                order_id=None,
                symbol=symbol,
                side="close",
                size=position.size,
                price=position.entry_price,
                message=f"Position closed: {reason}" if success else "Close failed"
            )
        
        return ExecutionResult(
            success=False,
            order_id=None,
            symbol=symbol,
            side="close",
            size=0,
            price=0,
            message="Execution disabled"
        )
    
    def get_status(self) -> Dict:
        """Get execution engine status"""
        return {
            'mode': self.mode.value,
            'configured': self.executor.is_configured,
            'killed': self.killed,
            'kill_reason': self.kill_reason,
            'account_balance': self.account_balance,
            'positions': len(self.positions),
            'daily_stats': {
                'date': str(self.daily_stats.date),
                'trades': self.daily_stats.trades,
                'pnl': self.daily_stats.pnl
            },
            'risk_config': {
                'max_position_pct': self.risk.max_position_pct,
                'max_total_exposure_pct': self.risk.max_total_exposure_pct,
                'max_positions': self.risk.max_positions,
                'max_daily_loss_pct': self.risk.max_daily_loss_pct,
                'default_leverage': self.risk.default_leverage
            }
        }
