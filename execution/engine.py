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
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum

from .kucoin import KuCoinFuturesExecutor, Position
from .strategy_tracker import tracker as strategy_tracker
from orchestrator.signal_aggregator import AggregatedSignal
from agents.base import SignalSide

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    LIVE = "live"
    PAPER = "paper"
    DISABLED = "disabled"

@dataclass
class RiskConfig:
    """Risk management configuration - NO CAPS / KELLY DECIDES"""
    max_position_pct: float = 100.0  # No cap - Kelly/risk manager decides size
    max_total_exposure_pct: float = 100.0  # No cap - full send
    max_positions: int = 50  # Effectively unlimited
    max_daily_loss_pct: float = 15.0  # Kill switch at 15% daily loss - emergency only
    default_leverage: int = 5  # Safe default - some symbols cap at 5x
    default_stop_loss_pct: float = 2.0  # 2% stop loss
    default_take_profit_pct: float = 8.0  # 8% take profit - let winners run far
    min_order_value_usd: float = 10.0
    # DRAWDOWN PROTECTION: 20% account drop = reduce positions
    drawdown_threshold_pct: float = 20.0
    drawdown_reduction_factor: float = 0.8  # Only 20% reduction in drawdown
    # STRATEGY PERFORMANCE MULTIPLIERS (dynamic sizing based on track record)
    strategy_boost_multipliers: Dict = None  # Set at runtime
    
    def __post_init__(self):
        if self.strategy_boost_multipliers is None:
            # Default multipliers: star performers get bigger positions
            self.strategy_boost_multipliers = {
                # Original strategies
                'trend-following': 1.5,  # STAR: 51% WR, +$208 - BOOST 50%
                'mean-reversion': 1.0,
                'turtle': 1.0,
                'weinstein': 1.0,
                'livermore': 1.0,
                'bts-lynch': 0.8,
                'zweig': 0.0,  # DISABLED: 14% WR
                'rsi-divergence': 1.0,
                'stat-arb': 1.0,
                # New futures-specific strategies (Feb 2026) - start conservative
                'funding-fade': 0.8,  # Contrarian - start smaller
                'oi-divergence': 0.8,  # Needs validation
                'liquidation-hunter': 0.6,  # High risk - small size
                'volatility-breakout': 1.0,  # Standard
                'correlation-pairs': 0.7,  # Pairs = split across 2 positions
            }

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
        self.account_balance: float = 0.0  # marginBalance from KuCoin
        self.account_equity: float = 0.0   # accountEquity from KuCoin (includes unrealized)
        self.available_balance: float = 0.0  # availableBalance from KuCoin (free funds)
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.execution_history: List[ExecutionResult] = []
        self.daily_stats: DailyStats = DailyStats(date=date.today())
        
        # Kill switch
        self.killed = False
        self.kill_reason: Optional[str] = None
        
        # DRAWDOWN PROTECTION: Track peak balance for drawdown calculation
        # Reset daily - drawdown is per-day, not all-time
        self.peak_balance: float = 0.0
        self.peak_balance_date: date = date.today()
        self.in_drawdown_mode: bool = False
        self.drawdown_pct: float = 0.0
    
    def refresh_state(self):
        """Refresh account state from exchange"""
        if not self.executor.is_configured:
            logger.warning("Executor not configured - running in paper mode")
            return
        
        try:
            # Get account balance - use marginBalance (actual cash), NOT availableBalance
            # availableBalance = free funds only (excludes margin)
            # marginBalance = actual deposited funds
            # accountEquity = marginBalance + unrealizedPnL
            overview = self.executor.get_account_overview()
            self.account_balance = float(overview.get('marginBalance', 0))
            self.account_equity = float(overview.get('accountEquity', 0))
            self.available_balance = float(overview.get('availableBalance', 0))
            
            # Get positions
            positions = self.executor.get_positions()
            self.positions = {p.symbol: p for p in positions}
            
            # Check daily loss limit
            self._check_daily_loss()
            
            # DRAWDOWN PROTECTION: Check and update drawdown state
            self._check_drawdown()
            
            drawdown_status = f" [DRAWDOWN MODE: {self.drawdown_pct:.1f}%]" if self.in_drawdown_mode else ""
            logger.info(f"State refreshed: ${self.account_balance:.2f} balance, {len(self.positions)} positions{drawdown_status}")
            
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
    
    def _check_drawdown(self):
        """
        DRAWDOWN PROTECTION: Monitor account drawdown from DAILY peak.
        Resets each day - we measure drawdown from day start, not all-time high.
        If drawdown exceeds threshold, reduce position sizes by factor.
        """
        total_equity = self.account_balance + sum(p.unrealized_pnl for p in self.positions.values())
        
        # Reset peak at start of new day
        today = date.today()
        if self.peak_balance_date != today:
            self.peak_balance = total_equity
            self.peak_balance_date = today
            self.in_drawdown_mode = False
            self.drawdown_pct = 0.0
            logger.info(f"📅 New day - daily peak reset to ${self.peak_balance:.2f}")
        
        # Update peak balance (only goes up during the day)
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
            if self.in_drawdown_mode:
                logger.info(f"📈 Exited drawdown mode - new daily peak: ${self.peak_balance:.2f}")
            self.in_drawdown_mode = False
            self.drawdown_pct = 0.0
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            self.drawdown_pct = (self.peak_balance - total_equity) / self.peak_balance * 100
            
            # Enter drawdown mode if threshold exceeded
            if self.drawdown_pct >= self.risk.drawdown_threshold_pct and not self.in_drawdown_mode:
                self.in_drawdown_mode = True
                logger.warning(f"⚠️ DRAWDOWN PROTECTION ACTIVATED: {self.drawdown_pct:.1f}% drawdown from peak ${self.peak_balance:.2f}")
                logger.warning(f"   Position sizes reduced by {(1-self.risk.drawdown_reduction_factor)*100:.0f}%")
    
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
        """
        Calculate position size based on risk parameters.
        
        OPTIMIZATIONS:
        - Strategy performance multipliers (boost winners, reduce losers)
        - Drawdown protection (reduce size when account is down)
        - Consensus bonus (multiple strategies = higher conviction)
        """
        if self.account_balance <= 0:
            return 0
        
        # Base position value
        position_value = self.account_balance * self.risk.max_position_pct / 100
        
        # Adjust by signal confidence
        position_value *= signal.adjusted_confidence
        
        # Adjust by consensus (more strategies agree = larger position)
        position_value *= (0.8 + 0.2 * signal.consensus_score)
        
        # STRATEGY BOOST: Amplify winners, reduce losers based on track record
        strategy_id = signal.signal.strategy_id
        strategy_multiplier = self.risk.strategy_boost_multipliers.get(strategy_id, 1.0)
        if strategy_multiplier == 0:
            logger.info(f"🚫 Blocking signal from disabled strategy: {strategy_id}")
            return 0
        position_value *= strategy_multiplier
        
        # DRAWDOWN PROTECTION: Reduce size when in drawdown
        if self.in_drawdown_mode:
            position_value *= self.risk.drawdown_reduction_factor
            logger.info(f"📉 Drawdown mode: Position reduced by {(1-self.risk.drawdown_reduction_factor)*100:.0f}%")
        
        # Apply leverage
        leveraged_value = position_value * self.risk.default_leverage
        
        # Calculate contracts (for KuCoin, size is in contracts)
        # This is simplified - real implementation needs contract specs
        contracts = int(leveraged_value / price)
        
        # Ensure minimum order value
        if contracts * price < self.risk.min_order_value_usd:
            return 0
        
        # Log sizing decision
        logger.debug(f"Position sizing: {symbol} via {strategy_id}: "
                    f"base=${self.account_balance * self.risk.max_position_pct / 100:.2f}, "
                    f"confidence={signal.adjusted_confidence:.2f}, "
                    f"strategy_mult={strategy_multiplier:.1f}, "
                    f"final=${position_value:.2f}")
        
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
            logger.info(f"   Strategy: {signal.strategy_id}")
            
            # Track strategy attribution
            trade_side = 'long' if side == 'buy' else 'short'
            strategy_tracker.open_position(
                symbol=symbol,
                strategy_id=signal.strategy_id,
                side=trade_side,
                entry_price=price,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
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
                # Track strategy close
                current_price = position.entry_price  # TODO: get actual close price
                strategy_tracker.close_position(symbol, current_price, reason)
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
            'account_equity': self.account_equity,
            'available_balance': self.available_balance,
            'positions': len(self.positions),
            'daily_stats': {
                'date': str(self.daily_stats.date),
                'trades': self.daily_stats.trades,
                'pnl': self.daily_stats.pnl
            },
            'drawdown_protection': {
                'peak_balance': self.peak_balance,
                'current_drawdown_pct': self.drawdown_pct,
                'in_drawdown_mode': self.in_drawdown_mode,
                'threshold_pct': self.risk.drawdown_threshold_pct,
                'reduction_factor': self.risk.drawdown_reduction_factor
            },
            'strategy_multipliers': self.risk.strategy_boost_multipliers,
            'risk_config': {
                'max_position_pct': self.risk.max_position_pct,
                'max_total_exposure_pct': self.risk.max_total_exposure_pct,
                'max_positions': self.risk.max_positions,
                'max_daily_loss_pct': self.risk.max_daily_loss_pct,
                'default_leverage': self.risk.default_leverage
            }
        }
    
    def update_strategy_multipliers(self, performance_data: Dict[str, Dict]):
        """
        Dynamically update strategy multipliers based on actual performance.
        Called by orchestrator when it has fresh performance data.
        
        Args:
            performance_data: Dict of strategy_id -> {'win_rate': float, 'total_pnl': float, 'trades': int}
        """
        for strategy_id, perf in performance_data.items():
            win_rate = perf.get('win_rate', 0.5)
            total_pnl = perf.get('total_pnl', 0)
            trades = perf.get('trades', 0)
            
            # Need minimum trades to adjust
            if trades < 10:
                continue
            
            # Calculate multiplier based on performance
            # Base: 1.0, Range: 0.0 (disabled) to 2.0 (double size)
            if win_rate < 0.35:
                # Terrible WR (<35%) - disable this strategy
                multiplier = 0.0
                logger.warning(f"🚫 Disabling {strategy_id}: {win_rate:.0%} WR is unacceptable")
            elif win_rate < 0.45:
                # Poor WR - heavily reduce
                multiplier = 0.5
            elif total_pnl < 0:
                # Negative P&L despite decent WR - bad R:R, reduce
                multiplier = 0.7
            elif win_rate >= 0.50 and total_pnl > 0:
                # Winner: good WR + positive P&L
                multiplier = 1.0 + min(0.5, total_pnl / 500)  # Up to 1.5x
            else:
                multiplier = 1.0
            
            self.risk.strategy_boost_multipliers[strategy_id] = multiplier
            logger.info(f"📊 Strategy {strategy_id}: WR={win_rate:.0%}, PnL=${total_pnl:.0f} -> multiplier={multiplier:.1f}x")
