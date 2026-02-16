"""
Risk Manager - Central risk control for Cash Town

Responsibilities:
1. Position sizing (Kelly Criterion / Fixed Fractional)
2. Portfolio heat tracking (total risk exposure)
3. Correlation detection (avoid overloading same bet)
4. Circuit breakers (halt trading on drawdown)
5. Volatility-adjusted sizing

Philosophy: Protect capital first, maximize risk-adjusted returns second.
"""
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from enum import Enum
import os

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))

# Asset correlation groups - assets that move together
CORRELATION_GROUPS = {
    'btc_ecosystem': {'XBTUSDTM', 'BTCUSDTM'},
    'eth_ecosystem': {'ETHUSDTM'},
    'alt_l1': {'SOLUSDTM', 'AVAXUSDTM', 'NEARUSDTM', 'APTUSDTM', 'SUIUSDTM', 'TONUSDTM', 'ICPUSDTM'},
    'defi': {'UNIUSDTM', 'LINKUSDTM', 'INJUSDTM'},
    'l2': {'MATICUSDTM', 'ARBUSDTM', 'OPUSDTM'},
    'cosmos': {'ATOMUSDTM', 'TIAUSDTM'},
    'old_guard': {'LTCUSDTM', 'BCHUSDTM', 'XRPUSDTM', 'ADAUSDTM', 'DOTUSDTM'},
    'storage': {'FILUSDTM', 'RENDERUSDTM'},
}

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VolatilityData:
    """Volatility tracking for an asset"""
    symbol: str
    current_vol: float = 0.0  # Rolling 24h volatility %
    avg_vol: float = 0.0      # 30-day average volatility
    vol_regime: str = "normal"  # low, normal, high, extreme
    last_updated: datetime = None
    
    def is_high_vol(self) -> bool:
        return self.vol_regime in ("high", "extreme")


@dataclass
class PortfolioHeat:
    """Track total portfolio risk exposure"""
    total_risk_pct: float = 0.0      # Total portfolio % at risk
    max_risk_pct: float = 10.0       # Maximum allowed
    position_count: int = 0
    correlated_exposure: Dict[str, float] = field(default_factory=dict)  # Group -> exposure %
    
    def is_overheated(self) -> bool:
        return self.total_risk_pct >= self.max_risk_pct * 0.9
    
    def headroom_pct(self) -> float:
        return max(0, self.max_risk_pct - self.total_risk_pct)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    is_triggered: bool = False
    trigger_reason: str = ""
    triggered_at: datetime = None
    cooldown_until: datetime = None
    daily_loss_pct: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    
    def can_trade(self) -> Tuple[bool, str]:
        if self.is_triggered:
            if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.utcnow()).total_seconds() / 60
                return False, f"Circuit breaker: {self.trigger_reason} (cooldown {remaining:.0f}m)"
            # Auto-reset after cooldown
            self.is_triggered = False
            self.trigger_reason = ""
        return True, "OK"


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position sizing
    max_position_risk_pct: float = 2.0      # Max risk per position (% of equity)
    max_total_risk_pct: float = 10.0        # Max total portfolio risk
    default_stop_loss_pct: float = 2.0      # Default stop distance
    
    # Kelly Criterion settings
    use_kelly: bool = True
    kelly_fraction: float = 0.25            # Fraction of Kelly to use (conservative)
    min_trades_for_kelly: int = 20          # Need this many trades before using Kelly
    
    # Correlation limits
    max_correlated_exposure_pct: float = 4.0  # Max exposure to correlated group
    max_same_direction_positions: int = 4     # Max all-long or all-short
    
    # Circuit breakers
    max_daily_loss_pct: float = 5.0         # Halt if daily loss exceeds this
    max_drawdown_pct: float = 15.0          # Halt if drawdown from peak exceeds this
    circuit_breaker_cooldown_hours: float = 4.0  # How long to pause after trigger
    
    # Volatility adjustments
    high_vol_reduction: float = 0.5         # Reduce size by 50% in high vol
    extreme_vol_reduction: float = 0.25     # Reduce size by 75% in extreme vol
    vol_lookback_hours: int = 24


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    risk_amount: float          # $ at risk
    risk_pct: float             # % of portfolio at risk
    correlation_group: str
    strategy_id: str


class RiskManager:
    """
    Central risk management for Cash Town.
    
    All signals must pass through here before execution.
    Protects capital through:
    - Position sizing limits
    - Portfolio heat tracking
    - Correlation awareness
    - Circuit breakers
    - Volatility scaling
    """
    
    def __init__(self, config: RiskConfig = None, equity: float = 0.0):
        self.config = config or RiskConfig()
        self.equity = equity
        self.peak_equity = equity
        
        # State
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_heat = PortfolioHeat(max_risk_pct=self.config.max_total_risk_pct)
        self.circuit_breaker = CircuitBreakerState()
        self.volatility: Dict[str, VolatilityData] = {}
        
        # Performance tracking for Kelly
        self.strategy_stats: Dict[str, Dict] = {}  # strategy_id -> {wins, losses, avg_win, avg_loss}
        
        # Daily tracking
        self.daily_stats = {
            'date': date.today().isoformat(),
            'starting_equity': equity,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0
        }
        
        # Load persisted state
        self._load_state()
        
        logger.info(f"🛡️ RiskManager initialized: equity=${equity:.2f}, max_risk={self.config.max_total_risk_pct}%")
    
    def update_equity(self, new_equity: float):
        """Update current equity and check drawdown"""
        self.equity = new_equity
        
        # Track peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            if self.circuit_breaker.current_drawdown_pct > 5:
                logger.info(f"📈 New equity peak: ${new_equity:.2f} - recovered from drawdown")
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.circuit_breaker.current_drawdown_pct = (
                (self.peak_equity - new_equity) / self.peak_equity * 100
            )
        
        # Check circuit breaker
        self._check_circuit_breakers()
        
        # Reset daily stats if new day
        today = date.today().isoformat()
        if self.daily_stats['date'] != today:
            self._reset_daily_stats()
    
    def update_volatility(self, symbol: str, prices: List[float]):
        """Update volatility data for a symbol from price history"""
        if len(prices) < 2:
            return
        
        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 
                   for i in range(1, len(prices))]
        
        # Calculate realized volatility (std dev of returns)
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            vol = math.sqrt(variance)
            
            # Classify regime
            if vol < 1.0:
                regime = "low"
            elif vol < 3.0:
                regime = "normal"
            elif vol < 6.0:
                regime = "high"
            else:
                regime = "extreme"
            
            # Update or create
            if symbol not in self.volatility:
                self.volatility[symbol] = VolatilityData(symbol=symbol)
            
            self.volatility[symbol].current_vol = vol
            self.volatility[symbol].vol_regime = regime
            self.volatility[symbol].last_updated = datetime.utcnow()
    
    def can_open_position(self, symbol: str, side: str, strategy_id: str) -> Tuple[bool, str]:
        """
        Check if a new position is allowed.
        
        Returns:
            (allowed, reason)
        """
        # Check circuit breaker
        can_trade, reason = self.circuit_breaker.can_trade()
        if not can_trade:
            return False, reason
        
        # Check portfolio heat
        if self.portfolio_heat.is_overheated():
            return False, f"Portfolio overheated: {self.portfolio_heat.total_risk_pct:.1f}% risk (max {self.config.max_total_risk_pct}%)"
        
        # Check correlation exposure
        group = self._get_correlation_group(symbol)
        if group:
            current_exposure = self.portfolio_heat.correlated_exposure.get(group, 0)
            if current_exposure >= self.config.max_correlated_exposure_pct:
                return False, f"Max exposure to {group}: {current_exposure:.1f}% (limit {self.config.max_correlated_exposure_pct}%)"
        
        # Check same-direction concentration
        long_count = sum(1 for p in self.positions.values() if p.side == 'long')
        short_count = sum(1 for p in self.positions.values() if p.side == 'short')
        
        if side == 'long' and long_count >= self.config.max_same_direction_positions:
            return False, f"Max long positions reached ({long_count})"
        if side == 'short' and short_count >= self.config.max_same_direction_positions:
            return False, f"Max short positions reached ({short_count})"
        
        # Check if already in position
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing.side == side:
                return False, f"Already {side} {symbol}"
            # Opposite direction = close existing first
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        side: str,
        stop_loss: float,
        strategy_id: str,
        base_confidence: float
    ) -> Tuple[float, Dict]:
        """
        Calculate optimal position size.
        
        Uses Kelly Criterion if enough data, otherwise fixed fractional.
        Adjusts for volatility and portfolio heat.
        
        Returns:
            (position_value_usd, metadata_dict)
        """
        if self.equity <= 0:
            return 0, {'reason': 'no_equity'}
        
        metadata = {
            'method': 'fixed_fractional',
            'base_risk_pct': self.config.max_position_risk_pct,
            'adjustments': []
        }
        
        # Calculate risk per share (distance to stop)
        if stop_loss:
            if side == 'long':
                risk_per_unit = max(0.001, (price - stop_loss) / price * 100)
            else:
                risk_per_unit = max(0.001, (stop_loss - price) / price * 100)
        else:
            risk_per_unit = self.config.default_stop_loss_pct
        
        # Base position size using fixed fractional
        risk_amount = self.equity * self.config.max_position_risk_pct / 100
        position_value = risk_amount / (risk_per_unit / 100)
        
        # Try Kelly Criterion if we have enough data
        kelly_size = self._calculate_kelly_size(strategy_id, risk_per_unit)
        if kelly_size is not None:
            # Use minimum of Kelly and fixed fractional (conservative)
            if kelly_size < position_value:
                position_value = kelly_size
                metadata['method'] = 'kelly_criterion'
                metadata['kelly_fraction'] = self.config.kelly_fraction
        
        # Adjust for confidence
        confidence_mult = 0.5 + 0.5 * base_confidence  # Range: 0.5 to 1.0
        position_value *= confidence_mult
        metadata['adjustments'].append(f"confidence={base_confidence:.0%} -> {confidence_mult:.2f}x")
        
        # Adjust for volatility
        vol_data = self.volatility.get(symbol)
        if vol_data and vol_data.is_high_vol():
            if vol_data.vol_regime == "extreme":
                vol_mult = self.config.extreme_vol_reduction
            else:
                vol_mult = self.config.high_vol_reduction
            position_value *= vol_mult
            metadata['adjustments'].append(f"volatility={vol_data.vol_regime} -> {vol_mult:.2f}x")
        
        # Adjust for portfolio heat (reduce size as we approach limit)
        headroom = self.portfolio_heat.headroom_pct()
        if headroom < self.config.max_position_risk_pct:
            heat_mult = headroom / self.config.max_position_risk_pct
            position_value *= heat_mult
            metadata['adjustments'].append(f"portfolio_heat -> {heat_mult:.2f}x")
        
        # Adjust for correlation exposure
        group = self._get_correlation_group(symbol)
        if group:
            current_exposure = self.portfolio_heat.correlated_exposure.get(group, 0)
            remaining = self.config.max_correlated_exposure_pct - current_exposure
            if remaining < self.config.max_position_risk_pct:
                corr_mult = max(0, remaining / self.config.max_position_risk_pct)
                position_value *= corr_mult
                metadata['adjustments'].append(f"correlation_{group} -> {corr_mult:.2f}x")
        
        # Ensure minimum viable size
        min_size = 10.0  # $10 minimum
        if position_value < min_size:
            return 0, {'reason': f'size_too_small: ${position_value:.2f}'}
        
        # Calculate final risk %
        final_risk_pct = (risk_per_unit / 100) * position_value / self.equity * 100
        metadata['final_risk_pct'] = final_risk_pct
        metadata['position_value'] = position_value
        
        logger.info(f"📐 Position size: {symbol} ${position_value:.2f} ({metadata['method']}, risk={final_risk_pct:.2f}%)")
        for adj in metadata['adjustments']:
            logger.debug(f"   ↳ {adj}")
        
        return position_value, metadata
    
    def _calculate_kelly_size(self, strategy_id: str, risk_pct: float) -> Optional[float]:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - (1-W)/R
        Where:
            W = Win rate
            R = Win/Loss ratio (avg win / avg loss)
        
        Returns None if not enough data.
        """
        stats = self.strategy_stats.get(strategy_id)
        if not stats or stats.get('trades', 0) < self.config.min_trades_for_kelly:
            return None
        
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        total = wins + losses
        
        if total == 0 or losses == 0:
            return None
        
        win_rate = wins / total
        avg_win = stats.get('avg_win_pct', 2.0)
        avg_loss = abs(stats.get('avg_loss_pct', 2.0))
        
        if avg_loss == 0:
            return None
        
        # Kelly formula
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fraction (never bet full Kelly)
        kelly_pct *= self.config.kelly_fraction
        
        # Clamp to reasonable range
        kelly_pct = max(0, min(kelly_pct, self.config.max_position_risk_pct))
        
        if kelly_pct <= 0:
            logger.debug(f"Kelly says don't trade {strategy_id}: W={win_rate:.0%}, R={win_loss_ratio:.2f}")
            return 0
        
        # Convert to position value
        risk_amount = self.equity * kelly_pct / 100
        position_value = risk_amount / (risk_pct / 100)
        
        logger.debug(f"Kelly for {strategy_id}: W={win_rate:.0%}, R={win_loss_ratio:.2f} -> {kelly_pct:.2f}% risk")
        
        return position_value
    
    def register_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        strategy_id: str
    ):
        """Register a new position with the risk manager"""
        # Calculate risk
        if side == 'long':
            risk_pct = (entry_price - stop_loss) / entry_price * 100
        else:
            risk_pct = (stop_loss - entry_price) / entry_price * 100
        
        risk_amount = size * entry_price * risk_pct / 100
        portfolio_risk_pct = risk_amount / self.equity * 100 if self.equity > 0 else 0
        
        position = PositionRisk(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            risk_amount=risk_amount,
            risk_pct=portfolio_risk_pct,
            correlation_group=self._get_correlation_group(symbol) or 'other',
            strategy_id=strategy_id
        )
        
        self.positions[symbol] = position
        self._update_portfolio_heat()
        self.daily_stats['trades'] += 1
        
        logger.info(f"🛡️ Registered position: {side} {symbol}, risk={portfolio_risk_pct:.2f}% of portfolio")
        self._save_state()
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> Optional[Dict]:
        """
        Close a position and record the result.
        
        Returns trade result dict for learning.
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == 'long':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100
        
        pnl_amount = position.size * position.entry_price * pnl_pct / 100
        won = pnl_pct > 0
        
        # Update daily stats
        self.daily_stats['pnl'] += pnl_amount
        if won:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1
        
        # Update strategy stats for Kelly
        self._update_strategy_stats(position.strategy_id, won, pnl_pct)
        
        # Remove position
        del self.positions[symbol]
        self._update_portfolio_heat()
        
        # Check circuit breakers
        self._check_circuit_breakers()
        
        result = {
            'symbol': symbol,
            'side': position.side,
            'strategy_id': position.strategy_id,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'won': won,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        emoji = "✅" if won else "❌"
        logger.info(f"{emoji} Closed {position.side} {symbol}: {pnl_pct:+.2f}% (${pnl_amount:+.2f})")
        
        self._save_state()
        return result
    
    def _update_strategy_stats(self, strategy_id: str, won: bool, pnl_pct: float):
        """Update strategy statistics for Kelly calculation"""
        if strategy_id not in self.strategy_stats:
            self.strategy_stats[strategy_id] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_win_pct': 0,
                'total_loss_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0
            }
        
        stats = self.strategy_stats[strategy_id]
        stats['trades'] += 1
        
        if won:
            stats['wins'] += 1
            stats['total_win_pct'] += pnl_pct
            stats['avg_win_pct'] = stats['total_win_pct'] / stats['wins']
        else:
            stats['losses'] += 1
            stats['total_loss_pct'] += abs(pnl_pct)
            stats['avg_loss_pct'] = stats['total_loss_pct'] / stats['losses']
        
        logger.debug(f"Strategy {strategy_id} stats: {stats['wins']}/{stats['trades']} wins, "
                    f"avg_win={stats['avg_win_pct']:.2f}%, avg_loss={stats['avg_loss_pct']:.2f}%")
    
    def _update_portfolio_heat(self):
        """Recalculate portfolio heat"""
        total_risk = sum(p.risk_pct for p in self.positions.values())
        
        # Calculate correlation group exposure
        group_exposure = {}
        for position in self.positions.values():
            group = position.correlation_group
            group_exposure[group] = group_exposure.get(group, 0) + position.risk_pct
        
        self.portfolio_heat.total_risk_pct = total_risk
        self.portfolio_heat.position_count = len(self.positions)
        self.portfolio_heat.correlated_exposure = group_exposure
        
        if total_risk > self.config.max_total_risk_pct * 0.8:
            logger.warning(f"⚠️ Portfolio heat high: {total_risk:.1f}% (max {self.config.max_total_risk_pct}%)")
    
    def _check_circuit_breakers(self):
        """Check and trigger circuit breakers if needed"""
        now = datetime.utcnow()
        
        # Check daily loss
        if self.daily_stats['starting_equity'] > 0:
            daily_loss_pct = -self.daily_stats['pnl'] / self.daily_stats['starting_equity'] * 100
            self.circuit_breaker.daily_loss_pct = daily_loss_pct
            
            if daily_loss_pct >= self.config.max_daily_loss_pct and not self.circuit_breaker.is_triggered:
                self._trigger_circuit_breaker(f"Daily loss limit: {daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%")
                return
        
        # Check drawdown
        if self.circuit_breaker.current_drawdown_pct >= self.config.max_drawdown_pct and not self.circuit_breaker.is_triggered:
            self._trigger_circuit_breaker(f"Max drawdown: {self.circuit_breaker.current_drawdown_pct:.1f}% >= {self.config.max_drawdown_pct}%")
            return
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger the circuit breaker"""
        now = datetime.utcnow()
        cooldown_hours = self.config.circuit_breaker_cooldown_hours
        
        self.circuit_breaker.is_triggered = True
        self.circuit_breaker.trigger_reason = reason
        self.circuit_breaker.triggered_at = now
        self.circuit_breaker.cooldown_until = now + timedelta(hours=cooldown_hours)
        
        logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"   Trading halted until {self.circuit_breaker.cooldown_until.isoformat()}")
        
        self._save_state()
    
    def _get_correlation_group(self, symbol: str) -> Optional[str]:
        """Find which correlation group a symbol belongs to"""
        for group, symbols in CORRELATION_GROUPS.items():
            if symbol in symbols:
                return group
        return None
    
    def _reset_daily_stats(self):
        """Reset daily statistics for new trading day"""
        self.daily_stats = {
            'date': date.today().isoformat(),
            'starting_equity': self.equity,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0
        }
        
        # Reset daily circuit breaker (but not drawdown-based)
        if 'Daily loss' in self.circuit_breaker.trigger_reason:
            self.circuit_breaker.is_triggered = False
            self.circuit_breaker.trigger_reason = ""
            logger.info("🌅 New trading day - daily loss circuit breaker reset")
    
    def get_risk_check(self, symbol: str, side: str, strategy_id: str) -> Dict:
        """
        Comprehensive risk check for a potential trade.
        
        Returns dict with all risk factors and recommendation.
        """
        can_trade, reason = self.can_open_position(symbol, side, strategy_id)
        
        group = self._get_correlation_group(symbol)
        vol_data = self.volatility.get(symbol)
        
        return {
            'symbol': symbol,
            'side': side,
            'strategy_id': strategy_id,
            'allowed': can_trade,
            'reason': reason,
            'portfolio_heat': {
                'current': self.portfolio_heat.total_risk_pct,
                'max': self.config.max_total_risk_pct,
                'headroom': self.portfolio_heat.headroom_pct()
            },
            'correlation': {
                'group': group,
                'group_exposure': self.portfolio_heat.correlated_exposure.get(group, 0) if group else 0,
                'max_group_exposure': self.config.max_correlated_exposure_pct
            },
            'volatility': {
                'regime': vol_data.vol_regime if vol_data else 'unknown',
                'current': vol_data.current_vol if vol_data else 0
            },
            'circuit_breaker': {
                'triggered': self.circuit_breaker.is_triggered,
                'reason': self.circuit_breaker.trigger_reason,
                'daily_loss_pct': self.circuit_breaker.daily_loss_pct,
                'drawdown_pct': self.circuit_breaker.current_drawdown_pct
            },
            'kelly': self.strategy_stats.get(strategy_id, {})
        }
    
    def get_status(self) -> Dict:
        """Get comprehensive risk manager status"""
        return {
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'portfolio_heat': {
                'total_risk_pct': self.portfolio_heat.total_risk_pct,
                'max_risk_pct': self.portfolio_heat.max_risk_pct,
                'position_count': self.portfolio_heat.position_count,
                'correlated_exposure': self.portfolio_heat.correlated_exposure,
                'is_overheated': self.portfolio_heat.is_overheated()
            },
            'circuit_breaker': {
                'is_triggered': self.circuit_breaker.is_triggered,
                'trigger_reason': self.circuit_breaker.trigger_reason,
                'daily_loss_pct': self.circuit_breaker.daily_loss_pct,
                'drawdown_pct': self.circuit_breaker.current_drawdown_pct,
                'cooldown_until': self.circuit_breaker.cooldown_until.isoformat() if self.circuit_breaker.cooldown_until else None
            },
            'daily_stats': self.daily_stats,
            'strategy_stats': self.strategy_stats,
            'positions': {
                symbol: {
                    'side': p.side,
                    'risk_pct': p.risk_pct,
                    'correlation_group': p.correlation_group,
                    'strategy': p.strategy_id
                }
                for symbol, p in self.positions.items()
            },
            'config': {
                'max_position_risk_pct': self.config.max_position_risk_pct,
                'max_total_risk_pct': self.config.max_total_risk_pct,
                'use_kelly': self.config.use_kelly,
                'kelly_fraction': self.config.kelly_fraction,
                'max_daily_loss_pct': self.config.max_daily_loss_pct,
                'max_drawdown_pct': self.config.max_drawdown_pct
            }
        }
    
    def _load_state(self):
        """Load persisted state"""
        state_file = DATA_DIR / 'risk_manager_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.strategy_stats = state.get('strategy_stats', {})
                self.peak_equity = state.get('peak_equity', self.equity)
                
                # Restore circuit breaker if still in cooldown
                cb = state.get('circuit_breaker', {})
                if cb.get('cooldown_until'):
                    cooldown = datetime.fromisoformat(cb['cooldown_until'])
                    if cooldown > datetime.utcnow():
                        self.circuit_breaker.is_triggered = cb.get('is_triggered', False)
                        self.circuit_breaker.trigger_reason = cb.get('trigger_reason', '')
                        self.circuit_breaker.cooldown_until = cooldown
                        logger.warning(f"🛑 Circuit breaker still active: {self.circuit_breaker.trigger_reason}")
                
                logger.info(f"Loaded risk state: {len(self.strategy_stats)} strategies tracked")
            except Exception as e:
                logger.error(f"Error loading risk state: {e}")
    
    def _save_state(self):
        """Persist state to disk"""
        state_file = DATA_DIR / 'risk_manager_state.json'
        try:
            state = {
                'strategy_stats': self.strategy_stats,
                'peak_equity': self.peak_equity,
                'circuit_breaker': {
                    'is_triggered': self.circuit_breaker.is_triggered,
                    'trigger_reason': self.circuit_breaker.trigger_reason,
                    'cooldown_until': self.circuit_breaker.cooldown_until.isoformat() if self.circuit_breaker.cooldown_until else None
                },
                'daily_stats': self.daily_stats,
                'saved_at': datetime.utcnow().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")


def create_risk_manager(equity: float = 0.0, aggressive: bool = False) -> RiskManager:
    """Factory function to create risk manager with preset configs"""
    if aggressive:
        config = RiskConfig(
            max_position_risk_pct=3.0,
            max_total_risk_pct=15.0,
            kelly_fraction=0.35,
            max_daily_loss_pct=7.0,
            max_drawdown_pct=20.0
        )
    else:
        config = RiskConfig()  # Conservative defaults
    
    return RiskManager(config=config, equity=equity)
