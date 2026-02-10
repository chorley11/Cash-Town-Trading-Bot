"""
Backtesting Engine for Gas Town Agents

Allows agents to:
1. Test strategy logic against historical data
2. Compare parameter variations
3. Validate improvements before deployment
"""
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
import numpy as np
from copy import deepcopy

from .base import BaseStrategyAgent, Signal, SignalSide, Trade

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    position_size_pct: float = 2.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    max_positions: int = 5
    
@dataclass
class BacktestTrade:
    """Record of a simulated trade"""
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime = None
    exit_price: float = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    close_reason: str = ''
    signal_confidence: float = 0.0

@dataclass 
class BacktestResult:
    """Results from a backtest run"""
    agent_id: str
    config_hash: str
    period: str
    
    # Performance metrics
    total_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    
    # Trade stats
    avg_win: float
    avg_loss: float
    avg_hold_time_hours: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Risk metrics
    win_loss_ratio: float
    expectancy: float
    
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['trades'] = [asdict(t) for t in self.trades]
        return result

class Backtester:
    """
    Backtesting engine for strategy agents.
    
    Usage:
        backtester = Backtester(agent)
        result = backtester.run(historical_data, config)
    """
    
    def __init__(self, agent: BaseStrategyAgent):
        self.agent = agent
        self.original_config = deepcopy(agent.config)
    
    def run(self, historical_data: Dict[str, Dict], 
            config: BacktestConfig,
            parameter_overrides: Dict[str, Any] = None) -> BacktestResult:
        """
        Run backtest with given historical data.
        
        Args:
            historical_data: Dict of symbol -> OHLCV data
            config: BacktestConfig with settings
            parameter_overrides: Optional dict of strategy parameters to test
        
        Returns:
            BacktestResult with performance metrics
        """
        # Apply parameter overrides if provided
        if parameter_overrides:
            for key, value in parameter_overrides.items():
                self.agent.config[key] = value
        
        config_hash = self._hash_config(self.agent.config)
        
        # Initialize state
        capital = config.initial_capital
        equity_curve = [capital]
        positions: Dict[str, BacktestTrade] = {}
        closed_trades: List[BacktestTrade] = []
        
        # Get all timestamps from data
        all_timestamps = self._get_timestamps(historical_data)
        
        for i, timestamp in enumerate(all_timestamps):
            # Build market data snapshot
            market_snapshot = self._build_snapshot(historical_data, i)
            
            if not market_snapshot:
                equity_curve.append(equity_curve[-1])
                continue
            
            # Check existing positions for exits
            for symbol, position in list(positions.items()):
                if symbol in market_snapshot:
                    current_price = market_snapshot[symbol]['close'][-1]
                    should_exit, reason = self._check_exit(position, current_price, market_snapshot[symbol])
                    
                    if should_exit:
                        # Close position
                        position.exit_time = timestamp
                        position.exit_price = current_price * (1 - config.slippage_pct/100 if position.side == 'long' else 1 + config.slippage_pct/100)
                        
                        if position.side == 'long':
                            position.pnl = (position.exit_price - position.entry_price) * position.size
                        else:
                            position.pnl = (position.entry_price - position.exit_price) * position.size
                        
                        position.pnl -= position.size * position.entry_price * config.commission_pct / 100 * 2  # Entry + exit
                        position.pnl_percent = position.pnl / (position.entry_price * position.size) * 100
                        position.close_reason = reason
                        
                        capital += position.pnl
                        closed_trades.append(position)
                        del positions[symbol]
            
            # Generate new signals
            signals = self.agent.generate_signals(market_snapshot)
            
            # Process signals (entry)
            for signal in signals:
                if signal.symbol in positions:
                    continue  # Already have position
                
                if len(positions) >= config.max_positions:
                    continue  # Max positions reached
                
                # Calculate position size
                position_value = capital * config.position_size_pct / 100
                entry_price = signal.price * (1 + config.slippage_pct/100 if signal.side == SignalSide.LONG else 1 - config.slippage_pct/100)
                size = position_value / entry_price
                
                # Open position
                trade = BacktestTrade(
                    symbol=signal.symbol,
                    side='long' if signal.side == SignalSide.LONG else 'short',
                    entry_time=timestamp,
                    entry_price=entry_price,
                    size=size,
                    signal_confidence=signal.confidence
                )
                
                # Store stop loss / take profit in metadata
                trade._stop_loss = signal.stop_loss
                trade._take_profit = signal.take_profit
                
                positions[signal.symbol] = trade
            
            # Update equity
            unrealized = sum(
                (market_snapshot[p.symbol]['close'][-1] - p.entry_price) * p.size if p.side == 'long'
                else (p.entry_price - market_snapshot[p.symbol]['close'][-1]) * p.size
                for p in positions.values()
                if p.symbol in market_snapshot
            )
            equity_curve.append(capital + unrealized)
        
        # Close any remaining positions
        for symbol, position in positions.items():
            if symbol in market_snapshot:
                position.exit_time = all_timestamps[-1]
                position.exit_price = market_snapshot[symbol]['close'][-1]
                if position.side == 'long':
                    position.pnl = (position.exit_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - position.exit_price) * position.size
                position.pnl_percent = position.pnl / (position.entry_price * position.size) * 100
                position.close_reason = 'backtest_end'
                closed_trades.append(position)
        
        # Restore original config
        self.agent.config = deepcopy(self.original_config)
        
        # Calculate metrics
        return self._calculate_metrics(
            agent_id=self.agent.agent_id,
            config_hash=config_hash,
            config=config,
            trades=closed_trades,
            equity_curve=equity_curve
        )
    
    def _get_timestamps(self, data: Dict) -> List[datetime]:
        """Get all unique timestamps from data"""
        timestamps = set()
        for symbol_data in data.values():
            if 'timestamp' in symbol_data:
                timestamps.update(symbol_data['timestamp'])
        return sorted(timestamps)
    
    def _build_snapshot(self, data: Dict, index: int) -> Dict[str, Dict]:
        """Build market data snapshot at given index"""
        snapshot = {}
        for symbol, symbol_data in data.items():
            if index < len(symbol_data.get('close', [])):
                # Include data up to this point
                snapshot[symbol] = {
                    'open': symbol_data['open'][:index+1],
                    'high': symbol_data['high'][:index+1],
                    'low': symbol_data['low'][:index+1],
                    'close': symbol_data['close'][:index+1],
                    'volume': symbol_data['volume'][:index+1],
                }
        return snapshot
    
    def _check_exit(self, position: BacktestTrade, current_price: float, 
                    data: Dict) -> tuple[bool, str]:
        """Check if position should be exited"""
        stop_loss = getattr(position, '_stop_loss', None)
        take_profit = getattr(position, '_take_profit', None)
        
        if position.side == 'long':
            if stop_loss and current_price <= stop_loss:
                return True, 'stop_loss'
            if take_profit and current_price >= take_profit:
                return True, 'take_profit'
        else:
            if stop_loss and current_price >= stop_loss:
                return True, 'stop_loss'
            if take_profit and current_price <= take_profit:
                return True, 'take_profit'
        
        return False, ''
    
    def _calculate_metrics(self, agent_id: str, config_hash: str,
                          config: BacktestConfig, trades: List[BacktestTrade],
                          equity_curve: List[float]) -> BacktestResult:
        """Calculate all backtest metrics"""
        if not trades:
            return BacktestResult(
                agent_id=agent_id,
                config_hash=config_hash,
                period=f"{config.start_date.date()} to {config.end_date.date()}",
                total_trades=0,
                win_rate=0,
                total_pnl=0,
                total_return_pct=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                avg_hold_time_hours=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                win_loss_ratio=0,
                expectancy=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        # Basic stats
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100
        
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Hold time
        hold_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                hold_times.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        # Consecutive wins/losses
        max_wins, max_losses = 0, 0
        current_wins, current_losses = 0, 0
        for t in trades:
            if t.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Returns for Sharpe
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Total return
        total_return_pct = (equity_curve[-1] - config.initial_capital) / config.initial_capital * 100
        
        return BacktestResult(
            agent_id=agent_id,
            config_hash=config_hash,
            period=f"{config.start_date.date()} to {config.end_date.date()}",
            total_trades=len(trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_time_hours=avg_hold_time,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def _hash_config(self, config: Dict) -> str:
        """Create hash of config for comparison"""
        import hashlib
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def compare_parameters(self, historical_data: Dict[str, Dict],
                          config: BacktestConfig,
                          parameter_variations: Dict[str, List[Any]]) -> List[BacktestResult]:
        """
        Compare multiple parameter variations.
        
        Args:
            parameter_variations: Dict of parameter name -> list of values to test
                                  e.g. {'stop_loss_atr': [1.5, 2.0, 2.5]}
        
        Returns:
            List of BacktestResults for each variation
        """
        results = []
        
        # Generate all combinations
        from itertools import product
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())
        
        for values in product(*param_values):
            overrides = dict(zip(param_names, values))
            result = self.run(historical_data, config, overrides)
            result.config_hash = str(overrides)  # Store variation for reference
            results.append(result)
        
        # Sort by total return
        results.sort(key=lambda r: r.total_return_pct, reverse=True)
        
        return results
