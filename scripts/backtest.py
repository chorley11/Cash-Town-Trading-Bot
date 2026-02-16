#!/usr/bin/env python3
"""
Cash Town Backtesting Framework

Backtest trading strategies on historical data to validate signal generation
and estimate performance before deploying to production.

Usage:
    python scripts/backtest.py --strategy trend-following --symbol BTC-USDT --days 30
    python scripts/backtest.py --strategy all --symbol BTC-USDT --days 90 --output results.json
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests

from agents.base import Signal, SignalSide, Position
from agents.strategies.trend_following import TrendFollowingAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a simulated trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy_id: str
    confidence: float
    holding_time_hours: float


@dataclass 
class BacktestResult:
    """Results from a backtest run"""
    strategy_id: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_hours: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    
    Features:
    - Simulates trading with realistic execution
    - Tracks equity curve and drawdowns
    - Calculates key performance metrics
    - Supports multiple strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_pct: float = 5.0,
        slippage_pct: float = 0.1,
        commission_pct: float = 0.1
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        
        self.capital = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def run(
        self,
        strategy,
        historical_data: Dict[str, Any],
        start_idx: int = 50
    ) -> BacktestResult:
        """
        Run backtest for a strategy on historical data.
        
        Args:
            strategy: Strategy instance (must implement generate_signals)
            historical_data: Dict with OHLCV data
            start_idx: Index to start backtesting (allow warmup for indicators)
        
        Returns:
            BacktestResult with performance metrics
        """
        self._reset()
        
        symbol = strategy.symbols[0]
        data = historical_data.get(symbol, {})
        
        if not data or 'close' not in data:
            logger.error(f"No data found for {symbol}")
            return self._empty_result(strategy.agent_id, symbol)
        
        closes = data['close']
        highs = data['high']
        lows = data['low']
        n_candles = len(closes)
        
        logger.info(f"Running backtest for {strategy.agent_id} on {symbol}")
        logger.info(f"Data: {n_candles} candles, starting at index {start_idx}")
        
        for i in range(start_idx, n_candles):
            # Prepare slice of data up to current candle
            current_data = {
                symbol: {
                    'open': data['open'][:i+1],
                    'high': data['high'][:i+1],
                    'low': data['low'][:i+1],
                    'close': data['close'][:i+1],
                    'volume': data['volume'][:i+1],
                }
            }
            
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            
            # Check for position exits first
            if self.position:
                exit_reason = self._check_exit(
                    self.position, 
                    current_price,
                    current_high,
                    current_low,
                    strategy
                )
                if exit_reason:
                    self._close_position(current_price, exit_reason, i)
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            
            # Execute signals if no position
            if not self.position and signals:
                best_signal = max(signals, key=lambda s: s.confidence)
                self._open_position(best_signal, current_price, i)
            
            # Update equity curve
            equity = self.capital
            if self.position:
                equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
        
        # Close any remaining position
        if self.position:
            self._close_position(closes[-1], 'end_of_data', n_candles - 1)
        
        return self._calculate_result(strategy.agent_id, symbol, data)
    
    def _reset(self):
        """Reset engine state for new backtest"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
    
    def _open_position(self, signal: Signal, price: float, candle_idx: int):
        """Open a new position"""
        # Apply slippage
        if signal.side == SignalSide.LONG:
            entry_price = price * (1 + self.slippage_pct / 100)
        else:
            entry_price = price * (1 - self.slippage_pct / 100)
        
        # Calculate position size
        position_value = self.capital * (self.position_size_pct / 100)
        size = position_value / entry_price
        
        # Apply commission
        commission = position_value * (self.commission_pct / 100)
        self.capital -= commission
        
        self.position = Position(
            id=f"pos-{candle_idx}",
            symbol=signal.symbol,
            side=signal.side.value,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            value=position_value,
            unrealized_pnl=0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=datetime.utcnow(),  # Placeholder
            strategy_id=signal.strategy_id
        )
        self.position._confidence = signal.confidence
        self.position._entry_idx = candle_idx
        
        logger.debug(f"Opened {signal.side.value} @ {entry_price:.2f}")
    
    def _close_position(self, price: float, reason: str, candle_idx: int):
        """Close current position"""
        if not self.position:
            return
        
        # Apply slippage
        if self.position.side == 'long':
            exit_price = price * (1 - self.slippage_pct / 100)
        else:
            exit_price = price * (1 + self.slippage_pct / 100)
        
        # Calculate PnL
        if self.position.side == 'long':
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size
        
        pnl_pct = (pnl / self.position.value) * 100
        
        # Apply commission
        commission = self.position.value * (self.commission_pct / 100)
        
        # Update capital
        self.capital += self.position.value + pnl - commission
        
        # Record trade
        entry_idx = getattr(self.position, '_entry_idx', 0)
        holding_candles = candle_idx - entry_idx
        
        trade = BacktestTrade(
            entry_time=self.position.opened_at,
            exit_time=datetime.utcnow(),  # Placeholder
            symbol=self.position.symbol,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            strategy_id=self.position.strategy_id,
            confidence=getattr(self.position, '_confidence', 0),
            holding_time_hours=holding_candles  # Assuming 1h candles
        )
        self.trades.append(trade)
        
        logger.debug(f"Closed {self.position.side} @ {exit_price:.2f}, PnL: {pnl_pct:+.2f}%")
        
        self.position = None
    
    def _check_exit(
        self, 
        position: Position, 
        current_price: float,
        high: float,
        low: float,
        strategy
    ) -> Optional[str]:
        """Check if position should be closed"""
        # Check intra-candle stop loss / take profit
        if position.stop_loss:
            if position.side == 'long' and low <= position.stop_loss:
                return 'stop_loss'
            elif position.side == 'short' and high >= position.stop_loss:
                return 'stop_loss'
        
        if position.take_profit:
            if position.side == 'long' and high >= position.take_profit:
                return 'take_profit'
            elif position.side == 'short' and low <= position.take_profit:
                return 'take_profit'
        
        # Use strategy's exit logic
        return strategy.should_exit(position, current_price)
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized PnL"""
        equity = self.capital
        if self.position:
            if self.position.side == 'long':
                unrealized = (current_price - self.position.entry_price) * self.position.size
            else:
                unrealized = (self.position.entry_price - current_price) * self.position.size
            equity += self.position.value + unrealized
        return equity
    
    def _calculate_result(
        self, 
        strategy_id: str, 
        symbol: str,
        data: Dict[str, Any]
    ) -> BacktestResult:
        """Calculate backtest metrics"""
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio()
        
        avg_holding = np.mean([t.holding_time_hours for t in self.trades]) if self.trades else 0
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        return BacktestResult(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=str(datetime.utcnow() - timedelta(days=len(data.get('close', [])) / 24)),
            end_date=str(datetime.utcnow()),
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return_pct=total_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            avg_holding_hours=avg_holding,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # Annualize (assuming hourly data)
        annual_return = np.mean(returns) * 24 * 365
        annual_std = np.std(returns) * np.sqrt(24 * 365)
        
        return (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0
    
    def _empty_result(self, strategy_id: str, symbol: str) -> BacktestResult:
        """Return empty result for failed backtest"""
        return BacktestResult(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date='',
            end_date='',
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            avg_holding_hours=0
        )


def fetch_historical_data(
    symbol: str,
    days: int = 30,
    interval: str = '1hour'
) -> Dict[str, Any]:
    """
    Fetch historical data from KuCoin API.
    
    Args:
        symbol: Trading pair (e.g., 'BTC-USDT')
        days: Number of days of history
        interval: Candle interval
    
    Returns:
        Dict with OHLCV arrays
    """
    logger.info(f"Fetching {days} days of {interval} data for {symbol}")
    
    # KuCoin API endpoint
    base_url = 'https://api.kucoin.com'
    endpoint = '/api/v1/market/candles'
    
    # Convert symbol format
    api_symbol = symbol.replace('-', '-')
    
    # Calculate time range
    end_time = int(datetime.utcnow().timestamp())
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    
    # Map interval
    interval_map = {
        '1hour': '1hour',
        '4hour': '4hour',
        '1day': '1day'
    }
    api_interval = interval_map.get(interval, '1hour')
    
    params = {
        'symbol': api_symbol,
        'type': api_interval,
        'startAt': start_time,
        'endAt': end_time
    }
    
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if result.get('code') != '200000':
            logger.error(f"API error: {result.get('msg')}")
            return {}
        
        candles = result.get('data', [])
        if not candles:
            logger.warning(f"No data returned for {symbol}")
            return {}
        
        # KuCoin returns: [time, open, close, high, low, volume, turnover]
        # Reverse to chronological order
        candles = candles[::-1]
        
        data = {
            symbol: {
                'timestamp': [int(c[0]) for c in candles],
                'open': [float(c[1]) for c in candles],
                'close': [float(c[2]) for c in candles],
                'high': [float(c[3]) for c in candles],
                'low': [float(c[4]) for c in candles],
                'volume': [float(c[5]) for c in candles],
            }
        }
        
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return {}


def create_strategy(strategy_name: str, symbol: str):
    """Create strategy instance by name"""
    strategies = {
        'trend-following': TrendFollowingAgent,
        # Add more strategies as they're implemented
    }
    
    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategy_class(symbols=[symbol])


def print_result(result: BacktestResult):
    """Print backtest result in readable format"""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {result.strategy_id}")
    print("=" * 60)
    print(f"Symbol:           {result.symbol}")
    print(f"Period:           {result.start_date} to {result.end_date}")
    print("-" * 60)
    print(f"Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"Final Capital:    ${result.final_capital:,.2f}")
    print(f"Total Return:     {result.total_return_pct:+.2f}%")
    print("-" * 60)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Winning Trades:   {result.winning_trades}")
    print(f"Losing Trades:    {result.losing_trades}")
    print(f"Win Rate:         {result.win_rate*100:.1f}%")
    print("-" * 60)
    print(f"Avg Win:          {result.avg_win_pct:+.2f}%")
    print(f"Avg Loss:         {result.avg_loss_pct:.2f}%")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print("-" * 60)
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Avg Hold Time:    {result.avg_holding_hours:.1f} hours")
    print("=" * 60)
    
    if result.trades:
        print("\nRecent Trades:")
        for trade in result.trades[-5:]:
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(f"  {emoji} {trade.side.upper()} @ {trade.entry_price:.2f} -> "
                  f"{trade.exit_price:.2f} ({trade.pnl_pct:+.2f}%) [{trade.exit_reason}]")


def main():
    parser = argparse.ArgumentParser(description='Cash Town Backtesting Framework')
    parser.add_argument('--strategy', type=str, default='trend-following',
                       help='Strategy to backtest (or "all")')
    parser.add_argument('--symbol', type=str, default='BTC-USDT',
                       help='Trading symbol')
    parser.add_argument('--days', type=int, default=30,
                       help='Days of historical data')
    parser.add_argument('--interval', type=str, default='1hour',
                       help='Candle interval (1hour, 4hour, 1day)')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital')
    parser.add_argument('--position-size', type=float, default=5.0,
                       help='Position size as percentage of capital')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for JSON results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Fetch historical data
    data = fetch_historical_data(args.symbol, args.days, args.interval)
    if not data:
        logger.error("Failed to fetch data. Exiting.")
        sys.exit(1)
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=args.capital,
        position_size_pct=args.position_size
    )
    
    # Run backtest
    strategies = ['trend-following'] if args.strategy == 'all' else [args.strategy]
    results = []
    
    for strategy_name in strategies:
        try:
            strategy = create_strategy(strategy_name, args.symbol)
            result = engine.run(strategy, data)
            results.append(result)
            print_result(result)
        except Exception as e:
            logger.error(f"Failed to backtest {strategy_name}: {e}")
    
    # Save results if output file specified
    if args.output and results:
        output_data = [asdict(r) for r in results]
        # Convert trades to serializable format
        for r in output_data:
            r['trades'] = [asdict(t) for t in r.get('trades', [])]
            for t in r['trades']:
                t['entry_time'] = str(t['entry_time'])
                t['exit_time'] = str(t['exit_time'])
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
