#!/usr/bin/env python3
"""
Cash Town - Comprehensive Backtest for New Strategies (Feb 2026)

Backtests the 5 new futures-specific strategies:
- funding-fade: Fade extreme funding rates
- oi-divergence: Trade price/OI divergences  
- liquidation-hunter: Ride and fade liquidation cascades
- volatility-breakout: Trade Bollinger squeeze breakouts
- correlation-pairs: Pairs trading on correlated assets

Realistic assumptions:
- 0.06% trading fees (taker)
- 5x default leverage
- $200 position size per trade
- Max 3 concurrent positions per strategy

Usage:
    python scripts/backtest_new_strategies.py --days 60
    python scripts/backtest_new_strategies.py --days 90 --strategies funding-fade,oi-divergence
"""
import argparse
import json
import logging
import os
import sys
import random
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
from agents.strategies import (
    FundingFadeAgent,
    OIDivergenceAgent,
    LiquidationHunterAgent,
    VolatilityBreakoutAgent,
    CorrelationPairsAgent,
    STRATEGY_REGISTRY
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

BACKTEST_CONFIG = {
    # Trading parameters
    'initial_capital': 10000.0,
    'position_size': 200.0,  # $200 per trade
    'leverage': 5,
    'taker_fee_pct': 0.06,  # 0.06% per side
    'max_concurrent_positions': 3,
    
    # Symbols to backtest (KuCoin futures format)
    'symbols': [
        'XBTUSDTM',   # BTC
        'ETHUSDTM',   # ETH
        'SOLUSDTM',   # SOL
        'AVAXUSDTM',  # AVAX
    ],
    
    # Data parameters
    'candle_interval': '1hour',
    'warmup_periods': 60,  # Candles needed for indicators
}

# New strategies to backtest
NEW_STRATEGIES = [
    'funding-fade',
    'oi-divergence',
    'liquidation-hunter',
    'volatility-breakout',
    'correlation-pairs',
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestTrade:
    """Record of a simulated trade"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    leverage: int
    pnl: float
    pnl_pct: float
    fees_paid: float
    exit_reason: str
    strategy_id: str
    confidence: float
    holding_hours: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class StrategyResult:
    """Results from backtesting one strategy"""
    strategy_id: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_pnl: float
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
    avg_trades_per_day: float
    total_fees: float
    best_trade_pct: float
    worst_trade_pct: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    signals_generated: int = 0
    signals_skipped: int = 0  # Due to max positions


@dataclass
class OpenPosition:
    """Tracks an open position during backtest"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    size: float
    leverage: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy_id: str
    confidence: float
    entry_fee: float
    candle_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Data Fetching
# =============================================================================

class KuCoinFuturesData:
    """Fetch historical data from KuCoin Futures API"""
    
    BASE_URL = "https://api-futures.kucoin.com"
    
    @classmethod
    def fetch_candles(cls, symbol: str, days: int, interval: str = '1hour') -> Dict[str, List]:
        """
        Fetch historical candles from KuCoin Futures.
        
        Args:
            symbol: e.g., 'XBTUSDTM'
            days: Number of days of history
            interval: Candle interval
        
        Returns:
            Dict with OHLCV arrays
        """
        logger.info(f"Fetching {days} days of {interval} data for {symbol}")
        
        # Map interval to granularity (minutes)
        interval_map = {
            '1hour': 60,
            '4hour': 240,
            '1day': 1440,
            '15min': 15,
        }
        granularity = interval_map.get(interval, 60)
        
        # Calculate time range
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        
        all_candles = []
        current_end = end_time
        
        # Fetch in chunks (API limit)
        while current_end > start_time:
            chunk_start = max(start_time, current_end - (200 * granularity * 60 * 1000))
            
            try:
                url = f"{cls.BASE_URL}/api/v1/kline/query"
                params = {
                    'symbol': symbol,
                    'granularity': granularity,
                    'from': chunk_start,
                    'to': current_end
                }
                
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                
                if data.get('code') != '200000':
                    logger.warning(f"API error for {symbol}: {data.get('msg')}")
                    break
                
                candles = data.get('data', [])
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Move to earlier time period
                earliest = min(c[0] for c in candles)
                current_end = earliest - 1
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break
        
        if not all_candles:
            logger.warning(f"No data returned for {symbol}")
            return {}
        
        # Remove duplicates and sort
        seen = set()
        unique_candles = []
        for c in all_candles:
            if c[0] not in seen:
                seen.add(c[0])
                unique_candles.append(c)
        
        unique_candles.sort(key=lambda x: x[0])
        
        # KuCoin returns: [time, open, high, low, close, volume]
        result = {
            'timestamp': [datetime.fromtimestamp(c[0] / 1000) for c in unique_candles],
            'open': [float(c[1]) for c in unique_candles],
            'high': [float(c[2]) for c in unique_candles],
            'low': [float(c[3]) for c in unique_candles],
            'close': [float(c[4]) for c in unique_candles],
            'volume': [float(c[5]) for c in unique_candles],
        }
        
        logger.info(f"Fetched {len(unique_candles)} candles for {symbol}")
        return result
    
    @classmethod
    def fetch_all_symbols(cls, symbols: List[str], days: int, 
                         interval: str = '1hour') -> Dict[str, Dict]:
        """Fetch data for multiple symbols"""
        data = {}
        for symbol in symbols:
            symbol_data = cls.fetch_candles(symbol, days, interval)
            if symbol_data:
                data[symbol] = symbol_data
        return data


# =============================================================================
# Synthetic Data Generation (for funding, OI, etc.)
# =============================================================================

class SyntheticFuturesData:
    """
    Generate synthetic futures-specific data for backtesting.
    In production, this would come from actual API data.
    """
    
    @staticmethod
    def generate_funding_rates(closes: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Generate synthetic funding rates that correlate with price action.
        - Positive when price rallies (longs pay)
        - Negative when price dumps (shorts pay)
        - Mean reverts around 0
        """
        np.random.seed(seed)
        n = len(closes)
        
        # Base random walk
        base = np.cumsum(np.random.randn(n) * 0.0001)
        
        # Add price correlation
        returns = np.diff(closes, prepend=closes[0]) / closes
        price_component = returns * 0.01  # Price influence
        
        # Combine and mean-revert
        funding = base + price_component
        funding = funding - np.mean(funding)  # Center around 0
        
        # Add occasional extremes
        extreme_idx = np.random.choice(n, size=n//50, replace=False)
        funding[extreme_idx] *= np.random.uniform(2, 5, size=len(extreme_idx))
        
        # Clip to realistic range
        funding = np.clip(funding, -0.003, 0.003)  # -0.3% to +0.3%
        
        return funding
    
    @staticmethod
    def generate_open_interest(closes: np.ndarray, volumes: np.ndarray, 
                               seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generate synthetic OI data that shows realistic patterns:
        - OI tends to rise during trends
        - OI tends to fall during capitulation
        - Volume correlates with OI changes
        """
        np.random.seed(seed + 1)
        n = len(closes)
        
        # Start with base OI
        base_oi = closes[0] * 1000000  # Arbitrary starting OI
        
        oi = np.zeros(n)
        oi[0] = base_oi
        
        returns = np.diff(closes, prepend=closes[0]) / closes
        vol_normalized = volumes / np.mean(volumes)
        
        for i in range(1, n):
            # OI change influenced by price action and volume
            price_influence = returns[i] * 0.1 * vol_normalized[i]
            random_component = np.random.randn() * 0.02
            
            change = price_influence + random_component
            oi[i] = oi[i-1] * (1 + change)
        
        # Ensure positive
        oi = np.maximum(oi, base_oi * 0.5)
        
        # Calculate previous OI for divergence detection
        prev_oi = np.roll(oi, 6)  # 6-period lookback
        prev_oi[:6] = oi[:6]
        
        return {
            'current': oi,
            'previous': prev_oi
        }
    
    @staticmethod
    def generate_order_book_levels(closes: np.ndarray, seed: int = 42) -> Dict:
        """Generate synthetic order book data (simplified)"""
        np.random.seed(seed + 2)
        
        # For liquidation hunter, we mainly need price levels
        # This is simplified - in production we'd have actual order book data
        return {
            'bid_depth': closes[-1] * 0.98,
            'ask_depth': closes[-1] * 1.02,
        }


# =============================================================================
# Backtest Engine
# =============================================================================

class NewStrategyBacktester:
    """
    Backtest engine specifically designed for the new futures strategies.
    
    Features:
    - Handles synthetic futures data (funding, OI)
    - Proper fee calculation with leverage
    - Max concurrent position tracking
    - Detailed trade logging
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or BACKTEST_CONFIG
        self.capital = self.config['initial_capital']
        self.positions: List[OpenPosition] = []
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_counter = 0
        self.signals_generated = 0
        self.signals_skipped = 0
    
    def _reset(self):
        """Reset engine state for new backtest"""
        self.capital = self.config['initial_capital']
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.trade_counter = 0
        self.signals_generated = 0
        self.signals_skipped = 0
    
    def _calculate_fee(self, notional_value: float) -> float:
        """Calculate trading fee"""
        return notional_value * (self.config['taker_fee_pct'] / 100)
    
    def _open_position(self, signal: Signal, current_price: float,
                      current_time: datetime, candle_idx: int) -> bool:
        """
        Open a new position from a signal.
        
        Returns:
            True if position was opened, False if skipped
        """
        # Check max positions
        if len(self.positions) >= self.config['max_concurrent_positions']:
            self.signals_skipped += 1
            return False
        
        # Check if we already have a position in this symbol
        existing = [p for p in self.positions if p.symbol == signal.symbol]
        if existing:
            self.signals_skipped += 1
            return False
        
        # Calculate position size
        position_value = self.config['position_size']
        leverage = self.config['leverage']
        notional_value = position_value * leverage
        
        # Calculate entry fee
        entry_fee = self._calculate_fee(notional_value)
        
        # Apply slippage (0.05%)
        if signal.side == SignalSide.LONG:
            entry_price = current_price * 1.0005
        else:
            entry_price = current_price * 0.9995
        
        # Calculate position size in contracts
        size = notional_value / entry_price
        
        self.trade_counter += 1
        
        position = OpenPosition(
            trade_id=f"trade-{self.trade_counter}",
            symbol=signal.symbol,
            side=signal.side.value,
            entry_price=entry_price,
            entry_time=current_time,
            size=size,
            leverage=leverage,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_id=signal.strategy_id,
            confidence=signal.confidence,
            entry_fee=entry_fee,
            candle_idx=candle_idx,
            metadata=signal.metadata
        )
        
        self.positions.append(position)
        self.capital -= entry_fee
        
        logger.debug(f"Opened {signal.side.value} {signal.symbol} @ {entry_price:.2f}")
        return True
    
    def _close_position(self, position: OpenPosition, exit_price: float,
                       exit_time: datetime, reason: str, candle_idx: int) -> BacktestTrade:
        """Close a position and record the trade"""
        # Apply slippage
        if position.side == 'long':
            actual_exit = exit_price * 0.9995
        else:
            actual_exit = exit_price * 1.0005
        
        # Calculate PnL
        notional_value = self.config['position_size'] * position.leverage
        
        if position.side == 'long':
            pnl = (actual_exit - position.entry_price) / position.entry_price * notional_value
        else:
            pnl = (position.entry_price - actual_exit) / position.entry_price * notional_value
        
        # Exit fee
        exit_fee = self._calculate_fee(notional_value)
        total_fees = position.entry_fee + exit_fee
        
        # Net PnL
        net_pnl = pnl - exit_fee
        pnl_pct = (net_pnl / self.config['position_size']) * 100
        
        # Holding time
        holding_hours = (exit_time - position.entry_time).total_seconds() / 3600
        
        trade = BacktestTrade(
            trade_id=position.trade_id,
            entry_time=position.entry_time,
            exit_time=exit_time,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=actual_exit,
            size=position.size,
            leverage=position.leverage,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees_paid=total_fees,
            exit_reason=reason,
            strategy_id=position.strategy_id,
            confidence=position.confidence,
            holding_hours=holding_hours,
            metadata=position.metadata
        )
        
        # Update capital
        self.capital += self.config['position_size'] + net_pnl
        
        # Remove from positions
        self.positions = [p for p in self.positions if p.trade_id != position.trade_id]
        self.closed_trades.append(trade)
        
        logger.debug(f"Closed {position.side} {position.symbol} @ {actual_exit:.2f}, "
                    f"PnL: ${net_pnl:.2f} ({pnl_pct:+.2f}%)")
        
        return trade
    
    def _check_exit_conditions(self, position: OpenPosition, 
                               current_high: float, current_low: float,
                               current_close: float) -> Optional[str]:
        """Check if position should be closed"""
        # Stop loss check (uses high/low for proper simulation)
        if position.stop_loss:
            if position.side == 'long' and current_low <= position.stop_loss:
                return 'stop_loss'
            elif position.side == 'short' and current_high >= position.stop_loss:
                return 'stop_loss'
        
        # Take profit check
        if position.take_profit:
            if position.side == 'long' and current_high >= position.take_profit:
                return 'take_profit'
            elif position.side == 'short' and current_low <= position.take_profit:
                return 'take_profit'
        
        return None
    
    def _calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current equity including unrealized PnL"""
        equity = self.capital
        
        for position in self.positions:
            price = current_prices.get(position.symbol, position.entry_price)
            notional = self.config['position_size'] * position.leverage
            
            if position.side == 'long':
                unrealized = (price - position.entry_price) / position.entry_price * notional
            else:
                unrealized = (position.entry_price - price) / position.entry_price * notional
            
            equity += self.config['position_size'] + unrealized
        
        return equity
    
    def run_strategy(self, strategy_id: str, market_data: Dict[str, Dict],
                    days: int) -> StrategyResult:
        """
        Run backtest for a single strategy.
        
        Args:
            strategy_id: Strategy to test
            market_data: Dict of symbol -> OHLCV data
            days: Number of days being tested
        
        Returns:
            StrategyResult with all metrics
        """
        self._reset()
        
        # Get strategy class and create instance
        strategy_class = STRATEGY_REGISTRY.get(strategy_id)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        
        symbols = list(market_data.keys())
        strategy = strategy_class(symbols=symbols)
        
        # Find the common time range across all symbols
        min_len = min(len(d['close']) for d in market_data.values())
        start_idx = self.config['warmup_periods']
        
        if min_len <= start_idx:
            logger.error(f"Not enough data for {strategy_id}")
            return self._empty_result(strategy_id, strategy.name)
        
        logger.info(f"Running backtest for {strategy_id} ({strategy.name})")
        logger.info(f"Data: {min_len} candles, starting at index {start_idx}")
        
        # Generate synthetic futures data for each symbol
        synthetic_data = {}
        for symbol, data in market_data.items():
            closes = np.array(data['close'])
            volumes = np.array(data['volume'])
            seed = hash(symbol) % 10000
            
            synthetic_data[symbol] = {
                'funding': SyntheticFuturesData.generate_funding_rates(closes, seed),
                'oi': SyntheticFuturesData.generate_open_interest(closes, volumes, seed),
            }
        
        # Get timestamps from first symbol
        first_symbol = symbols[0]
        timestamps = market_data[first_symbol]['timestamp']
        
        # Main backtest loop
        for i in range(start_idx, min_len):
            current_time = timestamps[i]
            
            # Prepare current market data slice
            current_data = {}
            current_prices = {}
            
            for symbol, data in market_data.items():
                current_data[symbol] = {
                    'open': data['open'][:i+1],
                    'high': data['high'][:i+1],
                    'low': data['low'][:i+1],
                    'close': data['close'][:i+1],
                    'volume': data['volume'][:i+1],
                }
                current_prices[symbol] = data['close'][i]
            
            # Set synthetic data on strategy
            self._set_strategy_data(strategy, strategy_id, synthetic_data, i)
            
            # Check exits first
            positions_to_close = []
            for position in self.positions:
                symbol_data = market_data.get(position.symbol)
                if not symbol_data:
                    continue
                
                exit_reason = self._check_exit_conditions(
                    position,
                    symbol_data['high'][i],
                    symbol_data['low'][i],
                    symbol_data['close'][i]
                )
                
                if exit_reason:
                    positions_to_close.append((position, exit_reason))
            
            # Close positions
            for position, reason in positions_to_close:
                exit_price = market_data[position.symbol]['close'][i]
                # For stop loss, use the stop price
                if reason == 'stop_loss' and position.stop_loss:
                    exit_price = position.stop_loss
                elif reason == 'take_profit' and position.take_profit:
                    exit_price = position.take_profit
                
                self._close_position(position, exit_price, current_time, reason, i)
            
            # Generate signals
            try:
                signals = strategy.generate_signals(current_data)
                self.signals_generated += len(signals)
                
                # Process signals
                for signal in signals:
                    if signal.confidence >= 0.55:  # Minimum confidence
                        price = current_prices.get(signal.symbol, signal.price)
                        self._open_position(signal, price, current_time, i)
                        
            except Exception as e:
                logger.warning(f"Error generating signals at {i}: {e}")
            
            # Record equity
            equity = self._calculate_equity(current_prices)
            self.equity_curve.append((current_time, equity))
        
        # Close any remaining positions
        for position in list(self.positions):
            exit_price = market_data[position.symbol]['close'][-1]
            self._close_position(position, exit_price, timestamps[-1], 'end_of_backtest', min_len - 1)
        
        return self._calculate_result(strategy_id, strategy.name, timestamps[start_idx], timestamps[-1], days)
    
    def _set_strategy_data(self, strategy, strategy_id: str, 
                          synthetic_data: Dict, idx: int):
        """Set synthetic data on strategy based on its type"""
        if strategy_id == 'funding-fade':
            funding_data = {s: d['funding'][idx] for s, d in synthetic_data.items()}
            strategy.set_funding_data(funding_data)
        
        elif strategy_id == 'oi-divergence':
            oi_data = {}
            for symbol, data in synthetic_data.items():
                oi_data[symbol] = {
                    'current': data['oi']['current'][idx],
                    'previous': data['oi']['previous'][idx]
                }
            strategy.set_oi_data(oi_data)
        
        elif strategy_id == 'liquidation-hunter':
            # OI and order book data
            oi_data = {s: d['oi']['current'][idx] for s, d in synthetic_data.items()}
            strategy.set_oi_data(oi_data)
            # Order book is generated dynamically in the strategy
    
    def _calculate_result(self, strategy_id: str, strategy_name: str,
                         start_time: datetime, end_time: datetime,
                         days: int) -> StrategyResult:
        """Calculate backtest metrics"""
        total_trades = len(self.closed_trades)
        
        if total_trades == 0:
            return self._empty_result(strategy_id, strategy_name)
        
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_fees = sum(t.fees_paid for t in self.closed_trades)
        total_pnl = sum(t.pnl for t in self.closed_trades)
        
        max_dd = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio()
        
        avg_holding = np.mean([t.holding_hours for t in self.closed_trades])
        avg_trades_per_day = total_trades / max(days, 1)
        
        pnl_pcts = [t.pnl_pct for t in self.closed_trades]
        best_trade = max(pnl_pcts) if pnl_pcts else 0
        worst_trade = min(pnl_pcts) if pnl_pcts else 0
        
        final_capital = self.config['initial_capital'] + total_pnl
        total_return = (total_pnl / self.config['initial_capital']) * 100
        
        return StrategyResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            start_date=str(start_time.date()),
            end_date=str(end_time.date()),
            initial_capital=self.config['initial_capital'],
            final_capital=final_capital,
            total_return_pct=total_return,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_holding_hours=avg_holding,
            avg_trades_per_day=avg_trades_per_day,
            total_fees=total_fees,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            signals_generated=self.signals_generated,
            signals_skipped=self.signals_skipped
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0
        
        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
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
        
        equities = [e[1] for e in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # Annualize (assuming hourly data)
        annual_return = np.mean(returns) * 24 * 365
        annual_std = np.std(returns) * np.sqrt(24 * 365)
        
        return (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0
    
    def _empty_result(self, strategy_id: str, strategy_name: str) -> StrategyResult:
        """Return empty result for failed backtest"""
        return StrategyResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            start_date='',
            end_date='',
            initial_capital=self.config['initial_capital'],
            final_capital=self.config['initial_capital'],
            total_return_pct=0,
            total_pnl=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            avg_holding_hours=0,
            avg_trades_per_day=0,
            total_fees=0,
            best_trade_pct=0,
            worst_trade_pct=0
        )


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(results: List[StrategyResult], 
                            output_path: str,
                            config: Dict[str, Any]) -> str:
    """Generate a comprehensive markdown report"""
    
    report = []
    report.append("# Cash Town - New Strategies Backtest Report")
    report.append(f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    report.append(f"\n**Period:** {results[0].start_date} to {results[0].end_date}")
    report.append("")
    
    # Configuration summary
    report.append("## Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Initial Capital | ${config['initial_capital']:,.2f} |")
    report.append(f"| Position Size | ${config['position_size']:,.2f} |")
    report.append(f"| Leverage | {config['leverage']}x |")
    report.append(f"| Taker Fee | {config['taker_fee_pct']}% |")
    report.append(f"| Max Concurrent Positions | {config['max_concurrent_positions']} |")
    report.append(f"| Symbols | {', '.join(config['symbols'])} |")
    report.append("")
    
    # Executive summary
    report.append("## Executive Summary")
    report.append("")
    
    # Sort by total return
    sorted_results = sorted(results, key=lambda r: r.total_return_pct, reverse=True)
    
    profitable = [r for r in results if r.total_pnl > 0]
    report.append(f"- **Profitable Strategies:** {len(profitable)}/{len(results)}")
    report.append(f"- **Best Performer:** {sorted_results[0].strategy_name} ({sorted_results[0].total_return_pct:+.2f}%)")
    if len(sorted_results) > 1:
        report.append(f"- **Worst Performer:** {sorted_results[-1].strategy_name} ({sorted_results[-1].total_return_pct:+.2f}%)")
    
    total_pnl = sum(r.total_pnl for r in results)
    avg_win_rate = np.mean([r.win_rate for r in results if r.total_trades > 0])
    report.append(f"- **Combined PnL (if all run):** ${total_pnl:+,.2f}")
    report.append(f"- **Average Win Rate:** {avg_win_rate*100:.1f}%")
    report.append("")
    
    # Performance comparison table
    report.append("## Strategy Performance Comparison")
    report.append("")
    report.append("| Strategy | Return | PnL | Trades | Win Rate | Sharpe | Max DD | Profit Factor |")
    report.append("|----------|--------|-----|--------|----------|--------|--------|---------------|")
    
    for r in sorted_results:
        emoji = "✅" if r.total_pnl > 0 else "❌"
        report.append(
            f"| {emoji} {r.strategy_name} | {r.total_return_pct:+.2f}% | ${r.total_pnl:+,.2f} | "
            f"{r.total_trades} | {r.win_rate*100:.1f}% | {r.sharpe_ratio:.2f} | "
            f"{r.max_drawdown_pct:.1f}% | {r.profit_factor:.2f} |"
        )
    report.append("")
    
    # Detailed results per strategy
    report.append("## Detailed Strategy Analysis")
    report.append("")
    
    for r in sorted_results:
        emoji = "🟢" if r.total_pnl > 0 else "🔴"
        report.append(f"### {emoji} {r.strategy_name} (`{r.strategy_id}`)")
        report.append("")
        
        report.append("**Performance Metrics:**")
        report.append("")
        report.append(f"- Initial Capital: ${r.initial_capital:,.2f}")
        report.append(f"- Final Capital: ${r.final_capital:,.2f}")
        report.append(f"- Total Return: **{r.total_return_pct:+.2f}%**")
        report.append(f"- Total PnL: ${r.total_pnl:+,.2f}")
        report.append(f"- Total Fees Paid: ${r.total_fees:.2f}")
        report.append("")
        
        report.append("**Trade Statistics:**")
        report.append("")
        report.append(f"- Total Trades: {r.total_trades}")
        report.append(f"- Winning: {r.winning_trades} | Losing: {r.losing_trades}")
        report.append(f"- Win Rate: {r.win_rate*100:.1f}%")
        report.append(f"- Avg Win: {r.avg_win_pct:+.2f}% | Avg Loss: {r.avg_loss_pct:.2f}%")
        report.append(f"- Best Trade: {r.best_trade_pct:+.2f}% | Worst: {r.worst_trade_pct:.2f}%")
        report.append(f"- Avg Holding Time: {r.avg_holding_hours:.1f} hours")
        report.append(f"- Trades per Day: {r.avg_trades_per_day:.2f}")
        report.append("")
        
        report.append("**Risk Metrics:**")
        report.append("")
        report.append(f"- Max Drawdown: {r.max_drawdown_pct:.2f}%")
        report.append(f"- Sharpe Ratio: {r.sharpe_ratio:.2f}")
        report.append(f"- Profit Factor: {r.profit_factor:.2f}")
        report.append(f"- Signals Generated: {r.signals_generated}")
        report.append(f"- Signals Skipped (max positions): {r.signals_skipped}")
        report.append("")
        
        # Recent trades
        if r.trades:
            report.append("**Last 5 Trades:**")
            report.append("")
            report.append("| Symbol | Side | Entry | Exit | PnL | Reason |")
            report.append("|--------|------|-------|------|-----|--------|")
            
            for trade in r.trades[-5:]:
                trade_emoji = "✅" if trade.pnl > 0 else "❌"
                report.append(
                    f"| {trade.symbol} | {trade.side.upper()} | "
                    f"${trade.entry_price:,.2f} | ${trade.exit_price:,.2f} | "
                    f"{trade_emoji} {trade.pnl_pct:+.2f}% | {trade.exit_reason} |"
                )
            report.append("")
        
        report.append("---")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    viable = [r for r in results if r.win_rate >= 0.45 and r.sharpe_ratio > 0.5]
    if viable:
        report.append("**Strategies recommended for live testing:**")
        for r in sorted(viable, key=lambda x: x.sharpe_ratio, reverse=True):
            report.append(f"- {r.strategy_name}: Sharpe {r.sharpe_ratio:.2f}, Win Rate {r.win_rate*100:.1f}%")
    else:
        report.append("⚠️ No strategies meet the minimum criteria for live deployment.")
        report.append("Consider adjusting parameters or extending the test period.")
    
    report.append("")
    report.append("---")
    report.append(f"*Report generated by Cash Town Backtest Engine v2.0*")
    
    # Write to file
    report_text = "\n".join(report)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {output_path}")
    return report_text


def print_summary(results: List[StrategyResult]):
    """Print summary to console"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY - NEW STRATEGIES")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda r: r.total_return_pct, reverse=True)
    
    for r in sorted_results:
        emoji = "✅" if r.total_pnl > 0 else "❌"
        print(f"\n{emoji} {r.strategy_name}")
        print(f"   Return: {r.total_return_pct:+.2f}% | PnL: ${r.total_pnl:+,.2f}")
        print(f"   Trades: {r.total_trades} | Win Rate: {r.win_rate*100:.1f}%")
        print(f"   Sharpe: {r.sharpe_ratio:.2f} | Max DD: {r.max_drawdown_pct:.1f}%")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Backtest new Cash Town futures strategies'
    )
    parser.add_argument(
        '--days', type=int, default=60,
        help='Days of historical data (30-90 recommended)'
    )
    parser.add_argument(
        '--strategies', type=str, default=None,
        help='Comma-separated list of strategies to test (default: all)'
    )
    parser.add_argument(
        '--output', type=str, 
        default=str(PROJECT_ROOT / 'data' / 'backtest_results.md'),
        help='Output path for markdown report'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate days
    if args.days < 30:
        logger.warning("Less than 30 days may not provide reliable results")
    if args.days > 90:
        logger.warning("More than 90 days may hit API rate limits")
    
    # Determine strategies to test
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
        # Validate
        for s in strategies:
            if s not in NEW_STRATEGIES:
                logger.error(f"Unknown strategy: {s}")
                logger.info(f"Available: {', '.join(NEW_STRATEGIES)}")
                sys.exit(1)
    else:
        strategies = NEW_STRATEGIES
    
    print(f"\n🚀 Cash Town New Strategies Backtest")
    print(f"   Strategies: {', '.join(strategies)}")
    print(f"   Period: {args.days} days")
    print(f"   Symbols: {', '.join(BACKTEST_CONFIG['symbols'])}")
    print()
    
    # Fetch historical data
    print("📊 Fetching historical data from KuCoin Futures...")
    market_data = KuCoinFuturesData.fetch_all_symbols(
        BACKTEST_CONFIG['symbols'],
        args.days,
        BACKTEST_CONFIG['candle_interval']
    )
    
    if not market_data:
        logger.error("Failed to fetch market data. Exiting.")
        sys.exit(1)
    
    print(f"   Loaded data for {len(market_data)} symbols")
    
    # Run backtests
    backtester = NewStrategyBacktester(BACKTEST_CONFIG)
    results = []
    
    for strategy_id in strategies:
        print(f"\n🔄 Testing {strategy_id}...")
        try:
            result = backtester.run_strategy(strategy_id, market_data, args.days)
            results.append(result)
            print(f"   ✓ Completed: {result.total_trades} trades, "
                  f"Return: {result.total_return_pct:+.2f}%")
        except Exception as e:
            logger.error(f"Failed to backtest {strategy_id}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        logger.error("No successful backtests. Exiting.")
        sys.exit(1)
    
    # Generate report
    print("\n📝 Generating report...")
    generate_markdown_report(results, args.output, BACKTEST_CONFIG)
    
    # Print summary
    print_summary(results)
    
    print(f"\n✅ Full report saved to: {args.output}")


if __name__ == '__main__':
    main()
