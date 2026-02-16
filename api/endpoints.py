"""
Bloomberg-style Dashboard API Endpoints

Provides comprehensive API for the family office dashboard:
- Portfolio overview (equity, P&L, exposure)
- Position management (live data, drill-down)
- Trade history (filters, pagination)
- Strategy performance analytics
- Signal tracking (accepted + rejected)
- Risk metrics
- OHLCV charting data
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
SIGNALS_LOG = DATA_DIR / 'signals_history.jsonl'
TRADES_LOG = DATA_DIR / 'trades_history.jsonl'
COUNTERFACTUAL_LOG = DATA_DIR / 'counterfactual.jsonl'


@dataclass
class APIResponse:
    """Standard API response wrapper"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data,
            'metadata': self.metadata
        }


class DashboardAPI:
    """
    Family Office Dashboard API
    
    All data queryable, Bloomberg-style drill-down capability.
    """
    
    def __init__(self, orchestrator=None, executor=None, data_feed=None):
        self.orchestrator = orchestrator
        self.executor = executor
        self.data_feed = data_feed
        
        # Cache for expensive operations
        self._trades_cache = None
        self._trades_cache_time = None
        self._cache_ttl = 60  # 1 minute cache
    
    def set_orchestrator(self, orchestrator):
        """Set orchestrator reference"""
        self.orchestrator = orchestrator
    
    def set_executor(self, executor):
        """Set executor reference"""
        self.executor = executor
    
    def set_data_feed(self, data_feed):
        """Set data feed reference"""
        self.data_feed = data_feed
    
    # ==========================================
    # PORTFOLIO ENDPOINTS
    # ==========================================
    
    def get_portfolio(self) -> Dict:
        """
        GET /api/portfolio
        
        Current portfolio overview:
        - Total equity
        - Available balance
        - Total exposure
        - Unrealized P&L
        - Realized P&L (today)
        - Position count
        - Risk metrics summary
        """
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'equity': {
                'total': 0.0,
                'available': 0.0,
                'margin_used': 0.0,
                'currency': 'USDT'
            },
            'pnl': {
                'unrealized': 0.0,
                'unrealized_pct': 0.0,
                'realized_today': 0.0,
                'realized_all_time': 0.0
            },
            'exposure': {
                'total_notional': 0.0,
                'net_exposure': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'exposure_pct': 0.0
            },
            'positions': {
                'count': 0,
                'long_count': 0,
                'short_count': 0
            },
            'risk': {
                'max_drawdown_pct': 0.0,
                'current_drawdown_pct': 0.0,
                'daily_loss_pct': 0.0,
                'circuit_breaker_active': False
            },
            'performance': {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_estimate': 0.0,
                'trades_today': 0
            }
        }
        
        # Get executor state
        if self.executor:
            try:
                status = self.executor.get_status()
                balance = status.get('account_balance', 0)
                positions = self.executor.positions
                
                # Calculate equity components
                unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())
                margin_used = sum(p.margin for p in positions.values())
                total_equity = balance + unrealized_pnl
                
                response['equity']['total'] = total_equity
                response['equity']['available'] = balance - margin_used
                response['equity']['margin_used'] = margin_used
                
                # Calculate P&L
                response['pnl']['unrealized'] = unrealized_pnl
                response['pnl']['unrealized_pct'] = (unrealized_pnl / total_equity * 100) if total_equity > 0 else 0
                
                daily = status.get('daily_stats', {})
                response['pnl']['realized_today'] = daily.get('pnl', 0)
                
                # Calculate exposure
                long_exp = sum(p.size * p.entry_price for p in positions.values() if p.side == 'long')
                short_exp = sum(p.size * p.entry_price for p in positions.values() if p.side == 'short')
                response['exposure']['long_exposure'] = long_exp
                response['exposure']['short_exposure'] = short_exp
                response['exposure']['net_exposure'] = long_exp - short_exp
                response['exposure']['total_notional'] = long_exp + short_exp
                response['exposure']['exposure_pct'] = (long_exp + short_exp) / total_equity * 100 if total_equity > 0 else 0
                
                # Position counts
                response['positions']['count'] = len(positions)
                response['positions']['long_count'] = sum(1 for p in positions.values() if p.side == 'long')
                response['positions']['short_count'] = sum(1 for p in positions.values() if p.side == 'short')
                
                # Risk metrics
                dd = status.get('drawdown_protection', {})
                response['risk']['max_drawdown_pct'] = dd.get('threshold_pct', 10)
                response['risk']['current_drawdown_pct'] = dd.get('current_drawdown_pct', 0)
                response['risk']['circuit_breaker_active'] = status.get('killed', False)
                
                # Performance from trades
                perf = self._calculate_performance_metrics()
                response['performance'].update(perf)
                response['performance']['trades_today'] = daily.get('trades', 0)
                
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
        
        # Get risk manager status if available
        if self.orchestrator:
            try:
                risk_status = self.orchestrator.get_risk_status()
                can_trade, reason = self.orchestrator.can_trade()
                response['risk']['circuit_breaker_active'] = not can_trade
                if not can_trade:
                    response['risk']['circuit_breaker_reason'] = reason
            except Exception as e:
                logger.debug(f"Risk status unavailable: {e}")
        
        return APIResponse(
            success=True,
            data=response,
            metadata={'source': 'live', 'cached': False}
        ).to_dict()
    
    # ==========================================
    # POSITIONS ENDPOINTS
    # ==========================================
    
    def get_positions(self, include_closed: bool = False) -> Dict:
        """
        GET /api/positions
        
        All open positions with live data:
        - Symbol, side, size
        - Entry price, current price
        - Unrealized P&L (absolute and %)
        - Leverage, margin
        - Stop loss, take profit
        - Age, strategy attribution
        """
        positions_list = []
        
        if self.executor:
            try:
                from execution.strategy_tracker import tracker
                
                for symbol, pos in self.executor.positions.items():
                    # Get strategy attribution
                    tracked = tracker.get_all_positions().get(symbol)
                    strategy_id = tracked.strategy_id if tracked else 'unknown'
                    entry_time = tracked.entry_time if tracked else None
                    stop_loss = tracked.stop_loss if tracked else None
                    take_profit = tracked.take_profit if tracked else None
                    
                    # Calculate age
                    age_hours = 0
                    if entry_time:
                        try:
                            entry_dt = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
                            age_hours = (datetime.utcnow() - entry_dt).total_seconds() / 3600
                        except:
                            pass
                    
                    # Calculate P&L %
                    pnl_pct = 0
                    if pos.entry_price > 0:
                        if pos.side == 'long':
                            pnl_pct = (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100
                        else:
                            pnl_pct = (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100
                    
                    # Get current price from data feed
                    current_price = pos.entry_price  # Default to entry
                    if self.data_feed:
                        data = self.data_feed.get_symbol_data(symbol)
                        if data and data.get('close'):
                            current_price = data['close'][-1]
                    
                    position_data = {
                        'symbol': symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': current_price,
                        'leverage': pos.leverage,
                        'margin': pos.margin,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_pct': round(pnl_pct, 2),
                        'liquidation_price': pos.liquidation_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strategy_id': strategy_id,
                        'age_hours': round(age_hours, 1),
                        'entry_time': entry_time,
                        'status': 'open'
                    }
                    positions_list.append(position_data)
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
        
        # Sort by unrealized P&L
        positions_list.sort(key=lambda x: x['unrealized_pnl'], reverse=True)
        
        return APIResponse(
            success=True,
            data=positions_list,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'count': len(positions_list),
                'filters': {'include_closed': include_closed}
            }
        ).to_dict()
    
    def get_position(self, symbol: str) -> Dict:
        """
        GET /api/position/:symbol
        
        Detailed view of a single position with full history.
        """
        if self.executor and symbol in self.executor.positions:
            pos = self.executor.positions[symbol]
            
            # Get full details from position endpoint
            positions = self.get_positions()['data']
            for p in positions:
                if p['symbol'] == symbol:
                    # Add extra detail for single position view
                    p['signals'] = self._get_signals_for_symbol(symbol, limit=10)
                    p['price_history'] = self._get_price_history(symbol)
                    return APIResponse(
                        success=True,
                        data=p,
                        metadata={'timestamp': datetime.utcnow().isoformat()}
                    ).to_dict()
        
        return APIResponse(
            success=False,
            data=None,
            metadata={'error': f'Position not found: {symbol}'}
        ).to_dict()
    
    # ==========================================
    # TRADES ENDPOINTS
    # ==========================================
    
    def get_trades(self, 
                   strategy: str = None,
                   symbol: str = None,
                   side: str = None,
                   start_date: str = None,
                   end_date: str = None,
                   min_pnl: float = None,
                   max_pnl: float = None,
                   won_only: bool = None,
                   limit: int = 100,
                   offset: int = 0) -> Dict:
        """
        GET /api/trades
        
        Historical trades with comprehensive filters:
        - By strategy
        - By symbol
        - By date range
        - By P&L range
        - Winners/losers only
        - Pagination
        """
        trades = self._load_trades()
        
        # Apply filters
        filtered = []
        for trade in trades:
            # Strategy filter
            if strategy and trade.get('strategy_id') != strategy:
                continue
            
            # Symbol filter
            if symbol and trade.get('symbol') != symbol:
                continue
            
            # Side filter
            if side and trade.get('side') != side:
                continue
            
            # Date filters
            if start_date:
                trade_date = trade.get('timestamp', '')[:10]
                if trade_date < start_date:
                    continue
            
            if end_date:
                trade_date = trade.get('timestamp', '')[:10]
                if trade_date > end_date:
                    continue
            
            # P&L filters
            pnl = trade.get('pnl', 0)
            if min_pnl is not None and pnl < min_pnl:
                continue
            if max_pnl is not None and pnl > max_pnl:
                continue
            
            # Won filter
            if won_only is not None:
                if won_only and not trade.get('won'):
                    continue
                if not won_only and trade.get('won'):
                    continue
            
            filtered.append(trade)
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Calculate totals before pagination
        total_count = len(filtered)
        total_pnl = sum(t.get('pnl', 0) for t in filtered)
        win_count = sum(1 for t in filtered if t.get('won'))
        
        # Apply pagination
        paginated = filtered[offset:offset + limit]
        
        return APIResponse(
            success=True,
            data=paginated,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'total_count': total_count,
                'returned_count': len(paginated),
                'offset': offset,
                'limit': limit,
                'has_more': offset + limit < total_count,
                'filters_applied': {
                    'strategy': strategy,
                    'symbol': symbol,
                    'side': side,
                    'start_date': start_date,
                    'end_date': end_date,
                    'min_pnl': min_pnl,
                    'max_pnl': max_pnl,
                    'won_only': won_only
                },
                'aggregates': {
                    'total_pnl': round(total_pnl, 2),
                    'win_count': win_count,
                    'loss_count': total_count - win_count,
                    'win_rate': round(win_count / total_count * 100, 1) if total_count > 0 else 0
                }
            }
        ).to_dict()
    
    def get_trade(self, trade_id: str) -> Dict:
        """
        GET /api/trade/:id
        
        Single trade details with full context.
        """
        trades = self._load_trades()
        
        # Find by ID (we'll use timestamp as ID since trades don't have explicit IDs)
        for trade in trades:
            # Match by timestamp or construct a pseudo-ID
            pseudo_id = f"{trade.get('timestamp', '')}_{trade.get('symbol', '')}".replace(':', '-')
            if trade_id == pseudo_id or trade.get('timestamp') == trade_id:
                # Add extra context
                trade['related_signals'] = self._get_signals_for_symbol(
                    trade.get('symbol'), 
                    limit=5,
                    around_time=trade.get('timestamp')
                )
                return APIResponse(
                    success=True,
                    data=trade,
                    metadata={'timestamp': datetime.utcnow().isoformat()}
                ).to_dict()
        
        return APIResponse(
            success=False,
            data=None,
            metadata={'error': f'Trade not found: {trade_id}'}
        ).to_dict()
    
    # ==========================================
    # STRATEGIES ENDPOINTS
    # ==========================================
    
    def get_strategies(self) -> Dict:
        """
        GET /api/strategies
        
        Performance breakdown by strategy:
        - Trade count, win rate
        - Total P&L, avg P&L
        - Max win, max loss
        - Current multiplier
        - Active positions
        """
        strategies = {}
        
        # Get from orchestrator
        if self.orchestrator:
            try:
                perf = self.orchestrator.strategy_performance
                for strategy_id, data in perf.items():
                    strategies[strategy_id] = {
                        'strategy_id': strategy_id,
                        'trades': data.get('trades', 0),
                        'wins': data.get('wins', 0),
                        'win_rate': round(data.get('win_rate', 0) * 100, 1),
                        'total_pnl': round(data.get('total_pnl_pct', 0), 2),
                        'avg_pnl': round(data.get('total_pnl_pct', 0) / max(data.get('trades', 1), 1), 2),
                        'score': round(data.get('score', 0), 3),
                        'multiplier': data.get('multiplier', 1.0),
                        'status': 'active' if data.get('multiplier', 1) > 0 else 'disabled'
                    }
            except Exception as e:
                logger.debug(f"Could not get orchestrator performance: {e}")
        
        # Enrich from trades log
        trades = self._load_trades()
        for trade in trades:
            sid = trade.get('strategy_id', 'unknown')
            if sid not in strategies:
                strategies[sid] = {
                    'strategy_id': sid,
                    'trades': 0,
                    'wins': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'max_win': 0,
                    'max_loss': 0,
                    'score': 0,
                    'multiplier': 1.0,
                    'status': 'unknown'
                }
            
            s = strategies[sid]
            pnl = trade.get('pnl', 0)
            
            # Track max win/loss
            if 'max_win' not in s:
                s['max_win'] = 0
                s['max_loss'] = 0
            
            if pnl > s['max_win']:
                s['max_win'] = pnl
            if pnl < s['max_loss']:
                s['max_loss'] = pnl
        
        # Get active position counts
        if self.executor:
            from execution.strategy_tracker import tracker
            for symbol, tracked in tracker.get_all_positions().items():
                sid = tracked.strategy_id
                if sid in strategies:
                    if 'active_positions' not in strategies[sid]:
                        strategies[sid]['active_positions'] = 0
                    strategies[sid]['active_positions'] += 1
        
        # Convert to list and sort by total P&L
        strategies_list = list(strategies.values())
        strategies_list.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
        
        return APIResponse(
            success=True,
            data=strategies_list,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'count': len(strategies_list)
            }
        ).to_dict()
    
    def get_strategy(self, strategy_id: str) -> Dict:
        """
        GET /api/strategy/:id
        
        Detailed strategy view with trade history.
        """
        strategies = self.get_strategies()['data']
        
        for s in strategies:
            if s['strategy_id'] == strategy_id:
                # Add trade history for this strategy
                s['recent_trades'] = self.get_trades(strategy=strategy_id, limit=20)['data']
                s['signals'] = self._get_signals_for_strategy(strategy_id, limit=50)
                
                return APIResponse(
                    success=True,
                    data=s,
                    metadata={'timestamp': datetime.utcnow().isoformat()}
                ).to_dict()
        
        return APIResponse(
            success=False,
            data=None,
            metadata={'error': f'Strategy not found: {strategy_id}'}
        ).to_dict()
    
    # ==========================================
    # SIGNALS ENDPOINTS
    # ==========================================
    
    def get_signals(self, 
                    accepted: bool = None,
                    strategy: str = None,
                    symbol: str = None,
                    limit: int = 100,
                    offset: int = 0) -> Dict:
        """
        GET /api/signals
        
        Recent signals (both accepted and rejected):
        - Signal details
        - Selection outcome
        - Counterfactual analysis (for rejected)
        """
        signals = self._load_signals()
        
        # Apply filters
        filtered = []
        for sig in signals:
            if accepted is not None:
                if accepted and not sig.get('was_selected'):
                    continue
                if not accepted and sig.get('was_selected'):
                    continue
            
            if strategy and sig.get('strategy_id') != strategy:
                continue
            
            if symbol and sig.get('symbol') != symbol:
                continue
            
            filtered.append(sig)
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Stats before pagination
        total_count = len(filtered)
        accepted_count = sum(1 for s in filtered if s.get('was_selected'))
        
        # Pagination
        paginated = filtered[offset:offset + limit]
        
        return APIResponse(
            success=True,
            data=paginated,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'total_count': total_count,
                'returned_count': len(paginated),
                'offset': offset,
                'limit': limit,
                'has_more': offset + limit < total_count,
                'filters': {
                    'accepted': accepted,
                    'strategy': strategy,
                    'symbol': symbol
                },
                'aggregates': {
                    'accepted_count': accepted_count,
                    'rejected_count': total_count - accepted_count,
                    'acceptance_rate': round(accepted_count / total_count * 100, 1) if total_count > 0 else 0
                }
            }
        ).to_dict()
    
    # ==========================================
    # PYRAMID ENDPOINTS
    # ==========================================
    
    def get_pyramid_status(self) -> Dict:
        """
        GET /api/pyramid
        
        Current pyramiding status for all positions:
        - Pyramid levels and thresholds
        - Position sizes (base vs current)
        - Leverage progression
        - History of adds/deleverages
        - Opportunities for next pyramid
        """
        pyramid_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'enabled': True,
            'config': {
                'max_levels': 3,
                'level_2_threshold': 1.5,  # +1.5% ROE
                'level_3_threshold': 3.0,  # +3.0% ROE
                'level_2_add_pct': 50,
                'level_3_add_pct': 25,
                'leverage_bump_per_level': 1,
                'max_leverage': 10
            },
            'positions': [],
            'summary': {
                'total_pyramided_positions': 0,
                'total_pyramid_adds': 0,
                'total_deleverages': 0,
                'positions_at_max_level': 0
            }
        }
        
        # Get pyramid status from orchestrator's risk manager
        if self.orchestrator and hasattr(self.orchestrator, 'risk_manager'):
            try:
                rm = self.orchestrator.risk_manager
                pyr_status = rm.get_pyramid_status()
                
                pyramid_data['enabled'] = pyr_status.get('enabled', True)
                pyramid_data['config']['max_levels'] = pyr_status.get('max_levels', 3)
                
                # Enrich with current price data
                for symbol, pstate in pyr_status.get('positions', {}).items():
                    position_info = pstate.copy()
                    
                    # Get current price from executor
                    if self.executor and symbol in self.executor.positions:
                        pos = self.executor.positions[symbol]
                        position_info['entry_price'] = pos.entry_price
                        position_info['unrealized_pnl'] = pos.unrealized_pnl
                        position_info['side'] = pos.side
                        
                        # Calculate ROE
                        if pos.entry_price > 0:
                            if pos.side == 'long':
                                # For long, need current price to calculate ROE
                                # unrealized_pnl = (current - entry) * size
                                roe_pct = (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100 * pos.leverage
                            else:
                                roe_pct = (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100 * pos.leverage
                            position_info['current_roe_pct'] = round(roe_pct, 2)
                            
                            # Check if can pyramid
                            next_threshold = pstate.get('next_level_threshold')
                            if next_threshold and roe_pct >= next_threshold:
                                position_info['pyramid_ready'] = True
                            else:
                                position_info['pyramid_ready'] = False
                    
                    # Count history events
                    history = pstate.get('pyramid_history', [])
                    adds = sum(1 for h in history if h.get('action') != 'deleverage')
                    deleverages = sum(1 for h in history if h.get('action') == 'deleverage')
                    position_info['total_adds'] = adds
                    position_info['total_deleverages'] = deleverages
                    
                    pyramid_data['positions'].append(position_info)
                    
                    # Update summary
                    if pstate.get('current_level', 1) > 1:
                        pyramid_data['summary']['total_pyramided_positions'] += 1
                    if pstate.get('current_level', 1) >= 3:
                        pyramid_data['summary']['positions_at_max_level'] += 1
                    pyramid_data['summary']['total_pyramid_adds'] += adds
                    pyramid_data['summary']['total_deleverages'] += deleverages
                    
            except Exception as e:
                logger.error(f"Error getting pyramid status: {e}")
        
        return APIResponse(
            success=True,
            data=pyramid_data,
            metadata={'source': 'live'}
        ).to_dict()
    
    def get_leverage_info(self) -> Dict:
        """
        GET /api/leverage
        
        Current leverage configuration and usage:
        - Confidence-based leverage tiers
        - Strategy bonuses/penalties
        - Current leverage by position
        """
        leverage_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'tiers': {
                'low': {'confidence': '55-65%', 'leverage': '2-3x'},
                'medium': {'confidence': '65-80%', 'leverage': '4-6x'},
                'high': {'confidence': '80%+', 'leverage': '8-10x'}
            },
            'absolute_max': 10,
            'absolute_min': 1,
            'positions': [],
            'strategy_bonuses': {}
        }
        
        if self.orchestrator and hasattr(self.orchestrator, 'risk_manager'):
            try:
                rm = self.orchestrator.risk_manager
                
                # Get leverage config
                lev_config = rm.config.leverage_config
                leverage_data['absolute_max'] = lev_config.absolute_max_leverage
                leverage_data['absolute_min'] = lev_config.min_leverage
                
                # Get strategy bonuses from stats
                for strategy_id, stats in rm.strategy_stats.items():
                    bonus = rm._calculate_strategy_leverage_bonus(strategy_id)
                    if bonus != 0:
                        leverage_data['strategy_bonuses'][strategy_id] = {
                            'bonus': bonus,
                            'trades': stats.get('trades', 0),
                            'win_rate': stats.get('wins', 0) / max(stats.get('trades', 1), 1) * 100
                        }
                
                # Get current leverage by position
                for symbol, pstate in rm.pyramid_states.items():
                    leverage_data['positions'].append({
                        'symbol': symbol,
                        'base_leverage': pstate.base_leverage,
                        'current_leverage': pstate.current_leverage,
                        'level': pstate.current_level
                    })
                    
            except Exception as e:
                logger.error(f"Error getting leverage info: {e}")
        
        return APIResponse(
            success=True,
            data=leverage_data,
            metadata={'source': 'live'}
        ).to_dict()
    
    # ==========================================
    # RISK ENDPOINTS
    # ==========================================
    
    def get_risk(self) -> Dict:
        """
        GET /api/risk
        
        Comprehensive risk metrics:
        - Position limits
        - Exposure limits
        - Drawdown status
        - Circuit breaker status
        - Per-symbol concentration
        - Correlation risk
        """
        risk_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'limits': {
                'max_positions': 5,
                'max_position_pct': 2.0,
                'max_exposure_pct': 20.0,
                'max_daily_loss_pct': 5.0,
                'max_drawdown_pct': 10.0
            },
            'current': {
                'positions': 0,
                'position_pct': 0,
                'exposure_pct': 0,
                'daily_loss_pct': 0,
                'drawdown_pct': 0
            },
            'utilization': {
                'position_util': 0,
                'exposure_util': 0,
                'loss_util': 0,
                'drawdown_util': 0
            },
            'circuit_breaker': {
                'active': False,
                'reason': None,
                'triggered_at': None
            },
            'concentration': [],
            'var_estimate': 0,
            'stress_test': {}
        }
        
        # Get from orchestrator risk manager
        if self.orchestrator:
            try:
                status = self.orchestrator.get_risk_status()
                
                risk_data['limits'].update({
                    'max_positions': status.get('max_positions', 5),
                    'max_position_pct': status.get('max_position_pct', 2.0),
                    'max_drawdown_pct': status.get('max_drawdown_pct', 10.0)
                })
                
                risk_data['current']['positions'] = status.get('open_positions', 0)
                risk_data['current']['drawdown_pct'] = status.get('current_drawdown_pct', 0)
                
                # Circuit breaker
                can_trade, reason = self.orchestrator.can_trade()
                risk_data['circuit_breaker']['active'] = not can_trade
                risk_data['circuit_breaker']['reason'] = reason if not can_trade else None
                
            except Exception as e:
                logger.debug(f"Risk manager unavailable: {e}")
        
        # Get from executor
        if self.executor:
            try:
                status = self.executor.get_status()
                risk_config = status.get('risk_config', {})
                dd = status.get('drawdown_protection', {})
                
                risk_data['limits']['max_positions'] = risk_config.get('max_positions', 5)
                risk_data['limits']['max_position_pct'] = risk_config.get('max_position_pct', 2.0)
                risk_data['limits']['max_exposure_pct'] = risk_config.get('max_total_exposure_pct', 20.0)
                risk_data['limits']['max_daily_loss_pct'] = risk_config.get('max_daily_loss_pct', 5.0)
                
                risk_data['current']['positions'] = len(self.executor.positions)
                risk_data['current']['drawdown_pct'] = dd.get('current_drawdown_pct', 0)
                
                # Calculate utilizations
                risk_data['utilization']['position_util'] = round(
                    len(self.executor.positions) / max(risk_config.get('max_positions', 5), 1) * 100, 1
                )
                
                # Concentration by symbol
                total_exposure = sum(p.margin for p in self.executor.positions.values())
                for symbol, pos in self.executor.positions.items():
                    if total_exposure > 0:
                        concentration = pos.margin / total_exposure * 100
                        risk_data['concentration'].append({
                            'symbol': symbol,
                            'exposure_pct': round(concentration, 1),
                            'pnl': pos.unrealized_pnl
                        })
                
                # Sort by exposure
                risk_data['concentration'].sort(key=lambda x: x['exposure_pct'], reverse=True)
                
            except Exception as e:
                logger.debug(f"Executor unavailable: {e}")
        
        return APIResponse(
            success=True,
            data=risk_data,
            metadata={'source': 'live'}
        ).to_dict()
    
    # ==========================================
    # CHART ENDPOINTS
    # ==========================================
    
    def get_chart(self, symbol: str, interval: str = '15m', limit: int = 200) -> Dict:
        """
        GET /api/chart/:symbol
        
        OHLCV data for charting:
        - Timestamp, OHLC, Volume
        - Optional: indicators, signals overlay
        """
        candles = []
        
        if self.data_feed:
            try:
                data = self.data_feed.get_symbol_data(symbol)
                if data:
                    timestamps = data.get('timestamp', [])
                    opens = data.get('open', [])
                    highs = data.get('high', [])
                    lows = data.get('low', [])
                    closes = data.get('close', [])
                    volumes = data.get('volume', [])
                    
                    for i in range(len(closes)):
                        ts = timestamps[i] if i < len(timestamps) else None
                        if hasattr(ts, 'isoformat'):
                            ts = ts.isoformat()
                        elif ts is None:
                            ts = ''
                        
                        candles.append({
                            'timestamp': ts,
                            'open': opens[i] if i < len(opens) else 0,
                            'high': highs[i] if i < len(highs) else 0,
                            'low': lows[i] if i < len(lows) else 0,
                            'close': closes[i] if i < len(closes) else 0,
                            'volume': volumes[i] if i < len(volumes) else 0
                        })
            except Exception as e:
                logger.error(f"Error getting chart data: {e}")
        
        # Fallback: fetch directly from KuCoin
        if not candles:
            try:
                from data.feed import KuCoinDataFeed
                feed = KuCoinDataFeed([symbol], interval)
                fetched = feed.fetch_candles(symbol, limit)
                for c in fetched:
                    candles.append({
                        'timestamp': c.timestamp.isoformat() if hasattr(c.timestamp, 'isoformat') else str(c.timestamp),
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    })
            except Exception as e:
                logger.error(f"Error fetching chart data: {e}")
        
        # Limit results
        candles = candles[-limit:]
        
        # Get signals for overlay
        signals_overlay = self._get_signals_for_symbol(symbol, limit=20)
        
        # Get position info if exists
        position = None
        if self.executor and symbol in self.executor.positions:
            pos = self.executor.positions[symbol]
            position = {
                'side': pos.side,
                'entry_price': pos.entry_price,
                'size': pos.size
            }
        
        return APIResponse(
            success=True,
            data={
                'symbol': symbol,
                'interval': interval,
                'candles': candles,
                'signals': signals_overlay,
                'position': position
            },
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'count': len(candles)
            }
        ).to_dict()
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    def _load_trades(self) -> List[Dict]:
        """Load trades from history file with caching"""
        now = datetime.utcnow()
        
        # Check cache
        if (self._trades_cache is not None and 
            self._trades_cache_time is not None and
            (now - self._trades_cache_time).total_seconds() < self._cache_ttl):
            return self._trades_cache
        
        trades = []
        if TRADES_LOG.exists():
            try:
                with open(TRADES_LOG, 'r') as f:
                    for line in f:
                        try:
                            trades.append(json.loads(line.strip()))
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
        
        # Also load from strategy tracker
        try:
            from execution.strategy_tracker import tracker
            for trade in tracker.closed_trades:
                trades.append({
                    'timestamp': trade.exit_time,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'strategy_id': trade.strategy_id,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'pnl_pct': (trade.pnl / (trade.entry_price * trade.size) * 100) if trade.entry_price and trade.size else 0,
                    'won': trade.pnl > 0 if trade.pnl else False,
                    'status': trade.status
                })
        except Exception as e:
            logger.debug(f"Could not load from tracker: {e}")
        
        # Update cache
        self._trades_cache = trades
        self._trades_cache_time = now
        
        return trades
    
    def _load_signals(self) -> List[Dict]:
        """Load signals from history file"""
        signals = []
        if SIGNALS_LOG.exists():
            try:
                with open(SIGNALS_LOG, 'r') as f:
                    for line in f:
                        try:
                            signals.append(json.loads(line.strip()))
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error loading signals: {e}")
        return signals
    
    def _get_signals_for_symbol(self, symbol: str, limit: int = 10, around_time: str = None) -> List[Dict]:
        """Get recent signals for a symbol"""
        signals = self._load_signals()
        filtered = [s for s in signals if s.get('symbol') == symbol]
        filtered.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return filtered[:limit]
    
    def _get_signals_for_strategy(self, strategy_id: str, limit: int = 50) -> List[Dict]:
        """Get signals for a strategy"""
        signals = self._load_signals()
        filtered = [s for s in signals if s.get('strategy_id') == strategy_id]
        filtered.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return filtered[:limit]
    
    def _get_price_history(self, symbol: str) -> List[Dict]:
        """Get recent price history for a symbol"""
        if self.data_feed:
            data = self.data_feed.get_symbol_data(symbol)
            if data and data.get('close'):
                closes = data['close'][-50:]
                timestamps = data.get('timestamp', [])[-50:]
                return [
                    {
                        'timestamp': str(timestamps[i]) if i < len(timestamps) else '',
                        'price': closes[i]
                    }
                    for i in range(len(closes))
                ]
        return []
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        trades = self._load_trades()
        
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_estimate': 0
            }
        
        wins = sum(1 for t in trades if t.get('won'))
        total = len(trades)
        win_rate = wins / total * 100 if total > 0 else 0
        
        # Profit factor = gross profits / gross losses
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Simple Sharpe estimate (not annualized)
        returns = [t.get('pnl_pct', 0) for t in trades]
        if len(returns) > 1:
            import statistics
            try:
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                sharpe = avg_return / std_return if std_return > 0 else 0
            except:
                sharpe = 0
        else:
            sharpe = 0
        
        return {
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'sharpe_estimate': round(sharpe, 2)
        }
