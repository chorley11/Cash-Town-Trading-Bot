"""
Strategy Position Tracker
Maps positions to their originating strategies for performance tracking
"""
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TrackedTrade:
    symbol: str
    strategy_id: str
    side: str
    entry_price: float
    entry_time: str
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    status: str = 'open'  # open, closed, stopped

class StrategyTracker:
    """Tracks which strategy opened which position"""
    
    def __init__(self, data_path: str = None):
        # Use DATA_DIR env var (set on Railway) or fallback to local path
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        self.data_path = data_path or os.path.join(data_dir, 'strategy_positions.json')
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        self.positions: Dict[str, TrackedTrade] = {}
        self.closed_trades: List[TrackedTrade] = []
        self._load()
    
    def _load(self):
        """Load from disk"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.positions = {
                        k: TrackedTrade(**v) for k, v in data.get('positions', {}).items()
                    }
                    self.closed_trades = [
                        TrackedTrade(**t) for t in data.get('closed_trades', [])[-500:]  # Keep last 500
                    ]
            except Exception as e:
                logger.warning(f"Could not load strategy tracker: {e}")
    
    def _save(self):
        """Save to disk"""
        try:
            data = {
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'closed_trades': [asdict(t) for t in self.closed_trades[-500:]]
            }
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save strategy tracker: {e}")
    
    def open_position(self, symbol: str, strategy_id: str, side: str, 
                      entry_price: float, size: float,
                      stop_loss: float = None, take_profit: float = None):
        """Record a new position"""
        trade = TrackedTrade(
            symbol=symbol,
            strategy_id=strategy_id,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.utcnow().isoformat(),
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.positions[symbol] = trade
        self._save()
        logger.info(f"Tracked: {strategy_id} opened {side} {symbol}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'manual'):
        """Close a tracked position"""
        if symbol not in self.positions:
            return
        
        trade = self.positions.pop(symbol)
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow().isoformat()
        trade.status = 'closed'
        
        # Calculate PnL
        if trade.side == 'long':
            trade.pnl = (exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size
        
        self.closed_trades.append(trade)
        self._save()
        logger.info(f"Closed: {trade.strategy_id} {symbol} PnL=${trade.pnl:.2f}")
    
    def get_strategy(self, symbol: str) -> Optional[str]:
        """Get strategy for a position"""
        if symbol in self.positions:
            return self.positions[symbol].strategy_id
        return None
    
    def get_all_positions(self) -> Dict[str, TrackedTrade]:
        """Get all open positions with strategy info"""
        return self.positions
    
    def get_strategy_stats(self) -> Dict[str, dict]:
        """Get performance stats by strategy"""
        stats = {}
        
        # Open positions
        for trade in self.positions.values():
            if trade.strategy_id not in stats:
                stats[trade.strategy_id] = {
                    'open_positions': 0, 'closed_trades': 0,
                    'total_pnl': 0, 'wins': 0, 'losses': 0
                }
            stats[trade.strategy_id]['open_positions'] += 1
        
        # Closed trades
        for trade in self.closed_trades:
            if trade.strategy_id not in stats:
                stats[trade.strategy_id] = {
                    'open_positions': 0, 'closed_trades': 0,
                    'total_pnl': 0, 'wins': 0, 'losses': 0
                }
            stats[trade.strategy_id]['closed_trades'] += 1
            stats[trade.strategy_id]['total_pnl'] += trade.pnl or 0
            if (trade.pnl or 0) > 0:
                stats[trade.strategy_id]['wins'] += 1
            else:
                stats[trade.strategy_id]['losses'] += 1
        
        # Calculate win rates
        for s in stats.values():
            total = s['wins'] + s['losses']
            s['win_rate'] = s['wins'] / total * 100 if total > 0 else 0
        
        return stats

# Global instance
tracker = StrategyTracker()
