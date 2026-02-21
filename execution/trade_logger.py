"""
Trade Logger - Records completed trades to trades_history.jsonl

This module captures trade data at close time since the KuCoin API
doesn't reliably return historical trade data.
"""
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
import uuid

logger = logging.getLogger(__name__)

# Data directory - Railway sets DATA_DIR, local dev uses ./data
DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
TRADES_FILE = DATA_DIR / 'trades_history.jsonl'

def log_trade(
    symbol: str,
    side: str,  # 'long' or 'short'
    entry_price: float,
    exit_price: float,
    size: float,
    pnl: float,
    strategy_id: str = 'unknown',
    entry_time: Optional[str] = None,
    close_reason: str = 'unknown',
    leverage: int = 5,
    fees: float = 0.0,
    notes: str = ''
) -> Dict:
    """
    Log a completed trade to trades_history.jsonl
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDTM')
        side: 'long' or 'short'
        entry_price: Entry price
        exit_price: Exit price
        size: Position size (notional value in USD)
        pnl: Realized profit/loss
        strategy_id: Which strategy triggered this trade
        entry_time: ISO timestamp of entry (optional)
        close_reason: Why the trade was closed
        leverage: Leverage used
        fees: Trading fees paid
        notes: Any additional notes
    
    Returns:
        The trade record that was logged
    """
    now = datetime.now(timezone.utc)
    
    # Calculate PnL percentage
    pnl_pct = (pnl / size * 100) if size > 0 else 0
    
    trade = {
        'id': str(uuid.uuid4()),
        'symbol': symbol,
        'side': side,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'size': size,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 4),
        'strategy_id': strategy_id,
        'entry_time': entry_time or now.isoformat(),
        'exit_time': now.isoformat(),
        'close_reason': close_reason,
        'leverage': leverage,
        'fees': fees,
        'source': 'live_close',
        'notes': notes
    }
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Append to JSONL file
    try:
        with open(TRADES_FILE, 'a') as f:
            f.write(json.dumps(trade) + '\n')
        logger.info(f"📝 Trade logged: {symbol} {side} PnL=${pnl:+.2f} ({close_reason})")
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")
    
    return trade


def get_recent_trades(limit: int = 100) -> list:
    """Get most recent trades from history"""
    trades = []
    
    if not TRADES_FILE.exists():
        return trades
    
    try:
        with open(TRADES_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        
        # Return most recent
        return trades[-limit:]
    except Exception as e:
        logger.error(f"Failed to read trades: {e}")
        return []


def get_daily_pnl(date_str: str = None) -> float:
    """Get total PnL for a specific date (YYYY-MM-DD) or today"""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    total = 0.0
    
    if not TRADES_FILE.exists():
        return total
    
    try:
        with open(TRADES_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    exit_time = trade.get('exit_time', '')
                    if exit_time.startswith(date_str):
                        total += trade.get('pnl', 0)
        return total
    except Exception as e:
        logger.error(f"Failed to calculate daily PnL: {e}")
        return 0.0
