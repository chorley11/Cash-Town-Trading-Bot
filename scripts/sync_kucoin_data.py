#!/usr/bin/env python3
"""
Sync trading data from KuCoin Futures API.
Fetches trades since Jan 21, 2026 and updates local data files.
"""

import json
import hmac
import hashlib
import base64
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# KuCoin Futures API
BASE_URL = "https://api-futures.kucoin.com"

def load_credentials():
    """Load KuCoin credentials from config."""
    cred_path = Path.home() / ".config/kucoin/cash_town_credentials.json"
    with open(cred_path) as f:
        return json.load(f)

def sign_request(secret: str, timestamp: str, method: str, endpoint: str, body: str = ""):
    """Generate KuCoin API signature."""
    str_to_sign = timestamp + method + endpoint + body
    signature = base64.b64encode(
        hmac.new(secret.encode(), str_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    return signature

def make_request(method: str, endpoint: str, params: dict = None):
    """Make authenticated request to KuCoin Futures API."""
    creds = load_credentials()
    
    url = BASE_URL + endpoint
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        endpoint = f"{endpoint}?{query}"
        url = BASE_URL + endpoint
    
    timestamp = str(int(time.time() * 1000))
    signature = sign_request(creds["api_secret"], timestamp, method, endpoint)
    passphrase = base64.b64encode(
        hmac.new(creds["api_secret"].encode(), creds["api_passphrase"].encode(), hashlib.sha256).digest()
    ).decode()
    
    headers = {
        "KC-API-KEY": creds["api_key"],
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": timestamp,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }
    
    response = requests.request(method, url, headers=headers)
    return response.json()

def get_fills(start_time_ms: int):
    """Get all fills/trades since start time."""
    all_fills = []
    current_page = 1
    page_size = 100
    
    while True:
        params = {
            "startAt": start_time_ms,
            "pageSize": page_size,
            "currentPage": current_page
        }
        
        result = make_request("GET", "/api/v1/fills", params)
        
        if result.get("code") != "200000":
            print(f"Error fetching fills: {result}")
            break
        
        data = result.get("data", {})
        items = data.get("items", [])
        
        if not items:
            break
            
        all_fills.extend(items)
        print(f"  Page {current_page}: {len(items)} fills")
        
        total_pages = data.get("totalPage", 1)
        if current_page >= total_pages:
            break
            
        current_page += 1
        time.sleep(0.2)  # Rate limiting
    
    return all_fills

def get_positions():
    """Get current open positions."""
    result = make_request("GET", "/api/v1/positions")
    
    if result.get("code") != "200000":
        print(f"Error fetching positions: {result}")
        return []
    
    return result.get("data", [])

def get_account_overview():
    """Get account balance overview (USDT-M)."""
    # Get USDT-margined account
    result = make_request("GET", "/api/v1/account-overview", {"currency": "USDT"})
    
    if result.get("code") != "200000":
        print(f"Error fetching account: {result}")
        return {}
    
    return result.get("data", {})

def load_existing_trades(filepath: Path):
    """Load existing trades from JSONL file."""
    trades = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
    return trades

def convert_fill_to_trade(fill: dict):
    """Convert a KuCoin fill to our trade format."""
    return {
        "id": fill.get("tradeId") or fill.get("orderId"),
        "symbol": fill.get("symbol", "UNKNOWN"),
        "side": fill.get("side", "unknown").lower(),
        "entry_price": float(fill.get("price", 0)),
        "exit_price": float(fill.get("price", 0)),
        "size": float(fill.get("value", 0)),
        "pnl": float(fill.get("realizedPnl", 0)),
        "pnl_pct": 0,  # Calculated later if possible
        "strategy_id": "kucoin-sync",
        "entry_time": datetime.fromtimestamp(fill.get("tradeTime", 0) / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.fromtimestamp(fill.get("tradeTime", 0) / 1000, tz=timezone.utc).isoformat(),
        "close_reason": fill.get("type", "trade"),
        "source": "kucoin_api",
        "fee": float(fill.get("fee", 0)),
        "fee_currency": fill.get("feeCurrency", "USDT"),
        "order_id": fill.get("orderId"),
        "leverage": fill.get("leverage", 1)
    }

def calculate_strategy_performance(trades: list):
    """Calculate performance metrics per strategy."""
    stats = defaultdict(lambda: {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "pnls": []
    })
    
    for trade in trades:
        strategy = trade.get("strategy_id", "unknown")
        pnl = float(trade.get("pnl", 0))
        
        stats[strategy]["trades"] += 1
        stats[strategy]["total_pnl"] += pnl
        stats[strategy]["pnls"].append(pnl)
        
        if pnl > 0:
            stats[strategy]["wins"] += 1
        elif pnl < 0:
            stats[strategy]["losses"] += 1
    
    result = {}
    for strategy, data in stats.items():
        trades_count = data["trades"]
        pnls = data["pnls"]
        
        result[strategy] = {
            "trades": trades_count,
            "wins": data["wins"],
            "losses": data["losses"],
            "total_pnl": round(data["total_pnl"], 2),
            "win_rate": round(data["wins"] / trades_count, 4) if trades_count > 0 else 0,
            "avg_pnl": round(data["total_pnl"] / trades_count, 4) if trades_count > 0 else 0,
            "max_win": round(max(pnls), 2) if pnls else 0,
            "max_loss": round(min(pnls), 2) if pnls else 0
        }
    
    return result

def main():
    data_dir = Path(__file__).parent.parent / "data"
    
    print("=" * 60)
    print("Cash Town KuCoin Data Sync")
    print("=" * 60)
    
    # Start date: Jan 21, 2026
    start_date = datetime(2026, 1, 21, 0, 0, 0, tzinfo=timezone.utc)
    start_time_ms = int(start_date.timestamp() * 1000)
    
    print(f"\nFetching fills since {start_date.isoformat()}...")
    
    # Fetch fills from KuCoin
    fills = get_fills(start_time_ms)
    print(f"\nTotal fills fetched: {len(fills)}")
    
    # Load existing trades
    trades_file = data_dir / "trades_history.jsonl"
    existing_trades = load_existing_trades(trades_file)
    existing_ids = {t.get("id") for t in existing_trades}
    print(f"Existing trades in file: {len(existing_trades)}")
    
    # Convert and merge new fills
    new_trades = []
    for fill in fills:
        trade = convert_fill_to_trade(fill)
        if trade["id"] not in existing_ids:
            new_trades.append(trade)
    
    print(f"New trades to add: {len(new_trades)}")
    
    # Append new trades to file
    if new_trades:
        with open(trades_file, "a") as f:
            for trade in new_trades:
                f.write(json.dumps(trade) + "\n")
        print(f"Appended {len(new_trades)} trades to {trades_file}")
    
    # Get current positions
    print("\nFetching current positions...")
    positions = get_positions()
    account = get_account_overview()
    
    # Create positions snapshot
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": {
            "currency": account.get("currency", "USDT"),
            "accountEquity": float(account.get("accountEquity", 0)),
            "unrealisedPNL": float(account.get("unrealisedPNL", 0)),
            "marginBalance": float(account.get("marginBalance", 0)),
            "availableBalance": float(account.get("availableBalance", 0)),
            "positionMargin": float(account.get("positionMargin", 0)),
            "orderMargin": float(account.get("orderMargin", 0)),
        },
        "positions": []
    }
    
    for pos in positions:
        if pos.get("isOpen"):
            snapshot["positions"].append({
                "symbol": pos.get("symbol"),
                "side": "long" if pos.get("currentQty", 0) > 0 else "short",
                "size": abs(float(pos.get("currentQty", 0))),
                "entry_price": float(pos.get("avgEntryPrice", 0)),
                "mark_price": float(pos.get("markPrice", 0)),
                "liquidation_price": float(pos.get("liquidationPrice", 0)),
                "unrealized_pnl": float(pos.get("unrealisedPnl", 0)),
                "unrealized_pnl_pct": float(pos.get("unrealisedPnlPcnt", 0)) * 100,
                "leverage": pos.get("realLeverage", 0),
                "margin": float(pos.get("posMaint", 0)),
                "strategy": "manual"  # Can be updated based on tagging
            })
    
    positions_file = data_dir / "positions_snapshot.json"
    with open(positions_file, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Saved positions snapshot: {len(snapshot['positions'])} open positions")
    
    # Update strategy performance
    print("\nCalculating strategy performance...")
    all_trades = existing_trades + new_trades
    performance = calculate_strategy_performance(all_trades)
    
    perf_file = data_dir / "strategy_performance.json"
    with open(perf_file, "w") as f:
        json.dump(performance, f, indent=2)
    print(f"Updated strategy performance for {len(performance)} strategies")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"Total trades in history: {len(all_trades)}")
    print(f"New trades added: {len(new_trades)}")
    print(f"Open positions: {len(snapshot['positions'])}")
    print(f"Account equity: ${snapshot['account']['accountEquity']:.2f}")
    print(f"Unrealized PnL: ${snapshot['account']['unrealisedPNL']:.2f}")
    print("=" * 60)
    
    return len(new_trades)

if __name__ == "__main__":
    main()
