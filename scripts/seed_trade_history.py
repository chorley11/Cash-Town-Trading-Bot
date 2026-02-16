#!/usr/bin/env python3
"""
Seed Trade History - Fetch from Notion and create trades_history.jsonl

This script:
1. Fetches all trades from Notion 📊 Trade Log
2. Converts to Cash Town format
3. Writes to data/trades_history.jsonl
4. Aggregates strategy performance

Usage:
    python scripts/seed_trade_history.py
    python scripts/seed_trade_history.py --output /app/data/trades_history.jsonl
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import requests

NOTION_API_KEY = os.environ.get('NOTION_API_KEY', '')
NOTION_DATABASE_ID = '2fb5ede8110c81a2a229f28e58ba3679'  # No hyphens

# Try to load from file if not in env
if not NOTION_API_KEY:
    key_file = Path.home() / '.config/notion/api_key'
    if key_file.exists():
        NOTION_API_KEY = key_file.read_text().strip()


def fetch_notion_trades(limit: int = 500) -> list:
    """Fetch trades from Notion database"""
    headers = {
        'Authorization': f'Bearer {NOTION_API_KEY}',
        'Notion-Version': '2022-06-28',
        'Content-Type': 'application/json'
    }
    
    all_trades = []
    has_more = True
    start_cursor = None
    
    while has_more and len(all_trades) < limit:
        payload = {
            'page_size': min(100, limit - len(all_trades)),
            'sorts': [{'property': 'Closed', 'direction': 'descending'}]
        }
        if start_cursor:
            payload['start_cursor'] = start_cursor
        
        response = requests.post(
            f'https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Error fetching from Notion: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        all_trades.extend(data.get('results', []))
        has_more = data.get('has_more', False)
        start_cursor = data.get('next_cursor')
        
        print(f"Fetched {len(all_trades)} trades...")
    
    return all_trades


def convert_to_cash_town_format(notion_trade: dict) -> dict:
    """Convert Notion trade to Cash Town trades_history.jsonl format"""
    props = notion_trade.get('properties', {})
    
    # Extract properties safely
    def get_date(prop_name):
        prop = props.get(prop_name, {})
        if prop.get('date'):
            return prop['date'].get('start')
        return None
    
    def get_select(prop_name):
        prop = props.get(prop_name, {})
        if prop.get('select'):
            return prop['select'].get('name')
        return None
    
    def get_number(prop_name):
        prop = props.get(prop_name, {})
        return prop.get('number', 0) or 0
    
    def get_title(prop_name):
        prop = props.get(prop_name, {})
        if prop.get('title') and len(prop['title']) > 0:
            return prop['title'][0].get('plain_text', '')
        return ''
    
    # Infer symbol from entry price
    entry_price = get_number('Entry')
    symbol = get_select('Symbol')
    if not symbol:
        # Infer from price range
        if entry_price > 50000:
            symbol = 'XBTUSDTM'
        elif entry_price > 2000:
            symbol = 'ETHUSDTM'
        elif entry_price > 100:
            symbol = 'SOLUSDTM'
        elif entry_price > 50:
            symbol = 'LINKUSDTM'
        else:
            symbol = 'UNKNOWN'
    
    # Infer strategy if missing
    strategy = get_select('Strategy')
    if not strategy:
        strategy = 'cucurbit'  # Default to Cucurbit for unattributed
    
    pnl = get_number('PnL')
    
    return {
        'id': notion_trade.get('id', ''),
        'symbol': symbol,
        'side': get_select('Side') or 'unknown',
        'entry_price': entry_price,
        'exit_price': get_number('Exit'),
        'size': get_number('Size'),
        'pnl': pnl,
        'pnl_pct': get_number('PnL %'),
        'strategy_id': strategy.lower().replace(' ', '-'),
        'entry_time': get_date('Opened'),
        'exit_time': get_date('Closed'),
        'close_reason': 'tp' if pnl > 0 else 'sl',
        'source': 'notion_seed'
    }


def calculate_strategy_performance(trades: list) -> dict:
    """Calculate performance metrics per strategy"""
    performance = {}
    
    for trade in trades:
        sid = trade.get('strategy_id', 'unknown')
        if sid not in performance:
            performance[sid] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        pnl = trade.get('pnl', 0)
        performance[sid]['trades'] += 1
        performance[sid]['total_pnl'] += pnl
        
        if pnl > 0:
            performance[sid]['wins'] += 1
            performance[sid]['max_win'] = max(performance[sid]['max_win'], pnl)
        else:
            performance[sid]['losses'] += 1
            performance[sid]['max_loss'] = min(performance[sid]['max_loss'], pnl)
    
    # Calculate win rates
    for sid, data in performance.items():
        total = data['trades']
        data['win_rate'] = data['wins'] / total if total > 0 else 0
        data['avg_pnl'] = data['total_pnl'] / total if total > 0 else 0
    
    return performance


def main():
    parser = argparse.ArgumentParser(description='Seed trade history from Notion')
    parser.add_argument('--output', default='data/trades_history.jsonl',
                       help='Output file path')
    parser.add_argument('--limit', type=int, default=500,
                       help='Maximum trades to fetch')
    parser.add_argument('--perf-output', default='data/strategy_performance.json',
                       help='Strategy performance output')
    args = parser.parse_args()
    
    if not NOTION_API_KEY:
        print("Error: NOTION_API_KEY not set")
        sys.exit(1)
    
    print(f"Fetching up to {args.limit} trades from Notion...")
    notion_trades = fetch_notion_trades(args.limit)
    print(f"Got {len(notion_trades)} trades from Notion")
    
    # Convert to Cash Town format
    trades = []
    for nt in notion_trades:
        try:
            trade = convert_to_cash_town_format(nt)
            if trade.get('exit_time'):  # Only include closed trades
                trades.append(trade)
        except Exception as e:
            print(f"Error converting trade: {e}")
    
    print(f"Converted {len(trades)} valid trades")
    
    # Write trades JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for trade in trades:
            f.write(json.dumps(trade) + '\n')
    
    print(f"Wrote trades to {output_path}")
    
    # Calculate and write strategy performance
    performance = calculate_strategy_performance(trades)
    
    perf_path = Path(args.perf_output)
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(perf_path, 'w') as f:
        json.dump(performance, f, indent=2)
    
    print(f"Wrote strategy performance to {perf_path}")
    
    # Print summary
    print("\n=== Strategy Performance Summary ===")
    for sid, data in sorted(performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        print(f"{sid:20s}: {data['trades']:3d} trades, {data['win_rate']*100:5.1f}% WR, ${data['total_pnl']:8.2f} PnL")


if __name__ == '__main__':
    main()
