#!/usr/bin/env python3
"""
Cash Town Trader - Main entry point for the trading system

Usage:
    python run_trader.py paper    # Paper trading mode (default)
    python run_trader.py live     # Live trading mode (requires API keys)
    python run_trader.py status   # Show status only
"""
import sys
import os
import logging
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.loop import TradingLoop, LoopConfig
from execution.engine import ExecutionMode
from execution.kucoin import KuCoinFuturesExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/cash-town-trader.log')
    ]
)
logger = logging.getLogger('cash-town')

def print_banner():
    """Print Cash Town banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     💰 CASH TOWN TRADING SYSTEM 💰                       ║
    ║     Multi-Agent Autonomous Trader                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_credentials():
    """Check if API credentials are configured"""
    creds_path = os.path.expanduser("~/.config/kucoin/cash_town_credentials.json")
    if os.path.exists(creds_path):
        try:
            with open(creds_path) as f:
                creds = json.load(f)
                if all(k in creds for k in ['api_key', 'api_secret', 'api_passphrase']):
                    return True, creds_path
        except:
            pass
    return False, creds_path

def show_status():
    """Show current system status"""
    print_banner()
    
    # Check credentials
    has_creds, creds_path = check_credentials()
    print(f"📋 System Status")
    print(f"   API Credentials: {'✅ Configured' if has_creds else '❌ Not found'}")
    print(f"   Credentials path: {creds_path}")
    
    # Check orchestrator
    import requests
    try:
        resp = requests.get('http://localhost:8888/health', timeout=2)
        if resp.status_code == 200:
            print(f"   Orchestrator: ✅ Running")
        else:
            print(f"   Orchestrator: ❌ Error")
    except:
        print(f"   Orchestrator: ❌ Not running")
    
    # Test KuCoin connection
    if has_creds:
        executor = KuCoinFuturesExecutor(creds_path)
        try:
            account = executor.get_account_overview()
            balance = float(account.get('availableBalance', 0))
            print(f"   KuCoin Balance: ${balance:,.2f} USDT")
        except Exception as e:
            print(f"   KuCoin Connection: ❌ {e}")
    
    print()

def run_paper():
    """Run in paper trading mode"""
    print_banner()
    print("📝 Starting PAPER TRADING mode")
    print("   No real trades will be executed")
    print()
    
    config = LoopConfig(
        execution_mode=ExecutionMode.PAPER,
        symbols=['XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'AVAXUSDTM', 'LINKUSDTM'],
        enabled_strategies=['bts-lynch', 'zweig', 'trend-following', 'mean-reversion', 'turtle', 'weinstein', 'livermore'],
        signal_check_seconds=300,  # 5 minutes
        data_refresh_seconds=60
    )
    
    loop = TradingLoop(config)
    
    try:
        loop.start()
        
        print("✅ Trading loop started. Press Ctrl+C to stop.")
        print()
        
        while loop.running:
            time.sleep(60)
            status = loop.get_status()
            
            # Print periodic status
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                  f"Cycles: {status['stats']['cycles']} | "
                  f"Signals: {status['stats']['signals_generated']} | "
                  f"Trades: {status['stats']['trades_executed']} | "
                  f"Errors: {status['stats']['errors']}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    finally:
        loop.stop()
        print("✅ Stopped")

def run_live():
    """Run in live trading mode"""
    print_banner()
    
    # Check credentials
    has_creds, creds_path = check_credentials()
    if not has_creds:
        print("❌ Cannot start live trading - API credentials not configured")
        print(f"   Please create: {creds_path}")
        print('   With format: {"api_key": "...", "api_secret": "...", "api_passphrase": "..."}')
        sys.exit(1)
    
    # Confirm live trading
    print("🔥 Starting LIVE TRADING mode")
    print("   ⚠️  REAL TRADES WILL BE EXECUTED")
    print()
    
    # Auto-confirm if environment variable set (for systemd)
    if os.environ.get('CASH_TOWN_AUTO_CONFIRM') == '1':
        print("Auto-confirmed via CASH_TOWN_AUTO_CONFIRM")
    else:
        response = input("Type 'YES' to confirm: ")
        if response != 'YES':
            print("Cancelled")
            sys.exit(0)
    
    config = LoopConfig(
        execution_mode=ExecutionMode.LIVE,
        symbols=['XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM'],  # Fewer symbols for live
        enabled_strategies=['trend-following', 'turtle', 'weinstein'],  # Conservative strategies
        signal_check_seconds=300,
        data_refresh_seconds=60
    )
    
    loop = TradingLoop(config)
    
    try:
        loop.start()
        
        print("✅ LIVE trading started. Press Ctrl+C to stop.")
        print()
        
        while loop.running:
            time.sleep(60)
            status = loop.get_status()
            
            exec_status = status['components']['execution']
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                  f"Balance: ${exec_status['account_balance']:,.2f} | "
                  f"Positions: {exec_status['positions']} | "
                  f"Trades: {status['stats']['trades_executed']} | "
                  f"Mode: LIVE")
            
            if exec_status['killed']:
                print(f"🛑 KILL SWITCH: {exec_status['kill_reason']}")
                break
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    finally:
        loop.stop()
        print("✅ Stopped")

def main():
    if len(sys.argv) < 2:
        mode = 'paper'
    else:
        mode = sys.argv[1].lower()
    
    if mode == 'status':
        show_status()
    elif mode == 'paper':
        run_paper()
    elif mode == 'live':
        run_live()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run_trader.py [paper|live|status]")
        sys.exit(1)

if __name__ == '__main__':
    main()
