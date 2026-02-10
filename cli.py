#!/usr/bin/env python3
"""
Cash Town CLI - Command line interface for managing the trading swarm
"""
import argparse
import json
import os
import requests
import sys

def simple_table(rows, headers):
    """Simple table formatting without tabulate"""
    if not rows:
        return ""
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Build output
    lines = []
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    
    return "\n".join(lines)

DEFAULT_URL = "http://localhost:8888"

def get_base_url():
    return DEFAULT_URL

def cmd_status(args):
    """Show orchestrator status"""
    try:
        resp = requests.get(f"{get_base_url()}/summary", timeout=5)
        data = resp.json()
        
        print("\n🏭 CASH TOWN ORCHESTRATOR")
        print("=" * 50)
        print(f"Status: {'🟢 Running' if data['orchestrator']['running'] else '🔴 Stopped'}")
        print(f"Time: {data['orchestrator']['timestamp']}")
        
        # Portfolio summary
        portfolio = data['portfolio']
        print(f"\n📊 PORTFOLIO SUMMARY")
        print("-" * 50)
        print(f"Total Agents: {portfolio['total_agents']} ({portfolio['healthy_agents']} healthy)")
        print(f"Total Positions: {portfolio['total_positions']}")
        print(f"Total Value: ${portfolio['total_value']:,.2f}")
        print(f"Unrealized PnL: ${portfolio['total_unrealized_pnl']:,.2f}")
        print(f"Daily PnL: ${portfolio['total_daily_pnl']:,.2f}")
        
        # Agents table
        if data['agents']:
            print(f"\n🤖 AGENTS")
            print("-" * 50)
            rows = []
            for agent in data['agents']:
                status = data['statuses'].get(agent['id'], {})
                health = '🟢' if status.get('healthy') else '🔴' if status else '⚪'
                rows.append([
                    health,
                    agent['id'],
                    agent['name'],
                    agent['type'],
                    f"{agent['allocation_pct']}%",
                    status.get('open_positions', '-'),
                    f"${status.get('unrealized_pnl', 0):,.2f}" if status else '-'
                ])
            print(simple_table(rows, ['', 'ID', 'Name', 'Type', 'Alloc', 'Pos', 'PnL']))
        
        # Recent alerts
        if data['alerts']:
            print(f"\n⚠️ RECENT ALERTS")
            print("-" * 50)
            for alert in data['alerts'][-5:]:
                icon = '🔴' if alert['severity'] == 'critical' else '🟡' if alert['severity'] == 'error' else '🟠'
                print(f"{icon} [{alert['timestamp'][:19]}] {alert['message']}")
        
        print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator. Is it running?")
        sys.exit(1)

def cmd_agents(args):
    """List all agents"""
    try:
        resp = requests.get(f"{get_base_url()}/agents", timeout=5)
        data = resp.json()
        
        if not data['agents']:
            print("No agents registered")
            return
        
        print(json.dumps(data, indent=2))
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_register(args):
    """Register a new agent"""
    agent_data = {
        'id': args.id,
        'name': args.name,
        'type': args.type,
        'endpoint': args.endpoint,
        'api_key': args.api_key,
        'allocation_pct': args.allocation or 0,
        'max_positions': args.max_positions or 10,
        'max_drawdown_pct': args.max_drawdown or 5.0
    }
    
    try:
        resp = requests.post(f"{get_base_url()}/agents", json=agent_data, timeout=5)
        if resp.status_code == 201:
            print(f"✅ Agent '{args.name}' registered successfully")
        else:
            print(f"❌ Error: {resp.json().get('error', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_remove(args):
    """Remove an agent"""
    try:
        resp = requests.delete(f"{get_base_url()}/agents/{args.id}", timeout=5)
        if resp.status_code == 200:
            print(f"✅ Agent '{args.id}' removed")
        else:
            print(f"❌ Error: {resp.json().get('error', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_refresh(args):
    """Trigger health check refresh"""
    try:
        resp = requests.post(f"{get_base_url()}/refresh", timeout=30)
        data = resp.json()
        print("✅ Health check complete")
        print(json.dumps(data['statuses'], indent=2))
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_portfolio(args):
    """Show portfolio summary"""
    try:
        resp = requests.get(f"{get_base_url()}/portfolio", timeout=5)
        data = resp.json()
        print(json.dumps(data, indent=2))
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_positions(args):
    """Show position status and rotation info"""
    try:
        resp = requests.get(f"{get_base_url()}/positions", timeout=5)
        data = resp.json()
        
        print(f"\n📊 POSITION STATUS")
        print("=" * 60)
        print(f"Total Positions: {data['total_positions']}")
        print(f"Active Cooldowns: {data['cooldowns_active']}")
        
        state_icons = {
            'new': '🆕', 'winning': '✅', 'losing': '📉',
            'stuck': '🔒', 'fallen': '📉⬇️', 'stale': '⏰'
        }
        
        for state, positions in data['by_state'].items():
            icon = state_icons.get(state, '❓')
            print(f"\n{icon} {state.upper()} ({len(positions)})")
            print("-" * 40)
            for p in positions:
                pnl_color = '+' if p['pnl_pct'] >= 0 else ''
                print(f"  {p['agent']:15} {p['symbol']:12} {p['side']:5} "
                      f"{pnl_color}{p['pnl_pct']:.1f}% (peak {p['peak_pnl_pct']:+.1f}%) "
                      f"{p['age_min']:.0f}min")
        
        print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_rotate(args):
    """Trigger position rotation evaluation"""
    try:
        resp = requests.post(f"{get_base_url()}/rotate", timeout=10)
        data = resp.json()
        
        print(f"\n🔄 ROTATION EVALUATION")
        print("=" * 60)
        print(f"Decisions: {data['rotation_decisions']}")
        
        if data['decisions']:
            for d in data['decisions']:
                urgency_icon = {'immediate': '🚨', 'soon': '⚠️', 'optional': '💡'}.get(d['urgency'], '❓')
                replacement = '✅ replacement' if d['has_replacement'] else '❌ no replacement'
                print(f"\n{urgency_icon} {d['symbol']} ({d['agent']})")
                print(f"   Reason: {d['reason']}")
                print(f"   Urgency: {d['urgency']}, {replacement}")
        else:
            print("\n✅ No rotations needed - all positions healthy")
        
        print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_close(args):
    """Force close a position"""
    try:
        data = {
            'agent_id': args.agent,
            'symbol': args.symbol,
            'reason': args.reason or 'Manual close from CLI'
        }
        resp = requests.post(f"{get_base_url()}/positions/close", json=data, timeout=5)
        result = resp.json()
        
        if result.get('success'):
            print(f"✅ Close command sent for {args.symbol}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to orchestrator")
        sys.exit(1)

def cmd_reflect(args):
    """Run reflection analysis on an agent"""
    from agents.strategies import STRATEGY_REGISTRY
    from agents.reflection import AgentReflector
    
    agent_id = args.agent
    days = args.days
    
    if agent_id not in STRATEGY_REGISTRY:
        print(f"❌ Unknown agent: {agent_id}")
        print(f"Available: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)
    
    # Create agent instance
    agent_class = STRATEGY_REGISTRY[agent_id]
    agent = agent_class(symbols=['BTC/USDT', 'ETH/USDT'])  # Default symbols
    
    # Run reflection
    reflector = AgentReflector(agent)
    report = reflector.reflect(lookback_days=days)
    
    print(f"\n🧠 REFLECTION REPORT: {report.agent_id}")
    print("=" * 60)
    print(f"Period: {report.period_start[:10]} to {report.period_end[:10]}")
    print(f"Health Score: {report.health_score:.0f}/100")
    print(f"Priority: {report.improvement_priority.upper()}")
    
    print(f"\n📊 PERFORMANCE")
    print("-" * 60)
    print(f"Total Trades: {report.total_trades}")
    print(f"Win Rate: {report.win_rate:.1f}%")
    print(f"Total PnL: ${report.total_pnl:,.2f}")
    print(f"Profit Factor: {report.profit_factor:.2f}")
    print(f"Avg Win: ${report.avg_win:,.2f}")
    print(f"Avg Loss: ${report.avg_loss:,.2f}")
    
    if report.patterns:
        print(f"\n🔍 PATTERNS IDENTIFIED")
        print("-" * 60)
        for pattern in report.patterns:
            print(f"• {pattern}")
    
    if report.parameter_suggestions:
        print(f"\n💡 PARAMETER SUGGESTIONS")
        print("-" * 60)
        for suggestion in report.parameter_suggestions:
            print(f"\n{suggestion.parameter}:")
            print(f"  Current: {suggestion.current_value}")
            print(f"  Suggested: {suggestion.suggested_value}")
            print(f"  Reason: {suggestion.reason}")
            print(f"  Confidence: {suggestion.confidence:.0%}")
    
    if report.strategy_recommendations:
        print(f"\n📋 RECOMMENDATIONS")
        print("-" * 60)
        for rec in report.strategy_recommendations:
            print(f"• {rec}")
    
    if args.apply:
        print(f"\n🔧 APPLYING SUGGESTIONS...")
        changes = reflector.apply_suggestions(confidence_threshold=args.threshold)
        if changes:
            for param, change in changes.items():
                print(f"  ✅ {param}: {change['old']} → {change['new']}")
        else:
            print("  No suggestions met the confidence threshold")
    
    print()

def cmd_trade(args):
    """Start trading loop"""
    import subprocess
    import os
    
    mode = args.mode
    script = os.path.expanduser("~/.openclaw/workspace/projects/cash-town/run_trader.py")
    
    if args.background:
        # Run in background
        subprocess.Popen(
            ['python3', script, mode],
            stdout=open('/tmp/cash-town-trader.log', 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        print(f"✅ Trading loop started in background ({mode} mode)")
        print("   Logs: tail -f /tmp/cash-town-trader.log")
    else:
        # Run in foreground
        subprocess.call(['python3', script, mode])

def cmd_trade_status(args):
    """Show trading status"""
    import subprocess
    script = os.path.expanduser("~/.openclaw/workspace/projects/cash-town/run_trader.py")
    subprocess.call(['python3', script, 'status'])

def cmd_trade_stop(args):
    """Stop trading loop"""
    import subprocess
    result = subprocess.run(['pkill', '-f', 'run_trader.py'], capture_output=True)
    if result.returncode == 0:
        print("✅ Trading loop stopped")
    else:
        print("ℹ️ No trading loop running")

def cmd_backtest(args):
    """Run backtest for an agent"""
    from agents.strategies import STRATEGY_REGISTRY
    from agents.backtester import Backtester, BacktestConfig
    from datetime import datetime, timedelta
    
    agent_id = args.agent
    
    if agent_id not in STRATEGY_REGISTRY:
        print(f"❌ Unknown agent: {agent_id}")
        sys.exit(1)
    
    # Create agent
    agent_class = STRATEGY_REGISTRY[agent_id]
    agent = agent_class(symbols=args.symbols.split(',') if args.symbols else ['BTC/USDT'])
    
    # For now, show what would happen (need historical data integration)
    print(f"\n📊 BACKTEST: {agent_id}")
    print("=" * 60)
    print(f"Symbols: {', '.join(agent.symbols)}")
    print(f"Period: {args.days} days")
    print(f"Initial Capital: ${args.capital:,.0f}")
    
    print("\n⚠️ Historical data integration needed")
    print("Backtest framework ready - connect to data source to run")
    
    # Show config that would be tested
    print(f"\n⚙️ STRATEGY CONFIG")
    print("-" * 60)
    for key, value in agent.config.items():
        print(f"  {key}: {value}")

def cmd_strategies(args):
    """List available strategies"""
    from agents.strategies import STRATEGY_REGISTRY
    
    print("\n📚 AVAILABLE STRATEGIES")
    print("=" * 60)
    
    for strategy_id, agent_class in STRATEGY_REGISTRY.items():
        # Get default config
        agent = agent_class(symbols=['BTC/USDT'])
        
        print(f"\n🎯 {agent.name} ({strategy_id})")
        print(f"   {agent_class.__doc__.strip().split(chr(10))[0] if agent_class.__doc__ else 'No description'}")
        print(f"   Key params: {', '.join(list(agent.config.keys())[:5])}")

def main():
    parser = argparse.ArgumentParser(description='Cash Town CLI - Trading Swarm Manager')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show orchestrator status')
    status_parser.set_defaults(func=cmd_status)
    
    # Agents command
    agents_parser = subparsers.add_parser('agents', help='List all agents')
    agents_parser.set_defaults(func=cmd_agents)
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new agent')
    register_parser.add_argument('--id', required=True, help='Agent ID')
    register_parser.add_argument('--name', required=True, help='Agent name')
    register_parser.add_argument('--type', required=True, help='Agent type (cucurbit, mm-bot, etc)')
    register_parser.add_argument('--endpoint', required=True, help='Agent API endpoint')
    register_parser.add_argument('--api-key', help='API key for agent')
    register_parser.add_argument('--allocation', type=float, help='Capital allocation %')
    register_parser.add_argument('--max-positions', type=int, help='Max positions')
    register_parser.add_argument('--max-drawdown', type=float, help='Max drawdown %')
    register_parser.set_defaults(func=cmd_register)
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove an agent')
    remove_parser.add_argument('id', help='Agent ID to remove')
    remove_parser.set_defaults(func=cmd_remove)
    
    # Refresh command
    refresh_parser = subparsers.add_parser('refresh', help='Trigger health check')
    refresh_parser.set_defaults(func=cmd_refresh)
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Show portfolio summary')
    portfolio_parser.set_defaults(func=cmd_portfolio)
    
    # Positions command
    positions_parser = subparsers.add_parser('positions', help='Show position status')
    positions_parser.set_defaults(func=cmd_positions)
    
    # Rotate command
    rotate_parser = subparsers.add_parser('rotate', help='Evaluate position rotations')
    rotate_parser.set_defaults(func=cmd_rotate)
    
    # Close command
    close_parser = subparsers.add_parser('close', help='Force close a position')
    close_parser.add_argument('--agent', required=True, help='Agent ID')
    close_parser.add_argument('--symbol', required=True, help='Symbol to close')
    close_parser.add_argument('--reason', help='Reason for closing')
    close_parser.set_defaults(func=cmd_close)
    
    # Strategies command
    strategies_parser = subparsers.add_parser('strategies', help='List available strategies')
    strategies_parser.set_defaults(func=cmd_strategies)
    
    # Reflect command
    reflect_parser = subparsers.add_parser('reflect', help='Run reflection analysis on an agent')
    reflect_parser.add_argument('agent', help='Agent ID to reflect on')
    reflect_parser.add_argument('--days', type=int, default=7, help='Lookback days (default: 7)')
    reflect_parser.add_argument('--apply', action='store_true', help='Apply suggested parameter changes')
    reflect_parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for applying (default: 0.7)')
    reflect_parser.set_defaults(func=cmd_reflect)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest for an agent')
    backtest_parser.add_argument('agent', help='Agent ID to backtest')
    backtest_parser.add_argument('--symbols', help='Comma-separated symbols')
    backtest_parser.add_argument('--days', type=int, default=30, help='Backtest period in days')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Start trading loop')
    trade_parser.add_argument('mode', nargs='?', default='paper', choices=['paper', 'live'], help='Trading mode')
    trade_parser.add_argument('--background', '-b', action='store_true', help='Run in background')
    trade_parser.set_defaults(func=cmd_trade)
    
    # Trade status
    trade_status_parser = subparsers.add_parser('trade-status', help='Show trading status')
    trade_status_parser.set_defaults(func=cmd_trade_status)
    
    # Trade stop
    trade_stop_parser = subparsers.add_parser('trade-stop', help='Stop trading loop')
    trade_stop_parser.set_defaults(func=cmd_trade_stop)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n🏭 Cash Town - Multi-Agent Trading Swarm")
        print("Run 'cashctl strategies' to see available strategies")
        sys.exit(1)
    
    args.func(args)

if __name__ == '__main__':
    main()
