#!/usr/bin/env python3
"""
Quick status check for Gas Town
Run: python3 status.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.registry import AgentRegistry
from orchestrator.health import HealthMonitor

def main():
    registry = AgentRegistry()
    monitor = HealthMonitor(registry)
    statuses = monitor.check_all()

    print("🏭 GAS TOWN ORCHESTRATOR")
    print("=" * 60)

    for agent in registry.list_all():
        status = statuses.get(agent.id)
        health = '🟢' if status and status.healthy else '🔴'
        enabled = '✓' if agent.enabled else '✗'
        print(f"\n{health} [{enabled}] {agent.name}")
        print(f"   Type: {agent.type} | Allocation: {agent.allocation_pct}%")
        if status:
            print(f"   Positions: {status.open_positions} | Value: ${status.total_value:,.2f}")
            print(f"   Unrealized PnL: ${status.unrealized_pnl:,.2f} | Daily: ${status.daily_pnl:,.2f}")
            print(f"   Can Trade: {'Yes' if status.can_trade else 'NO'}")
            if status.error:
                print(f"   ⚠️ Error: {status.error}")
        else:
            print("   Status: Unknown (check failed)")

    summary = monitor.get_portfolio_summary()
    print(f"\n{'='*60}")
    print(f"📊 PORTFOLIO TOTALS")
    print(f"   Agents: {summary['healthy_agents']}/{summary['total_agents']} healthy")
    print(f"   Positions: {summary['total_positions']}")
    print(f"   Value: ${summary['total_value']:,.2f}")
    print(f"   Unrealized PnL: ${summary['total_unrealized_pnl']:,.2f}")
    print(f"   Daily PnL: ${summary['total_daily_pnl']:,.2f}")
    print()

if __name__ == '__main__':
    main()
