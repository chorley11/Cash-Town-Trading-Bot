#!/usr/bin/env python3
"""
Cash Town Swarm Runner - Starts all strategy agents as separate processes

Each agent runs independently and reports signals to the orchestrator.
The orchestrator aggregates signals and the executor places trades.

Usage:
    python run_swarm.py start    # Start all agents
    python run_swarm.py stop     # Stop all agents
    python run_swarm.py status   # Show status
"""
import sys
import os
import subprocess
import time
import signal
import json
from pathlib import Path

# Agent configurations
AGENTS = [
    {'id': 'trend-following', 'interval': 300},
    {'id': 'mean-reversion', 'interval': 300},
    {'id': 'turtle', 'interval': 300},
    {'id': 'weinstein', 'interval': 300},
    {'id': 'livermore', 'interval': 300},
    {'id': 'bts-lynch', 'interval': 300},
    {'id': 'zweig', 'interval': 300},
    # stat-arb disabled by default (complex pairs trading)
]

PID_DIR = Path("/tmp/cash-town-agents")
LOG_DIR = Path("/tmp/cash-town-logs")

def ensure_dirs():
    PID_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

def start_agent(agent_id: str, interval: int = 300):
    """Start a single agent process"""
    ensure_dirs()
    
    pid_file = PID_DIR / f"{agent_id}.pid"
    log_file = LOG_DIR / f"{agent_id}.log"
    
    # Check if already running
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"  ⚠️  {agent_id} already running (PID {pid})")
            return False
        except OSError:
            pid_file.unlink()
    
    # Start the agent
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable, '-m', 'agents.runner',
        agent_id, str(interval)
    ]
    
    with open(log_file, 'a') as log:
        process = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Save PID
    pid_file.write_text(str(process.pid))
    print(f"  ✅ Started {agent_id} (PID {process.pid})")
    return True

def stop_agent(agent_id: str):
    """Stop a single agent process"""
    pid_file = PID_DIR / f"{agent_id}.pid"
    
    if not pid_file.exists():
        print(f"  ⚪ {agent_id} not running")
        return False
    
    pid = int(pid_file.read_text().strip())
    
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)  # Force kill if still running
        except OSError:
            pass
        print(f"  🛑 Stopped {agent_id} (PID {pid})")
    except OSError:
        print(f"  ⚪ {agent_id} was not running")
    
    pid_file.unlink(missing_ok=True)
    return True

def get_agent_status(agent_id: str) -> dict:
    """Get status of a single agent"""
    pid_file = PID_DIR / f"{agent_id}.pid"
    log_file = LOG_DIR / f"{agent_id}.log"
    
    status = {
        'id': agent_id,
        'running': False,
        'pid': None,
        'last_log': None
    }
    
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            status['running'] = True
            status['pid'] = pid
        except OSError:
            pass
    
    if log_file.exists():
        # Get last log line
        try:
            with open(log_file, 'rb') as f:
                f.seek(-500, 2)  # Last 500 bytes
                lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
                if lines:
                    status['last_log'] = lines[-1][:100]
        except:
            pass
    
    return status

def cmd_start():
    """Start all agents"""
    print("\n🚀 Starting Cash Town Agent Swarm")
    print("=" * 50)
    
    started = 0
    for agent in AGENTS:
        if start_agent(agent['id'], agent['interval']):
            started += 1
        time.sleep(0.5)  # Stagger starts
    
    print()
    print(f"✅ Started {started}/{len(AGENTS)} agents")
    print(f"📁 Logs: {LOG_DIR}")
    print()

def cmd_stop():
    """Stop all agents"""
    print("\n🛑 Stopping Cash Town Agent Swarm")
    print("=" * 50)
    
    for agent in AGENTS:
        stop_agent(agent['id'])
    
    print()
    print("✅ All agents stopped")
    print()

def cmd_status():
    """Show status of all agents"""
    print("\n💰 Cash Town Agent Swarm Status")
    print("=" * 50)
    
    running = 0
    for agent in AGENTS:
        status = get_agent_status(agent['id'])
        icon = "🟢" if status['running'] else "🔴"
        pid_str = f"PID {status['pid']}" if status['pid'] else "stopped"
        print(f"  {icon} {agent['id']:20} {pid_str}")
        if status['running']:
            running += 1
    
    print()
    print(f"Running: {running}/{len(AGENTS)} agents")
    
    # Check orchestrator
    import requests
    try:
        resp = requests.get('http://localhost:8888/health', timeout=2)
        if resp.status_code == 200:
            print(f"Orchestrator: 🟢 Running")
            
            # Get signal count
            resp = requests.get('http://localhost:8888/signals', timeout=2)
            data = resp.json()
            print(f"Pending Signals: {data.get('count', 0)}")
    except:
        print(f"Orchestrator: 🔴 Not running")
    
    print()

def cmd_logs(agent_id: str = None):
    """Show recent logs"""
    if agent_id:
        log_file = LOG_DIR / f"{agent_id}.log"
        if log_file.exists():
            os.system(f"tail -50 {log_file}")
        else:
            print(f"No logs for {agent_id}")
    else:
        print("Specify agent ID or use 'tail -f /tmp/cash-town-logs/*.log'")

def main():
    if len(sys.argv) < 2:
        cmd = 'status'
    else:
        cmd = sys.argv[1].lower()
    
    if cmd == 'start':
        cmd_start()
    elif cmd == 'stop':
        cmd_stop()
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'restart':
        cmd_stop()
        time.sleep(1)
        cmd_start()
    elif cmd == 'logs':
        agent_id = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_logs(agent_id)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python run_swarm.py [start|stop|status|restart|logs]")
        sys.exit(1)

if __name__ == '__main__':
    main()
