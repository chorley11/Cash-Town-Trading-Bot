"""
Health Monitor - Polls agents for status and detects issues
Uses requests (sync) instead of aiohttp for compatibility
"""
import logging
import requests
import threading
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from .registry import AgentConfig, AgentStatus, AgentRegistry

logger = logging.getLogger(__name__)

@dataclass
class HealthAlert:
    """Alert generated when agent health degrades"""
    agent_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: str
    data: dict = None

class HealthMonitor:
    """Monitors health of all registered agents"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.statuses: dict[str, AgentStatus] = {}
        self.alerts: list[HealthAlert] = []
        self.last_check: dict[str, datetime] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def check_agent(self, agent: AgentConfig) -> AgentStatus:
        """Check health of a single agent"""
        try:
            headers = {}
            if agent.api_key:
                headers['Authorization'] = f'Bearer {agent.api_key}'
            
            # Get health/status from agent
            resp = requests.get(
                f"{agent.endpoint}/health",
                headers=headers,
                timeout=10
            )
            if resp.status_code != 200:
                return AgentStatus(
                    agent_id=agent.id,
                    healthy=False,
                    last_heartbeat=datetime.utcnow().isoformat(),
                    open_positions=0,
                    total_value=0,
                    unrealized_pnl=0,
                    daily_pnl=0,
                    can_trade=False,
                    error=f"HTTP {resp.status_code}"
                )
            health_data = resp.json()
            
            # Get positions
            try:
                resp = requests.get(
                    f"{agent.endpoint}/positions",
                    headers=headers,
                    timeout=10
                )
                positions_data = resp.json() if resp.status_code == 200 else {'positions': []}
            except:
                positions_data = {'positions': []}
            
            # Get risk profile if available
            risk_data = {}
            try:
                resp = requests.get(
                    f"{agent.endpoint}/risk-profile",
                    headers=headers,
                    timeout=10
                )
                if resp.status_code == 200:
                    risk_data = resp.json()
            except:
                pass
            
            positions = positions_data.get('positions', [])
            risk_state = risk_data.get('riskState', {})
            
            return AgentStatus(
                agent_id=agent.id,
                healthy=health_data.get('status') == 'healthy',
                last_heartbeat=datetime.utcnow().isoformat(),
                open_positions=len(positions),
                total_value=sum(p.get('value', 0) for p in positions),
                unrealized_pnl=sum(p.get('unrealizedPnl', 0) for p in positions),
                daily_pnl=risk_state.get('dailyPnl', 0),
                can_trade=risk_state.get('canTrade', True),
                error=None
            )
                
        except requests.exceptions.Timeout:
            return AgentStatus(
                agent_id=agent.id,
                healthy=False,
                last_heartbeat=datetime.utcnow().isoformat(),
                open_positions=0,
                total_value=0,
                unrealized_pnl=0,
                daily_pnl=0,
                can_trade=False,
                error="Timeout"
            )
        except Exception as e:
            return AgentStatus(
                agent_id=agent.id,
                healthy=False,
                last_heartbeat=datetime.utcnow().isoformat(),
                open_positions=0,
                total_value=0,
                unrealized_pnl=0,
                daily_pnl=0,
                can_trade=False,
                error=str(e)
            )
    
    def check_all(self) -> dict[str, AgentStatus]:
        """Check health of all enabled agents"""
        agents = self.registry.list_enabled()
        
        # Check all agents in parallel
        futures = {self._executor.submit(self.check_agent, agent): agent for agent in agents}
        
        for future in futures:
            try:
                status = future.result(timeout=30)
                old_status = self.statuses.get(status.agent_id)
                self.statuses[status.agent_id] = status
                self.last_check[status.agent_id] = datetime.utcnow()
                
                # Check for state changes that need alerts
                self._check_alerts(old_status, status)
            except Exception as e:
                agent = futures[future]
                logger.error(f"Error checking agent {agent.id}: {e}")
        
        return self.statuses
    
    def _check_alerts(self, old: Optional[AgentStatus], new: AgentStatus):
        """Generate alerts for health changes"""
        agent = self.registry.get(new.agent_id)
        
        # Agent went unhealthy
        if old and old.healthy and not new.healthy:
            self.alerts.append(HealthAlert(
                agent_id=new.agent_id,
                severity='error',
                message=f"Agent {agent.name} is now unhealthy: {new.error}",
                timestamp=datetime.utcnow().isoformat()
            ))
        
        # Agent recovered
        if old and not old.healthy and new.healthy:
            self.alerts.append(HealthAlert(
                agent_id=new.agent_id,
                severity='info',
                message=f"Agent {agent.name} has recovered",
                timestamp=datetime.utcnow().isoformat()
            ))
        
        # Can't trade anymore
        if old and old.can_trade and not new.can_trade:
            self.alerts.append(HealthAlert(
                agent_id=new.agent_id,
                severity='warning',
                message=f"Agent {agent.name} can no longer trade",
                timestamp=datetime.utcnow().isoformat()
            ))
        
        # Drawdown check
        if agent and new.daily_pnl < 0:
            drawdown_pct = abs(new.daily_pnl) / max(new.total_value, 1) * 100
            if drawdown_pct > agent.max_drawdown_pct:
                self.alerts.append(HealthAlert(
                    agent_id=new.agent_id,
                    severity='critical',
                    message=f"Agent {agent.name} exceeded max drawdown: {drawdown_pct:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    data={'drawdown_pct': drawdown_pct, 'daily_pnl': new.daily_pnl}
                ))
    
    def get_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get last known status for an agent"""
        return self.statuses.get(agent_id)
    
    def get_all_statuses(self) -> dict[str, AgentStatus]:
        """Get all agent statuses"""
        return self.statuses.copy()
    
    def get_alerts(self, since: datetime = None, clear: bool = False) -> list[HealthAlert]:
        """Get alerts, optionally filtering by time and clearing"""
        if since:
            alerts = [a for a in self.alerts 
                     if datetime.fromisoformat(a.timestamp) > since]
        else:
            alerts = self.alerts.copy()
        
        if clear:
            self.alerts = []
        
        return alerts
    
    def get_portfolio_summary(self) -> dict:
        """Get aggregated portfolio summary across all agents"""
        total_positions = 0
        total_value = 0
        total_unrealized_pnl = 0
        total_daily_pnl = 0
        healthy_agents = 0
        unhealthy_agents = 0
        
        for status in self.statuses.values():
            total_positions += status.open_positions
            total_value += status.total_value
            total_unrealized_pnl += status.unrealized_pnl
            total_daily_pnl += status.daily_pnl
            if status.healthy:
                healthy_agents += 1
            else:
                unhealthy_agents += 1
        
        return {
            'total_agents': len(self.statuses),
            'healthy_agents': healthy_agents,
            'unhealthy_agents': unhealthy_agents,
            'total_positions': total_positions,
            'total_value': total_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_daily_pnl': total_daily_pnl,
            'last_updated': datetime.utcnow().isoformat()
        }
