"""
Agent Registry - Tracks all registered trading agents in the swarm
"""
import json
import os
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Optional
from pathlib import Path

@dataclass
class AgentConfig:
    """Configuration for a registered agent"""
    id: str
    name: str
    type: str  # 'cucurbit', 'mm-bot', 'strategy', etc.
    endpoint: Optional[str] = None  # API endpoint for health/status (None for local strategy agents)
    api_key: Optional[str] = None
    enabled: bool = True
    allocation_pct: float = 0.0  # Percentage of capital allocated
    max_positions: int = 10
    max_drawdown_pct: float = 5.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

@dataclass
class AgentStatus:
    """Current status of an agent"""
    agent_id: str
    healthy: bool
    last_heartbeat: str
    open_positions: int
    total_value: float
    unrealized_pnl: float
    daily_pnl: float
    can_trade: bool
    error: Optional[str] = None

class AgentRegistry:
    """Registry for managing trading agents"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.expanduser(
            "~/.openclaw/workspace/projects/cash-town/config/agents.json"
        )
        self.agents: dict[str, AgentConfig] = {}
        self._load()
    
    def _load(self):
        """Load agents from config file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Get valid fields from AgentConfig
                valid_fields = {f.name for f in fields(AgentConfig)}
                for agent_data in data.get('agents', []):
                    # Filter out unknown keys and store extras in metadata
                    filtered = {}
                    extras = {}
                    for k, v in agent_data.items():
                        if k in valid_fields:
                            filtered[k] = v
                        else:
                            extras[k] = v
                    # Merge extras into metadata
                    if extras:
                        filtered.setdefault('metadata', {}).update(extras)
                    agent = AgentConfig(**filtered)
                    self.agents[agent.id] = agent
    
    def _save(self):
        """Save agents to config file"""
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        data = {'agents': [asdict(a) for a in self.agents.values()]}
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register(self, agent: AgentConfig) -> bool:
        """Register a new agent"""
        if agent.id in self.agents:
            return False
        self.agents[agent.id] = agent
        self._save()
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from registry"""
        if agent_id not in self.agents:
            return False
        del self.agents[agent_id]
        self._save()
        return True
    
    def get(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def list_all(self) -> list[AgentConfig]:
        """List all registered agents"""
        return list(self.agents.values())
    
    def list_enabled(self) -> list[AgentConfig]:
        """List only enabled agents"""
        return [a for a in self.agents.values() if a.enabled]
    
    def update(self, agent_id: str, **updates) -> bool:
        """Update agent configuration"""
        if agent_id not in self.agents:
            return False
        agent = self.agents[agent_id]
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        self._save()
        return True
    
    def set_allocation(self, agent_id: str, allocation_pct: float) -> bool:
        """Set capital allocation for an agent"""
        return self.update(agent_id, allocation_pct=allocation_pct)
    
    def enable(self, agent_id: str) -> bool:
        """Enable an agent"""
        return self.update(agent_id, enabled=True)
    
    def disable(self, agent_id: str) -> bool:
        """Disable an agent"""
        return self.update(agent_id, enabled=False)
