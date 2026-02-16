#!/usr/bin/env python3
"""
Strategy Agent Runner - Runs a single strategy agent as an independent process

Each agent:
1. Fetches market data
2. Generates signals
3. Reports signals to orchestrator
4. Runs on its own schedule

Usage:
    python -m agents.runner trend-following
    python -m agents.runner turtle
"""
import sys
import os
import time
import json
import logging
import requests
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.strategies import STRATEGY_REGISTRY
from agents.base import Signal, SignalSide
from data.feed import KuCoinDataFeed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AgentRunner:
    """
    Runs a single strategy agent independently.
    Reports signals to the orchestrator.
    """
    
    ORCHESTRATOR_URL = os.environ.get('ORCHESTRATOR_URL', f"http://localhost:{os.environ.get('PORT', '8888')}")
    
    def __init__(self, strategy_id: str, symbols: list = None, interval_seconds: int = 300):
        self.strategy_id = strategy_id
        self.symbols = symbols or ['XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'AVAXUSDTM', 'LINKUSDTM']
        self.interval = interval_seconds
        self.running = False
        
        # Get strategy class
        if strategy_id not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_id}. Available: {list(STRATEGY_REGISTRY.keys())}")
        
        agent_class = STRATEGY_REGISTRY[strategy_id]
        self.agent = agent_class(symbols=self.symbols)
        self.logger = logging.getLogger(f"agent.{strategy_id}")
        
        # Data feed
        self.data_feed = KuCoinDataFeed(self.symbols, interval='15min')
        
        self.logger.info(f"Agent initialized: {self.agent.name}")
        self.logger.info(f"  Symbols: {', '.join(self.symbols)}")
        self.logger.info(f"  Interval: {self.interval}s")
    
    def start(self):
        """Start the agent loop"""
        self.running = True
        self.logger.info(f"🚀 Starting {self.agent.name} agent")
        
        # Register with orchestrator
        self._register()
        
        while self.running:
            try:
                self._cycle()
            except Exception as e:
                self.logger.error(f"Cycle error: {e}")
            
            # Sleep until next cycle
            self._sleep(self.interval)
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        self._unregister()
        self.logger.info(f"Agent stopped: {self.strategy_id}")
    
    def _cycle(self):
        """Run one signal generation cycle"""
        self.logger.debug("Running signal cycle...")
        
        # Refresh market data
        self.data_feed.refresh_all()
        market_data = self.data_feed.get_market_data()
        
        if not market_data:
            self.logger.warning("No market data available")
            return
        
        # Generate signals
        signals = self.agent.generate_signals(market_data)
        
        if signals:
            self.logger.info(f"Generated {len(signals)} signals")
            for signal in signals:
                self._report_signal(signal)
        else:
            self.logger.debug("No signals generated")
        
        # Report health
        self._report_health(len(signals))
    
    def _report_signal(self, signal: Signal):
        """Report a signal to the orchestrator"""
        try:
            # Convert metadata to JSON-safe types (handle numpy bools, etc.)
            safe_metadata = {}
            if signal.metadata:
                for k, v in signal.metadata.items():
                    if hasattr(v, 'item'):  # numpy types
                        safe_metadata[k] = v.item()
                    elif isinstance(v, bool):
                        safe_metadata[k] = bool(v)
                    elif isinstance(v, (int, float, str, type(None))):
                        safe_metadata[k] = v
                    else:
                        safe_metadata[k] = str(v)
            
            signal_data = {
                'strategy_id': self.strategy_id,
                'symbol': signal.symbol,
                'side': signal.side.value,
                'confidence': signal.confidence,
                'price': signal.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reason': signal.reason,
                'timestamp': signal.timestamp.isoformat(),
                'metadata': safe_metadata
            }
            
            self.logger.info(f"📊 Signal: {signal.side.value.upper()} {signal.symbol} @ ${signal.price:,.2f} ({signal.confidence:.0%})")
            self.logger.info(f"   Reason: {signal.reason}")
            
            # Send to orchestrator
            resp = requests.post(
                f"{self.ORCHESTRATOR_URL}/signals",
                json=signal_data,
                timeout=5
            )
            
            if resp.status_code == 200 or resp.status_code == 201:
                self.logger.debug(f"Signal reported to orchestrator")
            else:
                self.logger.warning(f"Failed to report signal: {resp.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.logger.warning("Cannot connect to orchestrator - signal not reported")
        except Exception as e:
            self.logger.error(f"Error reporting signal: {e}")
    
    def _report_health(self, signals_count: int):
        """Report health status to orchestrator"""
        try:
            health_data = {
                'agent_id': self.strategy_id,
                'healthy': True,
                'timestamp': datetime.utcnow().isoformat(),
                'signals_generated': signals_count,
                'symbols': self.symbols
            }
            
            requests.post(
                f"{self.ORCHESTRATOR_URL}/agents/{self.strategy_id}/health",
                json=health_data,
                timeout=5
            )
        except:
            pass  # Health reporting is best-effort
    
    def _register(self):
        """Register agent with orchestrator"""
        try:
            data = {
                'id': self.strategy_id,
                'name': self.agent.name,
                'type': 'strategy',
                'endpoint': None,  # Local agent
                'symbols': self.symbols
            }
            requests.post(f"{self.ORCHESTRATOR_URL}/agents", json=data, timeout=5)
            self.logger.info("Registered with orchestrator")
        except:
            self.logger.warning("Could not register with orchestrator")
    
    def _unregister(self):
        """Unregister from orchestrator"""
        try:
            requests.delete(f"{self.ORCHESTRATOR_URL}/agents/{self.strategy_id}", timeout=5)
        except:
            pass
    
    def _sleep(self, seconds: int):
        """Sleep with running check"""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agents.runner <strategy-id>")
        print(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)
    
    strategy_id = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # Default 5 minutes
    
    runner = AgentRunner(strategy_id, interval_seconds=interval)
    
    try:
        runner.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        runner.stop()


if __name__ == '__main__':
    main()
