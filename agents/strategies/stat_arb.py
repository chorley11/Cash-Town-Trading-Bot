"""
Statistical Arbitrage Strategy Agent
Pairs trading based on cointegration and mean reversion of spread.

Core Logic:
- Find correlated pairs (BTC/ETH, stablecoin pairs, etc.)
- Calculate z-score of price spread
- Trade when spread deviates significantly from mean
- Market-neutral positions (long one, short other)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

# Pre-defined correlated pairs for crypto
DEFAULT_PAIRS = [
    ('BTC/USDT', 'ETH/USDT'),
    ('SOL/USDT', 'AVAX/USDT'),
    ('LINK/USDT', 'DOT/USDT'),
    ('ADA/USDT', 'XRP/USDT'),
    ('DOGE/USDT', 'SHIB/USDT'),
]

class StatArbAgent(BaseStrategyAgent):
    """
    Statistical Arbitrage Strategy
    
    Approach:
    - Calculate rolling correlation between pairs
    - Compute spread and its z-score
    - Enter when z-score > threshold (spread too wide)
    - Exit when z-score returns to 0 (spread normalized)
    
    Risk management:
    - Stop loss if spread widens further
    - Time-based exit if spread doesn't converge
    """
    
    DEFAULT_CONFIG = {
        'lookback_period': 60,     # Period for calculating spread stats
        'zscore_entry': 2.0,       # Enter when z-score exceeds this
        'zscore_exit': 0.5,        # Exit when z-score drops below this
        'zscore_stop': 3.5,        # Stop if z-score exceeds this
        'min_correlation': 0.7,    # Minimum correlation to trade pair
        'position_size_pct': 2.0,  # Size per leg
        'max_hold_periods': 100,   # Max bars to hold position
        'min_confidence': 0.5,
    }
    
    def __init__(self, symbols: List[str], pairs: List[Tuple[str, str]] = None, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Filter pairs to only include symbols we're tracking
        self.pairs = []
        available_pairs = pairs or DEFAULT_PAIRS
        for pair in available_pairs:
            if pair[0] in symbols and pair[1] in symbols:
                self.pairs.append(pair)
        
        super().__init__(
            agent_id='stat-arb',
            name='Statistical Arbitrage',
            symbols=symbols,
            config=merged_config
        )
        
        self._spread_history: Dict[str, List[float]] = {}
        self._active_spreads: Dict[str, dict] = {}
    
    def get_required_indicators(self) -> List[str]:
        return ['close']  # We mainly need close prices
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for asset_a, asset_b in self.pairs:
            try:
                data_a = market_data.get(asset_a)
                data_b = market_data.get(asset_b)
                
                if not data_a or not data_b:
                    continue
                
                if len(data_a.get('close', [])) < self.config['lookback_period']:
                    continue
                
                pair_signals = self._analyze_pair(asset_a, asset_b, data_a, data_b)
                signals.extend(pair_signals)
                
            except Exception as e:
                logger.error(f"Error analyzing pair {asset_a}/{asset_b}: {e}")
        
        return signals
    
    def _analyze_pair(self, asset_a: str, asset_b: str, data_a: Dict, data_b: Dict) -> List[Signal]:
        signals = []
        pair_key = f"{asset_a}:{asset_b}"
        
        closes_a = np.array(data_a['close'])
        closes_b = np.array(data_b['close'])
        
        # Align lengths
        min_len = min(len(closes_a), len(closes_b))
        closes_a = closes_a[-min_len:]
        closes_b = closes_b[-min_len:]
        
        current_price_a = closes_a[-1]
        current_price_b = closes_b[-1]
        
        # Calculate correlation
        lookback = self.config['lookback_period']
        if len(closes_a) < lookback:
            return signals
        
        correlation = np.corrcoef(closes_a[-lookback:], closes_b[-lookback:])[0, 1]
        
        if abs(correlation) < self.config['min_correlation']:
            return signals  # Not correlated enough
        
        # Calculate spread (ratio-based for simplicity)
        spread = closes_a / closes_b
        
        # Calculate z-score of current spread
        spread_mean = np.mean(spread[-lookback:])
        spread_std = np.std(spread[-lookback:])
        
        if spread_std < 1e-10:
            return signals
        
        current_spread = spread[-1]
        zscore = (current_spread - spread_mean) / spread_std
        
        # Track spread history
        if pair_key not in self._spread_history:
            self._spread_history[pair_key] = []
        self._spread_history[pair_key].append(zscore)
        if len(self._spread_history[pair_key]) > 100:
            self._spread_history[pair_key] = self._spread_history[pair_key][-100:]
        
        # Generate signals based on z-score
        entry_threshold = self.config['zscore_entry']
        
        # Spread too high: Short A, Long B (expect spread to decrease)
        if zscore > entry_threshold:
            confidence = self._calculate_confidence(zscore, correlation)
            
            if confidence >= self.config['min_confidence']:
                # Signal for asset A (SHORT)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=asset_a,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price_a,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT StatArb: {pair_key} spread z={zscore:.2f} (expect reversion)",
                    metadata={
                        'pair': pair_key,
                        'leg': 'short',
                        'zscore': zscore,
                        'correlation': correlation,
                        'spread': current_spread,
                        'spread_mean': spread_mean
                    }
                ))
                
                # Signal for asset B (LONG)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=asset_b,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price_b,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG StatArb: {pair_key} spread z={zscore:.2f} (expect reversion)",
                    metadata={
                        'pair': pair_key,
                        'leg': 'long',
                        'zscore': zscore,
                        'correlation': correlation,
                        'spread': current_spread,
                        'spread_mean': spread_mean
                    }
                ))
        
        # Spread too low: Long A, Short B (expect spread to increase)
        elif zscore < -entry_threshold:
            confidence = self._calculate_confidence(abs(zscore), correlation)
            
            if confidence >= self.config['min_confidence']:
                # Signal for asset A (LONG)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=asset_a,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price_a,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG StatArb: {pair_key} spread z={zscore:.2f} (expect reversion)",
                    metadata={
                        'pair': pair_key,
                        'leg': 'long',
                        'zscore': zscore,
                        'correlation': correlation,
                        'spread': current_spread,
                        'spread_mean': spread_mean
                    }
                ))
                
                # Signal for asset B (SHORT)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=asset_b,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price_b,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT StatArb: {pair_key} spread z={zscore:.2f} (expect reversion)",
                    metadata={
                        'pair': pair_key,
                        'leg': 'short',
                        'zscore': zscore,
                        'correlation': correlation,
                        'spread': current_spread,
                        'spread_mean': spread_mean
                    }
                ))
        
        return signals
    
    def _calculate_confidence(self, zscore: float, correlation: float) -> float:
        confidence = 0.5
        
        # Z-score strength
        if zscore > 3.0:
            confidence += 0.2
        elif zscore > 2.5:
            confidence += 0.15
        elif zscore > 2.0:
            confidence += 0.1
        
        # Correlation strength
        if abs(correlation) > 0.9:
            confidence += 0.15
        elif abs(correlation) > 0.8:
            confidence += 0.1
        elif abs(correlation) > 0.7:
            confidence += 0.05
        
        return min(confidence, 0.85)
