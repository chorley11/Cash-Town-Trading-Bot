"""
Cross-Pair Correlation Strategy Agent

Core Logic:
- BTC/ETH (and other pairs) usually have high correlation
- When correlation breaks significantly, trade convergence
- Mean reversion on the spread between correlated assets
- Pairs trading adapted for crypto futures

Why it works:
- Crypto majors move together most of the time
- Temporary divergences revert as the market normalizes
- Market neutral when done correctly (long one, short other)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class CorrelationPairsAgent(BaseStrategyAgent):
    """
    Cross-Pair Correlation Strategy
    
    Pairs:
    - BTC/ETH (primary pair)
    - SOL/AVAX (L1s)
    - LINK/DOT (infrastructure)
    
    Entry conditions:
    - Z-score of spread exceeds threshold
    - Correlation still positive (just temporarily diverged)
    - Volume not spiking abnormally (avoid news events)
    
    Exit conditions:
    - Spread mean reverts (z-score normalizes)
    - Correlation breaks down (fundamental shift)
    - Stop loss hit
    """
    
    DEFAULT_CONFIG = {
        # Pairs to trade
        'pairs': [
            ('XBTUSDTM', 'ETHUSDTM'),  # BTC/ETH
            ('SOLUSDTM', 'AVAXUSDTM'),  # SOL/AVAX
        ],
        
        # Correlation parameters
        'correlation_lookback': 50,  # Periods for correlation calc
        'min_correlation': 0.6,  # Minimum correlation to trade
        
        # Spread parameters
        'spread_lookback': 30,  # Periods for spread mean/std
        'zscore_entry': 2.0,  # Z-score to enter
        'zscore_exit': 0.5,  # Z-score to exit
        'zscore_stop': 3.5,  # Z-score for stop loss
        
        # Risk management
        'max_holding_periods': 48,  # Max candles to hold
        'atr_period': 14,
        'stop_loss_zscore': 3.5,  # Exit if spread widens further
        
        # Position sizing
        'hedge_ratio_dynamic': True,  # Adjust hedge ratio dynamically
        
        # Confidence
        'min_confidence': 0.55,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Extract unique symbols from pairs
        all_symbols = set()
        for pair in merged_config.get('pairs', []):
            all_symbols.add(pair[0])
            all_symbols.add(pair[1])
        
        super().__init__(
            agent_id='correlation-pairs',
            name='Correlation Pairs',
            symbols=list(all_symbols),
            config=merged_config
        )
        
        # Track position state per pair
        self._pair_positions: Dict[Tuple[str, str], Dict] = {}
        self._holding_periods: Dict[Tuple[str, str], int] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"atr_{self.config['atr_period']}",
            'correlation',  # Custom: computed internally
            'spread_zscore',  # Custom: computed internally
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for pair in self.config['pairs']:
            symbol_a, symbol_b = pair
            
            try:
                data_a = market_data.get(symbol_a)
                data_b = market_data.get(symbol_b)
                
                if not data_a or not data_b:
                    continue
                
                if len(data_a.get('close', [])) < self.config['correlation_lookback']:
                    continue
                if len(data_b.get('close', [])) < self.config['correlation_lookback']:
                    continue
                
                pair_signals = self._analyze_pair(pair, data_a, data_b)
                signals.extend(pair_signals)
                
            except Exception as e:
                logger.error(f"Error analyzing pair {pair}: {e}")
        
        return signals
    
    def _analyze_pair(self, pair: Tuple[str, str],
                     data_a: Dict, data_b: Dict) -> List[Signal]:
        """Analyze a pair for correlation divergence"""
        symbol_a, symbol_b = pair
        
        closes_a = np.array(data_a['close'])
        closes_b = np.array(data_b['close'])
        
        # Align arrays to same length
        min_len = min(len(closes_a), len(closes_b))
        closes_a = closes_a[-min_len:]
        closes_b = closes_b[-min_len:]
        
        # Calculate correlation
        lookback = self.config['correlation_lookback']
        if len(closes_a) < lookback:
            return []
        
        correlation = np.corrcoef(closes_a[-lookback:], closes_b[-lookback:])[0, 1]
        
        # Skip if correlation too low (fundamental decorrelation)
        if correlation < self.config['min_correlation']:
            logger.debug(f"[correlation-pairs] {pair}: Correlation {correlation:.2f} below threshold")
            return []
        
        # Calculate spread (log returns ratio)
        # Spread = log(A) - log(B) * hedge_ratio
        returns_a = np.diff(np.log(closes_a))
        returns_b = np.diff(np.log(closes_b))
        
        # Calculate hedge ratio (beta)
        if self.config['hedge_ratio_dynamic']:
            hedge_ratio = self._calculate_hedge_ratio(returns_a, returns_b, lookback)
        else:
            hedge_ratio = 1.0
        
        # Calculate spread using prices
        spread = np.log(closes_a) - hedge_ratio * np.log(closes_b)
        
        # Calculate z-score of spread
        spread_lookback = self.config['spread_lookback']
        if len(spread) < spread_lookback:
            return []
        
        spread_recent = spread[-spread_lookback:]
        spread_mean = np.mean(spread_recent)
        spread_std = np.std(spread_recent)
        
        if spread_std < 1e-10:
            return []
        
        current_zscore = (spread[-1] - spread_mean) / spread_std
        
        # Get current prices
        price_a = closes_a[-1]
        price_b = closes_b[-1]
        
        signals = []
        
        # Check for entry conditions
        zscore_entry = self.config['zscore_entry']
        
        # SPREAD TOO HIGH: Long B, Short A (expect convergence)
        if current_zscore > zscore_entry:
            confidence = self._calculate_confidence(current_zscore, correlation, hedge_ratio)
            
            if confidence >= self.config['min_confidence']:
                # Short A
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol_a,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=price_a,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT {symbol_a}: Spread z={current_zscore:.2f} (pair with {symbol_b}), corr={correlation:.2f}",
                    metadata={
                        'pair': pair,
                        'pair_side': 'short_a_long_b',
                        'zscore': current_zscore,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio
                    }
                ))
                
                # Long B (hedge)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol_b,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=price_b,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG {symbol_b}: Spread z={current_zscore:.2f} (pair with {symbol_a}), corr={correlation:.2f}",
                    position_size_pct=hedge_ratio,  # Adjust size by hedge ratio
                    metadata={
                        'pair': pair,
                        'pair_side': 'short_a_long_b',
                        'zscore': current_zscore,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio
                    }
                ))
        
        # SPREAD TOO LOW: Long A, Short B (expect convergence)
        elif current_zscore < -zscore_entry:
            confidence = self._calculate_confidence(abs(current_zscore), correlation, hedge_ratio)
            
            if confidence >= self.config['min_confidence']:
                # Long A
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol_a,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=price_a,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG {symbol_a}: Spread z={current_zscore:.2f} (pair with {symbol_b}), corr={correlation:.2f}",
                    metadata={
                        'pair': pair,
                        'pair_side': 'long_a_short_b',
                        'zscore': current_zscore,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio
                    }
                ))
                
                # Short B (hedge)
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol_b,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=price_b,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT {symbol_b}: Spread z={current_zscore:.2f} (pair with {symbol_a}), corr={correlation:.2f}",
                    position_size_pct=hedge_ratio,
                    metadata={
                        'pair': pair,
                        'pair_side': 'long_a_short_b',
                        'zscore': current_zscore,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio
                    }
                ))
        
        return signals
    
    def _calculate_hedge_ratio(self, returns_a: np.ndarray, returns_b: np.ndarray,
                              lookback: int) -> float:
        """Calculate optimal hedge ratio using OLS regression"""
        if len(returns_a) < lookback or len(returns_b) < lookback:
            return 1.0
        
        recent_a = returns_a[-lookback:]
        recent_b = returns_b[-lookback:]
        
        # OLS: returns_a = beta * returns_b + alpha
        # beta = cov(a, b) / var(b)
        cov = np.cov(recent_a, recent_b)[0, 1]
        var_b = np.var(recent_b)
        
        if var_b < 1e-10:
            return 1.0
        
        beta = cov / var_b
        
        # Clip to reasonable range
        return np.clip(beta, 0.5, 2.0)
    
    def _calculate_confidence(self, zscore: float, correlation: float,
                             hedge_ratio: float) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        # Z-score strength (higher = more confident in mean reversion)
        if zscore > 3.0:
            confidence += 0.15
        elif zscore > 2.5:
            confidence += 0.1
        elif zscore > 2.0:
            confidence += 0.05
        
        # Correlation strength (higher = more reliable pair)
        if correlation > 0.85:
            confidence += 0.15
        elif correlation > 0.75:
            confidence += 0.1
        elif correlation > 0.65:
            confidence += 0.05
        
        # Hedge ratio reasonableness
        if 0.8 <= hedge_ratio <= 1.2:
            confidence += 0.05
        
        return min(confidence, 0.85)
    
    def should_exit_pair(self, pair: Tuple[str, str], current_zscore: float,
                        periods_held: int) -> Optional[str]:
        """Check if pair trade should be exited"""
        # Mean reversion complete
        if abs(current_zscore) < self.config['zscore_exit']:
            return 'mean_reversion'
        
        # Spread widened further (stop loss)
        if abs(current_zscore) > self.config['zscore_stop']:
            return 'stop_loss'
        
        # Time-based exit
        if periods_held > self.config['max_holding_periods']:
            return 'time_exit'
        
        return None


# Additional helper for tracking pair positions in the orchestrator
class PairPositionTracker:
    """Tracks paired positions for the correlation strategy"""
    
    def __init__(self):
        self.active_pairs: Dict[Tuple[str, str], Dict] = {}
    
    def add_pair_position(self, pair: Tuple[str, str], side: str,
                         entry_zscore: float, hedge_ratio: float):
        self.active_pairs[pair] = {
            'side': side,
            'entry_zscore': entry_zscore,
            'hedge_ratio': hedge_ratio,
            'entry_time': datetime.utcnow(),
            'periods_held': 0
        }
    
    def update_periods(self):
        for pair in self.active_pairs:
            self.active_pairs[pair]['periods_held'] += 1
    
    def remove_pair(self, pair: Tuple[str, str]):
        if pair in self.active_pairs:
            del self.active_pairs[pair]
    
    def get_active_pairs(self) -> List[Tuple[str, str]]:
        return list(self.active_pairs.keys())
