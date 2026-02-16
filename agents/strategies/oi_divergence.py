"""
Open Interest Divergence Strategy Agent

Core Logic:
- Price up + OI down = Weak rally → SHORT (longs closing, rally exhausting)
- Price down + OI down = Capitulation → LONG (forced selling, reversal setup)
- Price up + OI up = Strong trend confirmation
- Price down + OI up = Short accumulation

Why it works:
- OI tracks actual positioning, not just price
- Divergences reveal hidden weakness/strength
- Classic futures market structure signal
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class OIDivergenceAgent(BaseStrategyAgent):
    """
    Open Interest Divergence Strategy
    
    Entry conditions:
    - Significant OI divergence from price action
    - Volume confirmation
    - RSI not at extremes (avoid catching falling knives)
    
    Exit conditions:
    - OI starts confirming price direction
    - Stop loss / Take profit hit
    - RSI reversal
    """
    
    DEFAULT_CONFIG = {
        # Divergence thresholds
        'price_change_threshold': 1.5,  # % price change to consider
        'oi_change_threshold': 2.0,  # % OI change to consider
        'lookback_periods': 6,  # Candles to measure change over
        
        # Confirmation
        'rsi_period': 14,
        'rsi_oversold': 30,  # Don't short below this
        'rsi_overbought': 70,  # Don't long above this
        
        # Volume filter
        'volume_threshold': 1.2,  # Must be above avg volume
        
        # Risk management
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.5,
        
        # Confidence
        'min_confidence': 0.60,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='oi-divergence',
            name='OI Divergence',
            symbols=symbols,
            config=merged_config
        )
        # OI data: symbol -> {'current': float, 'history': [(timestamp, oi), ...]}
        self._oi_data: Dict[str, Dict] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"rsi_{self.config['rsi_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20',
            'open_interest',  # Custom: needs futures data feed
        ]
    
    def set_oi_data(self, oi_data: Dict[str, Dict]):
        """Update open interest data from external data feed"""
        self._oi_data = oi_data
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    continue
                
                # Get OI data for this symbol
                oi_info = self._oi_data.get(symbol, {})
                
                signal = self._analyze_symbol(symbol, data, oi_info)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict, oi_info: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        if len(closes) < self.config['lookback_periods'] + 20:
            return None
        
        current_price = closes[-1]
        lookback = self.config['lookback_periods']
        
        # Calculate price change over lookback period
        price_start = closes[-(lookback + 1)]
        price_change_pct = ((current_price - price_start) / price_start) * 100
        
        # Get OI change
        current_oi = oi_info.get('current', 0)
        prev_oi = oi_info.get('previous', current_oi)
        oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        
        # Calculate indicators
        rsi = self._rsi(closes, self.config['rsi_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if rsi is None or atr is None or volume_sma is None:
            return None
        
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Thresholds
        price_threshold = self.config['price_change_threshold']
        oi_threshold = self.config['oi_change_threshold']
        
        # Volume check
        volume_ok = current_volume >= avg_volume * self.config['volume_threshold']
        
        signal = None
        divergence_type = None
        
        # WEAK RALLY: Price up + OI down → SHORT
        if price_change_pct >= price_threshold and oi_change_pct <= -oi_threshold:
            divergence_type = 'weak_rally'
            
            # Don't short if already oversold
            if current_rsi > self.config['rsi_oversold'] and volume_ok:
                confidence = self._calculate_confidence(
                    abs(price_change_pct), abs(oi_change_pct),
                    current_rsi, divergence_type
                )
                
                if confidence >= self.config['min_confidence']:
                    stop_loss = current_price + (current_atr * self.config['stop_loss_atr'])
                    take_profit = current_price - (current_atr * self.config['take_profit_atr'])
                    
                    signal = Signal(
                        strategy_id=self.agent_id,
                        symbol=symbol,
                        side=SignalSide.SHORT,
                        confidence=confidence,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        reason=f"SHORT OI Divergence: Weak rally - Price +{price_change_pct:.1f}% but OI {oi_change_pct:.1f}%, RSI={current_rsi:.1f}",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'price_change': price_change_pct,
                            'oi_change': oi_change_pct,
                            'rsi': current_rsi,
                            'divergence_type': divergence_type
                        }
                    )
        
        # CAPITULATION: Price down + OI down → LONG (reversal play)
        elif price_change_pct <= -price_threshold and oi_change_pct <= -oi_threshold:
            divergence_type = 'capitulation'
            
            # Don't long if already overbought (shouldn't happen, but check)
            if current_rsi < self.config['rsi_overbought'] and volume_ok:
                # Extra check: RSI should be relatively low for capitulation
                if current_rsi < 50:
                    confidence = self._calculate_confidence(
                        abs(price_change_pct), abs(oi_change_pct),
                        current_rsi, divergence_type
                    )
                    
                    if confidence >= self.config['min_confidence']:
                        stop_loss = current_price - (current_atr * self.config['stop_loss_atr'])
                        take_profit = current_price + (current_atr * self.config['take_profit_atr'])
                        
                        signal = Signal(
                            strategy_id=self.agent_id,
                            symbol=symbol,
                            side=SignalSide.LONG,
                            confidence=confidence,
                            price=current_price,
                            timestamp=datetime.utcnow(),
                            reason=f"LONG OI Divergence: Capitulation - Price {price_change_pct:.1f}% and OI {oi_change_pct:.1f}%, RSI={current_rsi:.1f}",
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'price_change': price_change_pct,
                                'oi_change': oi_change_pct,
                                'rsi': current_rsi,
                                'divergence_type': divergence_type
                            }
                        )
        
        if signal:
            logger.info(f"[oi-divergence] {symbol}: {divergence_type} divergence detected")
        
        return signal
    
    def _calculate_confidence(self, price_change: float, oi_change: float,
                             rsi: float, divergence_type: str) -> float:
        """Calculate signal confidence based on divergence strength"""
        confidence = 0.5
        
        # Divergence strength bonus
        price_strength = min(price_change / 3.0, 0.15)  # Cap contribution
        oi_strength = min(oi_change / 5.0, 0.15)
        confidence += price_strength + oi_strength
        
        # RSI alignment bonus
        if divergence_type == 'weak_rally':
            # Better if RSI is high (overbought)
            if rsi > 60:
                confidence += 0.1
            elif rsi > 50:
                confidence += 0.05
        elif divergence_type == 'capitulation':
            # Better if RSI is low (oversold)
            if rsi < 35:
                confidence += 0.15
            elif rsi < 45:
                confidence += 0.1
        
        return min(confidence, 0.90)
    
    # Technical indicator helpers
    def _rsi(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period + 1:
            return None
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))
        
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period-1:]
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.convolve(tr, np.ones(period)/period, mode='valid')
