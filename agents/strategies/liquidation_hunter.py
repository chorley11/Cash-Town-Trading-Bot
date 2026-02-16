"""
Liquidation Hunter Strategy Agent

Core Logic:
- Detect when price is approaching major liquidation levels
- Trade into liquidation cascades for momentum
- Fade exhausted cascades for mean reversion
- Use order book depth to estimate liquidation zones

Why it works:
- Liquidations create forced selling/buying pressure
- Cascades accelerate price movement
- Exhausted cascades often lead to sharp reversals
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class LiquidationHunterAgent(BaseStrategyAgent):
    """
    Liquidation Hunter Strategy
    
    Two modes:
    1. CASCADE MODE: Ride into liquidation cascades
       - Price approaching liquidation cluster
       - Volume spike + momentum confirmation
       - Trade WITH the cascade
    
    2. FADE MODE: Fade exhausted cascades
       - Cascade just happened (volume spike + price gap)
       - RSI at extreme
       - Fade for mean reversion
    
    Exit conditions:
    - Cascade momentum fades
    - RSI reversal
    - Stop loss / Take profit hit
    """
    
    DEFAULT_CONFIG = {
        # Liquidation level estimation
        'leverage_levels': [10, 20, 50],  # Common leverage for liq estimation
        'liq_proximity_pct': 2.0,  # % distance to liq level to consider
        
        # Cascade detection
        'volume_spike_threshold': 2.5,  # Volume vs avg to detect cascade
        'price_move_threshold': 1.5,  # % price move to confirm cascade
        'momentum_period': 5,  # Candles for momentum calc
        
        # Fade conditions
        'rsi_extreme_low': 25,  # Oversold - potential long fade
        'rsi_extreme_high': 75,  # Overbought - potential short fade
        'fade_volume_ratio': 3.0,  # Volume spike needed for fade setup
        
        # Risk management
        'atr_period': 14,
        'stop_loss_atr_cascade': 1.5,  # Tighter stops for cascade trades
        'stop_loss_atr_fade': 2.5,  # Wider stops for fade trades
        'take_profit_atr_cascade': 2.5,
        'take_profit_atr_fade': 4.0,
        
        # Confidence
        'min_confidence_cascade': 0.60,
        'min_confidence_fade': 0.65,  # Higher bar for contrarian trades
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='liquidation-hunter',
            name='Liquidation Hunter',
            symbols=symbols,
            config=merged_config
        )
        # Order book data for liquidation estimation
        self._order_book: Dict[str, Dict] = {}
        # Open interest for position sizing
        self._oi_data: Dict[str, float] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"rsi_{14}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20',
            'order_book',  # Custom: needs futures data feed
        ]
    
    def set_order_book_data(self, ob_data: Dict[str, Dict]):
        """Update order book data from external data feed"""
        self._order_book = ob_data
    
    def set_oi_data(self, oi_data: Dict[str, float]):
        """Update OI data for sizing"""
        self._oi_data = oi_data
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    continue
                
                ob_info = self._order_book.get(symbol, {})
                
                # Try cascade mode first, then fade mode
                signal = self._analyze_cascade(symbol, data, ob_info)
                if not signal:
                    signal = self._analyze_fade(symbol, data, ob_info)
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_cascade(self, symbol: str, data: Dict, ob_info: Dict) -> Optional[Signal]:
        """Look for liquidation cascade opportunities"""
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if atr is None or volume_sma is None:
            return None
        
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Estimate liquidation levels
        liq_levels = self._estimate_liquidation_levels(current_price)
        
        # Check proximity to liquidation levels
        nearest_long_liq = None
        nearest_short_liq = None
        
        for lev, levels in liq_levels.items():
            long_liq = levels['long']
            short_liq = levels['short']
            
            long_dist_pct = ((current_price - long_liq) / current_price) * 100
            short_dist_pct = ((short_liq - current_price) / current_price) * 100
            
            # Track nearest levels
            if 0 < long_dist_pct < self.config['liq_proximity_pct']:
                if nearest_long_liq is None or long_liq > nearest_long_liq:
                    nearest_long_liq = long_liq
            
            if 0 < short_dist_pct < self.config['liq_proximity_pct']:
                if nearest_short_liq is None or short_liq < nearest_short_liq:
                    nearest_short_liq = short_liq
        
        # Calculate recent momentum
        momentum_period = self.config['momentum_period']
        if len(closes) > momentum_period:
            price_momentum = (closes[-1] - closes[-momentum_period]) / closes[-momentum_period] * 100
        else:
            price_momentum = 0
        
        # Volume spike check
        volume_spike = current_volume > avg_volume * self.config['volume_spike_threshold']
        
        signal = None
        
        # Approaching SHORT liquidations (price moving up toward them)
        if nearest_short_liq and price_momentum > 0.5 and volume_spike:
            confidence = self._calculate_cascade_confidence(
                price_momentum, current_volume / avg_volume,
                nearest_short_liq, current_price, is_long=True
            )
            
            if confidence >= self.config['min_confidence_cascade']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr_cascade'])
                take_profit = nearest_short_liq * 1.005  # Target just past liq level
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG Cascade: Short liquidations near ${nearest_short_liq:.2f}, momentum +{price_momentum:.1f}%",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'mode': 'cascade',
                        'target_liq_level': nearest_short_liq,
                        'momentum': price_momentum,
                        'volume_ratio': current_volume / avg_volume
                    }
                )
        
        # Approaching LONG liquidations (price moving down toward them)
        elif nearest_long_liq and price_momentum < -0.5 and volume_spike:
            confidence = self._calculate_cascade_confidence(
                abs(price_momentum), current_volume / avg_volume,
                nearest_long_liq, current_price, is_long=False
            )
            
            if confidence >= self.config['min_confidence_cascade']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr_cascade'])
                take_profit = nearest_long_liq * 0.995  # Target just past liq level
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT Cascade: Long liquidations near ${nearest_long_liq:.2f}, momentum {price_momentum:.1f}%",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'mode': 'cascade',
                        'target_liq_level': nearest_long_liq,
                        'momentum': price_momentum,
                        'volume_ratio': current_volume / avg_volume
                    }
                )
        
        return signal
    
    def _analyze_fade(self, symbol: str, data: Dict, ob_info: Dict) -> Optional[Signal]:
        """Look for exhausted cascade fade opportunities"""
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        rsi = self._rsi(closes, 14)
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if rsi is None or atr is None or volume_sma is None:
            return None
        
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Look for volume spike + RSI extreme = exhausted cascade
        volume_spike = current_volume > avg_volume * self.config['fade_volume_ratio']
        
        signal = None
        
        # Oversold + volume spike = potential long fade
        if current_rsi < self.config['rsi_extreme_low'] and volume_spike:
            # Check recent price action (should be down)
            recent_change = (closes[-1] - closes[-3]) / closes[-3] * 100
            
            if recent_change < -2:  # Confirm downward cascade happened
                confidence = self._calculate_fade_confidence(
                    current_rsi, current_volume / avg_volume, recent_change
                )
                
                if confidence >= self.config['min_confidence_fade']:
                    stop_loss = current_price - (current_atr * self.config['stop_loss_atr_fade'])
                    take_profit = current_price + (current_atr * self.config['take_profit_atr_fade'])
                    
                    signal = Signal(
                        strategy_id=self.agent_id,
                        symbol=symbol,
                        side=SignalSide.LONG,
                        confidence=confidence,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        reason=f"LONG Fade: Exhausted cascade - RSI={current_rsi:.1f}, Vol={current_volume/avg_volume:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'mode': 'fade',
                            'rsi': current_rsi,
                            'volume_ratio': current_volume / avg_volume,
                            'recent_move': recent_change
                        }
                    )
        
        # Overbought + volume spike = potential short fade
        elif current_rsi > self.config['rsi_extreme_high'] and volume_spike:
            recent_change = (closes[-1] - closes[-3]) / closes[-3] * 100
            
            if recent_change > 2:  # Confirm upward cascade happened
                confidence = self._calculate_fade_confidence(
                    100 - current_rsi, current_volume / avg_volume, abs(recent_change)
                )
                
                if confidence >= self.config['min_confidence_fade']:
                    stop_loss = current_price + (current_atr * self.config['stop_loss_atr_fade'])
                    take_profit = current_price - (current_atr * self.config['take_profit_atr_fade'])
                    
                    signal = Signal(
                        strategy_id=self.agent_id,
                        symbol=symbol,
                        side=SignalSide.SHORT,
                        confidence=confidence,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        reason=f"SHORT Fade: Exhausted cascade - RSI={current_rsi:.1f}, Vol={current_volume/avg_volume:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'mode': 'fade',
                            'rsi': current_rsi,
                            'volume_ratio': current_volume / avg_volume,
                            'recent_move': recent_change
                        }
                    )
        
        return signal
    
    def _estimate_liquidation_levels(self, current_price: float) -> Dict[int, Dict[str, float]]:
        """Estimate liquidation levels for common leverage tiers"""
        maintenance_margin = 0.005  # 0.5% typical
        
        levels = {}
        for lev in self.config['leverage_levels']:
            # Long liquidations (price drops)
            long_liq = current_price * (1 - (1/lev) + maintenance_margin)
            # Short liquidations (price rises)
            short_liq = current_price * (1 + (1/lev) - maintenance_margin)
            
            levels[lev] = {'long': long_liq, 'short': short_liq}
        
        return levels
    
    def _calculate_cascade_confidence(self, momentum: float, volume_ratio: float,
                                      liq_level: float, current_price: float,
                                      is_long: bool) -> float:
        confidence = 0.5
        
        # Momentum strength
        if momentum > 2:
            confidence += 0.15
        elif momentum > 1:
            confidence += 0.1
        
        # Volume strength
        if volume_ratio > 4:
            confidence += 0.15
        elif volume_ratio > 3:
            confidence += 0.1
        
        # Proximity to liq level (closer = higher confidence)
        dist = abs(liq_level - current_price) / current_price * 100
        if dist < 1:
            confidence += 0.1
        elif dist < 1.5:
            confidence += 0.05
        
        return min(confidence, 0.85)
    
    def _calculate_fade_confidence(self, rsi_extremeness: float,
                                  volume_ratio: float, price_move: float) -> float:
        confidence = 0.5
        
        # RSI extremeness (lower rsi_extremeness = more extreme)
        if rsi_extremeness < 20:
            confidence += 0.15
        elif rsi_extremeness < 25:
            confidence += 0.1
        
        # Volume spike
        if volume_ratio > 4:
            confidence += 0.1
        elif volume_ratio > 3:
            confidence += 0.05
        
        # Price move magnitude
        if abs(price_move) > 4:
            confidence += 0.1
        elif abs(price_move) > 3:
            confidence += 0.05
        
        return min(confidence, 0.85)
    
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
