"""
Funding Rate Fade Strategy Agent

Core Logic:
- When funding is extremely positive (longs paying), SHORT
- When funding is extremely negative (shorts paying), LONG
- Mean reversion play on crowded positioning
- Uses funding rate threshold + confirmation signals

Why it works:
- Extreme funding = crowded trade, likely to unwind
- Collecting funding while fading the crowd
- Works best in ranging markets
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class FundingFadeAgent(BaseStrategyAgent):
    """
    Funding Rate Fade Strategy
    
    Entry conditions:
    - Funding rate exceeds threshold (positive → short, negative → long)
    - Price not in strong trend (ADX < 30)
    - Volume not spiking (avoid momentum moves)
    
    Exit conditions:
    - Funding normalizes (< half threshold)
    - Stop loss / Take profit hit
    - Trend develops (ADX > 35)
    """
    
    DEFAULT_CONFIG = {
        # Funding thresholds
        'funding_threshold_high': 0.0005,  # 0.05% - short when funding above
        'funding_threshold_low': -0.0005,  # -0.05% - long when funding below
        'funding_extreme': 0.001,  # 0.1% - stronger signal
        
        # Trend filter (avoid fading strong trends)
        'adx_period': 14,
        'adx_max': 30,  # Don't fade if ADX above this
        'adx_exit': 35,  # Exit if trend develops
        
        # Risk management
        'atr_period': 14,
        'stop_loss_atr': 2.5,
        'take_profit_atr': 3.0,
        
        # Confidence
        'min_confidence': 0.55,
        
        # Position sizing based on funding extremeness
        'extreme_size_multiplier': 1.5,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='funding-fade',
            name='Funding Rate Fade',
            symbols=symbols,
            config=merged_config
        )
        self._funding_data: Dict[str, float] = {}  # symbol -> current funding rate
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"adx_{self.config['adx_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20',
            'funding_rate',  # Custom: needs futures data feed
        ]
    
    def set_funding_data(self, funding_data: Dict[str, float]):
        """Update funding rates from external data feed"""
        self._funding_data = funding_data
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    continue
                
                # Get funding rate for this symbol
                funding_rate = self._funding_data.get(symbol, 0)
                
                signal = self._analyze_symbol(symbol, data, funding_rate)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict, funding_rate: float) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate indicators
        adx = self._adx(highs, lows, closes, self.config['adx_period'])
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        volume_sma = self._sma(volumes, 20)
        
        if adx is None or atr is None or volume_sma is None:
            return None
        
        current_adx = adx[-1]
        current_atr = atr[-1]
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]
        
        # Check if funding is extreme enough
        high_threshold = self.config['funding_threshold_high']
        low_threshold = self.config['funding_threshold_low']
        extreme_threshold = self.config['funding_extreme']
        
        # Don't fade strong trends
        if current_adx > self.config['adx_max']:
            logger.debug(f"[funding-fade] {symbol}: Skipping, ADX={current_adx:.1f} too high")
            return None
        
        # Avoid volume spikes (momentum moves)
        if current_volume > avg_volume * 2.0:
            logger.debug(f"[funding-fade] {symbol}: Skipping, volume spike")
            return None
        
        signal = None
        
        # HIGH FUNDING -> SHORT (longs paying, fade them)
        if funding_rate >= high_threshold:
            is_extreme = funding_rate >= extreme_threshold
            confidence = self._calculate_confidence(
                funding_rate, high_threshold, extreme_threshold,
                current_adx, is_short=True
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price + (current_atr * self.config['stop_loss_atr'])
                take_profit = current_price - (current_atr * self.config['take_profit_atr'])
                
                size_mult = self.config['extreme_size_multiplier'] if is_extreme else 1.0
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"SHORT Funding Fade: rate={funding_rate*100:.4f}% (longs overleveraged), ADX={current_adx:.1f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size_pct=size_mult,
                    metadata={
                        'funding_rate': funding_rate,
                        'adx': current_adx,
                        'atr': current_atr,
                        'is_extreme': is_extreme
                    }
                )
        
        # LOW FUNDING -> LONG (shorts paying, fade them)
        elif funding_rate <= low_threshold:
            is_extreme = funding_rate <= -extreme_threshold
            confidence = self._calculate_confidence(
                abs(funding_rate), abs(low_threshold), extreme_threshold,
                current_adx, is_short=False
            )
            
            if confidence >= self.config['min_confidence']:
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr'])
                take_profit = current_price + (current_atr * self.config['take_profit_atr'])
                
                size_mult = self.config['extreme_size_multiplier'] if is_extreme else 1.0
                
                signal = Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    reason=f"LONG Funding Fade: rate={funding_rate*100:.4f}% (shorts overleveraged), ADX={current_adx:.1f}",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size_pct=size_mult,
                    metadata={
                        'funding_rate': funding_rate,
                        'adx': current_adx,
                        'atr': current_atr,
                        'is_extreme': is_extreme
                    }
                )
        
        return signal
    
    def _calculate_confidence(self, rate: float, threshold: float, extreme: float,
                             adx: float, is_short: bool) -> float:
        """Calculate signal confidence based on funding extremeness and conditions"""
        confidence = 0.5
        
        # Funding rate strength
        rate_strength = min(rate / extreme, 2.0)  # Cap at 2x extreme
        confidence += rate_strength * 0.2
        
        # ADX bonus (lower = better for mean reversion)
        if adx < 20:
            confidence += 0.15
        elif adx < 25:
            confidence += 0.1
        elif adx < 30:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def should_exit(self, position: 'Position', current_price: float,
                   current_funding: float = None) -> Optional[str]:
        """Check for exit conditions including funding normalization"""
        # Standard stop/take profit checks from base class
        base_exit = super().should_exit(position, current_price)
        if base_exit:
            return base_exit
        
        # Check if funding has normalized
        if current_funding is not None:
            threshold = self.config['funding_threshold_high'] / 2
            if abs(current_funding) < threshold:
                return 'funding_normalized'
        
        return None
    
    # Technical indicator helpers
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
    
    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        if len(high) < period * 2:
            return None
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, period) / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)
        
        return adx
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
