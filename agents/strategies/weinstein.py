"""
Weinstein Stage Analysis Strategy Agent
Based on Stan Weinstein's market stage analysis from "Secrets for Profiting in Bull and Bear Markets"

Core Logic:
- Identify market stages (1-4) using price vs 30-week MA
- Buy Stage 2 breakouts (accumulation -> markup)
- Short Stage 4 breakdowns (distribution -> markdown)
- Use weekly charts for trend, daily for entries
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide

logger = logging.getLogger(__name__)

class WeinsteinAgent(BaseStrategyAgent):
    """
    Weinstein Stage Analysis Strategy
    
    Stages:
    1. Basing (accumulation) - Price flat around MA
    2. Advancing (markup) - Price above rising MA - BUY
    3. Topping (distribution) - Price flat, MA flattening
    4. Declining (markdown) - Price below falling MA - SHORT
    
    Entry:
    - Stage 2: Price breaks above resistance on volume
    - Stage 4: Price breaks below support on volume
    """
    
    DEFAULT_CONFIG = {
        'ma_period': 30,        # 30-period MA (represents weekly on daily charts)
        'ma_slope_period': 5,   # Periods to measure MA slope
        'volume_period': 20,
        'volume_threshold': 1.3,
        'resistance_lookback': 20,
        'support_lookback': 20,
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.0,
        'min_confidence': 0.55,
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='weinstein',
            name='Weinstein Stage Analysis',
            symbols=symbols,
            config=merged_config
        )
        self._stage_history: Dict[str, List[int]] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['ma_period']}",
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 50:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate MA and its slope
        ma = self._sma(closes, self.config['ma_period'])
        if ma is None or len(ma) < self.config['ma_slope_period']:
            return None
        
        current_ma = ma[-1]
        ma_slope = (ma[-1] - ma[-self.config['ma_slope_period']]) / self.config['ma_slope_period']
        ma_slope_pct = ma_slope / current_ma * 100
        
        # ATR for stops
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        current_atr = atr[-1] if atr is not None else current_price * 0.02
        
        # Volume analysis
        volume_sma = self._sma(volumes, self.config['volume_period'])
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1] if volume_sma is not None else current_volume
        volume_ratio = current_volume / avg_volume
        
        # Determine current stage
        stage = self._determine_stage(current_price, current_ma, ma_slope_pct)
        
        # Track stage history
        if symbol not in self._stage_history:
            self._stage_history[symbol] = []
        self._stage_history[symbol].append(stage)
        if len(self._stage_history[symbol]) > 10:
            self._stage_history[symbol] = self._stage_history[symbol][-10:]
        
        # Check for stage transitions
        prev_stages = self._stage_history[symbol][-3:-1] if len(self._stage_history[symbol]) > 2 else []
        
        # Find support/resistance levels
        resistance = np.max(highs[-self.config['resistance_lookback']:])
        support = np.min(lows[-self.config['support_lookback']:])
        
        signal = None
        
        # Stage 2 entry: Transitioning from Stage 1, breaking resistance
        if stage == 2 and (1 in prev_stages or 2 in prev_stages):
            if current_price > resistance * 0.99 and volume_ratio > self.config['volume_threshold']:
                confidence = self._calculate_confidence(stage, ma_slope_pct, volume_ratio, True)
                
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
                        reason=f"LONG Weinstein: Stage 2 breakout + MA slope {ma_slope_pct:.2f}% + volume {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'stage': stage,
                            'ma': current_ma,
                            'ma_slope_pct': ma_slope_pct,
                            'resistance': resistance,
                            'volume_ratio': volume_ratio
                        }
                    )
        
        # Stage 4 entry: Transitioning from Stage 3, breaking support
        elif stage == 4 and (3 in prev_stages or 4 in prev_stages):
            if current_price < support * 1.01 and volume_ratio > self.config['volume_threshold']:
                confidence = self._calculate_confidence(stage, ma_slope_pct, volume_ratio, False)
                
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
                        reason=f"SHORT Weinstein: Stage 4 breakdown + MA slope {ma_slope_pct:.2f}% + volume {volume_ratio:.1f}x",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'stage': stage,
                            'ma': current_ma,
                            'ma_slope_pct': ma_slope_pct,
                            'support': support,
                            'volume_ratio': volume_ratio
                        }
                    )
        
        return signal
    
    def _determine_stage(self, price: float, ma: float, ma_slope_pct: float) -> int:
        """Determine Weinstein stage (1-4)"""
        price_above_ma = price > ma
        ma_rising = ma_slope_pct > 0.1
        ma_falling = ma_slope_pct < -0.1
        ma_flat = not ma_rising and not ma_falling
        
        if price_above_ma and ma_rising:
            return 2  # Advancing
        elif price_above_ma and ma_flat:
            return 3  # Topping
        elif not price_above_ma and ma_falling:
            return 4  # Declining
        elif not price_above_ma and ma_flat:
            return 1  # Basing
        elif price_above_ma and ma_falling:
            return 3  # Late topping
        else:
            return 1  # Default to basing
    
    def _calculate_confidence(self, stage: int, ma_slope: float, volume_ratio: float, is_long: bool) -> float:
        confidence = 0.5
        
        # Stage clarity
        if is_long and stage == 2:
            confidence += 0.1
        elif not is_long and stage == 4:
            confidence += 0.1
        
        # MA slope strength
        slope_strength = abs(ma_slope)
        if slope_strength > 0.5:
            confidence += 0.15
        elif slope_strength > 0.3:
            confidence += 0.1
        
        # Volume confirmation
        if volume_ratio > 2.0:
            confidence += 0.15
        elif volume_ratio > 1.5:
            confidence += 0.1
        elif volume_ratio > 1.3:
            confidence += 0.05
        
        return min(confidence, 0.85)
    
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
