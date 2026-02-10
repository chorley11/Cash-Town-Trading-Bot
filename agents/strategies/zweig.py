"""
Zweig Super Model Strategy Agent
Based on Martin Zweig's "Super Model" market timing system.

Core Logic:
- Multi-factor scoring system (0-10)
- Trend score, Momentum score, Health score
- Only trade when Super Model score reaches extremes
- Persistence check (signal must hold for confirmation)
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..base import BaseStrategyAgent, Signal, SignalSide, Position

logger = logging.getLogger(__name__)

class ZweigAgent(BaseStrategyAgent):
    """
    Zweig Super Model Strategy
    
    Generates signals based on multi-factor scoring:
    - Trend Score (0-3): MA alignment
    - Momentum Score (0-3): RSI + MACD
    - Health Score (0-2): Volume + volatility
    - Persistence (0-2): Signal holding time
    
    Total: 0-10, trade at extremes (<=3 for shorts, >=7 for longs)
    """
    
    DEFAULT_CONFIG = {
        'sma_short': 10,
        'sma_medium': 20,
        'sma_long': 50,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
        'volume_period': 20,
        'short_threshold': 3,    # Score <= 3 = short
        'long_threshold': 7,     # Score >= 7 = long
        'persistence_bars': 2,   # Signal must hold for N bars
        'stop_loss_pct': 2.0,    # 2% stop
        'take_profit_pct': 3.0,  # 3% take profit
    }
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='zweig',
            name='Zweig Super Model',
            symbols=symbols,
            config=merged_config
        )
        # Track scores for persistence check
        self._score_history: Dict[str, List[float]] = {}
    
    def get_required_indicators(self) -> List[str]:
        return [
            f"sma_{self.config['sma_short']}",
            f"sma_{self.config['sma_medium']}",
            f"sma_{self.config['sma_long']}",
            f"rsi_{self.config['rsi_period']}",
            'macd',
            f"atr_{self.config['atr_period']}",
            'volume_sma_20'
        ]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate Zweig Super Model signals"""
        signals = []
        
        for symbol in self.symbols:
            try:
                data = market_data.get(symbol)
                if not data or len(data.get('close', [])) < 100:
                    continue
                
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Analyze symbol and calculate Super Model score"""
        
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        current_price = closes[-1]
        
        # Calculate components
        trend_score = self._calculate_trend_score(closes)
        momentum_score = self._calculate_momentum_score(closes)
        health_score = self._calculate_health_score(closes, volumes)
        
        # Total score
        total_score = trend_score + momentum_score + health_score
        
        # Track score history for persistence
        if symbol not in self._score_history:
            self._score_history[symbol] = []
        self._score_history[symbol].append(total_score)
        if len(self._score_history[symbol]) > 10:
            self._score_history[symbol] = self._score_history[symbol][-10:]
        
        # Check persistence
        persistence_score = self._check_persistence(symbol, total_score)
        final_score = total_score + persistence_score
        
        # Calculate ATR for stop/take profit
        atr = self._atr(highs, lows, closes, self.config['atr_period'])
        current_atr = atr[-1] if atr is not None and len(atr) > 0 else current_price * 0.02
        
        signal = None
        
        # SHORT signal: Low score
        if final_score <= self.config['short_threshold']:
            confidence = (self.config['short_threshold'] - final_score + 1) / 4  # 0.25 - 1.0
            confidence = min(max(confidence, 0.3), 0.9)
            
            stop_loss = current_price * (1 + self.config['stop_loss_pct'] / 100)
            take_profit = current_price * (1 - self.config['take_profit_pct'] / 100)
            
            signal = Signal(
                strategy_id=self.agent_id,
                symbol=symbol,
                side=SignalSide.SHORT,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.utcnow(),
                reason=f"Zweig SHORT signal: Super Model score {final_score}/10 (threshold: {self.config['short_threshold']})",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'health_score': health_score,
                    'persistence_score': persistence_score,
                    'total_score': final_score
                }
            )
        
        # LONG signal: High score
        elif final_score >= self.config['long_threshold']:
            confidence = (final_score - self.config['long_threshold'] + 1) / 4
            confidence = min(max(confidence, 0.3), 0.9)
            
            stop_loss = current_price * (1 - self.config['stop_loss_pct'] / 100)
            take_profit = current_price * (1 + self.config['take_profit_pct'] / 100)
            
            signal = Signal(
                strategy_id=self.agent_id,
                symbol=symbol,
                side=SignalSide.LONG,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.utcnow(),
                reason=f"Zweig LONG signal: Super Model score {final_score}/10 (threshold: {self.config['long_threshold']})",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'health_score': health_score,
                    'persistence_score': persistence_score,
                    'total_score': final_score
                }
            )
        
        return signal
    
    def _calculate_trend_score(self, closes: np.ndarray) -> float:
        """Calculate trend score (0-3)"""
        score = 0
        
        sma_short = self._sma(closes, self.config['sma_short'])
        sma_medium = self._sma(closes, self.config['sma_medium'])
        sma_long = self._sma(closes, self.config['sma_long'])
        
        if sma_short is None or sma_medium is None or sma_long is None:
            return 1.5  # Neutral
        
        current_price = closes[-1]
        
        # Price above/below MAs
        if current_price > sma_short[-1]:
            score += 1
        if current_price > sma_medium[-1]:
            score += 1
        if current_price > sma_long[-1]:
            score += 1
        
        return score
    
    def _calculate_momentum_score(self, closes: np.ndarray) -> float:
        """Calculate momentum score (0-3)"""
        score = 0
        
        # RSI component
        rsi = self._rsi(closes, self.config['rsi_period'])
        if rsi is not None and len(rsi) > 0:
            current_rsi = rsi[-1]
            if current_rsi > 50:
                score += 1
            if current_rsi > 60:
                score += 0.5
            if current_rsi < 40:
                score -= 0.5
        
        # MACD component
        macd, signal_line, hist = self._macd(closes)
        if macd is not None:
            if macd[-1] > signal_line[-1]:
                score += 1
            if hist[-1] > 0 and hist[-1] > hist[-2]:  # Increasing histogram
                score += 0.5
        
        return max(0, min(3, score))
    
    def _calculate_health_score(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate market health score (0-2)"""
        score = 1  # Start neutral
        
        # Volume health
        volume_sma = self._sma(volumes, self.config['volume_period'])
        if volume_sma is not None and len(volume_sma) > 0:
            if volumes[-1] > volume_sma[-1]:
                score += 0.5
            else:
                score -= 0.5
        
        # Volatility check (lower recent volatility = healthier trend)
        if len(closes) >= 10:
            recent_std = np.std(closes[-10:]) / np.mean(closes[-10:])
            older_std = np.std(closes[-20:-10]) / np.mean(closes[-20:-10]) if len(closes) >= 20 else recent_std
            
            if recent_std < older_std:  # Volatility decreasing
                score += 0.5
        
        return max(0, min(2, score))
    
    def _check_persistence(self, symbol: str, current_score: float) -> float:
        """Check if signal has persisted (0-2)"""
        history = self._score_history.get(symbol, [])
        if len(history) < self.config['persistence_bars']:
            return 0
        
        recent_scores = history[-self.config['persistence_bars']:]
        
        # All recent scores in same direction?
        all_low = all(s <= self.config['short_threshold'] + 1 for s in recent_scores)
        all_high = all(s >= self.config['long_threshold'] - 1 for s in recent_scores)
        
        if all_low or all_high:
            return 2
        return 0
    
    # Indicator calculations
    def _sma(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices: np.ndarray):
        """Calculate MACD"""
        if len(prices) < self.config['macd_slow'] + self.config['macd_signal']:
            return None, None, None
        
        ema_fast = self._ema(prices, self.config['macd_fast'])
        ema_slow = self._ema(prices, self.config['macd_slow'])
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        # Align lengths
        min_len = min(len(ema_fast), len(ema_slow))
        macd = ema_fast[-min_len:] - ema_slow[-min_len:]
        
        signal_line = self._ema(macd, self.config['macd_signal'])
        if signal_line is None:
            return None, None, None
        
        min_len = min(len(macd), len(signal_line))
        hist = macd[-min_len:] - signal_line[-min_len:]
        
        return macd[-min_len:], signal_line[-min_len:], hist
    
    def _ema(self, data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(high) < period + 1:
            return None
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.convolve(tr, np.ones(period)/period, mode='valid')
