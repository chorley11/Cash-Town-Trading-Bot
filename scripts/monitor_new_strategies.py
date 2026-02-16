#!/usr/bin/env python3
"""
Real-time Monitoring for Cash Town's 5 New Strategies (Feb 2026)

Monitors:
1. Funding Fade - Current funding rates
2. OI Divergence - Open interest changes
3. Volatility Breakout - Squeeze patterns
4. Liquidation Hunter - Liquidation level proximity
5. Correlation Pairs - BTC/ETH correlation status

Outputs JSON-friendly data for the dashboard API.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logger = logging.getLogger(__name__)

# ============== Data Classes ==============

@dataclass
class FundingRateData:
    """Funding rate data for a symbol"""
    symbol: str
    current_rate: float
    rate_pct: float  # Rate as percentage
    annualized_pct: float  # Annualized rate
    next_funding_time: Optional[str] = None
    sentiment: str = 'neutral'  # bullish, bearish, neutral
    signal_strength: float = 0.0  # 0-1
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OpenInterestData:
    """Open interest data for a symbol"""
    symbol: str
    open_interest: float
    open_interest_value_usd: float
    mark_price: float
    change_1h_pct: float = 0.0
    change_4h_pct: float = 0.0
    change_24h_pct: float = 0.0
    divergence_signal: Optional[str] = None  # weak_rally, capitulation, strong_trend
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SqueezePattern:
    """Bollinger squeeze pattern detection"""
    symbol: str
    in_squeeze: bool
    squeeze_candles: int  # How many candles in squeeze
    bb_width_percentile: float  # Current BB width vs history
    kc_squeeze: bool  # BB inside Keltner
    breakout_direction: Optional[str] = None  # long, short, None
    momentum: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CorrelationStatus:
    """BTC/ETH and other pair correlations"""
    pair: str
    symbol_a: str
    symbol_b: str
    correlation: float
    spread_zscore: float
    hedge_ratio: float
    signal: Optional[str] = None  # long_a_short_b, short_a_long_b, None
    mean_reversion_expected: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class StrategyStatus:
    """Status of a single strategy"""
    strategy_id: str
    name: str
    enabled: bool
    active_signals: int
    last_signal_time: Optional[str] = None
    symbols_watching: int = 0
    current_conditions: Dict[str, Any] = field(default_factory=dict)
    health: str = 'ok'  # ok, warning, error
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============== Monitor Class ==============

class NewStrategiesMonitor:
    """
    Real-time monitor for the 5 new futures-specific strategies.
    """
    
    DEFAULT_SYMBOLS = [
        'XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'AVAXUSDTM',
        'LINKUSDTM', 'XRPUSDTM', 'DOGEUSDTM', 'BNBUSDTM',
        'ADAUSDTM', 'MATICUSDTM', 'DOTUSDTM', 'LTCUSDTM'
    ]
    
    CORRELATION_PAIRS = [
        ('XBTUSDTM', 'ETHUSDTM'),
        ('SOLUSDTM', 'AVAXUSDTM'),
    ]
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self._futures_data = None
        self._data_feed = None
        self._price_cache: Dict[str, Dict] = {}
        self._oi_history: Dict[str, List[tuple]] = {}
        
    def _get_futures_data(self):
        """Lazy load futures data fetcher"""
        if self._futures_data is None:
            try:
                from data.futures_data import KuCoinFuturesData
                self._futures_data = KuCoinFuturesData(self.symbols)
            except Exception as e:
                logger.warning(f"Could not load KuCoinFuturesData: {e}")
        return self._futures_data
    
    def _get_data_feed(self):
        """Lazy load data feed"""
        if self._data_feed is None:
            try:
                from data.feed import KuCoinDataFeed
                self._data_feed = KuCoinDataFeed(self.symbols, '15m')
            except Exception as e:
                logger.warning(f"Could not load KuCoinDataFeed: {e}")
        return self._data_feed

    # ==========================================
    # Funding Rate Monitoring
    # ==========================================
    
    def get_funding_rates(self) -> Dict[str, FundingRateData]:
        """Fetch current funding rates for all symbols"""
        results = {}
        futures = self._get_futures_data()
        
        if not futures:
            # Return mock data structure for API testing
            for symbol in self.symbols:
                results[symbol] = FundingRateData(
                    symbol=symbol,
                    current_rate=0.0,
                    rate_pct=0.0,
                    annualized_pct=0.0,
                    sentiment='unknown',
                    signal_strength=0.0
                )
            return results
        
        for symbol in self.symbols:
            try:
                funding = futures.fetch_funding_rate(symbol)
                
                if funding:
                    rate = funding.current_rate
                    rate_pct = rate * 100
                    # Funding is typically every 8 hours, so 3x per day, 365 days
                    annualized = rate_pct * 3 * 365
                    
                    # Determine sentiment and signal strength
                    if rate > 0.0005:  # 0.05%+
                        sentiment = 'bearish'  # Longs are overleveraged
                        signal_strength = min(1.0, rate / 0.001)
                    elif rate < -0.0005:
                        sentiment = 'bullish'  # Shorts are overleveraged
                        signal_strength = min(1.0, abs(rate) / 0.001)
                    else:
                        sentiment = 'neutral'
                        signal_strength = 0.0
                    
                    results[symbol] = FundingRateData(
                        symbol=symbol,
                        current_rate=rate,
                        rate_pct=round(rate_pct, 4),
                        annualized_pct=round(annualized, 2),
                        next_funding_time=funding.next_funding_time.isoformat() if funding.next_funding_time else None,
                        sentiment=sentiment,
                        signal_strength=round(signal_strength, 2)
                    )
                else:
                    results[symbol] = FundingRateData(
                        symbol=symbol,
                        current_rate=0.0,
                        rate_pct=0.0,
                        annualized_pct=0.0,
                        sentiment='unknown'
                    )
            except Exception as e:
                logger.error(f"Error fetching funding for {symbol}: {e}")
                results[symbol] = FundingRateData(
                    symbol=symbol,
                    current_rate=0.0,
                    rate_pct=0.0,
                    annualized_pct=0.0,
                    sentiment='error'
                )
            
            time.sleep(0.05)  # Rate limiting
        
        return results
    
    # ==========================================
    # Open Interest Monitoring
    # ==========================================
    
    def get_open_interest(self) -> Dict[str, OpenInterestData]:
        """Fetch current open interest for all symbols"""
        results = {}
        futures = self._get_futures_data()
        
        if not futures:
            for symbol in self.symbols:
                results[symbol] = OpenInterestData(
                    symbol=symbol,
                    open_interest=0.0,
                    open_interest_value_usd=0.0,
                    mark_price=0.0
                )
            return results
        
        for symbol in self.symbols:
            try:
                oi_data = futures.fetch_open_interest(symbol)
                
                if oi_data:
                    # Track history for change calculations
                    now = datetime.utcnow()
                    if symbol not in self._oi_history:
                        self._oi_history[symbol] = []
                    self._oi_history[symbol].append((now, oi_data.open_interest))
                    
                    # Keep last 96 data points (24h at 15min intervals)
                    self._oi_history[symbol] = self._oi_history[symbol][-96:]
                    
                    # Calculate changes
                    history = self._oi_history[symbol]
                    change_1h = self._calc_oi_change(history, 4)   # 4 x 15min = 1h
                    change_4h = self._calc_oi_change(history, 16)  # 16 x 15min = 4h
                    change_24h = self._calc_oi_change(history, 96) # 96 x 15min = 24h
                    
                    # Detect divergence signal
                    divergence = self._detect_oi_divergence(symbol, oi_data)
                    
                    results[symbol] = OpenInterestData(
                        symbol=symbol,
                        open_interest=oi_data.open_interest,
                        open_interest_value_usd=oi_data.open_interest_value,
                        mark_price=oi_data.mark_price,
                        change_1h_pct=round(change_1h, 2),
                        change_4h_pct=round(change_4h, 2),
                        change_24h_pct=round(change_24h, 2),
                        divergence_signal=divergence
                    )
                else:
                    results[symbol] = OpenInterestData(
                        symbol=symbol,
                        open_interest=0.0,
                        open_interest_value_usd=0.0,
                        mark_price=0.0
                    )
            except Exception as e:
                logger.error(f"Error fetching OI for {symbol}: {e}")
                results[symbol] = OpenInterestData(
                    symbol=symbol,
                    open_interest=0.0,
                    open_interest_value_usd=0.0,
                    mark_price=0.0
                )
            
            time.sleep(0.05)
        
        return results
    
    def _calc_oi_change(self, history: List[tuple], periods_back: int) -> float:
        """Calculate OI change percentage over N periods"""
        if len(history) < 2:
            return 0.0
        
        current_oi = history[-1][1]
        
        if len(history) >= periods_back:
            old_oi = history[-periods_back][1]
        else:
            old_oi = history[0][1]
        
        if old_oi > 0:
            return ((current_oi - old_oi) / old_oi) * 100
        return 0.0
    
    def _detect_oi_divergence(self, symbol: str, oi_data) -> Optional[str]:
        """Detect OI/price divergence patterns"""
        try:
            feed = self._get_data_feed()
            if not feed:
                return None
            
            data = feed.get_symbol_data(symbol)
            if not data or len(data.get('close', [])) < 5:
                return None
            
            closes = data['close']
            price_change = (closes[-1] - closes[-5]) / closes[-5] * 100
            oi_change = oi_data.oi_change_pct if hasattr(oi_data, 'oi_change_pct') else 0
            
            # Detect patterns
            if price_change > 1 and oi_change < -2:
                return 'weak_rally'
            elif price_change < -1 and oi_change < -2:
                return 'capitulation'
            elif price_change > 1 and oi_change > 2:
                return 'strong_uptrend'
            elif price_change < -1 and oi_change > 2:
                return 'strong_downtrend'
            
        except Exception as e:
            logger.debug(f"Error detecting OI divergence: {e}")
        
        return None
    
    # ==========================================
    # Squeeze Pattern Detection
    # ==========================================
    
    def get_squeeze_patterns(self) -> Dict[str, SqueezePattern]:
        """Check for Bollinger squeeze patterns (volatility-breakout strategy)"""
        results = {}
        feed = self._get_data_feed()
        
        for symbol in self.symbols:
            try:
                data = None
                if feed:
                    data = feed.get_symbol_data(symbol)
                
                if data and len(data.get('close', [])) >= 50:
                    closes = np.array(data['close'])
                    highs = np.array(data['high'])
                    lows = np.array(data['low'])
                    
                    squeeze_info = self._analyze_squeeze(closes, highs, lows)
                    results[symbol] = SqueezePattern(
                        symbol=symbol,
                        **squeeze_info
                    )
                else:
                    results[symbol] = SqueezePattern(
                        symbol=symbol,
                        in_squeeze=False,
                        squeeze_candles=0,
                        bb_width_percentile=50.0,
                        kc_squeeze=False
                    )
            except Exception as e:
                logger.error(f"Error analyzing squeeze for {symbol}: {e}")
                results[symbol] = SqueezePattern(
                    symbol=symbol,
                    in_squeeze=False,
                    squeeze_candles=0,
                    bb_width_percentile=50.0,
                    kc_squeeze=False
                )
        
        return results
    
    def _analyze_squeeze(self, closes: np.ndarray, highs: np.ndarray, 
                         lows: np.ndarray, period: int = 20) -> Dict:
        """Analyze Bollinger Band squeeze"""
        # Calculate Bollinger Bands
        sma = np.convolve(closes, np.ones(period)/period, mode='valid')
        rolling_std = np.array([np.std(closes[i:i+period]) for i in range(len(closes)-period+1)])
        bb_upper = sma + (rolling_std * 2)
        bb_lower = sma - (rolling_std * 2)
        bb_width = (bb_upper - bb_lower) / sma * 100
        
        # Calculate Keltner Channels (simplified)
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        
        # Align lengths
        min_len = min(len(sma), len(atr))
        sma = sma[-min_len:]
        bb_upper = bb_upper[-min_len:]
        bb_lower = bb_lower[-min_len:]
        bb_width = bb_width[-min_len:]
        atr = atr[-min_len:]
        
        kc_upper = sma + (atr * 1.5)
        kc_lower = sma - (atr * 1.5)
        
        # Check squeeze (BB inside KC)
        in_squeeze = bb_lower[-1] > kc_lower[-1] and bb_upper[-1] < kc_upper[-1]
        
        # Count squeeze candles
        squeeze_candles = 0
        for i in range(len(bb_lower)-1, -1, -1):
            if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
                squeeze_candles += 1
            else:
                break
        
        # BB width percentile
        bb_width_percentile = (np.sum(bb_width < bb_width[-1]) / len(bb_width)) * 100
        
        # Check for breakout
        breakout_direction = None
        momentum = 0.0
        if not in_squeeze and squeeze_candles == 0:
            # Check if just broke out
            recent_closes = closes[-5:]
            momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
            if closes[-1] > bb_upper[-1]:
                breakout_direction = 'long'
            elif closes[-1] < bb_lower[-1]:
                breakout_direction = 'short'
        
        return {
            'in_squeeze': in_squeeze,
            'squeeze_candles': squeeze_candles,
            'bb_width_percentile': round(100 - bb_width_percentile, 1),  # Invert: low = compressed
            'kc_squeeze': in_squeeze,
            'breakout_direction': breakout_direction,
            'momentum': round(momentum, 2)
        }
    
    # ==========================================
    # Correlation Monitoring
    # ==========================================
    
    def get_correlation_status(self) -> Dict[str, CorrelationStatus]:
        """Check BTC/ETH and other pair correlations"""
        results = {}
        feed = self._get_data_feed()
        
        for pair in self.CORRELATION_PAIRS:
            symbol_a, symbol_b = pair
            pair_key = f"{symbol_a}/{symbol_b}"
            
            try:
                data_a = feed.get_symbol_data(symbol_a) if feed else None
                data_b = feed.get_symbol_data(symbol_b) if feed else None
                
                if data_a and data_b and len(data_a.get('close', [])) >= 50:
                    closes_a = np.array(data_a['close'][-50:])
                    closes_b = np.array(data_b['close'][-50:])
                    
                    # Calculate returns
                    returns_a = np.diff(closes_a) / closes_a[:-1]
                    returns_b = np.diff(closes_b) / closes_b[:-1]
                    
                    # Correlation
                    correlation = np.corrcoef(returns_a, returns_b)[0, 1]
                    
                    # Hedge ratio (beta)
                    if np.std(returns_a) > 0:
                        hedge_ratio = np.cov(returns_a, returns_b)[0, 1] / np.var(returns_a)
                    else:
                        hedge_ratio = 1.0
                    
                    # Spread and z-score
                    spread = closes_a / closes_b
                    spread_mean = np.mean(spread[-30:])
                    spread_std = np.std(spread[-30:])
                    zscore = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
                    
                    # Generate signal
                    signal = None
                    mean_reversion = False
                    if correlation > 0.6:  # Only trade if correlated
                        if zscore > 2.0:
                            signal = 'short_a_long_b'  # A overvalued vs B
                            mean_reversion = True
                        elif zscore < -2.0:
                            signal = 'long_a_short_b'  # A undervalued vs B
                            mean_reversion = True
                    
                    results[pair_key] = CorrelationStatus(
                        pair=pair_key,
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        correlation=round(correlation, 3),
                        spread_zscore=round(zscore, 2),
                        hedge_ratio=round(hedge_ratio, 3),
                        signal=signal,
                        mean_reversion_expected=mean_reversion
                    )
                else:
                    results[pair_key] = CorrelationStatus(
                        pair=pair_key,
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        correlation=0.0,
                        spread_zscore=0.0,
                        hedge_ratio=1.0
                    )
            except Exception as e:
                logger.error(f"Error analyzing correlation for {pair_key}: {e}")
                results[pair_key] = CorrelationStatus(
                    pair=pair_key,
                    symbol_a=symbol_a,
                    symbol_b=symbol_b,
                    correlation=0.0,
                    spread_zscore=0.0,
                    hedge_ratio=1.0
                )
        
        return results
    
    # ==========================================
    # Strategy Status
    # ==========================================
    
    def get_all_strategy_status(self) -> List[StrategyStatus]:
        """Get status of all 5 new strategies"""
        strategies = []
        
        # 1. Funding Fade
        funding_data = self.get_funding_rates()
        extreme_funding = [s for s, d in funding_data.items() if d.signal_strength > 0.5]
        strategies.append(StrategyStatus(
            strategy_id='funding-fade',
            name='Funding Rate Fade',
            enabled=True,
            active_signals=len(extreme_funding),
            symbols_watching=len(self.symbols),
            current_conditions={
                'extreme_positive': len([s for s, d in funding_data.items() if d.sentiment == 'bearish']),
                'extreme_negative': len([s for s, d in funding_data.items() if d.sentiment == 'bullish']),
                'neutral': len([s for s, d in funding_data.items() if d.sentiment == 'neutral'])
            },
            health='ok'
        ))
        
        # 2. OI Divergence
        oi_data = self.get_open_interest()
        divergences = [s for s, d in oi_data.items() if d.divergence_signal]
        strategies.append(StrategyStatus(
            strategy_id='oi-divergence',
            name='OI Divergence',
            enabled=True,
            active_signals=len(divergences),
            symbols_watching=len(self.symbols),
            current_conditions={
                'weak_rallies': len([s for s, d in oi_data.items() if d.divergence_signal == 'weak_rally']),
                'capitulations': len([s for s, d in oi_data.items() if d.divergence_signal == 'capitulation']),
                'strong_trends': len([s for s, d in oi_data.items() if d.divergence_signal in ('strong_uptrend', 'strong_downtrend')])
            },
            health='ok'
        ))
        
        # 3. Liquidation Hunter
        strategies.append(StrategyStatus(
            strategy_id='liquidation-hunter',
            name='Liquidation Hunter',
            enabled=True,
            active_signals=0,  # Would need position data to determine
            symbols_watching=len(self.symbols),
            current_conditions={
                'cascade_mode': False,
                'fade_opportunities': 0
            },
            health='ok'
        ))
        
        # 4. Volatility Breakout
        squeeze_data = self.get_squeeze_patterns()
        squeezes = [s for s, d in squeeze_data.items() if d.in_squeeze]
        breakouts = [s for s, d in squeeze_data.items() if d.breakout_direction]
        strategies.append(StrategyStatus(
            strategy_id='volatility-breakout',
            name='Volatility Breakout',
            enabled=True,
            active_signals=len(breakouts),
            symbols_watching=len(self.symbols),
            current_conditions={
                'symbols_in_squeeze': len(squeezes),
                'active_breakouts': len(breakouts),
                'long_breakouts': len([s for s, d in squeeze_data.items() if d.breakout_direction == 'long']),
                'short_breakouts': len([s for s, d in squeeze_data.items() if d.breakout_direction == 'short'])
            },
            health='ok'
        ))
        
        # 5. Correlation Pairs
        corr_data = self.get_correlation_status()
        active_pairs = [p for p, d in corr_data.items() if d.signal]
        strategies.append(StrategyStatus(
            strategy_id='correlation-pairs',
            name='Correlation Pairs',
            enabled=True,
            active_signals=len(active_pairs),
            symbols_watching=len(self.CORRELATION_PAIRS) * 2,
            current_conditions={
                'pairs_tracked': len(self.CORRELATION_PAIRS),
                'diverged_pairs': len(active_pairs),
                'btc_eth_correlation': corr_data.get('XBTUSDTM/ETHUSDTM', CorrelationStatus(
                    pair='', symbol_a='', symbol_b='', correlation=0, spread_zscore=0, hedge_ratio=1
                )).correlation
            },
            health='ok'
        ))
        
        return strategies
    
    # ==========================================
    # Full Report
    # ==========================================
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get complete monitoring report - JSON-friendly output"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'strategies': [asdict(s) for s in self.get_all_strategy_status()],
            'funding_rates': {k: asdict(v) for k, v in self.get_funding_rates().items()},
            'open_interest': {k: asdict(v) for k, v in self.get_open_interest().items()},
            'squeeze_patterns': {k: asdict(v) for k, v in self.get_squeeze_patterns().items()},
            'correlations': {k: asdict(v) for k, v in self.get_correlation_status().items()},
            'summary': {
                'total_symbols_monitored': len(self.symbols),
                'total_active_signals': sum(s.active_signals for s in self.get_all_strategy_status()),
                'health_status': 'ok'
            }
        }


# ============== CLI Interface ==============

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Cash Town new strategies')
    parser.add_argument('--format', choices=['json', 'pretty'], default='json',
                       help='Output format')
    parser.add_argument('--section', choices=['all', 'funding', 'oi', 'squeeze', 'correlation', 'strategies'],
                       default='all', help='Section to display')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to monitor')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    monitor = NewStrategiesMonitor(symbols=args.symbols)
    
    if args.section == 'all':
        data = monitor.get_full_report()
    elif args.section == 'funding':
        data = {k: asdict(v) for k, v in monitor.get_funding_rates().items()}
    elif args.section == 'oi':
        data = {k: asdict(v) for k, v in monitor.get_open_interest().items()}
    elif args.section == 'squeeze':
        data = {k: asdict(v) for k, v in monitor.get_squeeze_patterns().items()}
    elif args.section == 'correlation':
        data = {k: asdict(v) for k, v in monitor.get_correlation_status().items()}
    elif args.section == 'strategies':
        data = [asdict(s) for s in monitor.get_all_strategy_status()]
    
    if args.format == 'json':
        print(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, indent=2, default=str))


if __name__ == '__main__':
    main()
