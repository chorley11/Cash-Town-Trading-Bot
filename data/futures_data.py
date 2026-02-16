"""
Enhanced KuCoin Futures Data - Funding Rates, Open Interest, Order Book

Provides additional futures-specific data to strategy agents:
- Funding rates (current + historical)
- Open interest (current + changes)
- Order book depth analysis
- Liquidation level estimation
"""
import logging
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class FundingData:
    """Funding rate data for a symbol"""
    symbol: str
    current_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    predicted_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None
    daily_interest_rate: float = 0.0003
    rate_cap: float = 0.003
    rate_floor: float = -0.003
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OpenInterestData:
    """Open interest data for a symbol"""
    symbol: str
    open_interest: float  # In contracts
    open_interest_value: float  # In USD (estimated)
    mark_price: float
    index_price: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Historical for divergence detection
    prev_open_interest: Optional[float] = None
    oi_change_pct: float = 0.0

@dataclass
class OrderBookData:
    """Order book depth analysis"""
    symbol: str
    bid_depth: float  # Total bid volume in top N levels
    ask_depth: float  # Total ask volume in top N levels
    imbalance: float  # (bid - ask) / (bid + ask), positive = more bids
    spread: float  # Best ask - best bid
    spread_pct: float  # Spread as percentage
    best_bid: float
    best_ask: float
    mid_price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Depth at price levels (for liquidation estimation)
    bid_levels: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    ask_levels: List[tuple] = field(default_factory=list)

class KuCoinFuturesData:
    """
    Fetches and tracks futures-specific data from KuCoin.
    """
    
    BASE_URL = "https://api-futures.kucoin.com"
    
    def __init__(self, symbols: List[str] = None):
        default_symbols = [
            'XBTUSDTM', 'ETHUSDTM', 'SOLUSDTM', 'AVAXUSDTM',
            'LINKUSDTM', 'XRPUSDTM', 'DOGEUSDTM', 'BNBUSDTM',
            'ADAUSDTM', 'MATICUSDTM', 'DOTUSDTM', 'LTCUSDTM'
        ]
        self.symbols = symbols or default_symbols
        
        # Data stores
        self.funding: Dict[str, FundingData] = {}
        self.open_interest: Dict[str, OpenInterestData] = {}
        self.order_book: Dict[str, OrderBookData] = {}
        
        # Historical OI for tracking changes
        self._oi_history: Dict[str, List[tuple]] = {}  # symbol -> [(timestamp, oi), ...]
    
    def fetch_funding_rate(self, symbol: str) -> Optional[FundingData]:
        """Fetch current funding rate for a symbol"""
        try:
            url = f"{self.BASE_URL}/api/v1/funding-rate/{symbol}/current"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('code') != '200000':
                logger.warning(f"Funding rate API error for {symbol}: {data}")
                return None
            
            rate_data = data.get('data', {})
            
            funding = FundingData(
                symbol=symbol,
                current_rate=float(rate_data.get('value', 0)),
                daily_interest_rate=float(rate_data.get('dailyInterestRate', 0.0003)),
                rate_cap=float(rate_data.get('fundingRateCap', 0.003)),
                rate_floor=float(rate_data.get('fundingRateFloor', -0.003)),
                next_funding_time=datetime.fromtimestamp(
                    rate_data.get('fundingTime', 0) / 1000
                ) if rate_data.get('fundingTime') else None,
                timestamp=datetime.utcnow()
            )
            
            self.funding[symbol] = funding
            return funding
            
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def fetch_open_interest(self, symbol: str) -> Optional[OpenInterestData]:
        """Fetch open interest and contract data"""
        try:
            url = f"{self.BASE_URL}/api/v1/contracts/{symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('code') != '200000':
                logger.warning(f"Contract API error for {symbol}: {data}")
                return None
            
            contract = data.get('data', {})
            
            oi = float(contract.get('openInterest', 0))
            mark_price = float(contract.get('markPrice', 0))
            multiplier = float(contract.get('multiplier', 0.001))  # Contract size
            
            # Calculate change from previous
            prev_oi = None
            oi_change = 0.0
            if symbol in self.open_interest:
                prev_oi = self.open_interest[symbol].open_interest
                if prev_oi and prev_oi > 0:
                    oi_change = ((oi - prev_oi) / prev_oi) * 100
            
            oi_data = OpenInterestData(
                symbol=symbol,
                open_interest=oi,
                open_interest_value=oi * multiplier * mark_price,
                mark_price=mark_price,
                index_price=float(contract.get('indexPrice', 0)),
                volume_24h=float(contract.get('volumeOf24h', 0)),
                prev_open_interest=prev_oi,
                oi_change_pct=oi_change,
                timestamp=datetime.utcnow()
            )
            
            # Track history
            if symbol not in self._oi_history:
                self._oi_history[symbol] = []
            self._oi_history[symbol].append((datetime.utcnow(), oi))
            # Keep last 100 data points
            self._oi_history[symbol] = self._oi_history[symbol][-100:]
            
            self.open_interest[symbol] = oi_data
            return oi_data
            
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None
    
    def fetch_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBookData]:
        """Fetch order book and analyze depth"""
        try:
            url = f"{self.BASE_URL}/api/v1/level2/depth{depth}"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('code') != '200000':
                logger.warning(f"Order book API error for {symbol}: {data}")
                return None
            
            book = data.get('data', {})
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Parse levels: [[price, size], ...]
            bid_levels = [(float(b[0]), float(b[1])) for b in bids]
            ask_levels = [(float(a[0]), float(a[1])) for a in asks]
            
            best_bid = bid_levels[0][0]
            best_ask = ask_levels[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100
            
            # Calculate depth
            bid_depth = sum(size for _, size in bid_levels)
            ask_depth = sum(size for _, size in ask_levels)
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            ob_data = OrderBookData(
                symbol=symbol,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                imbalance=imbalance,
                spread=spread,
                spread_pct=spread_pct,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                timestamp=datetime.utcnow()
            )
            
            self.order_book[symbol] = ob_data
            return ob_data
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def fetch_all(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all data for all symbols"""
        result = {}
        
        for symbol in self.symbols:
            funding = self.fetch_funding_rate(symbol)
            oi = self.fetch_open_interest(symbol)
            ob = self.fetch_order_book(symbol)
            
            result[symbol] = {
                'funding': funding,
                'open_interest': oi,
                'order_book': ob
            }
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        logger.info(f"Fetched futures data for {len(self.symbols)} symbols")
        return result
    
    def get_funding_extremes(self, threshold: float = 0.0005) -> Dict[str, FundingData]:
        """Get symbols with extreme funding rates"""
        extremes = {}
        for symbol, funding in self.funding.items():
            if abs(funding.current_rate) >= threshold:
                extremes[symbol] = funding
        return extremes
    
    def get_oi_divergences(self, price_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Detect OI divergences from price.
        
        Returns symbols where:
        - Price up + OI down = weak rally (bearish)
        - Price down + OI down = capitulation (potentially bullish)
        - Price up + OI up = strong trend
        - Price down + OI up = accumulation or short buildup
        """
        divergences = {}
        
        for symbol, oi_data in self.open_interest.items():
            if symbol not in price_data:
                continue
            
            prices = price_data[symbol].get('close', [])
            if len(prices) < 2:
                continue
            
            price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
            oi_change = oi_data.oi_change_pct
            
            signal = None
            if price_change > 0.5 and oi_change < -1:
                signal = 'weak_rally'  # Bearish divergence
            elif price_change < -0.5 and oi_change < -1:
                signal = 'capitulation'  # Potential reversal
            elif price_change > 0.5 and oi_change > 1:
                signal = 'strong_uptrend'
            elif price_change < -0.5 and oi_change > 1:
                signal = 'strong_downtrend'
            
            if signal:
                divergences[symbol] = {
                    'signal': signal,
                    'price_change': price_change,
                    'oi_change': oi_change,
                    'oi_data': oi_data
                }
        
        return divergences
    
    def estimate_liquidation_levels(self, symbol: str, leverage: int = 10) -> Dict[str, float]:
        """
        Estimate where major liquidation levels might be.
        
        This is a rough estimate based on:
        - Current price
        - Common leverage levels
        - Order book depth for potential cascade zones
        """
        oi_data = self.open_interest.get(symbol)
        ob_data = self.order_book.get(symbol)
        
        if not oi_data or not ob_data:
            return {}
        
        price = oi_data.mark_price
        
        # Estimate liquidation levels for various leverage
        # Liquidation happens when: loss > initial_margin
        # For longs: liq_price = entry * (1 - 1/leverage + maintenance)
        # For shorts: liq_price = entry * (1 + 1/leverage - maintenance)
        maintenance_margin = 0.005  # 0.5% typical
        
        levels = {}
        for lev in [5, 10, 20, 50, 100]:
            # Long liquidations (price drops)
            long_liq = price * (1 - (1/lev) + maintenance_margin)
            # Short liquidations (price rises)
            short_liq = price * (1 + (1/lev) - maintenance_margin)
            
            levels[f'long_liq_{lev}x'] = long_liq
            levels[f'short_liq_{lev}x'] = short_liq
        
        return levels
