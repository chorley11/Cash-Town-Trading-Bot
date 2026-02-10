"""
Cash Town Data Feed - KuCoin Futures WebSocket & REST data

Provides real-time and historical OHLCV data to strategy agents.
"""
import asyncio
import json
import logging
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import aiohttp
import requests

logger = logging.getLogger(__name__)

@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str = '15m'

@dataclass 
class MarketData:
    """Market data for a symbol"""
    symbol: str
    candles: List[Candle] = field(default_factory=list)
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    last_update: datetime = None
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dict format for strategy agents"""
        return {
            'open': [c.open for c in self.candles],
            'high': [c.high for c in self.candles],
            'low': [c.low for c in self.candles],
            'close': [c.close for c in self.candles],
            'volume': [c.volume for c in self.candles],
            'timestamp': [c.timestamp for c in self.candles]
        }

class KuCoinDataFeed:
    """
    KuCoin Futures data feed via REST API.
    Fetches historical candles and current prices.
    """
    
    BASE_URL = "https://api-futures.kucoin.com"
    
    def __init__(self, symbols: List[str], interval: str = '15min'):
        """
        Args:
            symbols: List of symbols to track (e.g., ['XBTUSDTM', 'ETHUSDTM'])
            interval: Candle interval (1min, 5min, 15min, 30min, 1hour, 4hour, 1day)
        """
        self.symbols = symbols
        self.interval = interval
        self.data: Dict[str, MarketData] = {}
        self.running = False
        self._callbacks: List[Callable] = []
        
        # Initialize market data for each symbol
        for symbol in symbols:
            self.data[symbol] = MarketData(symbol=symbol)
    
    def add_callback(self, callback: Callable[[str, MarketData], None]):
        """Add callback for data updates"""
        self._callbacks.append(callback)
    
    def _notify(self, symbol: str):
        """Notify callbacks of data update"""
        for callback in self._callbacks:
            try:
                callback(symbol, self.data[symbol])
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def fetch_candles(self, symbol: str, limit: int = 200) -> List[Candle]:
        """Fetch historical candles via REST API"""
        try:
            # Convert interval format
            interval_map = {
                '1m': 1, '1min': 1,
                '5m': 5, '5min': 5,
                '15m': 15, '15min': 15,
                '30m': 30, '30min': 30,
                '1h': 60, '1hour': 60,
                '4h': 240, '4hour': 240,
                '1d': 1440, '1day': 1440
            }
            granularity = interval_map.get(self.interval, 15)
            
            # Calculate time range
            end_time = int(time.time() * 1000)
            start_time = end_time - (limit * granularity * 60 * 1000)
            
            url = f"{self.BASE_URL}/api/v1/kline/query"
            params = {
                'symbol': symbol,
                'granularity': granularity,
                'from': start_time,
                'to': end_time
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('code') != '200000':
                logger.error(f"KuCoin API error: {data}")
                return []
            
            candles = []
            for item in data.get('data', []):
                # KuCoin returns: [time, open, high, low, close, volume]
                candle = Candle(
                    timestamp=datetime.fromtimestamp(item[0] / 1000),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    symbol=symbol,
                    interval=self.interval
                )
                candles.append(candle)
            
            # Sort by timestamp (oldest first)
            candles.sort(key=lambda c: c.timestamp)
            
            logger.info(f"Fetched {len(candles)} candles for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        try:
            url = f"{self.BASE_URL}/api/v1/ticker"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('code') != '200000':
                return None
            
            ticker = data.get('data', {})
            return {
                'price': float(ticker.get('price', 0)),
                'bid': float(ticker.get('bestBidPrice', 0)),
                'ask': float(ticker.get('bestAskPrice', 0)),
                'volume': float(ticker.get('size', 0))
            }
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def refresh_all(self) -> Dict[str, MarketData]:
        """Refresh data for all symbols"""
        for symbol in self.symbols:
            # Fetch candles
            candles = self.fetch_candles(symbol)
            if candles:
                self.data[symbol].candles = candles
                self.data[symbol].last_price = candles[-1].close
            
            # Fetch current ticker
            ticker = self.fetch_ticker(symbol)
            if ticker:
                self.data[symbol].last_price = ticker['price']
                self.data[symbol].bid = ticker['bid']
                self.data[symbol].ask = ticker['ask']
            
            self.data[symbol].last_update = datetime.utcnow()
            self._notify(symbol)
        
        return self.data
    
    def get_market_data(self) -> Dict[str, Dict]:
        """Get all market data in format for strategy agents"""
        result = {}
        for symbol, md in self.data.items():
            if md.candles:
                result[symbol] = md.to_dict()
        return result
    
    def start_polling(self, interval_seconds: int = 60):
        """Start polling for data updates"""
        self.running = True
        
        def poll_loop():
            while self.running:
                try:
                    self.refresh_all()
                    logger.debug(f"Data refreshed for {len(self.symbols)} symbols")
                except Exception as e:
                    logger.error(f"Poll error: {e}")
                time.sleep(interval_seconds)
        
        import threading
        self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"Data feed polling started ({interval_seconds}s interval)")
    
    def stop(self):
        """Stop the data feed"""
        self.running = False
        logger.info("Data feed stopped")


class DataFeedManager:
    """
    Manages data feeds and provides unified interface for strategy agents.
    """
    
    def __init__(self, symbols: List[str] = None, interval: str = '15min'):
        default_symbols = [
            'XBTUSDTM',   # BTC
            'ETHUSDTM',   # ETH
            'SOLUSDTM',   # SOL
            'AVAXUSDTM',  # AVAX
            'LINKUSDTM',  # LINK
            'XRPUSDTM',   # XRP
        ]
        self.symbols = symbols or default_symbols
        self.feed = KuCoinDataFeed(self.symbols, interval)
        self._last_refresh = None
    
    def start(self, poll_interval: int = 60):
        """Start the data feed"""
        # Initial fetch
        logger.info(f"Initializing data feed for {len(self.symbols)} symbols...")
        self.feed.refresh_all()
        self._last_refresh = datetime.utcnow()
        
        # Start polling
        self.feed.start_polling(poll_interval)
    
    def stop(self):
        """Stop the data feed"""
        self.feed.stop()
    
    def get_data(self) -> Dict[str, Dict]:
        """Get current market data for all symbols"""
        return self.feed.get_market_data()
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get data for a specific symbol"""
        data = self.feed.get_market_data()
        return data.get(symbol)
    
    def refresh(self) -> Dict[str, Dict]:
        """Force refresh all data"""
        self.feed.refresh_all()
        self._last_refresh = datetime.utcnow()
        return self.get_data()
    
    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._last_refresh
