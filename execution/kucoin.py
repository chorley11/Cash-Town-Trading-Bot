"""
KuCoin Futures Executor - Places and manages orders on KuCoin Futures

Handles:
- Order placement (market, limit)
- Position management
- Stop loss / Take profit orders
- Order status tracking
"""
import json
import time
import hmac
import hashlib
import base64
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

@dataclass
class Order:
    """Order record"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market' or 'limit'
    size: float
    price: Optional[float]
    status: str
    created_at: datetime
    filled_size: float = 0.0
    filled_price: float = 0.0
    
@dataclass
class Position:
    """Position record"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    margin: float
    liquidation_price: float

class KuCoinFuturesExecutor:
    """
    Executes trades on KuCoin Futures.
    """
    
    BASE_URL = "https://api-futures.kucoin.com"
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize with API credentials.
        
        Args:
            credentials_path: Path to JSON file with api_key, api_secret, api_passphrase
        """
        self.credentials_path = credentials_path or os.path.expanduser(
            "~/.config/kucoin/cash_town_credentials.json"
        )
        self.api_key = None
        self.api_secret = None
        self.api_passphrase = None
        self._load_credentials()
    
    def _load_credentials(self):
        """Load API credentials from file or environment variables"""
        # First try environment variables (for cloud deployment)
        self.api_key = os.environ.get('KUCOIN_API_KEY')
        self.api_secret = os.environ.get('KUCOIN_API_SECRET')
        self.api_passphrase = os.environ.get('KUCOIN_API_PASSPHRASE')
        
        if self.is_configured:
            logger.info("Loaded credentials from environment variables")
            return
        
        # Fall back to credentials file
        try:
            if os.path.exists(self.credentials_path):
                with open(self.credentials_path, 'r') as f:
                    creds = json.load(f)
                    self.api_key = creds.get('api_key')
                    self.api_secret = creds.get('api_secret')
                    self.api_passphrase = creds.get('api_passphrase')
                    logger.info(f"Loaded credentials from {self.credentials_path}")
            else:
                logger.warning(f"Credentials file not found: {self.credentials_path}")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
    
    @property
    def is_configured(self) -> bool:
        """Check if API credentials are configured"""
        return all([self.api_key, self.api_secret, self.api_passphrase])
    
    def _sign(self, timestamp: str, method: str, endpoint: str, body: str = '') -> Dict[str, str]:
        """Generate authentication headers"""
        str_to_sign = timestamp + method + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                str_to_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        passphrase = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                self.api_passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            'KC-API-KEY': self.api_key,
            'KC-API-SIGN': signature,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-PASSPHRASE': passphrase,
            'KC-API-KEY-VERSION': '2',
            'Content-Type': 'application/json'
        }
    
    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated API request"""
        if not self.is_configured:
            raise ValueError("API credentials not configured")
        
        timestamp = str(int(time.time() * 1000))
        body = json.dumps(data) if data else ''
        
        headers = self._sign(timestamp, method.upper(), endpoint, body)
        url = self.BASE_URL + endpoint
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=body, timeout=10)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            result = response.json()
            
            if result.get('code') != '200000':
                logger.error(f"API error: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {'code': 'error', 'msg': str(e)}
    
    def get_account_overview(self) -> Dict:
        """Get account balance and overview"""
        result = self._request('GET', '/api/v1/account-overview?currency=USDT')
        return result.get('data', {})
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        result = self._request('GET', '/api/v1/positions')
        positions = []
        
        for pos in result.get('data', []):
            if pos.get('currentQty', 0) != 0:
                positions.append(Position(
                    symbol=pos.get('symbol'),
                    side='long' if pos.get('currentQty', 0) > 0 else 'short',
                    size=abs(pos.get('currentQty', 0)),
                    entry_price=float(pos.get('avgEntryPrice', 0)),
                    leverage=int(pos.get('realLeverage', 1)),
                    unrealized_pnl=float(pos.get('unrealisedPnl', 0)),
                    margin=float(pos.get('posMargin', 0)),
                    liquidation_price=float(pos.get('liquidationPrice', 0))
                ))
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def place_market_order(self, symbol: str, side: str, size: float, 
                          leverage: int = 5, reduce_only: bool = False) -> Optional[str]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair (e.g., 'XBTUSDTM')
            side: 'buy' or 'sell'
            size: Position size in contracts
            leverage: Leverage to use
            reduce_only: If True, only reduces existing position
        
        Returns:
            Order ID if successful, None otherwise
        """
        # Set leverage first
        self._request('POST', '/api/v1/position/risk-limit-level/change', {
            'symbol': symbol,
            'level': leverage
        })
        
        data = {
            'clientOid': f"ct_{int(time.time()*1000)}",
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'size': int(size),
            'leverage': str(leverage),
            'reduceOnly': reduce_only
        }
        
        result = self._request('POST', '/api/v1/orders', data)
        
        if result.get('code') == '200000':
            order_id = result.get('data', {}).get('orderId')
            logger.info(f"Market order placed: {side} {size} {symbol} -> {order_id}")
            return order_id
        else:
            logger.error(f"Order failed: {result}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, size: float, price: float,
                         leverage: int = 5, reduce_only: bool = False,
                         post_only: bool = False) -> Optional[str]:
        """Place a limit order"""
        data = {
            'clientOid': f"ct_{int(time.time()*1000)}",
            'symbol': symbol,
            'side': side,
            'type': 'limit',
            'price': str(price),
            'size': int(size),
            'leverage': str(leverage),
            'reduceOnly': reduce_only,
            'postOnly': post_only
        }
        
        result = self._request('POST', '/api/v1/orders', data)
        
        if result.get('code') == '200000':
            order_id = result.get('data', {}).get('orderId')
            logger.info(f"Limit order placed: {side} {size} {symbol} @ {price} -> {order_id}")
            return order_id
        else:
            logger.error(f"Order failed: {result}")
            return None
    
    def place_stop_order(self, symbol: str, side: str, size: float, 
                        stop_price: float, leverage: int = 5) -> Optional[str]:
        """Place a stop market order"""
        data = {
            'clientOid': f"ct_stop_{int(time.time()*1000)}",
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'size': int(size),
            'leverage': str(leverage),
            'stop': 'down' if side == 'sell' else 'up',
            'stopPriceType': 'TP',  # Trade price
            'stopPrice': str(stop_price),
            'reduceOnly': True
        }
        
        result = self._request('POST', '/api/v1/orders', data)
        
        if result.get('code') == '200000':
            order_id = result.get('data', {}).get('orderId')
            logger.info(f"Stop order placed: {side} {size} {symbol} @ {stop_price} -> {order_id}")
            return order_id
        else:
            logger.error(f"Stop order failed: {result}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        result = self._request('DELETE', f'/api/v1/orders/{order_id}')
        return result.get('code') == '200000'
    
    def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders"""
        endpoint = '/api/v1/orders'
        if symbol:
            endpoint += f'?symbol={symbol}'
        result = self._request('DELETE', endpoint)
        return result.get('code') == '200000'
    
    def close_position(self, symbol: str) -> bool:
        """Close entire position for a symbol"""
        position = self.get_position(symbol)
        if not position:
            logger.info(f"No position to close for {symbol}")
            return True
        
        # Close by placing opposite market order
        close_side = 'sell' if position.side == 'long' else 'buy'
        order_id = self.place_market_order(
            symbol=symbol,
            side=close_side,
            size=position.size,
            reduce_only=True
        )
        
        return order_id is not None
    
    def get_contract_info(self, symbol: str) -> Optional[Dict]:
        """Get contract specifications"""
        result = self._request('GET', f'/api/v1/contracts/{symbol}')
        if result.get('code') == '200000':
            return result.get('data')
        return None
