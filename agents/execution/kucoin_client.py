"""
KuCoin Futures Client - Execution layer for Gas Town agents
"""
import hmac
import hashlib
import base64
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Contract multipliers for KuCoin Futures
CONTRACT_MULTIPLIERS = {
    'XBTUSDTM': 0.001,    # BTC
    'ETHUSDTM': 0.01,     # ETH
    'SOLUSDTM': 0.1,      # SOL
    'XRPUSDTM': 10,       # XRP
    'DOGEUSDTM': 100,     # DOGE
    'ADAUSDTM': 10,       # ADA
    'LINKUSDTM': 1,       # LINK
    'DOTUSDTM': 1,        # DOT
    'AVAXUSDTM': 0.1,     # AVAX
    'ATOMUSDTM': 0.1,     # ATOM
    # Default for most altcoins
    'DEFAULT': 1
}

def get_contract_multiplier(symbol: str) -> float:
    """Get the contract multiplier for a symbol"""
    return CONTRACT_MULTIPLIERS.get(symbol, CONTRACT_MULTIPLIERS['DEFAULT'])

@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_size: Optional[float] = None
    error: Optional[str] = None
    retries: int = 0

@dataclass
class KuCoinPosition:
    """Position from KuCoin"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: int
    margin_mode: str
    is_open: bool

class KuCoinFuturesClient:
    """
    KuCoin Futures API client for Gas Town.
    Handles authentication, order execution, and position management.
    """
    
    BASE_URL = "https://api-futures.kucoin.com"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        leverage: int = 5
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.leverage = leverage
        self.session = requests.Session()
        
        # Sign passphrase (KuCoin v2 API)
        self.signed_passphrase = base64.b64encode(
            hmac.new(
                api_secret.encode('utf-8'),
                passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
    
    def _sign(self, timestamp: str, method: str, endpoint: str, body: str = '') -> str:
        """Generate API signature"""
        str_to_sign = f"{timestamp}{method}{endpoint}{body}"
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                str_to_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        auth: bool = True
    ) -> Dict:
        """Make API request"""
        url = f"{self.BASE_URL}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        body = ''
        if data:
            body = json.dumps(data)
        
        if auth:
            signature = self._sign(timestamp, method, endpoint, body)
            headers.update({
                'KC-API-KEY': self.api_key,
                'KC-API-SIGN': signature,
                'KC-API-TIMESTAMP': timestamp,
                'KC-API-PASSPHRASE': self.signed_passphrase,
                'KC-API-KEY-VERSION': '2'
            })
        
        try:
            if method == 'GET':
                resp = self.session.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                resp = self.session.post(url, headers=headers, data=body, timeout=10)
            elif method == 'DELETE':
                resp = self.session.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            resp.raise_for_status()
            return resp.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances"""
        resp = self._request('GET', '/api/v1/account-overview?currency=USDT')
        data = resp.get('data', {})
        return {
            'available': float(data.get('availableBalance', 0)),
            'equity': float(data.get('accountEquity', 0)),
            'margin_used': float(data.get('positionMargin', 0)),
            'unrealized_pnl': float(data.get('unrealisedPNL', 0))
        }
    
    def get_positions(self) -> List[KuCoinPosition]:
        """Get all open positions"""
        resp = self._request('GET', '/api/v1/positions')
        positions = []
        
        for p in resp.get('data', []):
            if not p.get('isOpen'):
                continue
                
            qty = p.get('currentQty', 0)
            side = 'long' if qty > 0 else 'short'
            
            positions.append(KuCoinPosition(
                id=str(p.get('id')),
                symbol=p.get('symbol'),
                side=side,
                size=abs(qty),
                entry_price=float(p.get('avgEntryPrice', 0)),
                current_price=float(p.get('markPrice', 0)),
                unrealized_pnl=float(p.get('unrealisedPnl', 0)),
                leverage=int(p.get('leverage', self.leverage)),
                margin_mode=p.get('marginMode', 'CROSS'),
                is_open=True
            ))
        
        return positions
    
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker for a symbol"""
        # Convert symbol format: BTC/USDT -> XBTUSDTM
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        resp = self._request('GET', f'/api/v1/ticker?symbol={kucoin_symbol}', auth=False)
        data = resp.get('data', {})
        return {
            'price': float(data.get('price', 0)),
            'bid': float(data.get('bestBidPrice', 0)),
            'ask': float(data.get('bestAskPrice', 0)),
            'volume': float(data.get('volume', 0))
        }
    
    def _to_kucoin_symbol(self, symbol: str) -> str:
        """Convert internal symbol to KuCoin format"""
        # BTC/USDT -> XBTUSDTM, ETH/USDT -> ETHUSDTM
        base = symbol.split('/')[0]
        if base == 'BTC':
            base = 'XBT'
        return f"{base}USDTM"
    
    def _from_kucoin_symbol(self, symbol: str) -> str:
        """Convert KuCoin symbol to internal format"""
        # XBTUSDTM -> BTC/USDT
        base = symbol.replace('USDTM', '')
        if base == 'XBT':
            base = 'BTC'
        return f"{base}/USDT"
    
    def open_position(
        self,
        symbol: str,
        side: str,  # 'long' or 'short'
        size_usd: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OrderResult:
        """
        Open a position with market order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'long' or 'short'
            size_usd: Position size in USD
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        """
        try:
            kucoin_symbol = self._to_kucoin_symbol(symbol)
            ticker = self.get_ticker(symbol)
            price = ticker['price']
            
            # Calculate contract size
            multiplier = get_contract_multiplier(kucoin_symbol)
            contracts = int(size_usd / (price * multiplier))
            
            if contracts < 1:
                return OrderResult(
                    success=False,
                    error=f"Position size too small: {size_usd} USD = {contracts} contracts"
                )
            
            # Determine order side
            order_side = 'buy' if side == 'long' else 'sell'
            
            # Create order
            order_data = {
                'clientOid': f"gt-{int(time.time()*1000)}-{symbol.replace('/', '')}",
                'side': order_side,
                'symbol': kucoin_symbol,
                'type': 'market',
                'size': contracts,
                'leverage': self.leverage
            }
            
            resp = self._request('POST', '/api/v1/orders', order_data)
            
            if resp.get('code') == '200000':
                order_id = resp.get('data', {}).get('orderId')
                
                # Get fill info
                time.sleep(0.5)  # Wait for fill
                fill_resp = self._request('GET', f'/api/v1/orders/{order_id}')
                fill_data = fill_resp.get('data', {})
                
                executed_price = float(fill_data.get('dealValue', 0)) / max(float(fill_data.get('dealSize', 1)), 1)
                executed_size = float(fill_data.get('dealSize', 0)) * multiplier
                
                logger.info(f"Position opened: {symbol} {side} {contracts} contracts @ ~{price}")
                
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    executed_price=executed_price or price,
                    executed_size=executed_size or (contracts * multiplier)
                )
            else:
                return OrderResult(
                    success=False,
                    error=f"Order failed: {resp.get('msg', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return OrderResult(success=False, error=str(e))
    
    def close_position(self, symbol: str) -> OrderResult:
        """Close position for a symbol"""
        try:
            kucoin_symbol = self._to_kucoin_symbol(symbol)
            
            # Get current position
            positions = self.get_positions()
            position = next((p for p in positions if p.symbol == kucoin_symbol), None)
            
            if not position:
                return OrderResult(success=False, error='No position to close')
            
            # Close with opposite order
            close_side = 'sell' if position.side == 'long' else 'buy'
            
            order_data = {
                'clientOid': f"gt-close-{int(time.time()*1000)}",
                'side': close_side,
                'symbol': kucoin_symbol,
                'type': 'market',
                'size': int(position.size),
                'reduceOnly': True,
                'closeOrder': True
            }
            
            resp = self._request('POST', '/api/v1/orders', order_data)
            
            if resp.get('code') == '200000':
                logger.info(f"Position closed: {symbol}")
                return OrderResult(
                    success=True,
                    order_id=resp.get('data', {}).get('orderId')
                )
            else:
                return OrderResult(
                    success=False,
                    error=f"Close failed: {resp.get('msg', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return OrderResult(success=False, error=str(e))
    
    def get_klines(
        self,
        symbol: str,
        interval: str = '1hour',  # 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
        limit: int = 200
    ) -> List[Dict]:
        """Get OHLCV candlestick data"""
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        # Map interval
        interval_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440,
            '1min': 1, '5min': 5, '15min': 15, '30min': 30,
            '1hour': 60, '4hour': 240, '1day': 1440
        }
        granularity = interval_map.get(interval, 60)
        
        resp = self._request(
            'GET',
            f'/api/v1/kline/query?symbol={kucoin_symbol}&granularity={granularity}',
            auth=False
        )
        
        klines = []
        for k in resp.get('data', [])[:limit]:
            klines.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        
        return klines
