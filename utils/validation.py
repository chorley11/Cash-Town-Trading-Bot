"""
Input Validation - Sanitize and validate all external inputs

SECURITY: This module prevents injection attacks and malformed data
from corrupting the trading system.
"""
import re
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid symbols must match KuCoin futures format: XXXUSDTM
VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,10}USDTM$')

# Valid sides
VALID_SIDES = {'long', 'short', 'neutral'}

# Max string lengths to prevent DoS
MAX_REASON_LENGTH = 500
MAX_STRATEGY_ID_LENGTH = 50
MAX_METADATA_SIZE = 10000  # bytes

# Dangerous patterns that could indicate injection attempts
DANGEROUS_PATTERNS = [
    r'<script',
    r'javascript:',
    r'on\w+\s*=',
    r'\x00',  # Null bytes
    r'{{.*}}',  # Template injection
    r'\$\{.*\}',  # Template literals
]


def sanitize_string(value: str, max_length: int = 500, field_name: str = "field") -> str:
    """
    Sanitize a string input.
    
    - Strips whitespace
    - Truncates to max length
    - Removes dangerous patterns
    - Returns empty string for None/invalid
    """
    if value is None:
        return ""
    
    if not isinstance(value, str):
        logger.warning(f"Invalid type for {field_name}: {type(value)}")
        return str(value)[:max_length]
    
    # Strip and truncate
    result = value.strip()[:max_length]
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, result, re.IGNORECASE):
            logger.warning(f"Dangerous pattern detected in {field_name}: {pattern}")
            result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)
    
    return result


def validate_signal_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate incoming signal data from strategy agents.
    
    Returns:
        Tuple of (is_valid, error_message, sanitized_data)
    """
    if not isinstance(data, dict):
        return False, "Signal data must be a dictionary", None
    
    errors = []
    sanitized = {}
    
    # Required fields
    required = ['symbol', 'side', 'confidence']
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, "; ".join(errors), None
    
    # Validate symbol
    symbol = data.get('symbol', '')
    if not isinstance(symbol, str) or not VALID_SYMBOL_PATTERN.match(symbol):
        return False, f"Invalid symbol format: {symbol}", None
    sanitized['symbol'] = symbol
    
    # Validate side
    side = data.get('side', '').lower() if isinstance(data.get('side'), str) else ''
    if side not in VALID_SIDES:
        return False, f"Invalid side: {side}. Must be one of {VALID_SIDES}", None
    sanitized['side'] = side
    
    # Validate confidence (0.0 to 1.0)
    confidence = data.get('confidence')
    try:
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            return False, f"Confidence must be between 0.0 and 1.0, got {confidence}", None
        sanitized['confidence'] = confidence
    except (TypeError, ValueError):
        return False, f"Invalid confidence value: {confidence}", None
    
    # Validate and sanitize optional fields
    
    # strategy_id
    if 'strategy_id' in data:
        sanitized['strategy_id'] = sanitize_string(
            data['strategy_id'], 
            MAX_STRATEGY_ID_LENGTH, 
            'strategy_id'
        )
    
    # price (optional but if present must be positive number)
    if 'price' in data and data['price'] is not None:
        try:
            price = float(data['price'])
            if price < 0:
                return False, f"Price cannot be negative: {price}", None
            sanitized['price'] = price
        except (TypeError, ValueError):
            return False, f"Invalid price: {data['price']}", None
    
    # stop_loss (optional)
    if 'stop_loss' in data and data['stop_loss'] is not None:
        try:
            sanitized['stop_loss'] = float(data['stop_loss'])
        except (TypeError, ValueError):
            pass  # Optional, ignore invalid
    
    # take_profit (optional)
    if 'take_profit' in data and data['take_profit'] is not None:
        try:
            sanitized['take_profit'] = float(data['take_profit'])
        except (TypeError, ValueError):
            pass  # Optional, ignore invalid
    
    # reason (sanitized string)
    if 'reason' in data:
        sanitized['reason'] = sanitize_string(
            data['reason'],
            MAX_REASON_LENGTH,
            'reason'
        )
    
    # timestamp
    if 'timestamp' in data:
        sanitized['timestamp'] = sanitize_string(
            str(data['timestamp']),
            50,
            'timestamp'
        )
    
    # metadata (size-limited)
    if 'metadata' in data:
        import json
        try:
            meta_str = json.dumps(data['metadata'])
            if len(meta_str) <= MAX_METADATA_SIZE:
                sanitized['metadata'] = data['metadata']
            else:
                logger.warning("Metadata too large, truncating")
                sanitized['metadata'] = {}
        except:
            sanitized['metadata'] = {}
    
    return True, None, sanitized


def validate_trade_result(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate trade result data for learning/recording.
    """
    if not isinstance(data, dict):
        return False, "Trade result must be a dictionary", None
    
    required = ['symbol', 'side', 'pnl', 'pnl_pct', 'strategy_id']
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}", None
    
    sanitized = {}
    
    # Symbol
    symbol = data.get('symbol', '')
    if not isinstance(symbol, str) or not VALID_SYMBOL_PATTERN.match(symbol):
        return False, f"Invalid symbol: {symbol}", None
    sanitized['symbol'] = symbol
    
    # Side
    side = data.get('side', '').lower() if isinstance(data.get('side'), str) else ''
    if side not in {'long', 'short'}:
        return False, f"Invalid side: {side}", None
    sanitized['side'] = side
    
    # PnL values
    try:
        sanitized['pnl'] = float(data['pnl'])
        sanitized['pnl_pct'] = float(data['pnl_pct'])
    except (TypeError, ValueError) as e:
        return False, f"Invalid PnL values: {e}", None
    
    # Strategy ID
    sanitized['strategy_id'] = sanitize_string(
        data['strategy_id'],
        MAX_STRATEGY_ID_LENGTH,
        'strategy_id'
    )
    
    # Optional reason
    if 'reason' in data:
        sanitized['reason'] = sanitize_string(data['reason'], MAX_REASON_LENGTH, 'reason')
    
    return True, None, sanitized


def redact_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact sensitive data before logging.
    Safe to log the result.
    """
    sensitive_keys = {
        'api_key', 'api_secret', 'passphrase', 'password', 
        'token', 'secret', 'credential', 'auth'
    }
    
    def _redact(obj, depth=0):
        if depth > 10:  # Prevent deep recursion
            return "[MAX_DEPTH]"
        
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if any(sensitive in k.lower() for sensitive in sensitive_keys):
                    result[k] = "[REDACTED]"
                else:
                    result[k] = _redact(v, depth + 1)
            return result
        elif isinstance(obj, list):
            return [_redact(item, depth + 1) for item in obj[:100]]  # Limit list size
        else:
            return obj
    
    return _redact(data)
