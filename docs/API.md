# Cash Town Bloomberg Dashboard API

Comprehensive REST API for family office-grade trading dashboard.

## Base URL

```
http://localhost:8888/api
```

## Authentication

Currently no authentication required (internal use only).

---

## Portfolio Endpoints

### GET /api/portfolio

Complete portfolio overview including equity, P&L, exposure, and risk metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-02-16T12:00:00Z",
    "equity": {
      "total": 10000.00,
      "available": 8500.00,
      "margin_used": 1500.00,
      "currency": "USDT"
    },
    "pnl": {
      "unrealized": 250.00,
      "unrealized_pct": 2.5,
      "realized_today": 150.00,
      "realized_all_time": 2500.00
    },
    "exposure": {
      "total_notional": 15000.00,
      "net_exposure": 5000.00,
      "long_exposure": 10000.00,
      "short_exposure": 5000.00,
      "exposure_pct": 15.0
    },
    "positions": {
      "count": 3,
      "long_count": 2,
      "short_count": 1
    },
    "risk": {
      "max_drawdown_pct": 10.0,
      "current_drawdown_pct": 2.5,
      "daily_loss_pct": 0.5,
      "circuit_breaker_active": false
    },
    "performance": {
      "win_rate": 52.3,
      "profit_factor": 1.45,
      "sharpe_estimate": 0.85,
      "trades_today": 5
    }
  },
  "metadata": {"source": "live", "cached": false}
}
```

### GET /api/summary

Quick summary for dashboard header display.

**Response:**
```json
{
  "equity": 10000.00,
  "pnl": 250.00,
  "pnl_pct": 2.5,
  "positions": 3,
  "exposure_pct": 15.0,
  "win_rate": 52.3,
  "circuit_breaker": false,
  "timestamp": "2024-02-16T12:00:00Z"
}
```

---

## Position Endpoints

### GET /api/positions

All open positions with live data.

**Query Parameters:**
- `include_closed` (bool): Include recently closed positions

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "XBTUSDTM",
      "side": "long",
      "size": 100,
      "entry_price": 45000.00,
      "current_price": 45500.00,
      "leverage": 5,
      "margin": 900.00,
      "unrealized_pnl": 50.00,
      "unrealized_pnl_pct": 1.11,
      "liquidation_price": 40000.00,
      "stop_loss": 44000.00,
      "take_profit": 47000.00,
      "strategy_id": "trend-following",
      "age_hours": 4.5,
      "entry_time": "2024-02-16T07:30:00Z",
      "status": "open"
    }
  ],
  "metadata": {
    "timestamp": "2024-02-16T12:00:00Z",
    "count": 1,
    "filters": {"include_closed": false}
  }
}
```

### GET /api/position/:symbol

Detailed view of a single position.

**Path Parameters:**
- `symbol`: Trading symbol (e.g., XBTUSDTM)

---

## Trade History Endpoints

### GET /api/trades

Historical trades with comprehensive filters and pagination.

**Query Parameters:**
- `strategy` (string): Filter by strategy ID
- `symbol` (string): Filter by symbol
- `side` (string): Filter by side (long/short)
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `min_pnl` (float): Minimum P&L filter
- `max_pnl` (float): Maximum P&L filter
- `won_only` (bool): Filter winners/losers only
- `limit` (int): Results per page (default: 100)
- `offset` (int): Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-02-16T10:00:00Z",
      "symbol": "ETHUSDTM",
      "side": "long",
      "strategy_id": "mean-reversion",
      "entry_price": 2500.00,
      "exit_price": 2550.00,
      "size": 50,
      "pnl": 25.00,
      "pnl_pct": 2.0,
      "won": true,
      "status": "closed"
    }
  ],
  "metadata": {
    "timestamp": "2024-02-16T12:00:00Z",
    "total_count": 150,
    "returned_count": 100,
    "offset": 0,
    "limit": 100,
    "has_more": true,
    "filters_applied": {...},
    "aggregates": {
      "total_pnl": 2500.00,
      "win_count": 78,
      "loss_count": 72,
      "win_rate": 52.0
    }
  }
}
```

### GET /api/trade/:id

Single trade details with full context.

**Path Parameters:**
- `id`: Trade ID (timestamp_symbol format)

---

## Strategy Endpoints

### GET /api/strategies

Performance breakdown by strategy.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "strategy_id": "trend-following",
      "trades": 45,
      "wins": 24,
      "win_rate": 53.3,
      "total_pnl": 208.50,
      "avg_pnl": 4.63,
      "max_win": 85.00,
      "max_loss": -42.00,
      "score": 1.15,
      "multiplier": 1.5,
      "status": "active",
      "active_positions": 2
    }
  ],
  "metadata": {
    "timestamp": "2024-02-16T12:00:00Z",
    "count": 8
  }
}
```

### GET /api/strategy/:id

Detailed strategy view with trade history.

**Path Parameters:**
- `id`: Strategy ID (e.g., trend-following)

---

## Signal Endpoints

### GET /api/signals

Recent signals (both accepted and rejected) for analysis.

**Query Parameters:**
- `accepted` (bool): Filter by acceptance status
- `strategy` (string): Filter by strategy ID
- `symbol` (string): Filter by symbol
- `limit` (int): Results per page (default: 100)
- `offset` (int): Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-02-16T11:55:00Z",
      "strategy_id": "turtle",
      "symbol": "SOLUSDTM",
      "side": "long",
      "confidence": 0.72,
      "price": 105.50,
      "stop_loss": 102.00,
      "take_profit": 112.00,
      "reason": "20-day breakout with volume confirmation",
      "was_selected": true,
      "selection_reason": "Rank 1, consensus 85%",
      "aggregated_rank": 1,
      "consensus_score": 0.85
    }
  ],
  "metadata": {
    "timestamp": "2024-02-16T12:00:00Z",
    "total_count": 500,
    "aggregates": {
      "accepted_count": 120,
      "rejected_count": 380,
      "acceptance_rate": 24.0
    }
  }
}
```

---

## Risk Endpoints

### GET /api/risk

Comprehensive risk metrics and limits.

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-02-16T12:00:00Z",
    "limits": {
      "max_positions": 5,
      "max_position_pct": 2.0,
      "max_exposure_pct": 20.0,
      "max_daily_loss_pct": 5.0,
      "max_drawdown_pct": 10.0
    },
    "current": {
      "positions": 3,
      "position_pct": 1.5,
      "exposure_pct": 12.0,
      "daily_loss_pct": 0.5,
      "drawdown_pct": 2.5
    },
    "utilization": {
      "position_util": 60.0,
      "exposure_util": 60.0,
      "loss_util": 10.0,
      "drawdown_util": 25.0
    },
    "circuit_breaker": {
      "active": false,
      "reason": null,
      "triggered_at": null
    },
    "concentration": [
      {"symbol": "XBTUSDTM", "exposure_pct": 45.0, "pnl": 50.00},
      {"symbol": "ETHUSDTM", "exposure_pct": 35.0, "pnl": 25.00}
    ],
    "var_estimate": 0,
    "stress_test": {}
  },
  "metadata": {"source": "live"}
}
```

---

## Chart Endpoints

### GET /api/chart/:symbol

OHLCV candlestick data for charting.

**Path Parameters:**
- `symbol`: Trading symbol (e.g., XBTUSDTM)

**Query Parameters:**
- `interval` (string): Candle interval (1m, 5m, 15m, 1h, 4h, 1d) - default: 15m
- `limit` (int): Number of candles (default: 200)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "XBTUSDTM",
    "interval": "15m",
    "candles": [
      {
        "timestamp": "2024-02-16T11:45:00Z",
        "open": 45000.00,
        "high": 45150.00,
        "low": 44980.00,
        "close": 45100.00,
        "volume": 1250000
      }
    ],
    "signals": [...],
    "position": {
      "side": "long",
      "entry_price": 45000.00,
      "size": 100
    }
  },
  "metadata": {
    "timestamp": "2024-02-16T12:00:00Z",
    "count": 200
  }
}
```

---

## Real-Time Endpoints

### GET /api/stream

Server-Sent Events (SSE) stream for real-time updates.

**Usage:**
```javascript
const eventSource = new EventSource('http://localhost:8888/api/stream');

eventSource.addEventListener('snapshot', (e) => {
  console.log('Initial state:', JSON.parse(e.data));
});

eventSource.addEventListener('update', (e) => {
  console.log('Update:', JSON.parse(e.data));
});
```

**Event Types:**
- `connected` - Connection established
- `heartbeat` - Keep-alive signal
- `portfolio_update` - Portfolio state changed
- `equity_change` - Equity changed significantly
- `position_opened` - New position opened
- `position_closed` - Position closed
- `position_pnl` - Position P&L updated
- `price_update` - Price changed significantly
- `signal_received` - New signal from strategy
- `signal_accepted` - Signal accepted for execution
- `signal_rejected` - Signal rejected
- `trade_executed` - Trade successfully executed
- `risk_alert` - Risk limit warning
- `circuit_breaker` - Circuit breaker triggered/reset

### GET /api/ws/events

Get recent events for catch-up after reconnection.

**Query Parameters:**
- `limit` (int): Number of events (default: 50)
- `types` (string): Comma-separated event types to filter

### GET /api/ws/snapshot

Get current state snapshot.

---

## Internal Endpoints

These endpoints are used internally by the trading system:

- `GET /health` - Health check
- `GET /signals` - Internal signal aggregation
- `GET /learning` - Learning summary
- `GET /multipliers` - Strategy multipliers
- `POST /signals` - Receive signals from agents
- `POST /trade_result` - Record trade results

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "data": null,
  "metadata": {
    "error": "Error description"
  }
}
```

HTTP Status Codes:
- `200` - Success
- `400` - Bad request
- `404` - Not found
- `413` - Request too large
- `500` - Internal error

---

## Rate Limits

No rate limits currently enforced (internal use only).

---

## Versioning

Current API version: **2.0**

Check version via `/health` endpoint:
```json
{
  "status": "healthy",
  "api_version": "2.0",
  "dashboard": true
}
```
