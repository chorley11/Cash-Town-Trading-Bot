# 🔌 API Reference

Complete API documentation for Cash Town's HTTP endpoints.

---

## Base URL

```
http://localhost:8888
```

Or in production:
```
https://your-railway-deployment.railway.app
```

---

## Health & Status

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### GET /perf

Performance monitoring metrics.

**Response:**
```json
{
  "total_cycles": 150,
  "cycle_time_ms": {
    "avg": 1500,
    "min": 200,
    "max": 8000,
    "p50": 1200,
    "p95": 3500,
    "p99": 6000
  },
  "memory_mb": {
    "initial": 150.5,
    "current": 185.2,
    "peak": 210.0,
    "growth": 34.7
  },
  "signals": {
    "total_generated": 500,
    "total_executed": 50,
    "avg_per_cycle": 3.3
  },
  "errors": {
    "total_error_cycles": 3,
    "error_rate": 0.02
  },
  "last_10_cycles": [
    {"cycle_id": 150, "duration_ms": 1200, "signals": 2, "errors": 0},
    {"cycle_id": 149, "duration_ms": 1500, "signals": 3, "errors": 0}
  ]
}
```

---

### GET /risk

Risk manager status.

**Response:**
```json
{
  "equity": 10000.0,
  "peak_equity": 10500.0,
  "portfolio_heat": {
    "total_risk_pct": 4.5,
    "max_risk_pct": 10.0,
    "position_count": 3,
    "correlated_exposure": {
      "alt_l1": 2.0,
      "eth_ecosystem": 1.5
    },
    "is_overheated": false
  },
  "circuit_breaker": {
    "is_triggered": false,
    "trigger_reason": "",
    "daily_loss_pct": 1.2,
    "drawdown_pct": 3.0,
    "cooldown_until": null
  },
  "daily_stats": {
    "date": "2025-02-16",
    "starting_equity": 10200.0,
    "trades": 5,
    "wins": 3,
    "losses": 2,
    "pnl": 50.0
  },
  "strategy_stats": {
    "trend-following": {
      "trades": 25,
      "wins": 14,
      "losses": 11,
      "avg_win_pct": 2.5,
      "avg_loss_pct": 1.8
    }
  },
  "positions": {
    "ETHUSDTM": {
      "side": "long",
      "risk_pct": 1.5,
      "correlation_group": "eth_ecosystem",
      "strategy": "trend-following"
    }
  },
  "config": {
    "max_position_risk_pct": 2.0,
    "max_total_risk_pct": 10.0,
    "use_kelly": true,
    "kelly_fraction": 0.25,
    "max_daily_loss_pct": 5.0,
    "max_drawdown_pct": 15.0
  }
}
```

---

### GET /can_trade

Check if circuit breaker allows trading.

**Response (trading allowed):**
```json
{
  "can_trade": true,
  "reason": "OK"
}
```

**Response (trading halted):**
```json
{
  "can_trade": false,
  "reason": "Circuit breaker: Daily loss limit: 5.2% >= 5.0% (cooldown 180m)"
}
```

---

## Signals & Learning

### GET /signals

Get aggregated actionable signals for execution.

**Response:**
```json
{
  "count": 2,
  "signals": [
    {
      "symbol": "ETHUSDTM",
      "side": "long",
      "confidence": 0.72,
      "price": 3250.50,
      "stop_loss": 3185.00,
      "take_profit": 3510.00,
      "reason": "Trend following: EMA cross, ADX=32",
      "strategy_id": "trend-following",
      "rank": 1,
      "consensus": 1.0,
      "sources": ["trend-following"],
      "risk_position_size": 150.0,
      "risk_meta": {
        "method": "kelly_criterion",
        "final_risk_pct": 1.8,
        "adjustments": ["confidence=72% -> 0.86x", "volatility=normal -> 1.00x"]
      }
    },
    {
      "symbol": "SOLUSDTM",
      "side": "long",
      "confidence": 0.65,
      "price": 180.25,
      "stop_loss": 176.50,
      "take_profit": 195.00,
      "reason": "RSI Divergence: Bullish div (5.2%), RSI=35",
      "strategy_id": "rsi-divergence",
      "rank": 2,
      "consensus": 1.0,
      "sources": ["rsi-divergence"],
      "risk_position_size": 120.0,
      "risk_meta": {
        "method": "fixed_fractional",
        "final_risk_pct": 1.5
      }
    }
  ]
}
```

---

### GET /learning

Get learning summary (strategy performance and system state).

**Response:**
```json
{
  "strategy_performance": {
    "trend-following": {
      "trades": 50,
      "wins": 28,
      "total_pnl_pct": 45.5,
      "score": 1.12,
      "win_rate": 0.56,
      "multiplier": 1.5
    },
    "zweig": {
      "trades": 20,
      "wins": 5,
      "total_pnl_pct": -15.2,
      "score": 0.22,
      "win_rate": 0.25,
      "multiplier": 0.0
    }
  },
  "strategy_multipliers": {
    "trend-following": 1.5,
    "mean-reversion": 1.0,
    "zweig": 0.7
  },
  "pending_counterfactuals": 15,
  "second_chance": {
    "total_rescued": 8,
    "rescued_wins": 5,
    "rescued_losses": 3,
    "winning_patterns": 3,
    "pending_near_misses": 12
  },
  "risk_manager": {
    "equity": 10000.0,
    "peak_equity": 10500.0
  },
  "data_files": {
    "signals": "/app/data/signals_history.jsonl",
    "trades": "/app/data/trades_history.jsonl",
    "counterfactual": "/app/data/counterfactual.jsonl"
  }
}
```

---

### GET /multipliers

Get current dynamic strategy multipliers.

**Response:**
```json
{
  "trend-following": 1.5,
  "mean-reversion": 1.0,
  "turtle": 1.0,
  "weinstein": 1.0,
  "livermore": 1.0,
  "bts-lynch": 0.8,
  "zweig": 0.7,
  "rsi-divergence": 1.0
}
```

---

### GET /counterfactual

Analyze historical counterfactual data.

**Response:**
```json
{
  "timestamp": "2025-02-16T15:30:00.000Z",
  "counterfactual_analysis": {
    "total_rejections_tracked": 150,
    "would_have_won": 55,
    "would_have_lost": 95,
    "hypothetical_win_rate": 0.367,
    "by_rejection_reason": {
      "Low confidence (52% < 55%)": {
        "wins": 20,
        "losses": 35,
        "total_pnl": -15.5
      },
      "Already in long position": {
        "wins": 10,
        "losses": 25,
        "total_pnl": -8.2
      }
    }
  },
  "pattern_analysis": {
    "status": "analyzed",
    "total_rejections": 150,
    "winning_patterns": 2,
    "patterns": [
      {
        "pattern_id": "hist_1",
        "rejection_reason": "Low confidence",
        "confidence_range": [0.525, 0.575],
        "winning_rate": 0.62,
        "sample_size": 15,
        "strategy_ids": ["trend-following"],
        "boost_amount": 0.05
      }
    ]
  },
  "recommendations": [
    {
      "type": "REDUCE_FALSE_NEGATIVES",
      "reason": "Low confidence (52% < 55%)",
      "win_rate": "62%",
      "sample_size": 15,
      "suggestion": "Signals rejected for 'Low confidence' win 62% of the time. Consider relaxing this filter."
    }
  ]
}
```

---

### GET /rescue_stats

Get Second Chance rescue statistics.

**Response:**
```json
{
  "total_rescued": 25,
  "rescued_wins": 15,
  "rescued_losses": 10
}
```

---

## Profit Watchdog

### GET /watchdog

Full watchdog status.

**Response:**
```json
{
  "timestamp": "2025-02-16T15:30:00.000Z",
  "summary": {
    "total_decisions_tracked": 500,
    "pending_outcomes": 25,
    "active_alerts": 1
  },
  "accept_performance": {
    "total": 50,
    "winners": 28,
    "losers": 22,
    "win_rate": 0.56
  },
  "reject_counterfactual": {
    "total": 150,
    "would_have_won": 55,
    "would_have_lost": 95,
    "rejection_accuracy": 0.633
  },
  "strategy_drift": [
    {
      "strategy_id": "bts-lynch",
      "historical_win_rate": 0.48,
      "recent_win_rate": 0.35,
      "drift_pct": -27.1,
      "is_underperforming": true,
      "sample_size": 20,
      "confidence_in_drift": "medium"
    }
  ],
  "recommendations": [
    {
      "parameter": "strategy_multiplier:bts-lynch",
      "current_value": 0.8,
      "recommended_value": 0.5,
      "reason": "Reduce exposure - 35% win rate is below threshold",
      "expected_impact": "Better capital allocation",
      "confidence": "medium",
      "evidence": {"win_rate": 0.35, "sample_size": 20}
    }
  ],
  "alerts": [
    {
      "timestamp": "2025-02-16T15:00:00.000Z",
      "severity": "warning",
      "category": "drift",
      "message": "Strategy bts-lynch underperforming: 35% vs historical 48%",
      "details": {},
      "recommended_action": "Review and potentially reduce multiplier for bts-lynch"
    }
  ],
  "strategy_baselines": {
    "trend-following": {
      "win_rate": 0.51,
      "trades": 100,
      "total_pnl": 208.5
    }
  },
  "recent_performance": {
    "trend-following": {"wins": 15, "total": 25, "win_rate": 0.6}
  }
}
```

---

### GET /watchdog/decisions

Recent orchestrator decisions with outcomes.

**Query Parameters:**
- `limit` (optional): Number of decisions to return (default: 50)

**Response:**
```json
{
  "decisions": [
    {
      "timestamp": "2025-02-16T15:25:00.000Z",
      "signal_id": "trend-following:ETHUSDTM",
      "strategy_id": "trend-following",
      "symbol": "ETHUSDTM",
      "side": "long",
      "confidence": 0.72,
      "price_at_decision": 3250.50,
      "accepted": true,
      "rejection_reason": null,
      "predicted_direction": "up",
      "stop_loss": 3185.00,
      "take_profit": 3510.00,
      "price_1h": 3265.00,
      "price_4h": 3280.00,
      "price_24h": 3350.00,
      "actual_pnl_pct": 3.1,
      "was_winner": true,
      "outcome_filled": true
    }
  ]
}
```

---

### GET /watchdog/alerts

Active watchdog alerts.

**Response:**
```json
{
  "alerts": [
    {
      "timestamp": "2025-02-16T15:00:00.000Z",
      "severity": "warning",
      "category": "drift",
      "message": "Strategy bts-lynch underperforming: 35% vs historical 48%",
      "details": {
        "strategy_id": "bts-lynch",
        "historical_win_rate": 0.48,
        "recent_win_rate": 0.35,
        "drift_pct": -27.1
      },
      "recommended_action": "Review and potentially reduce multiplier for bts-lynch"
    }
  ]
}
```

**Alert Severities:**
- `info`: Informational, no action required
- `warning`: Attention needed, review recommended
- `critical`: Immediate action required

**Alert Categories:**
- `underperformance`: Overall system performance degradation
- `drift`: Strategy deviating from historical performance
- `missed_opportunity`: Rejecting too many winners
- `parameter`: Parameter tuning suggestion

---

### GET /watchdog/recommendations

Parameter tuning recommendations.

**Response:**
```json
{
  "recommendations": [
    {
      "parameter": "min_confidence",
      "current_value": 0.55,
      "recommended_value": 0.52,
      "reason": "Signals with confidence >=52% have 58% win rate",
      "expected_impact": "+8% edge improvement",
      "confidence": "medium",
      "evidence": {
        "win_rates_by_confidence": {
          "0.5": {"wins": 20, "losses": 15},
          "0.55": {"wins": 25, "losses": 18}
        }
      }
    }
  ]
}
```

---

### GET /watchdog/drift

Strategy drift analysis.

**Response:**
```json
{
  "strategy_drift": [
    {
      "strategy_id": "bts-lynch",
      "historical_win_rate": 0.48,
      "recent_win_rate": 0.35,
      "drift_pct": -27.1,
      "is_underperforming": true,
      "sample_size": 20,
      "confidence_in_drift": "medium"
    }
  ]
}
```

---

## Signal Submission

### POST /signals

Submit a signal from a strategy agent.

**Request:**
```json
{
  "strategy_id": "trend-following",
  "symbol": "ETHUSDTM",
  "side": "long",
  "confidence": 0.72,
  "price": 3250.50,
  "stop_loss": 3185.00,
  "take_profit": 3510.00,
  "reason": "EMA crossover with ADX confirmation",
  "timestamp": "2025-02-16T15:25:00.000Z",
  "metadata": {
    "adx": 32,
    "rsi": 55
  }
}
```

**Required Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | KuCoin symbol (e.g., `ETHUSDTM`) |
| `side` | string | `long`, `short`, or `neutral` |
| `confidence` | float | 0.0 to 1.0 |

**Optional Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | string | Identifier for the strategy |
| `price` | float | Current price at signal time |
| `stop_loss` | float | Stop loss price |
| `take_profit` | float | Take profit price |
| `reason` | string | Human-readable signal reason |
| `timestamp` | string | ISO 8601 timestamp |
| `metadata` | object | Additional signal data |

**Response (success):**
```json
{
  "status": "received"
}
```

**Response (validation error):**
```json
{
  "error": "Invalid symbol format: ETH"
}
```

**Security:**
- Max request size: 50KB
- Symbols validated against pattern `^[A-Z0-9]{2,10}USDTM$`
- Dangerous patterns (scripts, injections) are sanitised

---

### POST /trade_result

Record a trade outcome for learning.

**Request:**
```json
{
  "symbol": "ETHUSDTM",
  "side": "long",
  "pnl": 50.25,
  "pnl_pct": 1.5,
  "strategy_id": "trend-following",
  "reason": "Take profit hit"
}
```

**Required Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | KuCoin symbol |
| `side` | string | `long` or `short` |
| `pnl` | float | Absolute P&L in USDT |
| `pnl_pct` | float | P&L percentage |
| `strategy_id` | string | Strategy that generated the signal |

**Optional Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `reason` | string | Exit reason |

**Response:**
```json
{
  "status": "recorded"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Description of the error"
}
```

**HTTP Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (validation error) |
| 404 | Endpoint not found |
| 413 | Request too large |
| 500 | Internal server error |

---

## Rate Limiting

Currently no explicit rate limiting, but:
- Signal endpoints process one at a time
- Large requests (>50KB) are rejected
- Consider implementing rate limiting for production

---

## Dashboard API

For the full-featured dashboard API (portfolio, positions, trades, charts), see `api/endpoints.py`. Key additional endpoints:

```
GET /api/portfolio     → Portfolio overview
GET /api/positions     → All positions
GET /api/trades        → Trade history with filters
GET /api/strategies    → Strategy performance
GET /api/signals       → Signal history
GET /api/risk          → Risk metrics
GET /api/chart/:symbol → OHLCV data
```
