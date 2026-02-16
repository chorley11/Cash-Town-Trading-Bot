# 🏗 Architecture Overview

This document describes the system architecture of Cash Town, an institutional-grade multi-agent trading system.

---

## Design Philosophy

**Learning-First**: The system learns optimal behaviour from actual P&L, not from predetermined rules. Arbitrary limits and cooldowns have been removed in favour of data-driven adaptation.

**Capital Preservation**: All signals pass through centralised risk controls. The Risk Manager can veto any trade that would violate portfolio-level constraints.

**Self-Improvement**: The Profit Watchdog monitors every decision, tracks outcomes, and recommends parameter changes based on empirical evidence.

---

## System Components

### 1. Smart Orchestrator

The central brain that coordinates all system activity.

```
SmartOrchestrator
├── SignalAggregator      # Rank, filter, deduplicate signals
├── SecondChanceEvaluator # Rescue promising rejected signals  
├── RiskManager           # Position sizing and risk controls
└── ProfitWatchdog        # Self-improvement feedback loop
```

**Responsibilities:**
- Receive signals from strategy agents
- Aggregate and rank signals by quality
- Apply risk filters before execution
- Track all decisions for learning
- Maintain counterfactual records

**Key Methods:**
```python
receive_signal(strategy_id, signal_data)    # Receive raw signal
get_actionable_signals() -> List[Signal]    # Get filtered signals for execution
record_trade_result(...)                    # Record outcome for learning
update_equity(equity)                       # Update risk manager equity
```

---

### 2. Signal Aggregator

Intelligent signal selection and ranking.

**Pipeline:**
1. **Filter** - Remove signals below confidence threshold
2. **Deduplicate** - Handle same-symbol signals from multiple strategies
3. **Rank** - Score by adjusted confidence (strategy track record × raw confidence)
4. **Conflict Resolution** - Handle opposing signals on same symbol

**Configuration:**
```python
AggregatorConfig(
    min_confidence=0.55,      # Minimum raw confidence
    min_consensus=1,          # Minimum sources (strategies) required
    max_signals_per_cycle=99, # Effectively unlimited
    cooldown_minutes=0,       # Disabled - learn from data
)
```

---

### 3. Second Chance Evaluator

Rescues promising signals that were initially rejected.

**When it activates:**
- Signal was rejected for being just below thresholds
- Strategy has proven track record
- Multiple strategies agree on direction
- Signal matches a historical "winning rejection" pattern

**Boost Factors:**
| Factor | Boost |
|--------|-------|
| Trend-following signal | +8% |
| Multiple strategy consensus | +3% per agreeing strategy |
| Matches winning pattern | +6-12% |
| Strong ADX (>35) | +5% |

**Anti-Pattern:**
| Factor | Penalty |
|--------|---------|
| Zweig signal | -10% |
| High disagreement | -2-5% |

---

### 4. Risk Manager

Central risk control for the entire portfolio.

**Components:**
```
RiskManager
├── PortfolioHeat        # Total risk exposure tracking
├── CircuitBreakerState  # Emergency halt conditions
├── VolatilityData       # Per-symbol vol regime
└── PositionRisk         # Per-position risk metrics
```

**Position Sizing Logic:**
1. Calculate base size using Kelly Criterion (if 20+ trades available)
2. Fall back to fixed fractional (2% max risk) otherwise
3. Apply confidence multiplier (0.5× to 1.0×)
4. Apply volatility adjustment (up to 75% reduction in extreme vol)
5. Apply portfolio heat adjustment (reduce as approaching limit)
6. Apply correlation adjustment (reduce if correlated exposure high)

**Circuit Breakers:**
- **Daily Loss**: Halt at 5% daily loss, 4-hour cooldown
- **Max Drawdown**: Halt at 15% from peak equity
- Auto-reset on new trading day

---

### 5. Profit Watchdog

Self-improving feedback loop that monitors system performance.

**Tracking:**
- Every orchestrator decision (accept/reject)
- Actual vs predicted outcomes
- Strategy performance drift
- False positive/negative rates

**Outputs:**
- **Alerts**: Underperformance, drift, missed opportunities
- **Recommendations**: Parameter adjustments based on data
- **Auto-Tune**: Can automatically apply high-confidence recommendations

**Key Metrics:**
```python
# Acceptance Performance
accept_winners / (accept_winners + accept_losers)  # Should be >50%

# Rejection Quality
reject_would_have_lost / total_rejected  # Should be >50%
```

---

### 6. Strategy Agents

Independent agents running in parallel threads.

**Base Interface:**
```python
class BaseStrategyAgent:
    def generate_signals(self, market_data) -> List[Signal]
    def get_required_indicators(self) -> List[str]
```

**Signal Structure:**
```python
@dataclass
class Signal:
    strategy_id: str
    symbol: str
    side: SignalSide      # LONG, SHORT, NEUTRAL
    confidence: float     # 0.0 to 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime
    metadata: Dict
```

**Current Agents:**
| Agent | Thread Name | Interval |
|-------|-------------|----------|
| trend-following | agent-trend-following | 300s |
| mean-reversion | agent-mean-reversion | 300s |
| turtle | agent-turtle | 300s |
| weinstein | agent-weinstein | 300s |
| livermore | agent-livermore | 300s |
| bts-lynch | agent-bts-lynch | 300s |
| zweig | agent-zweig | 300s |
| rsi-divergence | agent-rsi-divergence | 300s |

---

### 7. Execution Engine

Handles order placement with risk integration.

**Modes:**
- `LIVE`: Real orders to KuCoin Futures
- `PAPER`: Simulated execution (logs only)
- `DISABLED`: No execution

**Execution Flow:**
1. Receive aggregated signal from orchestrator
2. Check execution-level risk (position limits, daily loss)
3. Calculate position size with strategy multiplier
4. Place market order with stop-loss and take-profit
5. Register position with Risk Manager
6. Record in Strategy Tracker for attribution

**Drawdown Protection:**
When account drops 10% from peak:
- Position sizes reduced by 50%
- Logs warning to alert operators

---

### 8. Data Flow

```
Market Data (KuCoin)
        │
        ▼
┌───────────────────┐
│   Data Feed       │ ◄── 24 symbols, 5-min bars
│   (KuCoinDataFeed)│
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Strategy Agents   │ ◄── 8 agents in parallel threads
│ (generate signals)│
└───────────────────┘
        │ POST /signals
        ▼
┌───────────────────┐
│ Smart Orchestrator│ ◄── Aggregate, filter, risk-check
│                   │
└───────────────────┘
        │ GET /signals
        ▼
┌───────────────────┐
│ Execution Engine  │ ◄── Place orders, manage positions
│                   │
└───────────────────┘
        │ POST /trade_result
        ▼
┌───────────────────┐
│ Learning Pipeline │ ◄── Update strategy performance
│ (counterfactual)  │
└───────────────────┘
```

---

### 9. Persistence

All learning data persisted to `DATA_DIR` (default `/app/data`):

| File | Purpose |
|------|---------|
| `signals_history.jsonl` | All signals (selected + rejected) |
| `trades_history.jsonl` | Completed trade outcomes |
| `counterfactual.jsonl` | What rejected signals would have done |
| `strategy_performance.json` | Aggregated strategy stats |
| `risk_manager_state.json` | Risk manager state (peak equity, circuit breakers) |
| `watchdog_state.json` | Watchdog stats and pending outcomes |
| `winning_patterns.json` | Identified second-chance patterns |
| `second_chance.jsonl` | Near-miss signal tracking |

---

### 10. Security

**Input Validation** (`utils/validation.py`):
- Symbol format validation (XXXUSDTM)
- Side validation (long/short/neutral)
- Confidence range check (0.0-1.0)
- String sanitisation (dangerous pattern removal)
- Request size limits (50KB signals, 10KB trade results)

**Dangerous Patterns Blocked:**
- `<script` (XSS)
- `{{...}}` (template injection)
- `${...}` (template literals)
- Null bytes
- Event handlers (`onclick`, etc.)

---

### 11. Performance Monitoring

**PerformanceMonitor** (`utils/monitoring.py`):
- Cycle execution timing
- Stage-level breakdown (fetch, execute, refresh)
- Memory tracking via `tracemalloc`
- Periodic garbage collection
- Alert thresholds (4-min max cycle, 100MB growth)

**Exposed at `/perf`:**
```json
{
  "cycle_time_ms": {"avg": 1500, "p95": 3000, "max": 8000},
  "memory_mb": {"initial": 150, "current": 180, "peak": 200},
  "signals": {"total_generated": 500, "total_executed": 50},
  "errors": {"error_rate": 0.02}
}
```

---

## Threading Model

```
Main Thread
├── HTTP Server Thread (receives signals, serves API)
├── Executor Thread (processes signals, places orders)
├── Agent Thread: trend-following
├── Agent Thread: mean-reversion
├── Agent Thread: turtle
├── Agent Thread: weinstein
├── Agent Thread: livermore
├── Agent Thread: bts-lynch
├── Agent Thread: zweig
└── Agent Thread: rsi-divergence
```

All threads are daemon threads—they terminate when the main process exits.

---

## Error Handling

**Strategy Agent Errors:**
- Caught per-symbol, logged, continue with other symbols
- Agent thread crashes are logged but don't crash system

**Execution Errors:**
- Order failures logged with full context
- Position state refreshed from exchange
- Circuit breaker triggered on repeated failures

**Risk Manager Errors:**
- Fail-safe: if risk check fails, trade is blocked
- State persisted frequently to survive restarts

---

## Deployment Topology

```
┌─────────────────────────────────────────────┐
│              Railway.app                     │
│  ┌─────────────────────────────────────┐    │
│  │         Cash Town Process           │    │
│  │  - Python 3.9                       │    │
│  │  - Single process, multi-threaded   │    │
│  │  - Persistent volume for /app/data  │    │
│  └─────────────────────────────────────┘    │
│              │                              │
│              ▼                              │
│  ┌─────────────────────────────────────┐    │
│  │     KuCoin Futures API               │    │
│  │     (External)                       │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Resource Requirements:**
- CPU: 1 vCPU minimum
- Memory: 512MB minimum, 1GB recommended
- Storage: 1GB for data files
- Network: Outbound to KuCoin API

---

## Future Architecture Considerations

1. **Horizontal Scaling**: Strategy agents could run as separate services with message queue
2. **Real-time Data**: WebSocket streams instead of polling
3. **Database**: PostgreSQL for structured learning data
4. **Monitoring**: Prometheus metrics, Grafana dashboards
5. **Event Sourcing**: Full audit trail of all state changes
