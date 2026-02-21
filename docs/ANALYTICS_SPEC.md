# Cash Town Analytics - Design Specification

**Version:** 1.0  
**Author:** Cyril  
**Date:** 2026-02-19  
**Status:** Draft for Review

---

## Executive Summary

Cash Town Analytics is a live data dashboard and agentic documentation system for the Cash Town multi-strategy crypto futures trading bot. It combines real-time performance monitoring with AI-accessible documentation, enabling both human users and AI assistants to query, analyze, and explain the trading system.

**Key Innovation:** Dual-layer content architecture where every data point is both visually rendered for humans AND structured for AI consumption, following the Alkimi IAB study pattern.

**Performance Context:**
- Started: January 21, 2026 with $6,378
- Current: ~$12,995 (+105% in ~30 days)
- Total Trades: 367+ completed
- Active Strategies: 10+
- Platform: KuCoin Futures

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Models](#2-data-models)
3. [Backtest Engine Spec](#3-backtest-engine-spec)
4. [Dashboard/MDX Structure](#4-dashboardmdx-structure)
5. [Agentic Docs Layer](#5-agentic-docs-layer)
6. [Tech Stack Recommendation](#6-tech-stack-recommendation)
7. [Implementation Phases](#7-implementation-phases)
8. [Open Questions](#8-open-questions)

---

## 1. System Architecture

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CASH TOWN TRADING BOT                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Agents    │→ │ Orchestrator│→ │  Executor   │→ │  KuCoin     │        │
│  │ (Strategies)│  │ (Risk Mgmt) │  │             │  │  Futures    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│        │                │                │                │                 │
│        ↓                ↓                ↓                ↓                 │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                     EVENT STREAM                                 │       │
│  │  • Signal Generated (executed or not)                           │       │
│  │  • Trade Opened                                                 │       │
│  │  • Trade Closed                                                 │       │
│  │  • Position Updated                                             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Signal Logger  │  │  Trade Tracker  │  │ Position Syncer │             │
│  │                 │  │                 │  │                 │             │
│  │ • All signals   │  │ • Entry/exit    │  │ • Real-time     │             │
│  │ • Executed Y/N  │  │ • PnL           │  │ • Mark prices   │             │
│  │ • Rejection     │  │ • Duration      │  │ • Unrealized    │             │
│  │   reasons       │  │ • Strategy      │  │   PnL           │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ↓                    ↓                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA STORAGE LAYER                                │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    Primary Storage (SQLite/JSON)                  │      │
│  ├──────────────┬──────────────┬──────────────┬──────────────┬──────┤      │
│  │   signals    │    trades    │  positions   │ counterfacts │ meta │      │
│  │              │              │              │              │      │      │
│  │ • All signal │ • Completed  │ • Current    │ • Backtest   │ • Eq │      │
│  │   events     │   trades     │   holdings   │   results on │   ui │      │
│  │ • 100% cap-  │ • Full PnL   │ • Live sync  │   non-exec'd │   ty │      │
│  │   ture       │   details    │              │   signals    │      │      │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYTICS ENGINE                                  │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Backtest Engine │  │  Metrics Calc   │  │   API Server    │             │
│  │                 │  │                 │  │                 │             │
│  │ • Replay non-   │  │ • Win rates     │  │ • REST + WS     │             │
│  │   executed      │  │ • Sharpe        │  │ • Real-time     │             │
│  │   signals       │  │ • Drawdown      │  │   updates       │             │
│  │ • Historical    │  │ • Per-strategy  │  │ • Auth          │             │
│  │   price data    │  │   breakdown     │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Next.js + MDX Dashboard                       │       │
│  │                                                                  │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │       │
│  │  │ Human View  │  │ Agent View  │  │ Interactive │              │       │
│  │  │             │  │             │  │   Console   │              │       │
│  │  │ • Charts    │  │ • JSON data │  │             │              │       │
│  │  │ • Tables    │  │ • Full      │  │ • Query AI  │              │       │
│  │  │ • Summaries │  │   datasets  │  │ • Explain   │              │       │
│  │  │ • Visuals   │  │ • Behavior  │  │   strategy  │              │       │
│  │  │             │  │   instruct. │  │ • Ask about │              │       │
│  │  │             │  │             │  │   trades    │              │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Details

#### Signal Logger
**Purpose:** Capture ALL signals, whether executed or not.

```python
class SignalLogger:
    """
    Hooks into SmartOrchestrator to log every signal event.
    """
    
    def log_signal(self, signal: Signal, executed: bool, reason: str = None):
        """
        Called for every signal generated by any strategy.
        
        Args:
            signal: The full signal object with all metadata
            executed: Whether this signal was converted to a trade
            reason: If not executed, why (risk_limit, correlation, low_conf, etc.)
        """
        pass
```

**Integration Point:** `SmartOrchestrator.receive_signal()` and `SignalAggregator.aggregate()`

#### Trade Tracker
**Purpose:** Record complete trade lifecycle.

```python
class TradeTracker:
    """
    Monitors trades from entry to exit.
    """
    
    def record_entry(self, trade_id: str, signal: Signal, entry_price: float):
        pass
    
    def record_exit(self, trade_id: str, exit_price: float, reason: str):
        pass
```

**Integration Point:** Execution layer websocket events

#### Backtest Engine
**Purpose:** Replay non-executed signals against historical data.

```python
class CounterfactualBacktester:
    """
    What would have happened if we had executed this signal?
    """
    
    async def backtest_signal(self, signal: Signal) -> CounterfactualResult:
        """
        Fetch historical price data and simulate the trade.
        """
        pass
```

---

## 2. Data Models

### 2.1 Trade Schema

```typescript
interface Trade {
  // Identity
  id: string;                    // UUID
  symbol: string;                // e.g., "SOLUSDTM"
  
  // Trade details
  side: "long" | "short";
  entry_price: number;
  exit_price: number;
  size: number;                  // Position size in quote currency
  leverage: number;
  
  // Performance
  pnl: number;                   // Realized PnL in USDT
  pnl_pct: number;               // Percentage return
  fees: number;                  // Trading fees paid
  
  // Strategy attribution
  strategy_id: string;           // e.g., "trend-following"
  signal_id: string;             // Link to original signal
  
  // Timing
  entry_time: ISO8601;
  exit_time: ISO8601;
  duration_minutes: number;
  
  // Exit info
  close_reason: "tp" | "sl" | "manual" | "risk" | "timeout";
  
  // Metadata
  source: string;                // "live" | "backtest" | "notion_seed"
  tags: string[];                // Optional categorization
}
```

**JSON-LD for Agent Consumption:**
```json
{
  "@context": "https://cashtown.dev/schema",
  "@type": "Trade",
  "id": "3065ede8-110c-810c-b0b1-dbcec54bf9be",
  "symbol": "SOLUSDTM",
  "side": "short",
  "entry_price": 0.9386,
  "exit_price": 0.9495,
  "pnl": -5.81,
  "pnl_pct": -1.16,
  "strategy_id": "cucurbit",
  "close_reason": "sl"
}
```

### 2.2 Signal Schema

```typescript
interface Signal {
  // Identity
  id: string;                    // UUID
  symbol: string;
  
  // Signal details
  side: "long" | "short";
  confidence: number;            // 0.0 - 1.0
  adjusted_confidence: number;   // After strategy track record adjustment
  
  // Price targets
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  
  // Risk metrics
  risk_reward_ratio: number;
  atr_at_signal: number;
  
  // Strategy info
  strategy_id: string;
  strategy_version: string;
  indicators: Record<string, number>;  // e.g., { rsi: 32, adx: 41 }
  
  // Timing
  generated_at: ISO8601;
  
  // Execution status
  executed: boolean;
  execution_time: ISO8601 | null;
  
  // If not executed, why
  rejection_reason: RejectionReason | null;
  rejection_details: string | null;
}

type RejectionReason = 
  | "low_confidence"        // Below min_confidence threshold
  | "risk_limit"            // Would exceed portfolio risk
  | "correlation"           // Too correlated with existing position
  | "max_positions"         // Already at position limit
  | "symbol_cooldown"       // Recently traded this symbol
  | "conflicting_signal"    // Opposing signal from higher-ranked strategy
  | "second_chance_fail"    // Didn't pass rescue evaluation
  | "manual_filter"         // Manually excluded
  ;
```

### 2.3 Counterfactual Schema

```typescript
interface Counterfactual {
  // Link to original signal
  signal_id: string;
  signal: Signal;
  
  // Backtest parameters
  backtest_start: ISO8601;
  backtest_end: ISO8601;
  
  // Simulated trade
  simulated_entry_price: number;   // Actual price at signal time
  simulated_exit_price: number;    // Where it would have exited
  simulated_exit_reason: "tp" | "sl" | "timeout";
  simulated_duration_minutes: number;
  
  // Would-be performance
  would_be_pnl: number;
  would_be_pnl_pct: number;
  
  // Price path during trade
  price_high: number;
  price_low: number;
  max_favorable_excursion: number;   // Best it ever got
  max_adverse_excursion: number;     // Worst it ever got
  
  // Verdict
  would_have_won: boolean;
  opportunity_cost: number;          // Positive = missed profit, negative = avoided loss
  
  // Data quality
  data_quality: "complete" | "partial" | "missing";
  confidence_in_result: number;      // 0.0 - 1.0
}
```

### 2.4 Position Schema

```typescript
interface Position {
  // Identity
  symbol: string;
  
  // Position details
  side: "long" | "short";
  size: number;
  entry_price: number;
  
  // Current state (updated in real-time)
  mark_price: number;
  liquidation_price: number;
  
  // Performance
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  
  // Risk
  leverage: number;
  margin: number;
  margin_ratio: number;
  
  // Attribution
  strategy: string;                  // Which strategy opened this
  trade_id: string | null;           // Link to trade record
  
  // Timing
  opened_at: ISO8601;
  last_updated: ISO8601;
}
```

### 2.5 Account Snapshot Schema

```typescript
interface AccountSnapshot {
  timestamp: ISO8601;
  
  // Balances
  currency: string;                  // "USDT"
  account_equity: number;            // Total account value
  margin_balance: number;            // Margin in use
  available_balance: number;         // Free margin
  
  // Unrealized
  unrealized_pnl: number;
  position_margin: number;
  order_margin: number;
  
  // Performance metrics (calculated)
  total_realized_pnl: number;
  all_time_high: number;
  current_drawdown_pct: number;
  
  // Position summary
  open_position_count: number;
  long_exposure: number;
  short_exposure: number;
  net_exposure: number;
}
```

---

## 3. Backtest Engine Spec

### 3.1 Purpose

The backtest engine answers: **"What would have happened if we had executed signals we didn't take?"**

This is crucial for:
1. **Validating risk management** - Are we correctly filtering bad signals?
2. **Tuning thresholds** - Is our confidence cutoff too high/low?
3. **Strategy evaluation** - Which strategies generate good signals we're missing?
4. **Building trust** - Show users the system makes sensible decisions

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTEST ENGINE                              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Signal     │    │  Price      │    │  Trade      │         │
│  │  Queue      │ →  │  Fetcher    │ →  │  Simulator  │         │
│  │             │    │             │    │             │         │
│  │ Non-exec'd  │    │ KuCoin API  │    │ Entry/Exit  │         │
│  │ signals     │    │ Historical  │    │ Logic       │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                  │                 │
│                            ↓                  ↓                 │
│                     ┌─────────────┐    ┌─────────────┐         │
│                     │  Price      │    │ Counterfact │         │
│                     │  Cache      │    │  Results    │         │
│                     │  (SQLite)   │    │             │         │
│                     └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Sources

**Primary: KuCoin Futures API**
```python
# Klines endpoint for historical OHLCV
GET /api/v1/kline/query
  ?symbol=SOLUSDTM
  &granularity=1      # 1-minute candles
  &from=1708300800    # Unix timestamp
  &to=1708387200

# Response contains: [time, open, close, high, low, volume]
```

**Rate Limits:**
- 30 requests/second for public endpoints
- Data available: ~2 years of 1-minute candles

**Caching Strategy:**
- Store fetched OHLCV in local SQLite
- Only fetch missing ranges
- Prune data older than 90 days (configurable)

### 3.4 Simulation Logic

```python
async def simulate_trade(signal: Signal, candles: List[Candle]) -> CounterfactualResult:
    """
    Replay a signal against actual price data.
    """
    # Find entry candle (signal generation time)
    entry_candle = find_candle_at(signal.generated_at, candles)
    entry_price = entry_candle.close  # Assume market order at close
    
    # Set stops based on signal
    stop_loss = signal.stop_loss
    take_profit = signal.take_profit
    
    # Walk forward through candles
    for candle in candles_after(entry_candle):
        # Check if stopped out
        if signal.side == "long":
            if candle.low <= stop_loss:
                return exit_at(stop_loss, "sl", candle)
            if candle.high >= take_profit:
                return exit_at(take_profit, "tp", candle)
        else:  # short
            if candle.high >= stop_loss:
                return exit_at(stop_loss, "sl", candle)
            if candle.low <= take_profit:
                return exit_at(take_profit, "tp", candle)
        
        # Check max duration (e.g., 24 hours)
        if candle.time > signal.generated_at + MAX_DURATION:
            return exit_at(candle.close, "timeout", candle)
    
    # Still open at end of data
    return exit_at(candles[-1].close, "timeout", candles[-1])
```

### 3.5 Metrics to Calculate

**Per-Signal Counterfactual:**
- Would-be PnL (absolute and %)
- Duration to exit
- Max favorable excursion (MFE) - best it ever got
- Max adverse excursion (MAE) - worst it ever got
- Exit reason (hit TP, hit SL, timed out)

**Aggregate Counterfactual Metrics:**
```typescript
interface CounterfactualSummary {
  // Volume
  total_non_executed_signals: number;
  backtest_coverage_pct: number;      // % with valid backtest
  
  // Would-be performance
  total_would_be_pnl: number;
  would_be_win_rate: number;
  would_be_avg_winner: number;
  would_be_avg_loser: number;
  
  // Decision quality
  correctly_rejected: number;         // Rejected signals that would have lost
  incorrectly_rejected: number;       // Rejected signals that would have won
  rejection_accuracy: number;         // correctly_rejected / total
  
  // Opportunity cost
  missed_profit_total: number;        // Sum of positive would-be PnL
  avoided_loss_total: number;         // Sum of negative would-be PnL
  net_opportunity_cost: number;       // missed - avoided
  
  // By rejection reason
  by_reason: Record<RejectionReason, {
    count: number;
    would_be_win_rate: number;
    total_would_be_pnl: number;
  }>;
}
```

### 3.6 Scheduling

**When to backtest:**
1. **On signal rejection** - Queue for backtest when market closes (price data complete)
2. **Daily batch** - Nightly job to process previous day's non-executed signals
3. **On demand** - API endpoint to trigger specific signal backtest

**Timing considerations:**
- Wait at least `MAX_TRADE_DURATION` after signal before backtesting
- Default: Wait 24 hours to ensure price data is complete
- Quick preview: Available 1 hour after signal (partial result)

---

## 4. Dashboard/MDX Structure

### 4.1 Information Architecture

```
cashtown-analytics/
├── app/
│   ├── page.mdx                 # Landing / Overview
│   ├── performance/
│   │   ├── page.mdx             # Performance Dashboard
│   │   └── [strategy]/page.mdx  # Per-Strategy Deep Dive
│   ├── trades/
│   │   ├── page.mdx             # Trade History
│   │   └── [id]/page.mdx        # Single Trade Analysis
│   ├── signals/
│   │   ├── page.mdx             # All Signals (executed + not)
│   │   └── counterfactuals.mdx  # "What if" Analysis
│   ├── positions/
│   │   └── page.mdx             # Live Positions
│   ├── methodology/
│   │   ├── page.mdx             # How It Works Overview
│   │   ├── strategies.mdx       # Strategy Documentation
│   │   ├── risk.mdx             # Risk Management
│   │   └── architecture.mdx     # Technical Architecture
│   └── ai/
│       └── page.mdx             # AI Query Console
├── components/
│   ├── AgentData.tsx            # Hidden structured data for AI
│   ├── EquityCurve.tsx          # Interactive equity chart
│   ├── TradeTable.tsx           # Sortable trade list
│   ├── StrategyCard.tsx         # Strategy performance card
│   ├── SignalFeed.tsx           # Live signal stream
│   └── CounterfactualCard.tsx   # "What if" result display
└── data/
    ├── trades.json              # Processed trade data
    ├── signals.json             # Signal log
    ├── counterfactuals.json     # Backtest results
    └── performance.json         # Aggregated metrics
```

### 4.2 Page Layouts

#### 4.2.1 Overview Page (`/`)

**Human View:**
```
┌─────────────────────────────────────────────────────────────────┐
│  CASH TOWN                                              🤖 AI  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    EQUITY CURVE                          │   │
│  │     $12,995 (+105%)                                     │   │
│  │     ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█████████████████████████████████   │   │
│  │     Jan 21                                      Feb 19   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │   367         │ │   48.7%       │ │   +$6,617     │        │
│  │   Total       │ │   Win Rate    │ │   Total PnL   │        │
│  │   Trades      │ │               │ │               │        │
│  └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                 │
│  STRATEGY BREAKDOWN                                             │
│  ┌──────────────────┬──────────┬──────────┬──────────┐        │
│  │ Strategy         │ Trades   │ Win Rate │ PnL      │        │
│  ├──────────────────┼──────────┼──────────┼──────────┤        │
│  │ 🌟 Cucurbit      │ 203      │ 47.8%    │ +$4,241  │        │
│  │    Trend-Follow  │ 48       │ 45.8%    │ +$206    │        │
│  │    Synced        │ 23       │ 56.5%    │ +$88     │        │
│  │    ...           │ ...      │ ...      │ ...      │        │
│  └──────────────────┴──────────┴──────────┴──────────┘        │
│                                                                 │
│  LIVE POSITIONS (7)                                             │
│  ┌────────────┬────────┬────────┬────────┬────────┐           │
│  │ Symbol     │ Side   │ Size   │ Entry  │ PnL    │           │
│  ├────────────┼────────┼────────┼────────┼────────┤           │
│  │ OPUSDTM    │ SHORT  │ 115k   │ 0.1474 │ +$626  │           │
│  │ ...        │ ...    │ ...    │ ...    │ ...    │           │
│  └────────────┴────────┴────────┴────────┴────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Agent Data (Hidden):**
```jsx
<AgentData type="overview">
{`
{
  "system": "Cash Town",
  "description": "Multi-strategy crypto futures trading bot",
  "platform": "KuCoin Futures",
  "inception_date": "2026-01-21",
  "starting_capital": 6378,
  "current_equity": 12995,
  "total_return_pct": 105.2,
  "snapshot_time": "2026-02-19T09:16:51Z",
  "strategy_count": 10,
  "total_trades": 367,
  "overall_win_rate": 0.487,
  "total_pnl": 6617,
  "open_positions": 7,
  "unrealized_pnl": 682
}
`}
</AgentData>
```

#### 4.2.2 Counterfactuals Page (`/signals/counterfactuals`)

**Human View:**
```
┌─────────────────────────────────────────────────────────────────┐
│  WHAT WE DIDN'T TRADE                                          │
│  Analysis of signals that were generated but not executed       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SUMMARY                                                        │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │   142         │ │   62.3%       │ │   -$847       │        │
│  │   Non-Exec'd  │ │   Would Have  │ │   Net Oppor-  │        │
│  │   Signals     │ │   Won         │ │   tunity Cost │        │
│  └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                 │
│  ℹ️  Negative opportunity cost means we correctly avoided more │
│      losses than profits we missed. Our risk filters are       │
│      working.                                                   │
│                                                                 │
│  BY REJECTION REASON                                            │
│  ┌────────────────────┬─────────┬──────────┬──────────┐        │
│  │ Reason             │ Count   │ Win Rate │ Avg PnL  │        │
│  ├────────────────────┼─────────┼──────────┼──────────┤        │
│  │ Low Confidence     │ 67      │ 41.2%    │ -$12.30  │ ✅     │
│  │ Risk Limit         │ 34      │ 58.8%    │ +$8.50   │ ⚠️     │
│  │ Correlation        │ 23      │ 52.1%    │ +$3.20   │        │
│  │ Max Positions      │ 18      │ 72.2%    │ +$18.40  │ ❌     │
│  └────────────────────┴─────────┴──────────┴──────────┘        │
│                                                                 │
│  NOTABLE MISSED TRADES                                          │
│  [Interactive list of highest would-be PnL signals]             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 MDX Component Structure

**AgentData Component:**
```tsx
// components/AgentData.tsx
interface AgentDataProps {
  type: 'overview' | 'trade' | 'signal' | 'counterfactual' | 'strategy';
  children: string;  // JSON string
}

export function AgentData({ type, children }: AgentDataProps) {
  // Render nothing visible, but include in DOM for AI parsing
  return (
    <script
      type="application/ld+json"
      data-agent-type={type}
      dangerouslySetInnerHTML={{ __html: children }}
    />
  );
}
```

**Dual-Layer Pattern:**
```mdx
# Strategy Performance

<AgentData type="strategy">
{JSON.stringify({
  strategy_id: "cucurbit",
  trades: 203,
  wins: 97,
  losses: 106,
  win_rate: 0.4778,
  total_pnl: 4240.6,
  avg_pnl: 20.89,
  max_win: 504.89,
  max_loss: -87.61,
  sharpe_ratio: 1.42,
  profit_factor: 2.31
})}
</AgentData>

## Cucurbit Strategy

Cucurbit is our primary strategy and largest PnL contributor.

| Metric | Value |
|--------|-------|
| Total Trades | 203 |
| Win Rate | 47.8% |
| Total PnL | +$4,241 |
| Avg Trade | +$20.89 |

The strategy uses [technical description]...
```

### 4.4 Interactive Components

**Filters:**
- Date range picker (preset: 24h, 7d, 30d, all)
- Strategy selector (multi-select)
- Symbol selector
- Side filter (long/short/all)
- Min/max PnL range

**Real-time updates:**
- WebSocket connection to API
- Live position PnL updates
- New trade notifications
- Signal stream

**Export options:**
- Download CSV (trades, signals)
- Copy JSON for AI (clipboard)
- Generate PDF report

---

## 5. Agentic Docs Layer

### 5.1 AI Behavioral Instructions

Following the Alkimi study pattern, we embed behavioral guidance for AI assistants directly in the content.

**Core Instructions Block (in layout):**
```mdx
<AgentInstructions>
{`
# Cash Town AI Assistant Guidelines

You are helping users understand Cash Town, a multi-strategy crypto futures trading bot.

## Your Role
- Explain how the system works in accessible terms
- Present data accurately, citing specific numbers
- Acknowledge limitations and risks honestly
- Adapt to the user's technical level

## Key Facts to Know
- System started January 21, 2026 with $6,378
- Currently at ~$12,995 (+105% in ~30 days)
- Trades on KuCoin Futures with 4-5x leverage
- Runs 10 strategies simultaneously
- Fully automated, no manual intervention

## When Asked About Performance
1. Always state the timeframe
2. Distinguish realized vs unrealized PnL
3. Acknowledge this is a short track record
4. Mention that past performance ≠ future results

## When Asked About Risk
- Be direct about risks (leverage, crypto volatility, technical failure)
- Explain the risk management system honestly
- Don't minimize or oversell safety

## Data Access
You have access to:
- Complete trade history (JSON)
- Strategy performance breakdown
- Current open positions
- Counterfactual analysis (what-if scenarios)

When citing data, reference the specific JSON blocks on the page.
`}
</AgentInstructions>
```

### 5.2 Audience Adaptation

**Trader Persona:**
```mdx
<AudienceContext persona="trader">
{`
For traders, emphasize:
- Entry/exit logic and timing
- Risk/reward ratios
- Position sizing methodology
- Win rate and expectancy
- Drawdown characteristics

Use trading terminology freely:
- ATR, ADX, RSI, EMA
- Stop loss, take profit
- Sharpe ratio, profit factor
- Leverage and margin

Answer questions about:
- "How does [strategy] generate signals?"
- "What's the avg hold time?"
- "How do you size positions?"
`}
</AudienceContext>
```

**Developer Persona:**
```mdx
<AudienceContext persona="developer">
{`
For developers, emphasize:
- System architecture
- API design and endpoints
- Data models and schemas
- Code structure
- Integration points

Be specific about:
- Python version, dependencies
- Database schema
- WebSocket protocols
- Deployment infrastructure (Railway)

Answer questions about:
- "How is the orchestrator implemented?"
- "What's the signal aggregation algorithm?"
- "How do you handle rate limits?"
`}
</AudienceContext>
```

**Investor Persona:**
```mdx
<AudienceContext persona="investor">
{`
For investors, emphasize:
- Risk-adjusted returns
- Drawdown and recovery
- Strategy diversification
- Scalability potential
- Track record and verification

Be clear about:
- This is ~30 days of live trading
- The capital is relatively small ($6k start)
- Crypto futures are high-risk
- Past performance disclaimers

Answer questions about:
- "What's the Sharpe ratio?"
- "How does this compare to holding BTC?"
- "What's the maximum drawdown so far?"
`}
</AudienceContext>
```

### 5.3 Objection Handling

```mdx
<ObjectionHandling>
{`
## Common Objections and Responses

### "30 days is too short to prove anything"
Acknowledge: True, this is a limited track record.
Respond: We agree. That's why we provide full transparency:
- Every trade is logged and verifiable
- Counterfactual analysis shows rejected signals
- We encourage skepticism and verification

### "Why should I trust automated trading?"
Respond honestly:
- Automated trading removes emotional bias
- But it introduces technical risk
- Our risk management limits position sizes and exposure
- The system has stop losses on every trade

### "Crypto is too volatile / risky"
Acknowledge: Crypto futures are high-risk instruments.
Clarify: This system uses:
- 4-5x leverage (moderate for futures)
- ATR-based stops (adapt to volatility)
- Portfolio-level risk limits
- Multiple uncorrelated strategies

### "What happens in a market crash?"
Honest answer:
- System would take losses
- Stop losses limit per-trade loss to ~1-2%
- Max position limits cap total exposure
- We've included the Feb 2026 market dip in our results
`}
</ObjectionHandling>
```

### 5.4 Data Citation Guidelines

```mdx
<CitationGuidelines>
{`
## How to Cite Data

When answering questions, reference specific data points:

✅ Good: "Cucurbit has a 47.8% win rate across 203 trades, with total PnL of +$4,241."

❌ Bad: "The strategies perform well."

Always include:
- The specific number
- The metric name
- The context (time period, strategy, etc.)

For counterfactuals, be clear about the hypothetical nature:
"If we had executed all low-confidence signals, the backtest shows we would have lost an additional $1,234."

For positions, note they are live:
"Currently holding 7 positions with +$682 unrealized PnL, as of [timestamp]."
`}
</CitationGuidelines>
```

### 5.5 Interactive AI Console

```mdx
## Ask AI About Cash Town

<AIConsole 
  systemPrompt={agentInstructions}
  dataContext={pageData}
  suggestedQuestions={[
    "How does Cash Town make money?",
    "What's the riskiest part of this system?",
    "Explain the Cucurbit strategy",
    "What signals did you reject today and why?",
    "How would performance change if we removed stop losses?"
  ]}
/>
```

---

## 6. Tech Stack Recommendation

### 6.1 Frontend

**Framework:** Next.js 14+ with App Router
- MDX support built-in
- Server components for data fetching
- Edge functions for real-time updates

**UI Components:**
- Tailwind CSS for styling
- shadcn/ui for component library
- Recharts or Tremor for charts
- TanStack Table for data grids

**MDX Processing:**
- @next/mdx for compilation
- remark-gfm for GitHub-flavored markdown
- Custom components for AgentData, AudienceContext

### 6.2 Data Storage

**Recommended: SQLite + JSON Exports**

**Why SQLite:**
- Zero infrastructure (file-based)
- Fast reads for dashboard queries
- Easy to backup and version control
- Supports full SQL for complex queries

**Schema:**
```sql
-- trades table
CREATE TABLE trades (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL NOT NULL,
  size REAL NOT NULL,
  pnl REAL NOT NULL,
  pnl_pct REAL NOT NULL,
  strategy_id TEXT NOT NULL,
  entry_time TEXT NOT NULL,
  exit_time TEXT NOT NULL,
  close_reason TEXT NOT NULL,
  source TEXT DEFAULT 'live'
);

-- signals table
CREATE TABLE signals (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  confidence REAL NOT NULL,
  entry_price REAL NOT NULL,
  stop_loss REAL NOT NULL,
  take_profit REAL NOT NULL,
  strategy_id TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  executed INTEGER NOT NULL,
  rejection_reason TEXT,
  trade_id TEXT REFERENCES trades(id)
);

-- counterfactuals table
CREATE TABLE counterfactuals (
  signal_id TEXT PRIMARY KEY REFERENCES signals(id),
  would_be_pnl REAL,
  would_be_pnl_pct REAL,
  exit_reason TEXT,
  duration_minutes INTEGER,
  backtest_time TEXT,
  data_quality TEXT
);

-- price_cache table (for backtest engine)
CREATE TABLE price_cache (
  symbol TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  PRIMARY KEY (symbol, timestamp)
);
```

**JSON Export Pipeline:**
- Cron job generates JSON files every 5 minutes
- JSON files placed in `/data/` for Next.js static generation
- Allows static site deployment (no runtime database)

**Alternative: Supabase**
- If real-time updates become critical
- Postgres with real-time subscriptions
- More complex but more scalable

### 6.3 API Layer

**Lightweight: Next.js API Routes**
```typescript
// app/api/trades/route.ts
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const strategy = searchParams.get('strategy');
  const limit = searchParams.get('limit') ?? '100';
  
  // Query SQLite or read JSON
  const trades = await getTrades({ strategy, limit: parseInt(limit) });
  
  return Response.json(trades);
}
```

**Endpoints:**
```
GET  /api/trades          - List trades (filterable)
GET  /api/trades/:id      - Single trade details
GET  /api/signals         - List signals
GET  /api/signals/:id/counterfactual - Backtest result
GET  /api/positions       - Current positions
GET  /api/performance     - Aggregate metrics
WS   /api/live            - Real-time updates
```

### 6.4 Hosting

**Vercel (Recommended)**
- Free tier sufficient for low-traffic dashboard
- Automatic deployments from GitHub
- Edge functions for API routes
- Built-in analytics

**Deployment Flow:**
```
GitHub Push → Vercel Build → Deploy
                 ↓
          Run data export scripts
          Compile MDX pages
          Generate static JSON
```

### 6.5 Update Frequency

**Tier 1: Real-time (WebSocket)**
- Open positions
- Live equity
- New signals (when generated)

**Tier 2: Near real-time (5-min polling)**
- Trade history
- Performance metrics
- Account snapshots

**Tier 3: Batched (Hourly/Daily)**
- Counterfactual backtests
- Strategy analytics
- Historical aggregations

---

## 7. Implementation Phases

### Phase 1: Data Infrastructure (Week 1-2)

**Objective:** Capture all data needed for analytics

**Tasks:**
1. [ ] Create SQLite database schema
2. [ ] Add SignalLogger to SmartOrchestrator
   - Hook into `receive_signal()`
   - Log all signals with execution status
   - Record rejection reasons
3. [ ] Enhance TradeTracker
   - Link trades to source signals
   - Capture full trade metadata
4. [ ] Set up position sync job
   - Poll KuCoin every minute
   - Update positions_snapshot.json
5. [ ] Create JSON export pipeline
   - Script to dump SQLite → JSON
   - Run every 5 minutes via cron

**Deliverables:**
- `data/signals.json` - All signals
- `data/trades.json` - Enhanced trade records
- `data/positions.json` - Live positions
- `data/performance.json` - Aggregated metrics

**Validation:**
- [ ] No signal is lost (compare signal count to expected)
- [ ] All trades link to signals
- [ ] JSON files update correctly

---

### Phase 2: Backtest Engine (Week 2-3)

**Objective:** Enable counterfactual analysis

**Tasks:**
1. [ ] Implement KuCoin price fetcher
   - Fetch 1-minute candles
   - Handle rate limits
   - Cache to SQLite
2. [ ] Build trade simulator
   - Replay signal against price data
   - Calculate would-be PnL
   - Determine exit reason
3. [ ] Create backtest scheduler
   - Queue non-executed signals
   - Process after 24-hour delay
   - Store results in DB
4. [ ] Build counterfactual aggregator
   - Calculate summary metrics
   - Group by rejection reason
   - Identify patterns

**Deliverables:**
- `CounterfactualBacktester` class
- `data/counterfactuals.json`
- Counterfactual summary endpoint

**Validation:**
- [ ] Backtest results match manual calculation
- [ ] Price cache is complete
- [ ] Edge cases handled (gaps, missing data)

---

### Phase 3: Dashboard MVP (Week 3-4)

**Objective:** Basic functional dashboard

**Tasks:**
1. [ ] Set up Next.js project with MDX
2. [ ] Create data loading layer
   - Static JSON import
   - API routes for dynamic data
3. [ ] Build core components
   - EquityCurve chart
   - TradeTable with filters
   - StrategyCard summaries
   - PositionList (live)
4. [ ] Implement pages
   - Overview dashboard
   - Trade history
   - Strategy breakdown
5. [ ] Add basic interactivity
   - Date range filter
   - Strategy filter
   - CSV export

**Deliverables:**
- Deployed dashboard on Vercel
- Working performance view
- Trade browsing capability

**Validation:**
- [ ] Data matches source files
- [ ] Filters work correctly
- [ ] Mobile responsive

---

### Phase 4: Agentic Docs Layer (Week 4-5)

**Objective:** AI-accessible documentation

**Tasks:**
1. [ ] Create AgentData component
   - Hidden JSON-LD blocks
   - Type annotations
2. [ ] Write behavioral instructions
   - Core AI guidelines
   - Audience personas
   - Objection handling
3. [ ] Add AgentData to all pages
   - Overview metrics
   - Per-trade data
   - Strategy descriptions
4. [ ] Implement methodology pages
   - How it works
   - Strategy docs
   - Risk management
5. [ ] Create "Copy for AI" feature
   - Clipboard-ready export
   - Include relevant context

**Deliverables:**
- Dual-layer MDX pages
- AI behavioral instructions
- Clipboard export function

**Validation:**
- [ ] Paste into Claude/ChatGPT works
- [ ] AI correctly interprets data
- [ ] Different audiences get appropriate responses

---

### Phase 5: Polish and Deploy (Week 5-6)

**Objective:** Production-ready release

**Tasks:**
1. [ ] Add real-time updates
   - WebSocket connection
   - Live position updates
   - Trade notifications
2. [ ] Implement AI console
   - Embedded chat interface
   - Pre-loaded context
   - Suggested questions
3. [ ] Polish UI/UX
   - Loading states
   - Error handling
   - Animations
4. [ ] Add authentication (if needed)
   - Protect sensitive data
   - API key for programmatic access
5. [ ] Documentation
   - README
   - API docs
   - Deployment guide
6. [ ] Final testing
   - Cross-browser
   - Mobile
   - AI interaction testing

**Deliverables:**
- Production dashboard
- Full documentation
- Monitoring setup

---

## 8. Open Questions

### 8.1 Data & Privacy

1. **Signal exposure:** Should non-executed signals be public?
   - They reveal strategy behavior
   - Could be valuable competitive intelligence
   - **Suggestion:** Make optional, default to hidden

2. **Position visibility:** Show live positions publicly?
   - Reveals current exposure
   - Could be front-run theoretically
   - **Suggestion:** Delayed by 1 hour, or require auth

3. **Historical depth:** How far back to show trades?
   - Full history vs rolling window
   - **Suggestion:** Show all, it's a short track record anyway

### 8.2 Technical

4. **Database choice:** SQLite vs Supabase?
   - SQLite: Simpler, cheaper, good enough for now
   - Supabase: Better for real-time, scales better
   - **Suggestion:** Start SQLite, migrate if needed

5. **Update frequency:** How real-time is "real-time"?
   - True WebSocket streaming vs polling
   - **Suggestion:** 5-second polling for positions, true WS later

6. **AI hosting:** Embed AI console or external only?
   - Embedding requires API costs
   - External (copy-paste) is free but less smooth
   - **Suggestion:** Start with copy-paste, add embedded later

### 8.3 Product

7. **Target audience:** Who is this for?
   - Internal monitoring?
   - Public marketing?
   - Investor due diligence?
   - **Impacts:** Design, privacy, detail level

8. **Branding:** Cash Town vs separate brand?
   - Dashboard URL/domain
   - Visual identity
   - **Suggestion:** cashtown-analytics.vercel.app or similar

9. **Interactivity:** How much filtering/exploration?
   - Simple overview vs deep analysis tool
   - **Suggestion:** Start simple, add based on usage

### 8.4 Counterfactuals

10. **Backtest assumptions:** How to handle slippage?
    - Assume perfect fills?
    - Add slippage estimate?
    - **Suggestion:** Document assumption, no slippage for v1

11. **Time window:** How long to wait before backtesting?
    - 1 hour (quick feedback) vs 24 hours (complete)
    - **Suggestion:** 24 hours default, optional quick mode

12. **Confidence threshold:** What counts as "correctly rejected"?
    - Would-be loss of any amount?
    - Would-be loss > fees?
    - **Suggestion:** Would-be loss after fees

---

## Appendix A: Current Data Samples

### A.1 Trade Record Example
```json
{
  "id": "3065ede8-110c-810c-b0b1-dbcec54bf9be",
  "symbol": "SOLUSDTM",
  "side": "short",
  "entry_price": 535.2,
  "exit_price": 535.3,
  "size": 497.74,
  "pnl": -0.09,
  "pnl_pct": -0.02,
  "strategy_id": "cucurbit",
  "entry_time": "2026-02-13T15:13:00.000+00:00",
  "exit_time": "2026-02-13T15:14:00.000+00:00",
  "close_reason": "sl",
  "source": "notion_seed"
}
```

### A.2 Strategy Performance Example
```json
{
  "cucurbit": {
    "trades": 203,
    "wins": 97,
    "losses": 106,
    "total_pnl": 4240.6,
    "win_rate": 0.4778,
    "avg_pnl": 20.89,
    "max_win": 504.89,
    "max_loss": -87.61
  }
}
```

### A.3 Position Snapshot Example
```json
{
  "symbol": "OPUSDTM",
  "side": "short",
  "size": 115593.0,
  "entry_price": 0.1474,
  "mark_price": 0.142,
  "liquidation_price": 0.1749,
  "unrealized_pnl": 625.73,
  "unrealized_pnl_pct": 3.67,
  "leverage": 4.08,
  "strategy": "manual"
}
```

---

## Appendix B: Reference Implementation

### B.1 SignalLogger Integration

```python
# In orchestrator/smart_orchestrator.py

class SmartOrchestrator:
    def __init__(self, ...):
        ...
        self.signal_logger = SignalLogger(db_path="data/analytics.db")
    
    async def receive_signal(self, strategy_id: str, signal: Signal):
        # Existing logic...
        
        # Log the signal regardless of outcome
        self.signal_logger.log_signal(
            signal=signal,
            strategy_id=strategy_id,
            received_at=datetime.utcnow()
        )
        
        # After aggregation
        result = self.aggregator.aggregate([signal])
        
        # Update with execution status
        self.signal_logger.update_execution_status(
            signal_id=signal.id,
            executed=result.executed,
            rejection_reason=result.rejection_reason
        )
```

### B.2 Counterfactual Backtest

```python
# In analytics/backtest.py

class CounterfactualBacktester:
    def __init__(self, price_fetcher: PriceFetcher):
        self.price_fetcher = price_fetcher
    
    async def backtest_signal(self, signal: Signal) -> CounterfactualResult:
        # Fetch price data
        candles = await self.price_fetcher.get_candles(
            symbol=signal.symbol,
            start=signal.generated_at,
            end=signal.generated_at + timedelta(hours=24)
        )
        
        if not candles:
            return CounterfactualResult(
                signal_id=signal.id,
                data_quality="missing"
            )
        
        # Simulate trade
        entry_price = candles[0].close
        
        for candle in candles[1:]:
            # Check exits
            if signal.side == "short":
                if candle.high >= signal.stop_loss:
                    return self._build_result(signal, entry_price, signal.stop_loss, "sl", candle)
                if candle.low <= signal.take_profit:
                    return self._build_result(signal, entry_price, signal.take_profit, "tp", candle)
            else:
                if candle.low <= signal.stop_loss:
                    return self._build_result(signal, entry_price, signal.stop_loss, "sl", candle)
                if candle.high >= signal.take_profit:
                    return self._build_result(signal, entry_price, signal.take_profit, "tp", candle)
        
        # Timeout
        exit_price = candles[-1].close
        return self._build_result(signal, entry_price, exit_price, "timeout", candles[-1])
```

---

## Appendix C: Alkimi Study Reference

The Alkimi IAB study (https://alkimi-study.vercel.app/) demonstrates the dual-layer pattern:

**Key patterns to adopt:**
1. **Clipboard-first:** Primary interaction is copy → paste into AI
2. **Structured JSON:** All data in parseable format
3. **Behavioral guidance:** Instructions for how AI should present info
4. **Audience awareness:** Different responses for different users
5. **Objection handling:** Preemptive answers to skeptical questions

**Differences for Cash Town:**
- Real-time data (Alkimi study is static)
- Interactive filtering (study is read-only)
- Live positions (study has none)
- Counterfactual analysis (unique to Cash Town)

---

*End of Specification*

*Next step: Review with Chorley, resolve open questions, begin Phase 1.*
