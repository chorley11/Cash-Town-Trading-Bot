# 💰 Cash Town — Intelligent Multi-Strategy Trading System

**Family Office Grade Automated Trading Infrastructure**

Cash Town is a self-improving multi-agent trading system designed for institutional-quality portfolio management. It coordinates multiple strategy agents, manages positions through a centralised risk framework, and continuously learns from market outcomes.

[![Railway Deployment](https://img.shields.io/badge/Railway-Deployed-blueviolet)](https://railway.app)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Proprietary-red)]()

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Strategy Suite](#-strategy-suite)
- [Risk Management](#-risk-management)
- [API Reference](#-api-reference)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Documentation](#-documentation)

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Agent Orchestration** | 8 independent strategy agents generating signals in parallel |
| **Intelligent Signal Selection** | Aggregator ranks, filters, and deconflicts signals |
| **Second Chance Logic** | Rescues promising signals initially rejected due to conservative thresholds |
| **Centralised Risk Manager** | Kelly Criterion sizing, correlation tracking, circuit breakers |
| **Profit Watchdog** | Self-improving feedback loop that tracks decisions vs outcomes |
| **Counterfactual Learning** | Tracks what rejected signals would have done—learns from mistakes |
| **Dynamic Multipliers** | Strategy position sizes scale with actual P&L track record |
| **Security Hardening** | Input validation, sanitisation, rate limiting |
| **Performance Monitoring** | Real-time cycle timing, memory tracking, `/perf` endpoint |

### What's New (February 2025)

- ✅ **Learning-first approach** — Removed arbitrary signal limits and cooldowns; the bot learns optimal behaviour from P&L
- ✅ **Synced strategy R:R fix** — All strategies now use 8% SL / 20% TP (2.5:1 reward-to-risk)
- ✅ **Zweig strategy rewrite** — Thrust detection, ADX filter, volume gate (no longer disabled)
- ✅ **RSI Divergence** — New strategy catching early reversals via price/RSI divergence
- ✅ **Second Chance logic** — Rescues promising signals that barely missed thresholds
- ✅ **Drawdown protection** — 50% position size reduction when account drops 10%
- ✅ **Profit Watchdog** — Monitors every decision, generates alerts and auto-tune recommendations
- ✅ **Risk Manager** — Kelly Criterion, correlation limits, circuit breakers
- ✅ **Security hardening** — Input validation, dangerous pattern detection
- ✅ **Performance monitoring** — `/perf` endpoint with cycle times and memory stats

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SMART ORCHESTRATOR                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Signal Aggregator│  │  Risk Manager   │  │  Profit Watchdog    │  │
│  │ - Rank & filter  │  │ - Kelly sizing  │  │ - Track decisions   │  │
│  │ - Consensus      │  │ - Correlation   │  │ - Drift detection   │  │
│  │ - Second Chance  │  │ - Circuit break │  │ - Auto-tune         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ Trend Following│         │ Mean Reversion │         │     Turtle    │
│    ⭐ STAR     │         │               │         │               │
└───────────────┘         └───────────────┘         └───────────────┘
        │                         │                         │
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   Weinstein   │         │   Livermore   │         │   BTS Lynch   │
└───────────────┘         └───────────────┘         └───────────────┘
        │                         │                         │
┌───────────────┐         ┌───────────────┐
│  Zweig v2 🔧  │         │ RSI Divergence│
│   (FIXED)     │         │   ✨ NEW      │
└───────────────┘         └───────────────┘
        │                         │
        └─────────────────────────┴─────────────────────────┐
                                                            ▼
                                                 ┌──────────────────┐
                                                 │ Execution Engine │
                                                 │ - KuCoin Futures │
                                                 │ - Position Track │
                                                 │ - P&L Recording  │
                                                 └──────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

---

## 📊 Strategy Suite

| Strategy | Based On | Style | Status | Default Multiplier |
|----------|----------|-------|--------|-------------------|
| **Trend Following** ⭐ | MA crossovers + ADX | Momentum | **STAR** | 1.5× |
| **Mean Reversion** | Bollinger Bands + RSI | Fade extremes | Active | 1.0× |
| **Turtle** | Richard Dennis | 20-day breakouts | Active | 1.0× |
| **Weinstein** | Stage Analysis | Buy Stage 2 | Active | 1.0× |
| **Livermore** | Jesse Livermore | Pivotal points | Active | 1.0× |
| **BTS Lynch** | Peter Lynch | High-momentum | Active | 0.8× |
| **Zweig v2** 🔧 | Martin Zweig | Breadth thrust | **FIXED** | 0.7× |
| **RSI Divergence** ✨ | Price/RSI divergence | Early reversal | **NEW** | 1.0× |

All strategies use ATR-based stops with **8% SL / 20% TP** (2.5:1 R:R).

See [docs/STRATEGIES.md](docs/STRATEGIES.md) for detailed strategy specifications.

---

## 🛡 Risk Management

### Position Sizing
- **Kelly Criterion** (25% Kelly fraction) for strategies with 20+ trades
- **Fixed Fractional** (2% max risk) as fallback
- **Confidence scaling** (0.5× to 1.0× based on signal strength)
- **Drawdown protection** (50% size reduction at 10% account drop)

### Portfolio Controls
- Max 10% portfolio at risk simultaneously
- Max 4% exposure per correlation group
- Max 4 positions in same direction (all long or all short)

### Circuit Breakers
- **Daily Loss**: Halt at 5% daily loss (4-hour cooldown)
- **Max Drawdown**: Halt at 15% drawdown from peak
- Auto-reset on new trading day or after cooldown

See [docs/RISK.md](docs/RISK.md) for complete risk framework.

---

## 🔌 API Reference

### Health & Status
```
GET /health           → {"status": "healthy"}
GET /perf             → Performance metrics (cycle times, memory)
GET /risk             → Risk manager status
GET /can_trade        → Circuit breaker check
```

### Signals & Learning
```
GET /signals          → Get aggregated actionable signals
GET /learning         → Learning summary (strategy performance)
GET /multipliers      → Dynamic strategy multipliers
GET /counterfactual   → Counterfactual analysis results
GET /rescue_stats     → Second-chance rescue statistics
```

### Profit Watchdog
```
GET /watchdog              → Full watchdog status
GET /watchdog/decisions    → Recent decisions with outcomes
GET /watchdog/alerts       → Active alerts
GET /watchdog/recommendations → Parameter tuning suggestions
GET /watchdog/drift        → Strategy drift analysis
```

### Signal Submission
```
POST /signals         → Submit signal from strategy agent
POST /trade_result    → Record trade outcome for learning
```

See [docs/API.md](docs/API.md) for complete API documentation with examples.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- KuCoin Futures API credentials

### Installation

```bash
git clone https://github.com/your-org/cash-town.git
cd cash-town
pip install -r requirements.txt
```

### Environment Variables

```bash
export KUCOIN_API_KEY="your-api-key"
export KUCOIN_API_SECRET="your-api-secret"
export KUCOIN_PASSPHRASE="your-passphrase"
export DATA_DIR="/app/data"
export PORT=8888
```

### Run

```bash
# Paper trading (default)
python run_cloud_v2.py

# Live trading
python run_cloud_v2.py --live
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup guide.

---

## ⚙ Configuration

### Orchestrator Config (AggregatorConfig)

```python
AggregatorConfig(
    min_confidence=0.55,      # Minimum signal confidence
    min_consensus=1,          # Minimum agreeing strategies
    max_signals_per_cycle=99, # Effectively unlimited
    cooldown_minutes=0,       # No cooldown (learn from data)
)
```

### Risk Config

```python
RiskConfig(
    max_position_pct=2.0,         # Max 2% equity per position
    max_total_exposure_pct=20.0,  # Max 20% total exposure
    max_positions=5,              # Max concurrent positions
    max_daily_loss_pct=5.0,       # Daily loss circuit breaker
    drawdown_threshold_pct=10.0,  # Drawdown protection trigger
    drawdown_reduction_factor=0.5 # 50% size reduction in drawdown
)
```

---

## 🚢 Deployment

### Railway (Recommended)

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy with included `railway.json` and `Procfile`

```json
// railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {"builder": "NIXPACKS"},
  "deploy": {"startCommand": "python run_cloud_v2.py --live"}
}
```

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_cloud_v2.py"]
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design overview |
| [STRATEGIES.md](docs/STRATEGIES.md) | All strategies with parameters |
| [API.md](docs/API.md) | Complete API reference |
| [RISK.md](docs/RISK.md) | Risk management rules |
| [QUICKSTART.md](docs/QUICKSTART.md) | Getting started guide |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history |

---

## 🏛 Project Structure

```
cash-town/
├── orchestrator/
│   ├── smart_orchestrator.py   # Main brain
│   ├── signal_aggregator.py    # Signal ranking/filtering
│   ├── risk_manager.py         # Central risk control
│   ├── profit_watchdog.py      # Self-improvement loop
│   └── second_chance.py        # Rescue promising rejects
├── agents/
│   ├── base.py                 # Base agent class
│   ├── runner.py               # Agent execution runner
│   └── strategies/             # Strategy implementations
├── execution/
│   ├── engine.py               # Execution engine
│   ├── kucoin.py               # KuCoin API client
│   └── strategy_tracker.py     # Position attribution
├── api/
│   └── endpoints.py            # Dashboard API
├── utils/
│   ├── validation.py           # Input sanitisation
│   └── monitoring.py           # Performance tracking
├── data/                       # Runtime data storage
├── docs/                       # Documentation
└── tests/                      # Test suite
```

---

## 📜 License

Proprietary. Family office use only.

---

**Built for serious traders. No shortcuts.**
