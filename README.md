# 🏭 Cash Town - Self-Improving Multi-Agent Trading Swarm

A sophisticated trading orchestration system that coordinates multiple strategy agents, actively manages positions, and improves itself over time.

## Features

### 🤖 8 Strategy Agents
Each agent runs independently with its own logic:

| Strategy | Based On | Style | Best Market |
|----------|----------|-------|-------------|
| **Trend Following** | MA crossovers + ADX | Ride momentum | Trending |
| **Mean Reversion** | Bollinger Bands + RSI | Fade extremes | Ranging |
| **Turtle** | Richard Dennis | 20-day breakouts | Any |
| **Weinstein** | Stage Analysis | Buy Stage 2, short Stage 4 | Weekly trends |
| **Livermore** | Jesse Livermore | Pivotal points | All |
| **BTS Lynch** | Peter Lynch | High-momentum picks | Growth |
| **Zweig** | Martin Zweig | Breadth thrust signals | Bottoms |
| **Stat Arb** | Pairs trading | Mean-revert spreads | Correlated pairs |

### 🔄 Active Position Rotation
The orchestrator actively manages positions:
- **Stuck positions**: Never reached profit threshold → close after grace period
- **Fallen positions**: Was winning, now negative → close immediately
- **Stale positions**: Held too long without result → close and rotate
- **Better opportunity**: Replace losing positions with higher-confidence signals

### 🧠 Self-Improvement
Each agent can reflect on its performance and improve:
- **Trade Analysis**: Reviews past trades for patterns
- **Parameter Suggestions**: Identifies parameter changes that might improve results
- **Backtesting**: Tests changes before applying
- **Auto-tuning**: Optionally applies high-confidence improvements

## Installation

```bash
cd ~/.openclaw/workspace/projects/cash-town
pip install -r requirements.txt
```

## Usage

### Start the Orchestrator
```bash
python run.py
# Or
python -m orchestrator.server
```

### CLI Commands

```bash
# Orchestrator status
cashctl status

# List available strategies
cashctl strategies

# Position management
cashctl positions          # View all positions by state
cashctl rotate             # Evaluate which positions to rotate
cashctl close --agent cucurbit --symbol BTCUSDTM  # Force close

# Agent reflection (self-improvement)
cashctl reflect zweig --days 7        # Analyze last 7 days
cashctl reflect bts-lynch --apply     # Apply suggested improvements

# Backtesting
cashctl backtest turtle --days 30 --capital 10000
```

### API Endpoints

```
GET  /health              - Health check
GET  /summary             - Full system summary
GET  /agents              - List all agents
GET  /positions           - Position status by state
GET  /portfolio           - Portfolio summary

POST /agents              - Register agent
POST /refresh             - Trigger health check
POST /rotate              - Evaluate rotations
POST /positions/close     - Force close position

PATCH /rotation-config    - Update rotation settings
```

## Position Rotation Rules

### Rotation Config (defaults)
```json
{
  "grace_period_minutes": 30,      // Don't judge new positions
  "stuck_threshold_pct": 0.5,      // Must reach +0.5% to not be "stuck"
  "stuck_max_minutes": 120,        // Close stuck positions after 2 hours
  "fallen_peak_threshold_pct": 1.0, // Must have been up 1% to be "fallen"
  "fallen_giveback_pct": 80,       // Close if gave back 80% of gains
  "fallen_negative_close": true,   // Close if fallen to negative
  "max_hold_hours": 48,            // Max time to hold any position
  "min_replacement_confidence": 0.6, // Min confidence for replacement signal
  "cooldown_minutes": 15           // Wait before re-entering same symbol
}
```

### Position States
- 🆕 **NEW** - Just opened, in grace period
- ✅ **WINNING** - Currently profitable
- 📉 **LOSING** - Currently at a loss
- 🔒 **STUCK** - Never reached profit, past grace period
- 📉⬇️ **FALLEN** - Was winning, now losing (or negative)
- ⏰ **STALE** - Held too long

## Architecture

```
cash-town/
├── orchestrator/
│   ├── server.py           # Main HTTP server
│   ├── registry.py         # Agent registry
│   ├── health.py           # Health monitoring
│   └── position_manager.py # Position rotation logic
├── agents/
│   ├── base.py             # Base agent class
│   ├── reflection.py       # Self-improvement module
│   ├── backtester.py       # Backtesting engine
│   └── strategies/         # Strategy implementations
├── config/
│   └── agents.json         # Agent configurations
├── cli.py                  # CLI tool
└── run.py                  # Entry point
```

## Integration with Cucurbit

Cash Town is designed to coordinate with your existing Cucurbit bot:
- Cucurbit remains the primary trader with its multi-strategy approach
- Cash Town agents can run alongside as A/B testing
- Eventually, top-performing Cash Town strategies could replace/complement Cucurbit strategies

## Next Steps

1. **Connect historical data** - KuCoin API or CSV for backtesting
2. **Paper trading mode** - Test strategies without real money
3. **Separate KuCoin sub-account** - Isolate Cash Town from Cucurbit
4. **Dashboard** - Visual monitoring of all agents

