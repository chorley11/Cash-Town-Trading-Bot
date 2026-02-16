# 🚀 Quick Start Guide

Get Cash Town running in under 10 minutes.

---

## Prerequisites

- Python 3.9 or higher
- KuCoin Futures account with API credentials
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/cash-town.git
cd cash-town
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `requests` - HTTP client for KuCoin API
- `numpy` - Numerical computations for indicators
- `python-dotenv` - Environment variable management (optional)

---

## 3. Configure API Credentials

### Option A: Environment Variables

```bash
export KUCOIN_API_KEY="your-api-key"
export KUCOIN_API_SECRET="your-api-secret"
export KUCOIN_PASSPHRASE="your-passphrase"
```

### Option B: .env File

Create `.env` in the project root:

```
KUCOIN_API_KEY=your-api-key
KUCOIN_API_SECRET=your-api-secret
KUCOIN_PASSPHRASE=your-passphrase
DATA_DIR=/app/data
PORT=8888
```

---

## 4. Set Data Directory

Cash Town stores learning data (signals, trades, counterfactuals) in `DATA_DIR`:

```bash
export DATA_DIR=/path/to/data
mkdir -p $DATA_DIR
```

Default: `/app/data`

---

## 5. Run in Paper Mode (Recommended First)

Paper mode simulates trades without real money:

```bash
python run_cloud_v2.py
```

You should see:

```
============================================================
💰 CASH TOWN v2 - LEARNING-FIRST TRADING
============================================================
Mode: 📝 PAPER
Port: 8888
Philosophy: MAKE MONEY. Learn from P&L, not rules.
Data dir: /app/data
============================================================
🌐 HTTP server started on port 8888
🤖 Starting 8 strategy agents...
  ✅ Started trend-following
  ✅ Started mean-reversion
  ✅ Started turtle
  ✅ Started weinstein
  ✅ Started livermore
  ✅ Started bts-lynch
  ✅ Started zweig
  ✅ Started rsi-divergence
🤖 8 agents running
⚡ Starting executor (PAPER)...
  ✅ Executor running
```

---

## 6. Verify It's Working

### Check Health

```bash
curl http://localhost:8888/health
```

Response:
```json
{"status": "healthy"}
```

### Check Risk Status

```bash
curl http://localhost:8888/risk
```

### Check Signals

```bash
curl http://localhost:8888/signals
```

### Check Performance

```bash
curl http://localhost:8888/perf
```

---

## 7. Run in Live Mode

⚠️ **Only after verifying paper mode works correctly!**

```bash
python run_cloud_v2.py --live
```

Or set environment variable:

```bash
export LIVE_MODE=true
python run_cloud_v2.py
```

---

## 8. Monitor the System

### Logs

Logs are printed to stdout. In production, redirect to a file:

```bash
python run_cloud_v2.py 2>&1 | tee cash-town.log
```

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Basic health check |
| `/perf` | Cycle times, memory usage |
| `/risk` | Risk manager status |
| `/can_trade` | Circuit breaker check |
| `/signals` | Current actionable signals |
| `/learning` | Strategy performance |
| `/watchdog` | Self-improvement status |

### Example: Check if Circuit Breaker is Active

```bash
curl http://localhost:8888/can_trade | jq
```

---

## 9. Deploy to Railway

### Setup

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set environment variables in Railway dashboard

### Environment Variables (Railway Dashboard)

| Variable | Value |
|----------|-------|
| `KUCOIN_API_KEY` | Your API key |
| `KUCOIN_API_SECRET` | Your API secret |
| `KUCOIN_PASSPHRASE` | Your passphrase |
| `DATA_DIR` | `/app/data` |
| `LIVE_MODE` | `true` (for live trading) |
| `PORT` | Leave blank (Railway assigns) |

### Configuration Files

**Procfile** (included):
```
web: python run_cloud_v2.py
```

**railway.json** (included):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {"builder": "NIXPACKS"},
  "deploy": {"startCommand": "python run_cloud_v2.py --live"}
}
```

### Deploy

Push to GitHub → Railway auto-deploys

---

## 10. Common Issues

### "No module named X"

```bash
pip install -r requirements.txt
```

### "Invalid API credentials"

1. Check environment variables are set correctly
2. Ensure API has Futures trading permissions
3. Check IP whitelist on KuCoin (if enabled)

### "Circuit breaker triggered"

Check why:
```bash
curl http://localhost:8888/can_trade | jq
```

Wait for cooldown or investigate the cause.

### "Signals empty"

Normal during low-volatility periods. The system only signals when strategies detect opportunities.

Check strategy status:
```bash
curl http://localhost:8888/learning | jq '.strategy_performance'
```

### Memory growing

Check `/perf` endpoint. If memory growth is concerning:
1. Check for log files filling disk
2. Restart the process (learning data is persisted)
3. Review `DATA_DIR` for large files

---

## 11. Next Steps

1. **Monitor for 24-48 hours** in paper mode
2. **Review `/watchdog`** for alerts and recommendations
3. **Start with small position sizes** in live mode
4. **Gradually increase** as you gain confidence
5. **Read the docs**:
   - [ARCHITECTURE.md](ARCHITECTURE.md) - How it works
   - [STRATEGIES.md](STRATEGIES.md) - Strategy details
   - [RISK.md](RISK.md) - Risk controls
   - [API.md](API.md) - All endpoints

---

## Project Structure

```
cash-town/
├── run_cloud_v2.py      # Main entry point
├── orchestrator/        # Signal selection & risk
├── agents/              # Strategy implementations
├── execution/           # Order execution
├── api/                 # Dashboard API
├── utils/               # Validation & monitoring
├── data/                # Runtime data (gitignored)
├── docs/                # Documentation
├── tests/               # Test suite
├── requirements.txt     # Python dependencies
├── Procfile             # Railway process definition
└── railway.json         # Railway config
```

---

## Getting Help

1. Check the [API documentation](API.md)
2. Review logs for error messages
3. Check `/watchdog/alerts` for system issues
4. File a GitHub issue with details

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design context.
