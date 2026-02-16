# 📜 Changelog

All notable changes to Cash Town are documented in this file.

---

## [2.0.0] - 2025-02-16

### Major Release: Learning-First Architecture

This release fundamentally shifts Cash Town from a rules-based system to a learning-first system. The bot now learns optimal behaviour from actual P&L rather than from predetermined rules.

### Added

#### New Strategy: RSI Divergence
- Detects price/RSI divergences for early reversal signals
- Bullish divergence: Price lower low + RSI higher low → LONG
- Bearish divergence: Price higher high + RSI lower high → SHORT
- Complements trend-following by catching reversals early
- Parameters: 14-period RSI, 20-bar lookback, 2% divergence threshold

#### Second Chance Logic
- Rescues promising signals that were initially rejected
- Boosts confidence based on:
  - Strategy track record (+8% for trend-following)
  - Multi-strategy consensus (+3% per agreeing strategy)
  - Historical "winning rejection" patterns (+6-12%)
  - Strong ADX (+5%)
- Penalties for known underperformers (-10% for zweig)
- Tracks near-miss signals for counterfactual analysis

#### Centralised Risk Manager
- Kelly Criterion position sizing (25% Kelly fraction)
- Fixed fractional fallback for new strategies
- Portfolio heat tracking (max 10% at risk)
- Correlation group limits (max 4% per group)
- Direction concentration limits (max 4 same direction)
- Volatility scaling (50-75% reduction in high vol)

#### Circuit Breakers
- Daily loss circuit breaker at 5% (4-hour cooldown)
- Max drawdown circuit breaker at 15%
- Auto-reset on new trading day
- Exposed via `GET /can_trade` endpoint

#### Drawdown Protection
- Separate from circuit breaker
- 50% position size reduction when account drops 10%
- Allows continued trading while recovering

#### Dynamic Strategy Multipliers
- Position sizes scale with strategy track record
- <35% win rate → disabled (0.0×)
- 35-45% win rate → reduced (0.5×)
- ≥50% win rate + positive P&L → boosted (up to 2.0×)
- Updates after each trade closes

#### Profit Watchdog
- Monitors every orchestrator decision (accept/reject)
- Tracks outcomes vs predictions
- Detects strategy performance drift
- Generates alerts for underperformance
- Recommends parameter adjustments
- Optional auto-tune based on data

#### Performance Monitoring
- Cycle execution timing
- Memory tracking via tracemalloc
- `/perf` endpoint for metrics
- Alerts for slow cycles (>4 min)
- Alerts for memory growth (>100MB)
- Periodic garbage collection

#### Security Hardening
- Input validation for all signal submissions
- Symbol format validation
- Confidence range checking
- String sanitisation (dangerous pattern removal)
- Request size limits (50KB signals, 10KB trade results)

### Changed

#### Learning-First Approach
- **REMOVED** arbitrary signal limits (`max_signals_per_cycle=99`)
- **REMOVED** cooldown periods (`cooldown_minutes=0`)
- Bot now learns when re-entry is good/bad from actual results
- Counterfactual tracking shows what rejected signals would have done

#### Synced Strategy R:R Fix
- All strategies now use consistent ATR-based stops
- Standard: 8% SL / 20% TP (2.5:1 reward-to-risk)
- Previously: Each strategy had different (often bad) R:R

#### Zweig Strategy Complete Rewrite
- **Problem**: 14% win rate, trading against trends, bad R:R
- **Fix 1**: Signal on ZONE TRANSITIONS only, not static extremes
- **Fix 2**: ADX > 25 required (trend confirmation)
- **Fix 3**: Volume > 1.2× average required
- **Fix 4**: ATR-based stops with 2:1 R:R
- **Fix 5**: Long-biased (Zweig is fundamentally bullish)
- **Fix 6**: 15% confidence penalty for short signals
- Status changed from DISABLED to PROBATIONARY (0.7× multiplier)

### Fixed

- Fixed potential memory leaks in long-running processes
- Fixed signal duplication when same symbol from multiple strategies
- Fixed position tracking desync with exchange

### API Changes

#### New Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/perf` | GET | Performance monitoring metrics |
| `/risk` | GET | Risk manager status |
| `/can_trade` | GET | Circuit breaker check |
| `/rescue_stats` | GET | Second Chance statistics |
| `/watchdog` | GET | Full watchdog status |
| `/watchdog/decisions` | GET | Recent decisions with outcomes |
| `/watchdog/alerts` | GET | Active alerts |
| `/watchdog/recommendations` | GET | Parameter suggestions |
| `/watchdog/drift` | GET | Strategy drift analysis |

#### Modified Endpoints
| Endpoint | Change |
|----------|--------|
| `/signals` | Now includes `risk_position_size` and `risk_meta` |
| `/learning` | Now includes `risk_manager` and `second_chance` sections |

### Documentation

- Complete README rewrite with current features
- New `docs/ARCHITECTURE.md` - System design overview
- New `docs/STRATEGIES.md` - All strategies with parameters
- New `docs/API.md` - Complete API reference
- New `docs/RISK.md` - Risk management rules
- New `docs/QUICKSTART.md` - Getting started guide
- New `docs/CHANGELOG.md` - This file

---

## [1.5.0] - 2025-02-13

### Added
- Stat Arb strategy for pairs trading
- Backtesting framework
- CLI tool (`cashctl`)
- Position rotation logic

### Changed
- Moved to Railway deployment
- Updated symbol universe to 24 large-caps

---

## [1.0.0] - 2025-02-10

### Initial Release

- Multi-agent architecture with 6 strategies
- Signal aggregation and deduplication
- KuCoin Futures integration
- Paper and live trading modes
- Basic risk management (position limits)

### Strategies
- Trend Following
- Mean Reversion
- Turtle
- Weinstein
- Livermore
- BTS Lynch

---

## Migration Guide

### From 1.x to 2.0

1. **Data Directory**: Ensure `DATA_DIR` environment variable is set
2. **New Dependencies**: Run `pip install -r requirements.txt`
3. **Config Changes**:
   - `AggregatorConfig.max_signals_per_cycle` now defaults to 99
   - `AggregatorConfig.cooldown_minutes` now defaults to 0
4. **API Changes**: Update any integrations using old endpoint formats
5. **Risk Config**: Review and adjust `RiskConfig` parameters

### Breaking Changes

- `POST /signals` now validates input more strictly
- Invalid symbols will be rejected (must match `^[A-Z0-9]{2,10}USDTM$`)
- Confidence must be between 0.0 and 1.0
- Request bodies larger than 50KB are rejected

---

## Upcoming

### Planned for 2.1.0
- [ ] Dashboard UI completion
- [ ] WebSocket real-time updates
- [ ] Multi-account support
- [ ] Enhanced backtesting with slippage simulation

### Planned for 2.2.0
- [ ] PostgreSQL for learning data
- [ ] Prometheus metrics export
- [ ] Alert webhooks (Slack, Discord)
- [ ] Strategy hotswap (add/remove without restart)
