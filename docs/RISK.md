# 🛡 Risk Management Framework

Complete documentation of Cash Town's risk controls.

---

## Philosophy

**Capital preservation first, returns second.**

Every signal passes through centralised risk controls before execution. The Risk Manager has absolute authority to veto any trade that would violate portfolio-level constraints.

---

## Position Sizing

### Primary: Kelly Criterion

For strategies with 20+ trades of history, position sizes are calculated using the Kelly Criterion:

```
Kelly % = W - (1-W) / R

Where:
  W = Win rate
  R = Win/Loss ratio (average win ÷ average loss)
```

**Safety Adjustments:**
- Use only 25% of calculated Kelly (conservative Kelly fraction)
- Cap at max 2% portfolio risk per position
- Minimum $10 position size

**Example:**
```
Strategy: trend-following
Win Rate: 56% (28/50)
Avg Win: 2.5%
Avg Loss: 1.8%
R = 2.5 / 1.8 = 1.39

Kelly % = 0.56 - (1-0.56) / 1.39
Kelly % = 0.56 - 0.317
Kelly % = 24.3%

Conservative Kelly (25%): 24.3% × 0.25 = 6.1%
Capped: min(6.1%, 2.0%) = 2.0%

With $10,000 equity and 2% stop distance:
Position = $10,000 × 2.0% / 2.0% = $10,000 × 1 = $1,000 notional
```

### Fallback: Fixed Fractional

For strategies with <20 trades:

```
Risk Amount = Equity × Max Position Risk %
Position Size = Risk Amount ÷ (Stop Distance %)

Default: 2% max risk per position
```

---

## Confidence Scaling

Position sizes are scaled by signal confidence:

| Confidence | Multiplier |
|------------|------------|
| 50% | 0.50× |
| 60% | 0.55× |
| 70% | 0.60× |
| 80% | 0.65× |
| 90% | 0.70× |
| 100% | 1.00× |

**Formula:** `multiplier = 0.5 + 0.5 × confidence`

---

## Volatility Scaling

Position sizes are reduced in high volatility environments:

| Regime | Criteria | Multiplier |
|--------|----------|------------|
| Low | Vol < 1% | 1.0× |
| Normal | 1% ≤ Vol < 3% | 1.0× |
| High | 3% ≤ Vol < 6% | 0.5× (50% reduction) |
| Extreme | Vol ≥ 6% | 0.25× (75% reduction) |

**Volatility Calculation:**
- Standard deviation of returns over 24 hours
- Updated from price history on each cycle

---

## Portfolio Heat Tracking

### Total Risk Limit

**Maximum 10% of portfolio at risk simultaneously.**

```python
portfolio_heat.total_risk_pct = sum(position.risk_pct for each position)

if total_risk_pct >= max_total_risk_pct × 0.9:
    portfolio_heat.is_overheated = True
```

When approaching the limit, new position sizes are scaled down:

```python
headroom = max_risk_pct - current_risk_pct
if headroom < max_position_risk_pct:
    heat_mult = headroom / max_position_risk_pct
    position_value *= heat_mult
```

---

## Correlation Groups

Assets are grouped by correlation to prevent overexposure:

| Group | Assets |
|-------|--------|
| `btc_ecosystem` | XBTUSDTM, BTCUSDTM |
| `eth_ecosystem` | ETHUSDTM |
| `alt_l1` | SOLUSDTM, AVAXUSDTM, NEARUSDTM, APTUSDTM, SUIUSDTM, TONUSDTM, ICPUSDTM |
| `defi` | UNIUSDTM, LINKUSDTM, INJUSDTM |
| `l2` | MATICUSDTM, ARBUSDTM, OPUSDTM |
| `cosmos` | ATOMUSDTM, TIAUSDTM |
| `old_guard` | LTCUSDTM, BCHUSDTM, XRPUSDTM, ADAUSDTM, DOTUSDTM |
| `storage` | FILUSDTM, RENDERUSDTM |

**Maximum 4% exposure per correlation group.**

If adding a position would exceed group limit:
```python
remaining = max_correlated_exposure_pct - current_group_exposure
if remaining < max_position_risk_pct:
    corr_mult = max(0, remaining / max_position_risk_pct)
    position_value *= corr_mult
```

---

## Direction Concentration

**Maximum 4 positions in the same direction.**

Prevents the portfolio from becoming:
- All long (vulnerable to market crash)
- All short (vulnerable to short squeeze)

```python
if side == 'long' and long_count >= max_same_direction_positions:
    return False, "Max long positions reached"
```

---

## Circuit Breakers

### Daily Loss Circuit Breaker

**Triggers at 5% daily loss.**

```python
daily_loss_pct = -daily_pnl / starting_equity × 100

if daily_loss_pct >= 5.0:
    trigger_circuit_breaker("Daily loss limit")
```

**Cooldown:** 4 hours
**Auto-Reset:** On new trading day

### Maximum Drawdown Circuit Breaker

**Triggers at 15% drawdown from peak equity.**

```python
current_drawdown_pct = (peak_equity - current_equity) / peak_equity × 100

if current_drawdown_pct >= 15.0:
    trigger_circuit_breaker("Max drawdown")
```

**Cooldown:** 4 hours (or until equity recovers)

### Circuit Breaker Behavior

When triggered:
1. All new positions blocked
2. Existing positions remain (no forced liquidation)
3. Cooldown timer starts
4. Log critical alert

After cooldown:
1. Circuit breaker resets
2. Trading resumes normally
3. If condition persists, will trigger again

---

## Drawdown Protection

**Separate from circuit breaker—reduces size, doesn't halt trading.**

When account drops 10% from peak:

```python
if current_drawdown_pct >= 10.0:
    position_size *= 0.5  # 50% reduction
```

This allows continued trading at reduced risk while recovering.

---

## Dynamic Strategy Multipliers

Position sizes are scaled by strategy track record:

| Track Record | Multiplier | Effect |
|--------------|------------|--------|
| Win rate < 35% | 0.0× | **DISABLED** |
| Win rate 35-45% | 0.5× | Half size |
| Negative P&L despite OK WR | 0.7× | Reduced |
| Win rate ≥ 50%, positive P&L | 1.0-2.0× | Boosted |

**Default Multipliers (before learning):**
```python
{
    'trend-following': 1.5,  # Known performer
    'mean-reversion': 1.0,
    'turtle': 1.0,
    'weinstein': 1.0,
    'livermore': 1.0,
    'bts-lynch': 0.8,
    'zweig': 0.7,            # Probationary
    'rsi-divergence': 1.0,
}
```

Multipliers update after each trade closes.

---

## Position Limits

| Limit | Default | Description |
|-------|---------|-------------|
| `max_positions` | 5 | Maximum concurrent positions |
| `max_position_pct` | 2% | Maximum risk per position |
| `max_total_exposure_pct` | 20% | Maximum total notional exposure |
| `max_daily_loss_pct` | 5% | Daily loss circuit breaker |
| `max_drawdown_pct` | 15% | Max drawdown circuit breaker |
| `max_correlated_exposure_pct` | 4% | Per-group limit |
| `max_same_direction_positions` | 4 | Direction concentration |

---

## Stop Loss / Take Profit

### Standard R:R

All strategies use **8% SL / 20% TP** (2.5:1 reward-to-risk) calculated from ATR:

```python
stop_loss_atr = 2.0   # 2 × ATR
take_profit_atr = 5.0 # 5 × ATR (≈2.5× stop distance)

stop_loss = entry_price - (current_atr × stop_loss_atr)  # For longs
take_profit = entry_price + (current_atr × take_profit_atr)
```

### Per-Strategy Variations

| Strategy | SL (ATR mult) | TP (ATR mult) | R:R |
|----------|---------------|---------------|-----|
| Trend Following | 2.0 | 5.0 | 2.5:1 |
| Mean Reversion | 2.0 | 4.0 | 2.0:1 |
| Turtle | 2.0 | 4.0 | 2.0:1 |
| Zweig v2 | 2.0 | 4.0 | 2.0:1 |
| RSI Divergence | 2.0 | 3.5 | 1.75:1 |

---

## Risk Check Flow

Every signal goes through this validation:

```
1. Circuit Breaker Check
   ├── Trading halted? → REJECT
   └── Continue ↓

2. Portfolio Heat Check
   ├── Overheated (≥90% of max)? → REJECT
   └── Continue ↓

3. Correlation Check
   ├── Group at limit? → REJECT
   └── Continue ↓

4. Direction Check
   ├── 4+ positions same direction? → REJECT
   └── Continue ↓

5. Existing Position Check
   ├── Already in position same direction? → REJECT
   ├── Opposite direction? → Will close existing
   └── Continue ↓

6. Calculate Position Size
   ├── Kelly or Fixed Fractional
   ├── × Confidence multiplier
   ├── × Volatility multiplier
   ├── × Heat multiplier
   ├── × Correlation multiplier
   └── × Strategy multiplier

7. Size Validation
   ├── Size < $10? → REJECT
   └── APPROVE with calculated size
```

---

## Monitoring Endpoints

| Endpoint | Data |
|----------|------|
| `GET /risk` | Full risk manager status |
| `GET /can_trade` | Circuit breaker check |
| `GET /watchdog` | Performance and drift analysis |

---

## Configuration

```python
RiskConfig(
    # Position sizing
    max_position_risk_pct=2.0,
    max_total_risk_pct=10.0,
    default_stop_loss_pct=2.0,
    
    # Kelly settings
    use_kelly=True,
    kelly_fraction=0.25,
    min_trades_for_kelly=20,
    
    # Correlation
    max_correlated_exposure_pct=4.0,
    max_same_direction_positions=4,
    
    # Circuit breakers
    max_daily_loss_pct=5.0,
    max_drawdown_pct=15.0,
    circuit_breaker_cooldown_hours=4.0,
    
    # Volatility
    high_vol_reduction=0.5,
    extreme_vol_reduction=0.25,
    vol_lookback_hours=24,
)
```

---

## Best Practices

1. **Don't override circuit breakers manually** — They exist for a reason
2. **Review strategy multipliers weekly** — Adjust based on performance data
3. **Monitor correlation groups** — Rebalance if one group dominates
4. **Check /risk endpoint daily** — Understand current portfolio state
5. **Don't increase position sizes after losses** — Let the system recover naturally
