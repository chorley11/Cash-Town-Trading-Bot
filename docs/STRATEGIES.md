# 📊 Strategy Documentation

Complete reference for all trading strategies in Cash Town.

---

## Overview

Cash Town runs 8 independent strategy agents in parallel. Each generates signals based on its own logic, which are then aggregated, filtered, and risk-checked by the Smart Orchestrator.

**Common Configuration:**
- All strategies use ATR-based stops
- Standard R:R is 8% SL / 20% TP (2.5:1)
- Signals include confidence (0.0-1.0), entry price, stop loss, take profit

---

## 1. Trend Following ⭐ (STAR)

**Based on:** Moving average crossovers with ADX trend strength confirmation

**Status:** STAR performer (+$208, 51% WR)

**Default Multiplier:** 1.5× (boosted position size)

### Logic

1. **EMA Crossover**: Fast EMA (12) crosses above/below Slow EMA (26)
2. **ADX Filter**: Only trade when ADX > 25 (strong trend)
3. **RSI Confirmation**: RSI supports direction (not overbought for longs, not oversold for shorts)
4. **Volume Confirmation**: Volume above 20-period average

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ema_fast` | 12 | Fast EMA period |
| `ema_slow` | 26 | Slow EMA period |
| `adx_period` | 14 | ADX calculation period |
| `adx_threshold` | 25 | Minimum ADX for signal |
| `rsi_period` | 14 | RSI period |
| `atr_period` | 14 | ATR for stop calculation |
| `stop_loss_atr` | 2.0 | Stop loss = 2 × ATR |
| `take_profit_atr` | 5.0 | Take profit = 5 × ATR |

### Signal Generation

```
LONG when:
  - EMA(12) > EMA(26) AND crosses up
  - ADX > 25
  - RSI < 70
  - Volume > Volume SMA(20)

SHORT when:
  - EMA(12) < EMA(26) AND crosses down
  - ADX > 25
  - RSI > 30
  - Volume > Volume SMA(20)
```

---

## 2. Mean Reversion

**Based on:** Bollinger Bands + RSI extremes

**Status:** Active

**Default Multiplier:** 1.0×

### Logic

1. **Bollinger Band Touch**: Price touches or exceeds band
2. **RSI Extreme**: RSI in overbought (>70) or oversold (<30) zone
3. **Reversion Expected**: Wait for price to show signs of reverting

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bb_period` | 20 | Bollinger Band period |
| `bb_std` | 2.0 | Standard deviations |
| `rsi_period` | 14 | RSI period |
| `rsi_oversold` | 30 | Oversold threshold |
| `rsi_overbought` | 70 | Overbought threshold |
| `atr_period` | 14 | ATR for stops |

### Signal Generation

```
LONG when:
  - Price < Lower Bollinger Band
  - RSI < 30
  - Price showing reversal candle

SHORT when:
  - Price > Upper Bollinger Band
  - RSI > 70
  - Price showing reversal candle
```

---

## 3. Turtle

**Based on:** Richard Dennis's Turtle Trading rules

**Status:** Active

**Default Multiplier:** 1.0×

### Logic

1. **20-Day Breakout**: Price breaks 20-day high (long) or low (short)
2. **System 1**: 20-day channel breakout for entry
3. **System 2**: 55-day channel for adding to winners

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_period` | 20 | Breakout lookback |
| `exit_period` | 10 | Exit lookback |
| `atr_period` | 20 | ATR period |
| `stop_atr_mult` | 2.0 | Stop = N × ATR |

### Signal Generation

```
LONG when:
  - Close > Highest High (20)
  - Not already long

SHORT when:
  - Close < Lowest Low (20)
  - Not already short
```

---

## 4. Weinstein

**Based on:** Stan Weinstein's Stage Analysis

**Status:** Active

**Default Multiplier:** 1.0×

### Logic

1. **Stage Detection**: Identify market stage (1-4)
2. **Stage 2 Entry**: Buy when transitioning to Stage 2 (markup)
3. **Stage 4 Entry**: Short when transitioning to Stage 4 (markdown)
4. **30-Week MA**: Primary trend indicator

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ma_period` | 30 | Moving average period |
| `breakout_pct` | 3.0 | Minimum breakout percentage |
| `volume_mult` | 1.5 | Volume multiplier for confirmation |

### Stages

| Stage | Market Phase | Action |
|-------|--------------|--------|
| 1 | Basing | Wait |
| 2 | Advancing | **BUY** |
| 3 | Topping | Exit longs |
| 4 | Declining | **SHORT** |

---

## 5. Livermore

**Based on:** Jesse Livermore's pivotal point methodology

**Status:** Active

**Default Multiplier:** 1.0×

### Logic

1. **Pivotal Points**: Identify key price levels where trend changes
2. **Momentum Confirmation**: Price and volume confirm direction
3. **Position Building**: Add to winners at new pivotal points

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Pivotal point detection window |
| `pivot_threshold` | 0.02 | Minimum price move (2%) |
| `volume_threshold` | 1.3 | Volume multiplier |

### Signal Generation

```
LONG when:
  - Price breaks above resistance pivotal point
  - Volume 1.3× average
  - Prior swing higher than previous

SHORT when:
  - Price breaks below support pivotal point
  - Volume 1.3× average
  - Prior swing lower than previous
```

---

## 6. BTS Lynch

**Based on:** Peter Lynch's growth investing adapted for crypto

**Status:** Active

**Default Multiplier:** 0.8× (slightly reduced)

### Logic

1. **Momentum Screening**: High relative strength
2. **Volume Surge**: Unusual volume activity
3. **Breakout Confirmation**: Price breaking consolidation

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_period` | 20 | Momentum calculation period |
| `min_momentum` | 5.0 | Minimum momentum (%) |
| `volume_surge_mult` | 2.0 | Volume surge threshold |
| `consolidation_days` | 10 | Minimum consolidation period |

---

## 7. Zweig v2 🔧 (FIXED)

**Based on:** Martin Zweig's breadth thrust methodology

**Status:** FIXED (was disabled at 14% WR, now re-enabled with major improvements)

**Default Multiplier:** 0.7× (probationary)

### Problems Fixed

| Issue | Old Behavior | Fix |
|-------|-------------|-----|
| Bad R:R | 2% SL / 3% TP | ATR-based with 2:1 R:R |
| No trend filter | Traded against trend | ADX > 25 required |
| Frequent signals | Scored every bar | Only on TRANSITIONS |
| Long/short parity | Equal treatment | Long-biased (Zweig is bullish) |
| No volume check | Ignored volume | 1.2× average required |

### Logic (v2)

1. **Multi-Factor Score**: Trend (0-3) + Momentum (0-3) + Health (0-2) = 0-8
2. **Zone Classification**: 
   - Bullish: Score ≥ 7
   - Neutral: 4-6
   - Bearish: Score ≤ 2
3. **THRUST Detection**: Signal only on zone TRANSITIONS (not staying in zone)
4. **Trend Confirmation**: ADX > 25 required
5. **Volume Gate**: Volume > 1.2× 20-period average

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sma_short` | 10 | Short MA for trend score |
| `sma_medium` | 20 | Medium MA |
| `sma_long` | 50 | Long MA |
| `rsi_period` | 14 | RSI for momentum |
| `adx_period` | 14 | ADX period |
| `adx_threshold` | 25 | Minimum ADX |
| `volume_threshold` | 1.2 | Volume multiplier required |
| `long_threshold` | 7 | Score for bullish zone |
| `short_threshold` | 2 | Score for bearish zone |
| `persistence_bars` | 3 | Bars to confirm zone |
| `long_bias` | true | Favor long signals |
| `short_penalty` | 0.15 | Confidence reduction for shorts |

### Signal Generation

```
LONG when:
  - Zone transitions from neutral/bearish → bullish
  - Score >= 7 for 3+ bars
  - ADX > 25
  - Volume > 1.2× average

SHORT when:
  - Zone transitions from neutral/bullish → bearish
  - Score <= 2 for 3+ bars
  - ADX > 25
  - Volume > 1.2× average
  - (Confidence reduced by 15% due to long bias)
```

---

## 8. RSI Divergence ✨ (NEW)

**Based on:** Price/RSI divergence for early reversal detection

**Status:** NEW

**Default Multiplier:** 1.0×

### Why Added

- **Trend Following** catches established trends but misses reversals
- **Mean Reversion** waits for extremes (reactive)
- **RSI Divergence** spots reversals EARLY (predictive)

### Logic

1. **Bullish Divergence**: Price makes lower low, RSI makes higher low → LONG
2. **Bearish Divergence**: Price makes higher high, RSI makes lower high → SHORT
3. **Confirmation**: RSI momentum shift required
4. **Extreme Zone Bonus**: Stronger if RSI in overbought/oversold

### Divergence Types

| Type | Price Action | RSI Action | Signal |
|------|--------------|------------|--------|
| Bullish Regular | Lower Low | Higher Low | LONG |
| Bearish Regular | Higher High | Lower High | SHORT |
| Bullish Hidden | Higher Low | Lower Low | LONG (trend continuation) |
| Bearish Hidden | Lower High | Higher High | SHORT (trend continuation) |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_period` | 14 | RSI calculation period |
| `lookback_bars` | 20 | Divergence detection window |
| `min_swing_bars` | 3 | Minimum bars between swings |
| `divergence_threshold` | 0.02 | Minimum divergence (2%) |
| `rsi_extreme_zone` | 30 | RSI extreme threshold |
| `atr_period` | 14 | ATR for stops |
| `stop_loss_atr` | 2.0 | Stop = 2 × ATR |
| `take_profit_atr` | 3.5 | TP = 3.5 × ATR |
| `require_confirmation` | true | Wait for RSI momentum shift |

### Signal Generation

```
LONG when:
  - Bullish divergence detected (price LL, RSI HL)
  - Divergence strength >= 2%
  - RSI showing upward momentum
  - Volume acceptable

SHORT when:
  - Bearish divergence detected (price HH, RSI LH)
  - Divergence strength >= 2%
  - RSI showing downward momentum
  - Volume acceptable
```

---

## Dynamic Multipliers

Position sizes are scaled by strategy track record:

| Win Rate | Total P&L | Resulting Multiplier |
|----------|-----------|---------------------|
| < 35% | Any | 0.0× (DISABLED) |
| 35-45% | Any | 0.5× |
| 45-50% | Negative | 0.7× |
| > 50% | Positive | 1.0× to 2.0× (scaled by P&L) |

**Formula:**
```python
if win_rate >= 0.50 and total_pnl > 0:
    multiplier = min(2.0, 1.0 + total_pnl / 500)
```

Multipliers update dynamically after each trade result.

---

## Symbol Universe

All strategies trade the same 24 large-cap symbols:

```
XBTUSDTM, ETHUSDTM, SOLUSDTM, XRPUSDTM, ADAUSDTM, LINKUSDTM,
DOTUSDTM, AVAXUSDTM, MATICUSDTM, ATOMUSDTM, UNIUSDTM, LTCUSDTM,
BCHUSDTM, NEARUSDTM, APTUSDTM, ARBUSDTM, OPUSDTM, FILUSDTM,
INJUSDTM, TIAUSDTM, RENDERUSDTM, SUIUSDTM, TONUSDTM, ICPUSDTM
```

---

## Adding New Strategies

1. Create `agents/strategies/your_strategy.py`
2. Inherit from `BaseStrategyAgent`
3. Implement `generate_signals(market_data) -> List[Signal]`
4. Register in `agents/strategies/__init__.py`
5. Add config to `AGENT_CONFIGS` in `run_cloud_v2.py`

**Template:**
```python
from ..base import BaseStrategyAgent, Signal, SignalSide

class YourStrategyAgent(BaseStrategyAgent):
    DEFAULT_CONFIG = {...}
    
    def __init__(self, symbols, config=None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            agent_id='your-strategy',
            name='Your Strategy',
            symbols=symbols,
            config=merged_config
        )
    
    def generate_signals(self, market_data):
        signals = []
        for symbol in self.symbols:
            data = market_data.get(symbol)
            # Your logic here
            if should_signal:
                signals.append(Signal(
                    strategy_id=self.agent_id,
                    symbol=symbol,
                    side=SignalSide.LONG,  # or SHORT
                    confidence=0.65,
                    price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason="Your reason",
                    timestamp=datetime.utcnow()
                ))
        return signals
```
