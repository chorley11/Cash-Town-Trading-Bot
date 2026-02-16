# Strategy Research Report - February 2026

## Executive Summary
Researched and proposed 3 new quantitative strategies to complement Cash Town's existing portfolio. Implemented the most promising one: **RSI Divergence Strategy**.

---

## Current Strategy Coverage Analysis

| Strategy | Market Condition | Signal Type |
|----------|------------------|-------------|
| trend-following | Trending (ADX > 25) | MA crossover |
| mean-reversion | Ranging (ADX < 25) | BB + RSI extremes |
| turtle | Trending | Donchian breakout |
| weinstein | Trending | Stage analysis |
| livermore | Trending/Swing | Pivotal points |
| bts-lynch | Any | Fundamentals-based |
| zweig | Momentum | Breadth thrust |
| stat-arb | Correlated pairs | Statistical |

### Gap Identified
**No reversal detection strategy** - Current strategies either:
- Follow the trend (trend-following, turtle, etc.)
- Trade at extremes reactively (mean-reversion)

Missing: **Early reversal signals before trends end or extremes are hit**

---

## 3 Strategy Proposals

### 1. RSI Divergence Strategy ✅ IMPLEMENTED
**Concept:** Detect price/momentum divergences to catch reversals early.

**Entry Rules:**
- **LONG:** Price makes lower low + RSI makes higher low (bullish divergence)
- **SHORT:** Price makes higher high + RSI makes lower high (bearish divergence)
- Confirmation: RSI momentum shift detected
- Volume filter: Volume should be at least 80% of average

**Exit Rules:**
- Stop loss: 2 ATR from entry
- Take profit: 3.5 ATR from entry
- Dynamic: Opposite divergence or divergence resolution

**Market Conditions:**
- Works at trend exhaustion points
- Effective in both trending and ranging markets
- Best when RSI is in extreme zones (< 30 or > 70)

**Why It Complements:**
- Catches reversals BEFORE trend-following exits (leads vs lags)
- Signals BEFORE mean-reversion triggers (price doesn't need to hit BB)
- Different signal timing than all existing strategies

---

### 2. VWAP Mean Reversion Strategy (Proposed)
**Concept:** Trade deviations from Volume-Weighted Average Price.

**Entry Rules:**
- LONG: Price > 2 std deviations below VWAP + reversal candle
- SHORT: Price > 2 std deviations above VWAP + reversal candle
- Time filter: Only during high-volume periods

**Exit Rules:**
- Target: Return to VWAP
- Stop: 1 std deviation beyond entry

**Market Conditions:**
- Best for intraday/shorter timeframes
- Works in liquid markets with clear volume patterns
- Institutional flow-based (where smart money operates)

**Why It Complements:**
- Different anchor (VWAP vs BB) for mean reversion
- Time-based filtering (session-aware)
- Volume-centric approach

---

### 3. Cross-Sectional Momentum Strategy (Proposed)
**Concept:** Long strongest momentum assets, short weakest.

**Entry Rules:**
- Calculate momentum score for all symbols (e.g., 20-day return)
- LONG: Top 20% by momentum score
- SHORT: Bottom 20% by momentum score
- Rebalance: Weekly

**Exit Rules:**
- Exit when asset drops out of top/bottom quintile
- Or on rebalance schedule

**Market Conditions:**
- Requires multiple tradeable assets (4+ symbols minimum)
- Works when there's dispersion across assets
- Market-neutral exposure

**Why It Complements:**
- Only multi-asset strategy besides stat-arb
- Relative strength vs absolute momentum
- Systematic rebalancing approach

---

## Implementation Details: RSI Divergence

**File:** `agents/strategies/rsi_divergence.py`
**Agent ID:** `rsi-divergence`
**Class:** `RSIDivergenceAgent`

**Configuration Options:**
```python
DEFAULT_CONFIG = {
    'rsi_period': 14,
    'lookback_bars': 20,          # Divergence lookback window
    'min_swing_bars': 3,          # Minimum bars between swings
    'divergence_threshold': 0.02,  # Minimum 2% divergence
    'rsi_extreme_zone': 30,       # RSI < 30 or > 70 for bonus confidence
    'atr_period': 14,
    'stop_loss_atr': 2.0,
    'take_profit_atr': 3.5,
    'min_confidence': 0.55,
    'require_confirmation': True,
    'volume_filter': True,
}
```

**Key Methods:**
- `_detect_bullish_divergence()` - Finds price lower low + RSI higher low
- `_detect_bearish_divergence()` - Finds price higher high + RSI lower high
- `_find_swing_lows/highs()` - Locates local minima/maxima
- `_check_rsi_confirmation()` - Validates momentum shift

---

## Expected Performance Characteristics

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| Win Rate | 45-55% | Reversal signals less frequent but higher conviction |
| Avg Win | 1.5-2x Avg Loss | 3.5 ATR TP vs 2.0 ATR SL |
| Signal Frequency | Low-Medium | Only on confirmed divergences |
| Drawdown Risk | Moderate | Can be wrong on timing |

**Best Used:** As a complementary signal that adds value when trend-following or mean-reversion isn't triggering.

---

## Recommended Next Steps

1. **Backtest RSI Divergence** on historical data (suggest 6-12 months)
2. **Paper trade** for 2-4 weeks to validate signal quality
3. **Implement VWAP Strategy** next (adds intraday capability)
4. **Consider Momentum Factor** if trading multiple symbols

---

## Research Sources
- Quantpedia cryptocurrency research
- Kraken RSI divergence guide
- Reddit r/quant systematic crypto discussion
- TradingView RSI divergence analysis
