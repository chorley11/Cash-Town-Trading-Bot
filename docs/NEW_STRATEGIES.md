# New Futures-Specific Strategies (Feb 2026)

Five new strategies that leverage KuCoin Futures-specific data: funding rates, open interest, order book depth, and cross-pair correlations.

## 1. Funding Rate Fade (`funding-fade`)

**Logic:** Fade extreme funding rates. When longs are paying excessive funding, short. When shorts are paying, long.

**Why it works:** Extreme funding indicates crowded positioning. These trades tend to unwind.

**Data needed:**
- Funding rate from `/api/v1/funding-rate/{symbol}/current`
- ADX to filter out strong trends

**Key parameters:**
- `funding_threshold_high`: 0.05% (short above this)
- `funding_threshold_low`: -0.05% (long below this)
- `adx_max`: 30 (don't fade strong trends)

**Best conditions:** Ranging markets with extreme funding.

---

## 2. OI Divergence (`oi-divergence`)

**Logic:** Trade divergences between price and open interest.

| Price | OI | Signal |
|-------|-----|--------|
| ↑ | ↓ | Weak rally → SHORT |
| ↓ | ↓ | Capitulation → LONG |
| ↑ | ↑ | Strong trend (no signal) |
| ↓ | ↑ | Accumulation (no signal) |

**Data needed:**
- Open interest from `/api/v1/contracts/{symbol}`
- RSI for confirmation

**Key parameters:**
- `price_change_threshold`: 1.5%
- `oi_change_threshold`: 2.0%
- `lookback_periods`: 6

**Best conditions:** After extended moves when OI starts declining.

---

## 3. Liquidation Hunter (`liquidation-hunter`)

**Logic:** Two modes:

1. **Cascade Mode:** Trade INTO liquidation cascades
   - Price approaching liq levels + volume spike + momentum
   - Ride the cascade for quick profits

2. **Fade Mode:** Fade EXHAUSTED cascades
   - Volume spike + RSI extreme + recent price gap
   - Mean reversion after forced liquidations

**Data needed:**
- Order book depth from `/api/v1/level2/depth20`
- Volume for spike detection
- RSI for exhaustion

**Key parameters:**
- `leverage_levels`: [10, 20, 50] for liq estimation
- `volume_spike_threshold`: 2.5x average
- `rsi_extreme_low/high`: 25/75

**Best conditions:** High volatility, leveraged markets.

---

## 4. Volatility Breakout (`volatility-breakout`)

**Logic:** Bollinger Squeeze pattern:
1. Detect squeeze: Bollinger Bands inside Keltner Channels
2. Wait for breakout: Price closes outside Bollinger Band
3. Confirm with momentum + volume

**Data needed:**
- OHLCV for Bollinger Bands & Keltner Channels
- Volume for breakout confirmation

**Key parameters:**
- `bb_period`: 20, `bb_std`: 2.0
- `kc_period`: 20, `kc_atr_mult`: 1.5
- `squeeze_lookback`: 6 candles
- `breakout_threshold`: 0.5% beyond band

**Best conditions:** After consolidation periods, any market.

---

## 5. Correlation Pairs (`correlation-pairs`)

**Logic:** Pairs trading on correlated crypto assets:
- BTC/ETH usually have ~0.85 correlation
- When spread z-score exceeds threshold, trade convergence
- Long the underperformer, short the outperformer

**Pairs:**
- BTC/ETH (primary)
- SOL/AVAX (L1s)

**Data needed:**
- Price data for both pairs
- Dynamic hedge ratio calculation

**Key parameters:**
- `correlation_lookback`: 50 periods
- `min_correlation`: 0.6
- `zscore_entry`: 2.0
- `zscore_exit`: 0.5

**Best conditions:** High correlation periods, temporary divergences.

---

## Integration

All strategies are registered in `agents/strategies/__init__.py`:

```python
from agents.strategies import STRATEGY_REGISTRY

# Access a strategy
FundingFadeClass = STRATEGY_REGISTRY['funding-fade']
agent = FundingFadeClass(symbols=['XBTUSDTM', 'ETHUSDTM'])
```

### Data Requirements

New strategies need futures-specific data from `data/futures_data.py`:

```python
from data.futures_data import KuCoinFuturesData

futures_data = KuCoinFuturesData(symbols=['XBTUSDTM', 'ETHUSDTM'])
futures_data.fetch_all()

# Set data on strategies
funding_agent.set_funding_data({
    'XBTUSDTM': futures_data.funding['XBTUSDTM'].current_rate
})
```

---

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run only new strategy tests
python3 -m pytest tests/test_new_strategies.py -v
```

---

## Performance Notes

- **Funding Fade:** Best in ranging markets, avoid during trend days
- **OI Divergence:** Slower signals, higher confidence
- **Liquidation Hunter:** High risk/reward, needs tight stops
- **Volatility Breakout:** Reliable but infrequent
- **Correlation Pairs:** Market neutral, lower returns but consistent

---

## TODO

- [ ] Add backtesting for each strategy
- [ ] Tune parameters per market regime
- [ ] Add real-time funding rate streaming
- [ ] Dashboard integration for new strategies
