# Cash Town - New Strategies Backtest Report

**Generated:** 2026-02-16 17:53:03 UTC

**Period:** 2025-12-21 to 2026-02-16

## Configuration

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000.00 |
| Position Size | $200.00 |
| Leverage | 5x |
| Taker Fee | 0.06% |
| Max Concurrent Positions | 3 |
| Symbols | XBTUSDTM, ETHUSDTM, SOLUSDTM, AVAXUSDTM |

## Executive Summary

- **Profitable Strategies:** 2/5
- **Best Performer:** Volatility Breakout (+1.17%)
- **Worst Performer:** Liquidation Hunter (-7.72%)
- **Combined PnL (if all run):** $-1,355.24
- **Average Win Rate:** 39.7%

## Strategy Performance Comparison

| Strategy | Return | PnL | Trades | Win Rate | Sharpe | Max DD | Profit Factor |
|----------|--------|-----|--------|----------|--------|--------|---------------|
| ✅ Volatility Breakout | +1.17% | $+116.91 | 14 | 50.0% | 8.95 | 0.2% | 2.30 |
| ✅ Funding Rate Fade | +0.09% | $+9.44 | 97 | 45.4% | 21.12 | 1.0% | 1.01 |
| ❌ Correlation Pairs | -3.15% | $-315.04 | 3 | 33.3% | 1.32 | 4.5% | 0.43 |
| ❌ OI Divergence | -3.95% | $-394.68 | 117 | 33.3% | 21.91 | 0.7% | 0.80 |
| ❌ Liquidation Hunter | -7.72% | $-771.87 | 230 | 36.5% | 24.31 | 0.3% | 0.66 |

## Detailed Strategy Analysis

### 🟢 Volatility Breakout (`volatility-breakout`)

**Performance Metrics:**

- Initial Capital: $10,000.00
- Final Capital: $10,116.91
- Total Return: **+1.17%**
- Total PnL: $+116.91
- Total Fees Paid: $16.80

**Trade Statistics:**

- Total Trades: 14
- Winning: 7 | Losing: 7
- Win Rate: 50.0%
- Avg Win: +14.79% | Avg Loss: -6.44%
- Best Trade: +39.28% | Worst: -9.70%
- Avg Holding Time: 6.1 hours
- Trades per Day: 0.23

**Risk Metrics:**

- Max Drawdown: 0.21%
- Sharpe Ratio: 8.95
- Profit Factor: 2.30
- Signals Generated: 14
- Signals Skipped (max positions): 0

**Last 5 Trades:**

| Symbol | Side | Entry | Exit | PnL | Reason |
|--------|------|-------|------|-----|--------|
| XBTUSDTM | SHORT | $77,159.40 | $75,175.94 | ✅ +12.55% | take_profit |
| SOLUSDTM | SHORT | $84.56 | $77.86 | ✅ +39.28% | take_profit |
| XBTUSDTM | LONG | $71,013.89 | $70,093.92 | ❌ -6.78% | stop_loss |
| ETHUSDTM | SHORT | $1,948.10 | $1,984.70 | ❌ -9.70% | stop_loss |
| ETHUSDTM | SHORT | $2,009.26 | $1,939.59 | ✅ +17.04% | take_profit |

---

### 🟢 Funding Rate Fade (`funding-fade`)

**Performance Metrics:**

- Initial Capital: $10,000.00
- Final Capital: $10,009.44
- Total Return: **+0.09%**
- Total PnL: $+9.44
- Total Fees Paid: $116.40

**Trade Statistics:**

- Total Trades: 97
- Winning: 44 | Losing: 53
- Win Rate: 45.4%
- Avg Win: +15.07% | Avg Loss: -12.42%
- Best Trade: +47.39% | Worst: -35.67%
- Avg Holding Time: 24.9 hours
- Trades per Day: 1.62

**Risk Metrics:**

- Max Drawdown: 0.96%
- Sharpe Ratio: 21.12
- Profit Factor: 1.01
- Signals Generated: 1394
- Signals Skipped (max positions): 1297

**Last 5 Trades:**

| Symbol | Side | Entry | Exit | PnL | Reason |
|--------|------|-------|------|-----|--------|
| AVAXUSDTM | LONG | $9.29 | $9.09 | ❌ -11.17% | stop_loss |
| XBTUSDTM | LONG | $69,023.59 | $67,890.35 | ❌ -8.51% | stop_loss |
| ETHUSDTM | LONG | $1,973.74 | $1,979.48 | ✅ +1.15% | end_of_backtest |
| AVAXUSDTM | LONG | $9.08 | $9.16 | ✅ +3.60% | end_of_backtest |
| XBTUSDTM | LONG | $67,911.04 | $67,922.22 | ❌ -0.22% | end_of_backtest |

---

### 🔴 Correlation Pairs (`correlation-pairs`)

**Performance Metrics:**

- Initial Capital: $10,000.00
- Final Capital: $9,684.96
- Total Return: **-3.15%**
- Total PnL: $-315.04
- Total Fees Paid: $3.60

**Trade Statistics:**

- Total Trades: 3
- Winning: 1 | Losing: 2
- Win Rate: 33.3%
- Avg Win: +117.33% | Avg Loss: -137.43%
- Best Trade: +117.33% | Worst: -159.27%
- Avg Holding Time: 1369.7 hours
- Trades per Day: 0.05

**Risk Metrics:**

- Max Drawdown: 4.53%
- Sharpe Ratio: 1.32
- Profit Factor: 0.43
- Signals Generated: 600
- Signals Skipped (max positions): 597

**Last 5 Trades:**

| Symbol | Side | Entry | Exit | PnL | Reason |
|--------|------|-------|------|-----|--------|
| SOLUSDTM | LONG | $123.67 | $84.35 | ❌ -159.27% | end_of_backtest |
| AVAXUSDTM | SHORT | $11.98 | $9.16 | ✅ +117.33% | end_of_backtest |
| XBTUSDTM | LONG | $88,275.02 | $67,922.22 | ❌ -115.58% | end_of_backtest |

---

### 🔴 OI Divergence (`oi-divergence`)

**Performance Metrics:**

- Initial Capital: $10,000.00
- Final Capital: $9,605.32
- Total Return: **-3.95%**
- Total PnL: $-394.68
- Total Fees Paid: $140.40

**Trade Statistics:**

- Total Trades: 117
- Winning: 39 | Losing: 78
- Win Rate: 33.3%
- Avg Win: +20.49% | Avg Loss: -12.77%
- Best Trade: +82.76% | Worst: -34.65%
- Avg Holding Time: 17.7 hours
- Trades per Day: 1.95

**Risk Metrics:**

- Max Drawdown: 0.65%
- Sharpe Ratio: 21.91
- Profit Factor: 0.80
- Signals Generated: 215
- Signals Skipped (max positions): 98

**Last 5 Trades:**

| Symbol | Side | Entry | Exit | PnL | Reason |
|--------|------|-------|------|-----|--------|
| SOLUSDTM | SHORT | $84.16 | $86.50 | ❌ -14.18% | stop_loss |
| ETHUSDTM | SHORT | $2,080.99 | $2,106.49 | ❌ -6.43% | stop_loss |
| ETHUSDTM | SHORT | $2,098.48 | $2,058.45 | ✅ +9.24% | take_profit |
| AVAXUSDTM | SHORT | $9.60 | $9.31 | ✅ +14.32% | take_profit |
| SOLUSDTM | LONG | $87.00 | $84.33 | ❌ -15.62% | stop_loss |

---

### 🔴 Liquidation Hunter (`liquidation-hunter`)

**Performance Metrics:**

- Initial Capital: $10,000.00
- Final Capital: $9,228.13
- Total Return: **-7.72%**
- Total PnL: $-771.87
- Total Fees Paid: $276.00

**Trade Statistics:**

- Total Trades: 230
- Winning: 84 | Losing: 146
- Win Rate: 36.5%
- Avg Win: +9.11% | Avg Loss: -7.88%
- Best Trade: +9.23% | Worst: -36.68%
- Avg Holding Time: 7.2 hours
- Trades per Day: 3.83

**Risk Metrics:**

- Max Drawdown: 0.27%
- Sharpe Ratio: 24.31
- Profit Factor: 0.66
- Signals Generated: 300
- Signals Skipped (max positions): 70

**Last 5 Trades:**

| Symbol | Side | Entry | Exit | PnL | Reason |
|--------|------|-------|------|-----|--------|
| AVAXUSDTM | LONG | $9.60 | $9.47 | ❌ -7.27% | stop_loss |
| XBTUSDTM | SHORT | $68,898.73 | $69,775.15 | ❌ -6.66% | stop_loss |
| XBTUSDTM | LONG | $69,787.58 | $69,005.02 | ❌ -5.91% | stop_loss |
| XBTUSDTM | SHORT | $68,352.31 | $67,990.18 | ✅ +2.35% | end_of_backtest |
| SOLUSDTM | SHORT | $84.12 | $84.43 | ❌ -2.19% | end_of_backtest |

---

## Recommendations

**Strategies recommended for live testing:**
- Funding Rate Fade: Sharpe 21.12, Win Rate 45.4%
- Volatility Breakout: Sharpe 8.95, Win Rate 50.0%

---
*Report generated by Cash Town Backtest Engine v2.0*