# Nuclear Pairs

A statistical-arbitrage pairs trading algorithm built on the QuantConnect / LEAN engine. The default pair is **Visa (V)** and **Mastercard (MA)**, traded on 1-second bars for the full year 2024.

## Strategy Overview

The algorithm models the long-run equilibrium between two highly correlated stocks and trades deviations from that equilibrium, betting that the spread will revert to its mean.

```
log(V) = α + β · log(MA) + spread
```

When the spread stretches far enough from zero, the algorithm shorts the "rich" leg and longs the "cheap" leg. It then unwinds the position when the spread either snaps back toward zero (take-profit) or blows out further (stop-loss).

## Files

| File | Role |
|---|---|
| `main.py` | LEAN algorithm — data ingestion, Kalman filter update, trade logic |
| `linear_regression.py` | Offline OLS regression on the prior year's daily bars used to initialize the Kalman filter |

## How It Works

### 1. Offline initialization (`linear_regression.py`)

Before the backtest starts, daily close prices for the previous calendar year are loaded for both tickers and an OLS regression is run in log-space:

- **α, β** — hedge-ratio coefficients (initial state estimate)
- **P** — initial state covariance, derived from the residual variance of the fit
- **Q** — process-noise covariance, estimated from the variance of **rolling** 20-day α/β differences (how quickly the relationship drifts)
- **R** — measurement-noise variance (OLS residual variance)

These five values seed the online Kalman filter used during live trading.

### 2. Online state estimation (`main.py`)

On every 1-second bar, a Kalman filter updates α and β in log-space:

- Predict: `P = P + Q`
- Observe: `error = log(V) - (α + β · log(MA))`
- Update: Kalman gain `K = P·Hᵀ / (H·P·Hᵀ + R)`, then `state += K · error`, `P -= K·Hᵀ·P`

This lets the hedge ratio drift slowly over the year rather than being held fixed at the initial OLS estimate.

### 3. Signal generation

The algorithm maintains two rolling buffers:

- **Spread buffer** — last 3600 seconds (1 hour) of residuals `log(V) - (α + β · log(MA))`
- **Return correlation** — rolling 300-second Pearson correlation of per-second log returns

Once both buffers are full, it computes a **z-score** of the current spread against the 1-hour distribution.

Entries are only permitted when rolling return correlation ≥ 0.5 — a sanity check that the pair is still co-moving.

### 4. Trade rules

| Condition | Action |
|---|---|
| `z > +2.0` and not in trade and corr ≥ 0.5 | **SELL** spread: short V 35%, long MA 35% |
| `z < -2.0` and not in trade and corr ≥ 0.5 | **BUY** spread: long V 35%, short MA 35% |
| In a SELL trade and `z > +5.0` | **Stop-loss**: liquidate |
| In a BUY trade and `z < -5.0` | **Stop-loss**: liquidate |
| In any trade and `\|z\| < 0.5` | **Take-profit**: liquidate |

Each leg is sized at 35% of portfolio value, so gross exposure is ~70% with roughly market-neutral dollar-notional between the two legs.

## Configuration

Edit the top of `main.py`:

```python
TICKER_A = "V"
TICKER_B = "MA"
YEAR = 2025
```

## Data Requirements

The algorithm reads custom CSV files from LEAN's data folder:

- `/Lean/Data/custom/{TICKER}_1s_bars_{YEAR}.csv` — second bars for the backtest year
- `/Lean/Data/custom/{TICKER}_1d_bars_{YEAR-1}.csv` — daily bars for the prior year (for OLS init)

Expected CSV columns: `timestamp,open,high,low,close,volume`.

## Execution Details

- Starting cash: $100,000
- Fee model: constant $0.50 per order
- Resolution: 1-second bars
- Backtest window: Jan 2 – Dec 31 of `YEAR`
