# Neural Volatility Surface Forecaster

An institutional-style quantitative research platform for forecasting the **future implied volatility (IV) surface** of options markets.

This project is built to feel like a real quant stack: data ingestion, pricing/inversion engines, surface construction, tensorized modeling, API serving, and a research dashboard.

---

## 1) Project Overview

### Core objective
Given a historical sequence of IV surfaces, predict the next surface (or multi-step horizon).

- **Input tensor**: `[time, expiry, strike]`
- **Output tensor**: future `[expiry, strike]` IV grid

### Why this matters
Volatility traders and risk desks care less about one option and more about **how the entire surface moves**:

- Smile steepening/flattening
- Skew shifts (downside risk repricing)
- Term structure kinks (near-term stress vs long-dated calm)
- Regime changes during market shocks

### End users
- Quant researchers
- Derivatives trading teams
- Volatility/risk managers
- ML engineers working on financial sequence modeling

---

## 2) What the System Does End-to-End

1. Pulls options chain snapshots (free mode: Yahoo Finance).
2. Computes implied volatilities robustly (Newton + Brent fallback).
3. Builds and smooths IV surfaces on a stable strike/expiry grid.
4. Stores surfaces and metadata in SQLite via SQLAlchemy.
5. Trains deep models (LSTM, GRU, CNN-LSTM, Transformer, Autoencoder, Conv3D).
6. Evaluates forecasts with quant-relevant metrics.
7. Serves results via FastAPI.
8. Visualizes market state and predictions via Streamlit.

---

## 3) Quant Glossary (Essential Terms)

### Option Chain
The set of listed options for an underlying ticker across strikes and expiries.

### Strike (`K`)
The fixed exercise price of the option contract.

### Expiry / Time-to-Maturity (`T`, `ttm`)
Remaining life of the option, usually expressed in years for pricing.

### Spot (`S`)
Current underlying price.

### Moneyness
Relative strike level vs spot (commonly `K/S` or log-moneyness `log(K/S)`), used to compare smile shape across price levels.

### Implied Volatility (IV)
The volatility value that, when plugged into Black-Scholes, reproduces the market price of an option.

### Smile / Skew
- **Smile**: curvature of IV across strike.
- **Skew**: directional slope of IV vs strike (equity markets often show downside put skew).

### Term Structure
How IV changes across expiries (short-dated vs long-dated vol levels).

### Greeks
Sensitivities of option price:
- Delta: `dV/dS`
- Gamma: `d2V/dS2`
- Theta: time decay
- Vega: volatility sensitivity
- Rho: rate sensitivity

### Risk-Neutral Pricing
Pricing framework where discounted asset prices are martingales; Black-Scholes is derived under this measure.

### Static Arbitrage (Surface Context)
No-arbitrage shape constraints across strike and maturity dimensions (calendar monotonicity / butterfly convexity proxies).

---

## 4) Financial/Mathematical Engine

### Black-Scholes PDE implemented and documented
`dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0`

### IV inversion strategy
For each option quote, solve:
`BSPrice(sigma) - MarketPrice = 0`

Pipeline:
1. Newton-Raphson for speed near root.
2. Brent fallback for robust bracketed convergence.
3. Numerical safeguards for low-vega and boundary cases.

### Surface construction
- Axes: strike/moneyness x expiry
- Grid aggregation: robust median binning
- Missing-point interpolation: cubic spline / RBF
- Smoothing: denoise while preserving market structure

---

## 5) Machine Learning Architecture

Implemented forecasting model families:

1. **LSTM Surface Forecaster**
   - Flatten each surface frame, model temporal evolution.
2. **GRU Surface Forecaster**
   - Lighter recurrent alternative.
3. **CNN + LSTM Hybrid**
   - CNN captures spatial smile/skew structure, LSTM captures temporal deformation.
4. **Transformer Surface Forecaster**
   - Attention over temporal sequence for non-local dependencies/regime behavior.
5. **Autoencoder + Forecast Head**
   - Latent compression + temporal latent forecasting.
6. **Conv3D (advanced)**
   - Joint spatio-temporal convolution on `[time, expiry, strike]`.

Loss:
- MSE/MAE/Huber + smoothness regularization.

Metrics:
- RMSE
- MAE
- Directional skew accuracy
- Surface cosine similarity

---

## 6) Streamlit Dashboard — Complete Page Guide

The dashboard entrypoint is:
`dashboard/app.py`

### Sidebar controls
- **Ticker**: underlying symbol (`SPY`, `QQQ`, `AAPL`, etc.).
- **Forecast Horizon slider**: UI control for target horizon context.
- **Page selector**: switches analytical view.
- **Refresh Data button**: pulls latest chain and persists newest surface snapshot.

### Page A — Live options chain
What it shows:
- Raw option rows (top slice of latest snapshot).
- Contract-level fields like strike, bid/ask, volume, open interest, IV, Greeks, maturity features.
- Quick KPIs:
  - Contract count
  - Median IV

How to interpret:
- Use spread/liquidity fields to sanity-check quote quality.
- Use volume/open interest to focus on active regions of the surface.

### Page B — Current IV surface
What it shows:
- 3D surface (`x`: strike/moneyness, `y`: expiry, `z`: IV).
- 2D heatmap of the same surface.

How to interpret:
- Vertical height differences show localized vol richness/cheapness.
- Left-right asymmetry reflects skew.
- Front-back gradient shows term structure shape.

### Page C — Historical surface playback
What it shows:
- Animated sequence of stored historical surfaces.

How to interpret:
- Observe how smile/skew deforms through time.
- Identify abrupt shape changes (shock regimes) vs gradual drift.

### Page D — Forecasted surface
What it shows:
- Latest model-predicted surface.
- Term-structure slice comparator (actual vs predicted) at user-selected strike index.

How to interpret:
- Compare structural match, not just pointwise values.
- Slice view helps inspect maturity-wise forecast bias at a chosen strike.

### Page E — Model performance
What it shows:
- Metrics history table from DB.
- RMSE/MAE trend chart over runs.

How to interpret:
- Declining RMSE/MAE across retrains suggests better calibration.
- Track whether improvements are stable or overfit/noisy.

### Page F — Regime analysis
What it shows:
- Derived surface-state features (level, skew, curvature, slope, shock proxy).
- Regime labels (KMeans-based state clustering).
- Regime frequency chart.

How to interpret:
- Regime clustering provides market-state segmentation (calm / stressed / transition-like states).
- Useful for conditional modeling or stress-driven strategy overlays.

---

## 7) Repository Structure

```text
neural-vol-surface/
├── data/
│   ├── raw/                # snapshots
│   ├── processed/          # DB and exported artifacts
│   └── cached/             # provider/cache intermediates
├── notebooks/              # research demos
├── configs/                # config templates
├── src/
│   ├── ingestion/          # market data providers + pipeline
│   ├── pricing/            # BS, Greeks, IV inversion, QuantLib hooks
│   ├── iv_surface/         # grid/interpolation/arbitrage/tensor storage
│   ├── preprocessing/      # cleaning/normalization/regimes
│   ├── features/           # engineered features
│   ├── datasets/           # supervised windows & dataloaders
│   ├── models/             # deep architectures
│   ├── training/           # loss/trainer/checkpointing
│   ├── evaluation/         # metrics/backtesting utilities
│   ├── visualization/      # plotly renderers
│   ├── api/                # FastAPI app/service/schemas
│   ├── database/           # SQLAlchemy models/repository
│   └── utils/              # logging/paths/config
├── dashboard/              # Streamlit UI
├── tests/                  # pytest suite
├── docker/                 # containerization
├── PROJECT_FLOW_AND_CONCEPTS.md
├── main.py
└── requirements.txt
```

---

## 8) Local Run Commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Collect data:
```bash
python main.py ingest --ticker SPY
python main.py ingest --ticker SPY --backfill-days 40 --frequency 1D
```

Train:
```bash
python main.py train --ticker SPY --model transformer --lookback 20 --horizon 1 --epochs 30 --batch-size 32
```

Run servers (separate terminals):
```bash
python main.py api
python main.py dashboard
```

---

## 9) Data Providers and Cost

### Fully free path (default)
- Yahoo Finance (`yfinance`)

### Optional provider modules included
- Polygon
- Alpaca

These are optional integrations and may require account/API credentials depending on your plan.

---

## 10) Research Extensions

- Regime-conditioned forecasting
- Arbitrage-constrained neural losses
- Uncertainty quantification (ensembles/Bayesian heads)
- Cross-asset transfer learning
- Volatility strategy/risk overlays
