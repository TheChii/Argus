# ARGUS QUANT: INSTITUTIONAL SYSTEM ARCHITECTURE, MATHEMATICAL FOUNDATIONS, ALGORITHMIC ENGINE & PROJECT ROADMAP

**Document Version**: 2.3.0  
**Classification**: Institutional Technical Specification & Algorithmic Blueprint  
**Target Universe**: S&P 100 Large-Cap Equities, Sector SPDR ETFs, and Benchmark ($SPY$)  
**Operating Cadence**: Daily EOD Ingestion, 4:00 PM ET Cutoff, Weekly Friday Rebalancing, Real-Time Intraday Risk Circuit Breakers  

---

## TABLE OF CONTENTS

1. [Executive Summary & System Philosophy](#1-executive-summary--system-philosophy)
2. [High-Level Architecture & End-to-End Workflow](#2-high-level-architecture--end-to-end-workflow)
3. [Data Ingestion & Integrity Pipelines](#3-data-ingestion--integrity-pipelines)
4. [Feature Engineering & Mathematical Formulations](#4-feature-engineering--mathematical-formulations)
5. [Natural Language Processing: FinBERT Sentiment Engine](#5-natural-language-processing-finbert-sentiment-engine)
6. [Market Volatility & Regime Detection Engine](#6-market-volatility--regime-detection-engine)
7. [Cross-Sectional Alpha Generation & Barra Factor Neutralization](#7-cross-sectional-alpha-generation--barra-factor-neutralization)
8. [Explainability & Attribution: TreeSHAP](#8-explainability--attribution-treeshap)
9. [Portfolio Construction, Risk Parity & Sector Constraints](#9-portfolio-construction-risk-parity--sector-constraints)
10. [Live Walk-Forward Monitoring & Intraday Risk Management](#10-live-walk-forward-monitoring--intraday-risk-management)
11. [Execution Layer & Broker Integrations](#11-execution-layer--broker-integrations)
12. [User Interface & Operations Terminal Architecture](#12-user-interface--operations-terminal-architecture)
13. [Comprehensive Project Roadmap (Phases 1 to 6)](#13-comprehensive-project-roadmap-phases-1-to-6)

---

## 1. EXECUTIVE SUMMARY & SYSTEM PHILOSOPHY

Argus Quant is an institutional-grade systematic quantitative investment engine designed for long-only and market-hedged equity strategies across the S&P 100 universe. The core mission of the platform is to generate persistent, cost-adjusted risk-adjusted alpha over the S&P 500 benchmark ($SPY$) by combining:

1. **Non-Linear Gradient-Boosted Alpha Modeling**: Capturing cross-sectional price momentum and fundamental factor interactions via constrained multi-seed LightGBM regression and linear Ridge shrinkage.
2. **Fundamental Corporate Valuations**: Systematically scoring earnings yield, book-to-price, sales-to-price, free cash flow yield, and enterprise value ratios to prevent growth-trap overconcentration.
3. **Barra Multi-Factor Risk Neutralization**: Projecting return forecasts onto an 11-sector and 4-style factor exposure matrix, stripping systematic market, sector, size, volatility, and momentum beta to isolate pure idiosyncratic stock alpha.
4. **Deep Domain-Specific NLP Sentiment**: Ingesting high-frequency financial news headlines and extracting polarity through FinBERT with exponential time-decay weighting.
5. **Macro-Conditioned Dynamic Risk Budgeting**: Modulating gross equity exposure based on market volatility regimes (HAR-RV and 3-State Gaussian Hidden Markov Models).
6. **Sector-Neutral Portfolio Construction**: Enforcing a maximum 25% sector weight cap and at most 3 constituents per sector, alongside an 8% individual stock cap and dual-capping iterative redistribution.
7. **Real-Time Intraday Risk Circuit Breakers**: Protecting capital against midweek idiosyncratic gap-downs using hard drawdown (-7%), trailing high-water mark drawdown (-7%), and $2.5\times$ ATR volatility stops with automated liquidation to cash or hedged $SPY$.
8. **Institutional Turnover Suppression**: Mitigating execution friction, bid-ask spread slippage, and brokerage commissions through an asymmetric hysteresis buffer ($[0.60, 0.90)$ holding band).
9. **Continuous Out-of-Sample Gating**: Utilizing a real-time Spearman Information Coefficient (IC) state machine (Confidence Light) that automatically derisks or moves to cash if empirical predictive edge decays.

### Non-Negotiable Quantitative Tenets

* **Zero Lookahead Leakage Guarantee**: Features calculated for day $t$ utilize price, fundamental, and news data finalized strictly prior to 16:00 US Eastern Time on day $t$. All subsequent returns predicted are forward-looking ($t \to t+5$).
* **Strict Trading Calendar Synchronization**: The system adheres to NYSE/NASDAQ trading calendars. Weekend and holiday data gaps are handled without forward leakage or phantom interpolation.
* **Cost-Aware Optimization**: Every strategy evaluation incorporates realistic execution friction (7.5 basis points one-way commission and spread modeling).
* **Interpretability at Scale**: Every recommendation is decomposed via TreeSHAP into human-auditable quantitative factors.
* **Capital Preservation Priority**: Risk circuit breakers and factor orthogonality take absolute precedence over return maximization.

---

## 2. HIGH-LEVEL ARCHITECTURE & END-TO-END WORKFLOW

The end-to-end operational pipeline runs daily after US equity market close (16:00 ET):

1. **Data Layer**:
   - **Price Pipeline**: Ingests auto-adjusted OHLCV from Yahoo Finance for S&P 100 + Sector ETFs + $SPY$ into Snappy-compressed Parquet.
   - **Macroeconomic Pipeline**: Ingests FRED series ($T10Y2Y$, $DFF$, $VIXCLS$, $BAMLH0A0HYM2$).
   - **Headline Archive**: Ingests Finnhub & GDELT news with SHA-256 deduplication and 4:00 PM ET cutoff tagging.
   - **Fundamental Pipeline**: Ingests corporate valuation multiples ($E/P, B/P, S/P, FCF/P, EV/\text{EBITDA}$, dividend yield, market cap) with deterministic sector fallback benchmarks.

2. **Feature Engineering**:
   - Computes multi-timeframe returns ($1\text{d}, 5\text{d}, 20\text{d}, 60\text{d}$), realized volatilities, Wilder's RSI, trend ratios, sector-relative momentum, and volatility z-scores via C++ OpenMP kernels.
   - Computes FinBERT sentiment polarity with exponential time-decay weighting.
   - Normalizes corporate valuation multiples.
   - Standardizes all features cross-sectionally via daily uniform rank normalization $[-0.5, 0.5]$.

3. **Modeling & Forecasting**:
   - **HAR Volatility Model**: Forecasts 5-day forward realized volatility $\widehat{\text{RV}}_{t \to t+5}$.
   - **3-State Gaussian HMM**: Estimates filtered market regime probabilities (Low Vol Bull, Normal, High Vol Crisis).
   - **Constrained Multi-Seed LightGBM & Ridge Regression**: Generate forward 5-day excess return rankings.
   - **Blended 50/50 Percentile Ranker**: Combines linear shrinkage and non-linear tree ranking.
   - **Barra Factor Risk Neutralization**: Residualizes blended predictions against 11 GICS sector dummies and 4 standardized style factors (Size, Volatility, Momentum, Value) via regularized projection:
     $$\alpha_{\text{neutral}} = \alpha - X(X^T X + \lambda I)^{-1} X^T \alpha$$

4. **Risk Budgeting & Allocation**:
   - **TreeSHAP**: Decomposes predictions into category drivers (Risk, Momentum, Trend, News Sentiment, Macro, Valuation).
   - **Inverse Volatility Risk Parity**: Sizes positions with an $8.0\%$ individual stock concentration cap and a $25.0\%$ aggregate sector cap.
   - **Sector Quota Constraints**: Maximum 3 holdings per GICS sector in the target portfolio.
   - **Dual-Capping Iterative Redistribution**: Redistributes excess weight proportionally to uncapped assets whose sector remains below 25%.
   - **Turnover Hysteresis Buffer**: Filters entries (top $10\%$) and exits (drop below top $40\%$), suppressing unnecessary churn.
   - **VIX Exposure Scaling**: Throttles gross exposure during market volatility spikes.

5. **Monitoring, Circuit Breakers & Execution**:
   - **Rolling 60-day Spearman IC**: Evaluates predictive edge and triggers Confidence Light (Green / Amber / Red).
   - **Intraday Risk Monitor**: Continuously tracks open positions against entry costs and peak high-water marks, executing defensive liquidation upon -7% drawdown or $2.5\times$ ATR violations.
   - **Broker Dispatch**: Routes sized orders via native APIs (Alpaca, Trading 212, IBKR Basket Trader, Dry Run).
   - **SQLite Ledger**: Logs complete execution history, high-water marks, and circuit breaker audit trails (`portfolio_ledger.db`).

---

## 3. DATA INGESTION & INTEGRITY PIPELINES

### 3.1 Universe Specification
The target universe comprises the constituent equities of the S&P 100 index (OEX), capturing the 100 largest, most liquid US blue-chip companies, supplemented by 11 SPDR Sector ETFs and the $SPY$ benchmark:
* **Constituent Registry**: `src/config/sp100_default.json` with dynamic Wikipedia reconstitution fallback.
* **Sector SPDR ETFs**: `XLK`, `XLV`, `XLF`, `XLI`, `XLC`, `XLY`, `XLP`, `XLE`, `XLU`, `XLRE`, `XLB`.
* **Benchmark**: `SPY` (SPDR S&P 500 ETF Trust).

### 3.2 Ingestion Protocols & Guarantees

| Pipeline | Source API | Primary Guarantees | Storage Format |
| :--- | :--- | :--- | :--- |
| **Prices** | Yahoo Finance (`yfinance`) | Auto-adjusted for splits/dividends; skips repeat queries if up to date. | Parquet (`data/prices/{TICKER}.parquet`) |
| **Macro** | St. Louis Fed FRED REST/CSV | Tracks $T10Y2Y$, $DFF$, $VIXCLS$, $BAMLH0A0HYM2$; forward-fills non-trading gaps up to 5 days. | Parquet (`data/macro/fred_macro.parquet`) |
| **News Archive** | Finnhub REST v1 & GDELT DOC 2.0 | Deterministic SHA-256 deduplication; strict 4:00 PM ET cutoff mapping to next trading day. | Partitioned Parquet (`year=YYYY/month=MM/`) |
| **Fundamentals** | Yahoo Finance & Sector Medians | Slices valuation ratios with deterministic offline sector benchmark fallbacks. | JSON & Parquet (`data/fundamentals/`) |

### 3.3 Fundamental Valuation Pipeline (`src/data/fundamentals.py`)
To prevent the model from buying overvalued momentum bubbles, the fundamental pipeline captures corporate valuation multiples:
* **Earnings Yield**: $E/P = 1 / \text{P/E}$
* **Book-to-Price**: $B/P = 1 / \text{P/B}$
* **Sales-to-Price**: $S/P = 1 / \text{P/S}$
* **Free Cash Flow Yield**: $FCF/P = \text{Free Cash Flow} / \text{Market Cap}$
* **Enterprise Value to EBITDA**: $EV / \text{EBITDA}$
* **Dividend Yield**
* **Log Market Capitalization**: $\ln(\text{MarketCap})$

**Offline Fallback Guarantee**: If remote financial statement APIs are unreachable or in continuous integration test environments, the pipeline immediately falls back to calibrated sector medians (`SECTOR_FUNDAMENTAL_BENCHMARKS`) perturbed deterministically by ticker hash. Synthetic mock tickers are identified instantly, ensuring sub-second execution without network lag.

---

## 4. FEATURE ENGINEERING & MATHEMATICAL FORMULATIONS

All features are engineered with strict zero lookahead guarantees. The feature vector for stock $i$ at trading day $t$ spans technical momentum, volatility, news sentiment, and fundamental valuation multiples:

### 4.1 Technical Momentum & Volatility Features

* **Normalized Log Returns**:
  $$R_{i, N, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-N}}\right) \quad \text{for } N \in \{1, 5, 20, 60\}$$

* **Annualized Realized Volatility**:
  $$\sigma_{i, N, t} = \sqrt{\frac{252}{N-1} \sum_{k=0}^{N-1} \left(r_{i, t-k} - \bar{r}_{i, t}\right)^2} \quad \text{for } N \in \{5, 20, 60\}$$

* **Wilder's Relative Strength Index (RSI 14-Day)**:
  $$\text{RS}_t = \frac{\text{EMA}_{14}(\max(0, \Delta P_t))}{\text{EMA}_{14}(\max(0, -\Delta P_t))}, \quad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$

* **Wilder's Average True Range ($\text{ATR}_{14}$)**:
  $$\text{TR}_t = \max\left(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|\right)$$
  $$\text{ATR}_t = \frac{13 \times \text{ATR}_{t-1} + \text{TR}_t}{14}$$

* **Trend Divergence Ratio**:
  $$\text{Trend}_{i, t} = \frac{\text{SMA}_{20}(P_{i, t})}{\text{SMA}_{50}(P_{i, t})} - 1.0$$

* **Volatility Z-Score**:
  $$Z_{\sigma, i, t} = \frac{\sigma_{i, 20, t} - \mu_{\sigma, i, 252}}{\text{std}(\sigma_{i, 252})}$$

* **Sector-Relative Momentum**:
  $$R^{\text{rel}}_{i, N, t} = R_{i, N, t} - R_{\text{SectorETF}(i), N, t}$$

### 4.2 Cross-Sectional Rank Normalization

To eliminate market-wide level shifts and isolate purely idiosyncratic ranking power, features are transformed via daily cross-sectional uniform rank normalization $[-0.5, +0.5]$:
$$\text{Rank}_{\%}(x_{i, t}) = \frac{\text{Rank}(x_{i, t}) - 1}{M - 1} - 0.5$$

---

## 5. NATURAL LANGUAGE PROCESSING: FINBERT SENTIMENT ENGINE

### 5.1 Architecture & Tokenization
Argus Quant utilizes `ProsusAI/finbert`, a domain-adapted BERT model fine-tuned on Financial PhraseBank. For each headline, it outputs class probabilities: $p_{\text{pos}}$, $p_{\text{neu}}$, $p_{\text{neg}}$.

**Scalar Sentiment Polarity**:
$$S = p_{\text{positive}} - p_{\text{negative}} \in [-1.0, +1.0]$$

### 5.2 Time-Decay Exponential Weighting
News impact diminishes exponentially with time. When aggregating all headlines $\mathcal{H}_i$ associated with stock $i$ over a rolling 7-day lookback window:
$$w_k = \exp\left(-\frac{t - t_k}{\tau}\right) = \exp\left(-\frac{\Delta t_k}{2.0\text{ days}}\right)$$
$$\bar{S}_{i, t} = \frac{\sum_{k \in \mathcal{H}_i} w_k S_k}{\sum_{k \in \mathcal{H}_i} w_k}$$

### 5.3 Persistent Inference Cache
Headline text is hashed via MD5/SHA256 into a persistent cache. If an identical headline is encountered, its score vector is returned in $\mathcal{O}(1)$ time without invoking transformer forward passes, speeding up pipeline execution by over $95\%$.

---

## 6. MARKET VOLATILITY & REGIME DETECTION ENGINE

### 6.1 HAR-RV Volatility Model (`src/models/vol_model/har.py`)
Predicts forward 5-day realized volatility $\text{RV}_{t \to t+5}$ using Corsi's Heterogeneous Autoregressive specification with Ridge regularization:
$$\ln(\text{RV}_{t \to t+5}) = \beta_0 + \beta_d \ln(\text{RV}_d) + \beta_w \ln(\text{RV}_w) + \beta_m \ln(\text{RV}_m) + \beta_v \ln(\text{VIX}_t) + \epsilon_t$$
Where $\text{RV}_d$, $\text{RV}_w$, and $\text{RV}_m$ are 1-day, 5-day, and 20-day historical realized volatilities.

The logarithmic formulation mathematically guarantees strictly positive volatility forecasts:
$$\widehat{\text{RV}}_{t \to t+5} = \exp\left(\widehat{\ln \text{RV}}_{t \to t+5}\right) > 0$$

### 6.2 3-State Gaussian Hidden Markov Model (`src/features/regime.py`)
Models underlying market regimes using unobservable latent Markov states $\mathcal{S}_t \in \{0, 1, 2\}$ fitted on daily $SPY$ returns and VIX levels:
* **State 0 (Low Volatility Bull)**: Low VIX ($\mu_{\text{vix}} < 16$), positive drift. Full gross exposure enabled.
* **State 1 (Normal / Transition)**: Moderate VIX ($16 \le \mu_{\text{vix}} \le 24$). Balanced positioning.
* **State 2 (High Volatility Crisis / Bear)**: Elevated VIX ($\mu_{\text{vix}} > 24$), negative returns, high kurtosis. Exposure throttled.

**Filtering Rule**: State probabilities $P(S_t = k \mid x_{1:t})$ are computed using forward filtering only. Backward smoothing algorithms across the entire sample are strictly prohibited to prevent lookahead bias.

---

## 7. CROSS-SECTIONAL ALPHA GENERATION & BARRA FACTOR NEUTRALIZATION

### 7.1 Target Variable Formulation
The predictive target is the 5-day forward excess return relative to the benchmark $SPY$:
$$y_{i, t} = \frac{P_{i, t+5}}{P_{i, t}} - \frac{P_{\text{SPY}, t+5}}{P_{\text{SPY}, t}}$$

### 7.2 Constrained LightGBM Return Ranker (`src/models/return_ranker/lgbm.py`)

| Hyperparameter | Value | Institutional Design Rationale |
| :--- | :--- | :--- |
| `max_depth` | `3` | Prevents high-order non-linear interactions from fitting financial noise |
| `num_leaves` | `8` | Enforces shallow, generalizable decision surfaces |
| `min_child_samples` | `250` | Requires broad statistical support across multiple market dates |
| `learning_rate` | `0.03` | Conservative gradient descent step size |
| `subsample` | `0.70` | Row bagging to decorrelate individual trees |
| `colsample_bytree` | `0.80` | Feature bagging to prevent dominance by a single factor |
| `n_estimators` | `300` | Controlled training depth with early stopping |

**Multi-Seed Averaging**: The model fits $K=5$ distinct instances initialized with decorrelated random seeds:
$$\hat{y}_{\text{LGBM}}(x) = \frac{1}{K} \sum_{k=1}^K \hat{f}_k(x; \text{seed}_k)$$

### 7.3 Blended 50/50 Percentile Ranker (`src/models/return_ranker/ensemble.py`)
Linear Ridge regression provides shrinkage stability against non-linear drift:
$$\hat{\beta} = \arg\min_\beta \left( \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \right) = \left( X^T X + \alpha I \right)^{-1} X^T y$$
$$\text{Score}_{\text{raw}}(i) = 0.5 \times \text{Rank}_{\%}(\hat{y}_{\text{Ridge}, i}) + 0.5 \times \text{Rank}_{\%}(\hat{y}_{\text{LGBM}, i})$$

### 7.4 Barra Multi-Factor Model & Alpha Neutralization (`src/features/barra.py`)
Without factor neutralization, raw quantitative forecasts frequently suffer from massive unintended style and sector biases (e.g. buying 65% Tech during momentum expansions), resulting in catastrophic factor rotations.

Argus Quant constructs a daily factor exposure matrix $X \in \mathbb{R}^{N \times 15}$:
1. **11 GICS Sector Dummy Variables**: Information Technology, Health Care, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials.
2. **4 Cross-Sectional Style Factors** (cross-sectionally Z-scored $\mu=0, \sigma=1$, clipped to $[-3.0, 3.0]$):
   - *Size*: Log market capitalization ($\ln(\text{MarketCap})$).
   - *Volatility*: 20-day annualized realized volatility ($\sigma_{20d}$).
   - *Momentum*: 60-day price momentum ($R_{60d}$).
   - *Value*: Composite score of Book-to-Price, Earnings Yield, and Sales-to-Price.

**Regularized Projection Neutralization**:
$$\alpha_{\text{neutral}} = \alpha - X(X^T X + \lambda I)^{-1} X^T \alpha$$
Where $\lambda = 10^{-4}$ provides ridge regularization guaranteeing strictly positive-definite matrix inversion.

**Mathematical Guarantees**:
* **Factor Orthogonality**: The cross-sectional correlation between $\alpha_{\text{neutral}}$ and all sector dummies and style columns satisfies $\max |\rho| < 10^{-6}$.
* **Dispersion Preservation**: Residual scores preserve the rank dispersion and scale of raw predictions while isolating pure idiosyncratic stock-selection edge.

---

## 8. EXPLAINABILITY & ATTRIBUTION: TREESHAP

For institutional trust and risk auditability, black-box predictions are decomposed via TreeSHAP:
$$\hat{f}(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
Where $\phi_j(x)$ represents the exact contribution of feature $j$ to the prediction relative to the baseline expectation $\phi_0$.

### Investor Translation Engine (`translate_shap_feature_to_plain_english`)

| Factor Category | Tag | Example Driver Description |
| :--- | :--- | :--- |
| **Risk & Stability** | `[VOL]` | "Smooth price action: 20-day realized volatility is lower and more stable than peer stocks." |
| **Technical Momentum**| `[RSI]` | "Healthy momentum sweet-spot: upward buying demand without being severely overbought." |
| **Trend Momentum** | `[MOM]` | "Strong 1-month upward price trend consistently outperforming the broader market." |
| **News Sentiment** | `[SENT]` | "FinBERT NLP sentiment: Positive headline tone over the rolling weekly news window." |
| **Macro Interaction** | `[MACRO]`| "Favorable macro positioning under current Treasury yield spread conditions." |
| **Fundamental Valuation** | `[VAL]` | "Attractive earnings yield and cash flow valuation discount relative to sector peers." |

---

## 9. PORTFOLIO CONSTRUCTION, RISK PARITY & SECTOR CONSTRAINTS

### 9.1 Stock Selection Universe
At each weekly rebalance date:
* All stocks in the S&P 100 universe are ranked by neutralized score $\alpha_{\text{neutral}}$.
* The top decile (top $10\%$, $\sim 10-15$ names) forms the target portfolio basket subject to sector quotas.

### 9.2 Inverse Volatility Sizing & Position Caps
To avoid concentration risk in high-beta names, position sizing uses inverse volatility weighting:
$$w_i^{\text{raw}} = \frac{1}{\hat{\sigma}_{i, \text{HAR}}}$$
$$w_i^* = \frac{w_i^{\text{raw}}}{\sum_{j=1}^N w_j^{\text{raw}}}$$

### 9.3 Sector-Neutral Portfolio Constraints & Dual-Capping
To eliminate unintended sector concentration, [`PortfolioAllocator`](src/portfolio/allocator.py) enforces:
1. **Single Sector Weight Cap**: No GICS sector may aggregate more than $w_{\text{sector}} \le 25.0\%$ of total capital.
2. **Constituent Sector Quota**: No single sector may hold more than $N_{\text{sector}} \le 3$ constituents in the target 15-stock basket.
3. **Single Stock Hard Cap**: No individual holding may exceed $w_i \le 8.0\%$.

**Dual-Capping Iterative Redistribution**:
Standard single-cap redistribution frequently violates sector limits when redistributing excess stock weight. Argus Quant runs an iterative dual-capping algorithm:
- Checks both individual limits ($w_i \le 0.08$) and aggregate sector limits ($\sum_{i \in S} w_i \le 0.25$).
- Redistributes excess capital proportionally *only* to uncapped holdings whose parent sector has remaining headroom under the 25% boundary.
- Guaranteed convergence within 100 iterations.

### 9.4 Turnover Hysteresis Buffer (Suppression Engine)
Unchecked rebalancing incurs substantial slippage and execution fees. Argus Quant applies an asymmetric buffer rule:
* **Buy Trigger**: An equity is bought only if it enters the **Top 10%** (percentile $\ge 0.90$) and satisfies sector quotas.
* **Sell Trigger**: An existing position is liquidated only if it drops below the **Top 40%** (percentile $< 0.60$).
* **Buffer Zone $[0.60, 0.90)$**: Positions within this zone are retained, cutting turnover by over $50\%$ without degrading alpha.
* **Delta Threshold**: If an existing holding's target weight changes by less than $\delta = 0.5\%$, the rebalance order is omitted.

### 9.5 Macro Volatility Exposure Scaling
When broader market volatility rises, gross equity exposure is scaled downward:
$$\gamma_{\text{vix}} = \min\left(1.0, \frac{\text{VIX}_{\text{baseline}}}{\text{VIX}_t}\right) = \min\left(1.0, \frac{20.0}{\text{VIX}_t}\right)$$
$$\text{Target Gross Exposure} = \text{Budget} \times \gamma_{\text{vix}} \times \gamma_{\text{confidence}}$$
The residual allocation is held in cash or cash-equivalents ($BIL$ / Short-Term Treasuries).

---

## 10. LIVE WALK-FORWARD MONITORING & INTRADAY RISK MANAGEMENT

### 10.1 Out-of-Sample Spearman IC State Machine
To detect regime degradation or model breakdown in production, the system computes the rolling 60-day out-of-sample Spearman Rank Correlation (Information Coefficient, IC) between model predictions $\hat{y}$ and realized 5-day returns $y$:
$$\text{IC}_t = \frac{\sum_{i=1}^M \left(R(\hat{y}_i) - \bar{R}_{\hat{y}}\right)\left(R(y_i) - \bar{R}_y\right)}{\sqrt{\sum_{i=1}^M \left(R(\hat{y}_i) - \bar{R}_{\hat{y}}\right)^2 \sum_{i=1}^M \left(R(y_i) - \bar{R}_y\right)^2}}$$
$$t_{\text{stat}} = \text{IC}_t \times \sqrt{\frac{N - 2}{1 - \text{IC}_t^2}}$$

### 10.2 Confidence Light State Machine Rules

| State Code | Badge | Thresholds | Portfolio Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **GREEN** | `[SAFE]` | $\text{IC}_{60d} \ge +0.02$ and $t \ge 1.5$ | 100% Target Equity Exposure | Strategy edge is statistically robust |
| **AMBER** | `[CAUTION]`| $0.00 < \text{IC}_{60d} < +0.02$ or $t < 1.5$ | 50% Exposure Scaled / 50% Cash | Edge is weakening; reduce risk budget |
| **RED** | `[DEFENSIVE]`| $\text{IC}_{60d} \le 0.00$ | 0% Stock Exposure (100% Cash / SPY) | Negative predictive power; preserve capital |

### 10.3 Intraday Risk Circuit Breakers & Emergency Stop-Losses (`src/portfolio/risk_monitor.py`)
Weekly rebalancing creates an idiosyncratic vulnerability: unexpected earnings misses, regulatory subpoenas, or catastrophic guidance revisions can cause a held constituent to drop -15% midweek while the model passively holds until Friday.

Argus Quant implements an automated intraday risk monitor that tracks open positions in real time against entry cost and peak high-water marks (HWM):
1. **Hard Drawdown Stop**: Liquidates immediately if price falls $\le -7.0\%$ from initial entry cost:
   $$\text{Drawdown}_{\text{entry}} = \frac{P_t - P_{\text{entry}}}{P_{\text{entry}}} \le -0.07$$
2. **Trailing High-Water Mark Stop**: Liquidates immediately if price falls $\le -7.0\%$ from its post-entry peak:
   $$\text{Drawdown}_{\text{peak}} = \frac{P_t - P_{\text{peak}}}{P_{\text{peak}}} \le -0.07$$
3. **Volatility-Adjusted ATR Stop**: Liquidates immediately if price breaches the volatility floor:
   $$P_t \le P_{\text{peak}} - 2.5 \times \text{ATR}_{14}$$

**Execution & Hedging Modes**:
* `cash`: Executes an immediate market sell order for all open shares, records the fill, and holds cash.
* `hedged_spy`: Liquidates the troubled constituent and immediately executes a market buy order for $SPY$ with the realized proceeds, eliminating single-stock tail risk while maintaining broader market beta.

**Audit Trail & Alerting**:
Every trigger dispatches a high-priority `CRITICAL` Telegram notification and writes an immutable audit record to `circuit_breaker_events` and `position_high_water_marks` in SQLite.

---

## 11. EXECUTION LAYER & BROKER INTEGRATIONS

Argus Quant features an execution abstraction layer (`BaseBrokerClient`) supporting multiple institutional and retail broker APIs:

### 11.1 Alpaca Markets REST API v2 (`AlpacaBrokerClient`)
* **Protocol**: REST over HTTPS with `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY` headers.
* **Fractional Share Execution**: Automatically enforces `time_in_force='day'` and `type='market'` as required by Alpaca's fractional engine.
* **Symbol Formatting**: Normalizes class shares to dot notation (`BRK.B`).
* **Environment**: Seamless switching between paper trading (`paper-api.alpaca.markets`) and live trading (`api.alpaca.markets`).

### 11.2 Trading 212 REST API v0 (`Trading212BrokerClient`)
* **Protocol**: HTTP Basic Authentication (`Authorization: Basic <base64(KEY:SECRET)>`) with legacy raw token fallback.
* **Instrument Ticker Formatting**: Maps US equities to standard Trading 212 exchange identifiers (e.g. `AAPL_US_EQ`, `BRK_B_US_EQ`).
* **Signed Quantities**: Market orders transmit positive quantities for BUY orders and negative quantities for SELL orders.
* **Execution Price Tracking**: Extracts `filledQuantity` and `filledValue` from fill receipts to compute exact average execution prices.

### 11.3 Interactive Brokers (IBKR) Basket Trader
* **Protocol**: Formats rebalance orders into pure CSV tickets compatible with TWS Basket Trader:
  ```csv
  Action,Symbol,SecType,Exchange,Quantity,OrderType
  BUY,AAPL,STK,SMART,12,MKT
  BUY,MSFT,STK,SMART,6,MKT
  ```
* **Clean Mode**: Eliminates comment headers to ensure direct, error-free import into TWS.

### 11.4 Dry-Run Paper Trading Sandbox (`DryRunBrokerClient`)
* Runs completely offline with zero API keys required.
* Executes simulated market orders against live consolidated market prices.
* Persists transactions into `portfolio_ledger.db`.

---

## 12. USER INTERFACE & OPERATIONS TERMINAL ARCHITECTURE

### 12.1 Unified Web Terminal Architecture
* **Stack**: Native HTML5, modern vanilla JavaScript, Tailwind CSS (XTB xStation / Bloomberg dark-mode palette).
* **Backend**: FastAPI REST server (`src/dashboard/server.py`) serving static assets and JSON endpoints.
* **Latency Profile**: Sub-5ms response time backed by pre-computed atomic snapshots (`today_snapshot.json`).
* **Interactive Modules**:
  - **Stock Inspector**: Real-time TreeSHAP waterfall charts, fundamental valuation multiples ($E/P, P/E, B/P, P/B, S/P, P/S, FCF$, $EV/\text{EBITDA}$), and technical momentum indicators.
  - **Sector Exposure Drawer**: Dynamic progress bars against the 25% sector cap, constituent badges, and Barra factor orthogonality matrix.
  - **Intraday Risk Monitor**: Real-time position status (Healthy, Caution, Breached), trailing stop floors, and global panic liquidation controls.

### 12.2 Operations Alert Dispatcher (`src/dashboard/alerts.py`)
* Dispatches operational notifications via Telegram Bot API and local SQLite logging.
* **HTML Sanitization**: All error messages, stack traces, and dynamic variables are sanitized using `html.escape(..., quote=False)` to prevent Telegram entity parsing crashes on unescaped `<module>`, `&`, or brackets.

---

## 13. COMPREHENSIVE PROJECT ROADMAP (PHASES 1 TO 6)

### Phase 1: Data Foundations & Architectural Purity (Completed)
* Standardized S&P 100 constituent definition and sector ETF mapping.
* Incremental OHLCV Parquet caching with automatic split and dividend reconciliation.
* FRED macroeconomic pipeline with public CSV fallback.
* Headline news ingestion with deterministic SHA-256 deduplication and 4:00 PM ET cutoff rules.

### Phase 2: NLP Sentiment, Tree Ensembles & Explainability (Completed)
* FinBERT transformer integration with exponential time-decay aggregation.
* Constrained shallow LightGBM return ranker with multi-seed averaging.
* 50/50 Percentile rank blending with linear Ridge shrinkage baseline.
* TreeSHAP feature attribution with investor plain-English translation.

### Phase 3: Risk Budgeting, Turnover Hysteresis & Diagnostics (Completed)
* Inverse volatility risk parity sizing with $8.0\%$ individual stock caps.
* Asymmetric hysteresis buffer ($[0.60, 0.90)$ holding band) cutting turnover by over $50\%$.
* HAR-RV volatility forecasting model and 3-State Gaussian HMM market regime detector.
* Rolling 60-day Spearman IC walk-forward monitor and Confidence Light state machine.

### Phase 4: Institutional Terminal & Multi-Broker Execution (Completed)
* High-performance single-page HTML5/Tailwind trading terminal with XTB-inspired styling.
* Native Alpaca Trading API v2 bridge supporting fractional day orders.
* Native Trading 212 API v0 bridge supporting HTTP Basic Auth and share class normalization.
* Interactive Brokers (IBKR) TWS Basket Trader pure CSV generator.
* SQLite transactional ledger recording fills, snapshots, and alerts.
* Zero-emoji codebase audit and Telegram HTML sanitization.

### Phase 5: Factor Neutralization, Fundamentals & Intraday Risk (Completed)
* **Barra Factor Risk Model**: 11 GICS sector dummies and 4 standardized style factors (Size, Volatility, Momentum, Value) with regularized projection residualization ($\max |\rho| < 10^{-6}$).
* **Fundamental Valuations Pipeline**: Sourcing and standardizing $E/P, B/P, S/P, FCF/P, EV/\text{EBITDA}$, dividend yield, and log market cap with deterministic offline fallback benchmarks.
* **Sector-Neutral Portfolio Constraints**: Maximum 25% single sector cap, maximum 3 names per sector, and iterative dual-capping weight redistribution.
* **Intraday Risk Circuit Breakers**: -7% hard entry drawdown, -7% trailing high-water mark stop, and $2.5\times$ ATR volatility stops with automated liquidation to cash or hedged $SPY$.

### Phase 6: Institutional Scale & Multi-Asset Expansion (Active Roadmap)
* **Intraday Execution Slicing (TWAP / VWAP)**:
  Dynamic volume-weighted and time-weighted order slicing over 30-minute intervals to minimize market impact on larger equity orders.
* **Alternative Data Signals**:
  Integrate SEC Form 13F institutional ownership tracking, corporate insider buy/sell transactions, and Wikipedia pageview momentum.
* **Multi-Asset Macro Hedging**:
  Introduce liquid futures contracts (E-mini S&P 500, 10-Year Treasury Notes, CBOE VIX Futures) to dynamically hedge portfolio systematic beta during High Volatility Crisis regimes.
* **Multi-Tenant Deployment & SOC 2 Compliance**:
  Containerize the deployment stack with Docker and Kubernetes, adding OAuth2 role-based access control, encrypted audit trails, and multi-account broker routing.

---
*Argus Quant System Manual — Institutional Version 2.3.0*
