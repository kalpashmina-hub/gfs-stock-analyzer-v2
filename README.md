# GFS Stock Analyzer — Version 2

A Streamlit application implementing the user's Grandfather–Father–Son framework for Indian equities:

- **Grandfather:** Monthly trend (35% weight)
- **Father:** Weekly trend (35% weight)
- **Son:** Daily trend (30% weight)

## Included in V2

- EOD history via Yahoo Finance (`yfinance`) or user-uploaded OHLCV CSV.
- Independent calculation of EMA 10/20/50, SMA 50/200, RSI 14, RSI 9-period average, MACD histogram and 20-session volume average.
- Weighted GFS checklist score out of 100 and a transparent state classification for each timeframe.
- Daily candlestick chart, RSI and MACD panels.
- Confirmed multi-timeframe swing-pivot support/resistance and volatility-filtered daily Fibonacci retracement estimates.
- Separate provisional current-month recovery diagnostics (excluded from the confirmed GFS score).
- Data-freshness warnings and transparent score/checklist diagnostics.
- Provider-supplied fundamental snapshot when Yahoo returns those fields.
- Downloadable daily-indicator and GFS-summary CSVs.

## Run on Windows / Mac / Linux

1. Install Python 3.10 or newer.
2. Extract this ZIP and open a terminal in the `gfs_stock_analyzer` folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local address shown in the terminal (usually `http://localhost:8501`).
6. Enter an NSE symbol without `.NS` (for example `INFY`, `TCS`, `STLTECH`) or upload a daily OHLCV CSV.

## CSV format

A CSV should have these columns (case-insensitive):

```csv
Date,Open,High,Low,Close,Volume
2026-01-01,1500,1520,1490,1510,123456
```

Use at least 220 daily sessions for the daily SMA 200. Prefer 3–5 years of daily data to make monthly and weekly indicators more meaningful. Timezone-aware timestamps are converted to India time; timezone-naive dates are assumed to be Indian exchange dates. The app checks basic OHLC consistency but cannot verify the licensing, corporate-action adjustment or full correctness of an uploaded file; use data you are authorised to use.

## Data-source reliability

The Yahoo Finance route is **free and unofficial**. It may be delayed, incomplete, blocked, adjusted differently, or change without notice. It is useful for prototyping and personal EOD research, not as a guaranteed exchange feed. The CSV upload route lets you use an EOD export from a provider whose terms you have checked. The app does not claim that NSE or BSE offers an unrestricted free API.

For a production upgrade, connect a licensed EOD provider through a dedicated data adapter, verify corporate-action adjustment policy, record provider and last-updated timestamps, and compare indicator values with TradingView on a test basket before using real capital.

## How to interpret the score

Each timeframe scores the percentage of its checklist conditions met. Monthly and weekly each contribute 35%; daily contributes 30%. This is a rule-based checklist score, **not** a probability of profit and not a validated predictive model. RSI 40–60 is an assessment zone rather than a hard filter; RSI above 60 may reflect strong momentum, and RSI below 30 is not an automatic buy.

The labels (Strong Buy Candidate, Buy on Confirmation, Watchlist, Avoid Fresh Entry) are screening categories only. Check valuation, results, debt, cash flow, promoter/shareholding disclosures, corporate announcements, liquidity, portfolio concentration, entry, stop-loss and reward/risk independently. No orders are placed.

## Known limitations in V2

- Yahoo fundamentals can be missing, stale or defined differently by the provider. The most-recent-quarter field is only a reference date; individual fields may use different periods. Verify debt/equity, growth, cash and debt against company filings.
- EOD data freshness is checked using calendar-day age; holidays and provider delays can still require manual judgment.
- Support/resistance pivots and Fibonacci levels are mechanical estimates; validate swing anchors visually.
- No live feed, broker integration, automated alerts, backtesting or trade execution.
- No guarantee of data availability or investment outcome.


## Fundamental filing intake (beta)

The app includes a source-aware CSV intake for fundamental figures. It is intentionally separate from the existing Yahoo Finance snapshot and does not yet generate a fundamental score.

1. Open the **Fundamental snapshot** section and download the filing CSV template.
2. For each metric, enter its numeric value, period end, reporting basis (exactly `consolidated` or `standalone`), unit, source name, source URL and filing date.
3. Prefer the original NSE/BSE filing or the company's investor-relations filing. Use free secondary sites only as cross-checks, not as the primary authority.
4. Do not combine consolidated and standalone values, quarterly and annual periods, or unlike units.
5. If an original filing has been revised, resolve which version is authoritative before uploading. The app flags duplicate metric/period/basis rows rather than silently choosing one.
6. The app reports differences only when metric, period, basis and unit match. A difference is a review flag, not proof that one source is wrong.

NSE filing pages:
- Financial Results: https://www.nseindia.com/companies-listing/corporate-filings-financial-results
- Integrated Filing – Financials: https://www.nseindia.com/companies-listing/corporate-integrated-filing
- XBRL filing information and taxonomy: https://www.nseindia.com/static/companies-listing/xbrl-information

Current limitation: this beta uses a user-supplied normalized CSV. It does not automatically download or scrape exchange filings. Automatic ingestion will be added only after stable access, file formats, source terms and revision handling are verified. No fundamental score is calculated from unverified or incomplete data.
