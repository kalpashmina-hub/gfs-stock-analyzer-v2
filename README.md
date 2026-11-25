# GFS Stock Analyzer — Version 2

A Streamlit application implementing the user's Grandfather–Father–Son framework for Indian equities:

- **Grandfather:** Monthly trend (35% weight)
- **Father:** Weekly trend (35% weight)
- **Son:** Daily trend (30% weight)

## Included in V2

- EOD history via Yahoo Finance (`yfinance`) or user-uploaded OHLCV CSV.
- Non-destructive OHLCV quality checks before cleaning/scoring, including duplicate dates, malformed candles, zero-volume bars, large moves and calendar gaps; structural errors stop analysis, while review flags remain visible.
- Corporate-action event availability and price-adjustment provenance reporting; adjusted prices remain unchanged in the existing GFS calculation path.
- Dynamic Yahoo Finance quarterly/annual income statement, balance-sheet and cash-flow tables for the selected ticker, when the provider supplies them.
- Per-ticker CSV exports for the provider fundamental snapshot and normalized raw financial-statement line items; the TCS pilot filing CSV is shown only when TCS is selected.
- Independent calculation of EMA 10/20/50, SMA 50/200, RSI 14, RSI 9-period average, MACD histogram and 20-session volume average.
- Bollinger Bands (20,2), %B, bandwidth and a trailing low-volatility squeeze diagnostic, displayed separately for entry/volatility context and excluded from GFS scoring.
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

- Yahoo fundamentals and statement tables can be missing, stale, incomplete or defined differently by the provider. Statement line items retain provider labels and reporting columns but are not independently verified exchange filings. Verify periods, units, debt/equity, growth, cash and debt against company filings.
- Provider statement CSV exports preserve reported labels and period columns but do not infer missing units, consolidated/standalone basis, or accounting standards. Use official filing intake for source-linked, period-specific figures.
- Market-data quality checks are diagnostic, not exchange authentication. Calendar gaps are not classified as missing sessions without an authoritative session calendar; unusual moves require news/corporate-action review. Yahoo's adjustment policy is recorded as requested, not independently proven.
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


### TCS Q2 FY27 pilot dataset — interpretation note

The prepared pilot CSV is a primary-source extract, not a complete fundamental model. It keeps separate measures for:
- `net_profit_for_period`: consolidated profit for the period before attribution between owners and non-controlling interests.
- `net_profit_attributable_to_owners`: profit attributable to equity owners of the parent.

These are not interchangeable. For Q2 FY27 the reported consolidated values are ₹13,934 crore and ₹13,884 crore respectively. For Q1 FY27 comparative values they are ₹13,420 crore and ₹13,349 crore respectively. The CSV uses the quarter-end dates and the audited filing date, not a generic provider quarter-reference date.

The pilot contains primary-source rows from two distinct publishers: NSE Integrated Filing rows for Ind AS financial-statement measures, and TCS investor-relations rows for selected IFRS cash-flow, margin and balance-sheet measures. These are both primary disclosures, but they are not interchangeable accounting presentations. Secondary-source rows were removed because matching numbers copied into a secondary-source row would create a misleading appearance of independent corroboration. A genuine secondary cross-check must be independently retrieved, with its own source record, and must match metric definition, period, basis and unit before comparison.

Source reference: TCS audited consolidated interim financial results for the quarter and six months ended 30 September 2026, approved 8 October 2026. See the NSE Integrated Filing record and the company's investor-relations results. No fundamental score is produced from this pilot.


### TCS pilot extension: cash flow and balance-sheet coverage

The TCS Q2 FY27 pilot now includes quarterly operating cash flow, capital expenditure, free cash flow, operating-cash-flow conversion ratios, reported operating/net margins, and selected IFRS consolidated balance-sheet measures. Cash-flow values are quarter-specific; balance-sheet values are point-in-time at 30 September 2026. Units and accounting presentation are recorded per row, so Ind AS and IFRS values are not mixed in a single comparison.

The app also calculates like-for-like revenue and cash-flow growth only when the same metric, period basis, unit and exact comparison quarter exist. It deliberately withholds automated profit-growth calculations until exceptional-item treatment and Ind AS/IFRS comparability have been reconciled. A source label is metadata, not proof of independent verification. Secondary-source matching values are not added unless independently retrieved and checked.

No fundamental score is produced yet. The pilot remains a data-quality and reconciliation exercise; the original Yahoo snapshot is not overwritten.


### Fundamental intake quality checks

The source-aware intake now validates non-empty metadata, finite numeric values,
HTTP(S) source links, reporting dates, reporting basis and duplicate observations.
Source-domain labels are provenance hints only; they do not authenticate that a
filing was opened or independently checked. Conflicting observations are never
selected automatically.

Run the focused regression suite from the repository root:

```bash
python -m pip install "pandas>=2.2"
python -m unittest discover -s tests -p "test_*.py" -v
```

A GitHub Actions workflow runs the same tests for pushes and pull requests to
`v2-review-upgrade`. The suite covers the TCS pilot CSV, source-domain boundaries,
date and numeric validation, duplicate-safe growth and source comparisons.

**TCS pilot review caveat:** the CSV intentionally contains both Ind AS filing
metrics and IFRS release metrics. Keep these accounting presentations distinct.
The Q1 FY27 revenue, PBT and profit rows now point to the July 9, 2026 Q1 filing;
the Q1 profit values were corrected to ₹13,420 crore before non-controlling
interests and ₹13,349 crore attributable to owners; EPS is ₹36.90. The Q1
exceptional item of ₹668 crore is separately recorded, so reported PBT is
₹17,944 crore versus ₹18,612 crore before exceptional items. The Q1 filing URL
and reconciled values are asserted by regression tests. Continue to compare
only identical accounting standards, metric definitions, periods and units.
No fundamental score is calculated.


## Non-destructive market-data quality audit (review branch)

The optional `market_data_quality.py` module audits daily OHLCV files independently of the GFS indicator and scoring engine. It never repairs, deletes, adjusts, or silently selects among source observations. Every row-level finding includes the original input row number so a reviewer can trace it back to the uploaded/exported file.

Checks include:
- Invalid/missing dates, duplicate dates, and out-of-order observations.
- Missing, non-numeric, infinite, non-positive or structurally impossible OHLC values.
- Missing/non-numeric volume, negative volume, zero-volume observations and flat zero-volume bars.
- Large close-to-close movements (review flags, not automatic rejection).
- Long calendar gaps (informational only; not automatically called missing exchange sessions).
- Observations that fall on dates supplied as exchange non-trading days.
- Missing sessions only when the caller supplies an explicit calendar containing an `IsTradingSession` boolean for each date.
- Corporate-action event availability/counts and price-adjustment provenance metadata.

### Exchange-calendar boundary

The module does not download, maintain, or claim to authenticate an official NSE/BSE calendar. Supply an authoritative calendar from the exchange or a trusted licensed source. A simple iterable of dates identifies non-trading dates; a DataFrame with `Date` and `IsTradingSession` can additionally identify expected trading sessions. This distinction prevents weekends/holidays from being mistaken for missing market data and avoids treating an unverified holiday list as an authoritative session schedule.

### Corporate actions and adjusted prices

`audit_actions()` counts only dividend/split events actually supplied by the caller. Missing event data is explicitly reported as unavailable, not as proof that no corporate actions occurred. `audit_price_adjustment_metadata()` records provider, interval, adjustment flags and related provenance when supplied; it cannot independently prove that a provider applied those settings correctly. Yahoo Finance/yfinance `auto_adjust=True` adjusts OHLC prices, while `actions=False` omits dividend/split event columns. The optional yfinance `repair` setting is a provider-side repair attempt, not exchange verification. See the [yfinance history reference](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html).

Example:

```python
import pandas as pd
from market_data_quality import audit_ohlcv, audit_actions, audit_price_adjustment_metadata

prices = pd.read_csv("daily_ohlcv.csv")
report = audit_ohlcv(prices, exchange="NSE")
print(report["findings"].to_string(index=False))

# Pass an authoritative calendar when available:
# calendar columns: Date, IsTradingSession, Reason
# report = audit_ohlcv(prices, exchange_calendar=calendar, exchange="NSE")

actions_report = audit_actions(None)  # explicitly reports unavailable
provenance = audit_price_adjustment_metadata({
    "provider": "Yahoo Finance", "interval": "1d",
    "auto_adjust": True, "actions": False, "exchange": "NSE",
})
```

Run the full regression suite from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

This is a review-branch utility only. It is not integrated into the live app and does not alter any indicator formula, GFS score, Fibonacci calculation, or source data. Do not switch adjustment or repair settings in the live scoring path without a separate comparison and approval.
