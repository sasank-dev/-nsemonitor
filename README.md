# BOURSE — Equity Screener

A free stock screener for **NSE (India)** and **NYSE (US)** equities. Same engine as the YouTube tutorial — yfinance, P/E + Volume + RSI filters — but with a custom Flask + HTML frontend instead of Streamlit.

## What it does

Scans a universe of large-cap tickers and surfaces stocks where, **simultaneously**:
- P/E ratio is below your threshold (default 20) — not overpriced
- Today's volume is N× the 20-day average (default 2×) — interest is building
- RSI(14) is above your threshold (default 50) — momentum tilted bullish

Results are ranked by a composite score, exportable as CSV.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run it
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

The first scan will take 30–60 seconds because yfinance fetches each ticker's `.info` (P/E ratio data). Subsequent scans are faster thanks to yfinance's internal caching.

## Customize

- **Add tickers:** edit `NIFTY_TICKERS` or `SP_TICKERS` lists at the top of `app.py`. NSE tickers need the `.NS` suffix.
- **Change the universe:** replace either list with your own watchlist.
- **Tune filters:** the UI sliders cover sensible ranges, but you can change min/max in `templates/index.html`.
- **Add metrics:** the `analyze_ticker()` function returns a dict — add any field yfinance exposes (market cap, dividend yield, beta, etc.) and surface it in the table.

## Project structure

```
stock_screener/
├── app.py              # Flask backend + yfinance logic
├── requirements.txt    # flask, yfinance, pandas, numpy
├── README.md           # this file
└── templates/
    └── index.html      # the entire frontend (HTML + CSS + JS)
```

## Notes

- **Data source:** yfinance scrapes Yahoo Finance — it's free and unofficial. If a request batch hits a rate limit, wait a minute and retry.
- **Not investment advice.** This is a learning tool that ranks stocks by mechanical rules. Treat the output as a starting point for further research, not a buy list.
