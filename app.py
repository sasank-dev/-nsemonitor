"""
Sift — AI-ranked Stock Screener
Auto-ranks NSE/NYSE stocks by a composite momentum score.
Optional Claude AI explanations per pick (BYO API key).
"""

from flask import Flask, render_template, jsonify, request, Response
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import csv
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─── Ticker universe ─────────────────────────────────────────────────
NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "TITAN.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS",
    "COALINDIA.NS", "GRASIM.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "BPCL.NS", "IOC.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "TECHM.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "UPL.NS", "HINDALCO.NS",
    "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "PIDILITIND.NS", "AMBUJACEM.NS",
    "ACC.NS", "SHREECEM.NS", "DLF.NS", "GAIL.NS", "PETRONET.NS",
    "HAVELLS.NS", "VOLTAS.NS", "PAGEIND.NS", "BERGEPAINT.NS", "BIOCON.NS",
    "LUPIN.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS", "MUTHOOTFIN.NS",
    "ICICIPRULI.NS", "ICICIGI.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS",
]

SP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "CRM",
    "NFLX", "PFE", "KO", "PEP", "INTC", "CSCO", "VZ", "T", "ORCL", "ABT",
    "MRK", "XOM", "CVX", "NKE", "TMO", "ABBV", "ACN", "AVGO", "QCOM", "TXN",
    "COST", "MCD", "DHR", "LLY", "MDT", "WFC", "BMY", "PM", "UPS", "LOW",
    "AMGN", "SBUX", "HON", "IBM", "GS", "MS", "BLK", "AMD", "BA", "CAT",
    "DE", "GE", "MMM", "RTX", "LMT", "AMT", "PLD", "EQIX", "SPGI", "INTU",
    "ISRG", "NOW", "PYPL", "AMAT", "ADI", "BKNG", "GILD", "MO", "TGT", "DUK",
]


def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def analyze_ticker(ticker: str):
    """Fetch + compute all metrics for one ticker. Returns dict or None."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", auto_adjust=False)
        if hist.empty or len(hist) < 21:
            return None

        closes = hist["Close"].dropna()
        current_price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0

        volumes = hist["Volume"]
        avg_vol_20 = float(volumes.iloc[-21:-1].mean())
        current_vol = float(volumes.iloc[-1])
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0

        rsi = calculate_rsi(closes)

        sma_20 = float(closes.rolling(20).mean().iloc[-1])
        above_sma20 = current_price > sma_20

        # 1-month return (~22 trading days back)
        if len(closes) >= 22:
            month_ago = float(closes.iloc[-22])
            one_month_return = ((current_price - month_ago) / month_ago) * 100 if month_ago else 0.0
        else:
            one_month_return = 0.0

        info = stock.info or {}
        pe = info.get("trailingPE")
        try:
            pe = float(pe) if pe is not None else None
        except (TypeError, ValueError):
            pe = None

        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "—")
        is_nse = ticker.endswith(".NS")

        return {
            "ticker": ticker.replace(".NS", ""),
            "name": (name or ticker)[:40],
            "sector": sector,
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "one_month_return": round(one_month_return, 2),
            "pe": round(pe, 2) if pe is not None else None,
            "vol_ratio": round(vol_ratio, 2),
            "rsi": round(float(rsi), 2) if not np.isnan(rsi) else None,
            "above_sma20": above_sma20,
            "currency": "₹" if is_nse else "$",
            "raw_ticker": ticker,
        }
    except Exception:
        return None


def momentum_score(m: dict) -> float:
    """
    Composite momentum score (0–100ish).
    Rewards: bullish RSI (not overbought), volume confirmation,
    positive 1-month return, price above 20-day SMA, today's move.
    """
    score = 0.0

    # RSI: sweet spot 55–70, penalize overbought
    rsi = m.get("rsi")
    if rsi is not None:
        if rsi < 40:    score += 0
        elif rsi < 50:  score += (rsi - 40) * 1.5         # 0 → 15
        elif rsi <= 65: score += 15 + (rsi - 50) * 1.0    # 15 → 30
        elif rsi <= 75: score += 30 - (rsi - 65) * 1.0    # 30 → 20
        else:           score += 15                       # overbought

    # Volume confirmation
    vol = m.get("vol_ratio") or 0
    score += min(25, vol * 8)

    # 1-month return
    mret = m.get("one_month_return") or 0
    if mret > 0:
        score += min(20, mret * 1.2)
    else:
        score += max(-10, mret * 1.0)

    # Above 20-day SMA = trend
    if m.get("above_sma20"):
        score += 15

    # Today's move
    daily = m.get("change_pct") or 0
    if daily > 0:
        score += min(10, daily * 2)

    return round(max(0, score), 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True) or {}
    market = data.get("market", "NSE").upper()
    limit = int(data.get("limit", 10))

    tickers = NIFTY_TICKERS if market == "NSE" else SP_TICKERS
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(analyze_ticker, tickers):
            if r:
                r["score"] = momentum_score(r)
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:limit]

    return jsonify({
        "results": top,
        "matched": len(top),
        "scanned": len(tickers),
        "fetched": len(results),
        "market": market,
        "ai_env_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


@app.route("/api/explain", methods=["POST"])
def explain():
    """Use Claude Haiku to explain why a stock looks strong."""
    data = request.get_json(silent=True) or {}
    stock = data.get("stock")
    if not stock:
        return jsonify({"error": "Missing stock data"}), 400

    api_key = request.headers.get("X-API-Key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "No API key set. Open Settings to add one."}), 400

    prompt = (
        f"In exactly 2 short sentences, plain English, explain why {stock.get('ticker')} "
        f"({stock.get('name')}) is showing strong momentum right now. Reference the "
        f"specific numbers below. No financial advice, no jargon.\n\n"
        f"Metrics:\n"
        f"- Price: {stock.get('currency')}{stock.get('price')} ({stock.get('change_pct')}% today)\n"
        f"- 1-month return: {stock.get('one_month_return')}%\n"
        f"- RSI(14): {stock.get('rsi')}\n"
        f"- Volume vs 20-day average: {stock.get('vol_ratio')}x\n"
        f"- P/E ratio: {stock.get('pe')}\n"
        f"- Trading above 20-day moving average: {stock.get('above_sma20')}\n"
        f"- Sector: {stock.get('sector')}"
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if resp.status_code != 200:
            try:
                err = resp.json().get("error", {}).get("message", f"API error ({resp.status_code})")
            except Exception:
                err = f"API error ({resp.status_code})"
            return jsonify({"error": err}), resp.status_code
        body = resp.json()
        text = body["content"][0]["text"].strip()
        return jsonify({"explanation": text})
    except requests.Timeout:
        return jsonify({"error": "Anthropic API timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export_csv():
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])
    if not results:
        return Response("No data", status=400)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["ticker", "name", "sector", "price", "change_pct",
                    "one_month_return", "pe", "vol_ratio", "rsi", "score", "currency"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(results)

    filename = f"sift_picks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    print("\n  Sift — AI-ranked Stock Screener")
    print("  http://127.0.0.1:5000\n")
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  ✓ ANTHROPIC_API_KEY found in environment")
    else:
        print("  ⓘ No API key in env — add one in the UI Settings for AI explanations")
    print()
    app.run(debug=False, port=5000, host="127.0.0.1")
