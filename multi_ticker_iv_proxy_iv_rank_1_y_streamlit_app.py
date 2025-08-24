# Multi-Ticker IV Proxy & IV Rank (1y)
# ---------------------------------------------------------------
# NOTE ABOUT DATA: 
# This app computes a *proxy* for 30-day implied volatility (IV) using
# a market-standard EWMA (RiskMetrics) volatility model on daily returns
# and annualizes it, since free, reliable historical option-implied
# volatility time series per ticker are not available via yfinance.
# The "IV Rank" is then computed over the last 252 trading days using
# this IV proxy. If you have an options IV data source, you can plug it
# into `get_iv_series()` to replace the proxy with true option-implied IV.
# ---------------------------------------------------------------

import io
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st
import plotly.graph_objects as go

# ------------------------------
# Config
# ------------------------------
st.set_page_config(
    page_title="IV & IV Rank (1y) – Multi-Ticker",
    layout="wide",
)

# ------------------------------
# Helpers
# ------------------------------

def fetch_prices(ticker: str, period_days: int = 400) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV for ~last `period_days` calendar days.
    We pull a bit more than 252 trading days to handle holidays/missing days."""
    start = (datetime.utcnow() - timedelta(days=period_days)).date().isoformat()
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)  # Ensure 'Close'
    return df


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA daily volatility (std)."""
    # Initialize with sample variance to avoid long warmup
    var = []
    prev = returns.var() if returns.notna().sum() > 1 else 0.0
    for r in returns.fillna(0.0):
        v = lam * prev + (1 - lam) * (r ** 2)
        var.append(v)
        prev = v
    vol = np.sqrt(np.array(var))
    return pd.Series(vol, index=returns.index)


def annualize_daily_vol(daily_vol: pd.Series, trading_days: int = 252) -> pd.Series:
    return daily_vol * np.sqrt(trading_days)


def get_iv_series(ticker: str, use_proxy: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns ['Close', 'IV'] indexed by date.

    - If use_proxy=True (default):
      Compute IV proxy = annualized EWMA of daily log returns (RiskMetrics).
    - If you have true option-implied IV (e.g., 30d ATM), replace the
      implementation here with your data fetch and set use_proxy=False if desired.
    """
    prices = fetch_prices(ticker)
    if prices.empty or 'Close' not in prices.columns:
        return pd.DataFrame()

    close = prices['Close']
    # Log returns to avoid negative pricing issues
    logret = np.log(close / close.shift(1))

    if use_proxy:
        daily_vol = ewma_vol(logret)
        iv_series = annualize_daily_vol(daily_vol)
    else:
        # Placeholder: replace with real implied volatility series if available.
        daily_vol = ewma_vol(logret)
        iv_series = annualize_daily_vol(daily_vol)

    out = pd.DataFrame({
        'Close': close,
        'IV': iv_series,
    }).dropna()
    return out


def compute_iv_rank(iv: pd.Series, lookback: int = 252) -> pd.Series:
    """Compute IV Rank over a rolling `lookback` window:
       IVRank_t = (IV_t - min(IV_{t-lookback+1..t})) / (max(...) - min(...))
    Returns a series aligned to `iv`.
    """
    roll_min = iv.rolling(window=lookback, min_periods=lookback//2).min()
    roll_max = iv.rolling(window=lookback, min_periods=lookback//2).max()
    rank = (iv - roll_min) / (roll_max - roll_min)
    return rank.clip(0, 1)


def summarize_current_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"price": np.nan, "iv": np.nan, "iv_rank": np.nan}
    iv_rank_series = compute_iv_rank(df['IV'])
    latest_date = df.index.max()
    return {
        "price": float(df.loc[latest_date, 'Close']),
        "iv": float(df.loc[latest_date, 'IV']),
        "iv_rank": float(iv_rank_series.loc[latest_date]) if not np.isnan(iv_rank_series.loc[latest_date]) else np.nan,
    }


def make_iv_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['IV'], mode='lines', name=f'{ticker} IV (proxy)'))
    fig.update_layout(
        title=f"{ticker} – IV (proxy) last 1y",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def make_ivrank_chart(ticker: str, iv_rank: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iv_rank.index, y=iv_rank, mode='lines', name=f'{ticker} IV Rank'))
    fig.update_layout(
        title=f"{ticker} – IV Rank (rolling 252d)",
        xaxis_title="Date",
        yaxis_title="IV Rank (0–1)",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        yaxis=dict(range=[0,1]),
    )
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

# ------------------------------
# UI
# ------------------------------

st.title("📈 Multi-Ticker IV (Proxy) & IV Rank – Last Year")

with st.expander("About this app"):
    st.markdown(
        """
        - **IV Source**: This app uses an **EWMA volatility proxy** for implied volatility because
          historical option-implied volatility time series are not available through `yfinance`.
          If you have a provider for true IV (e.g., 30d ATM option-implied vol), you can
          replace `get_iv_series()` with your data call.
        - **IV Rank**: Computed over a 252-trading-day rolling window using the proxy IV.
        - **Tip**: Longer lookbacks may need more history; we fetch ~400 days of prices to
          comfortably cover a year of trading days.
        """
    )

colA, colB = st.columns([3, 1])
with colA:
    tickers_input = st.text_area(
        "Enter tickers (comma, space, or newline separated)",
        value="AAPL, MSFT, NVDA, TSLA",
        height=90,
    )
with colB:
    lookback = st.number_input("IV Rank lookback (days)", min_value=100, max_value=504, value=252, step=1)
    lam = st.slider("EWMA lambda (decay)", min_value=0.80, max_value=0.99, value=0.94, step=0.01)

# Allow custom lambda by monkey-patching inside ewma call

def get_iv_series_with_lambda(ticker: str) -> pd.DataFrame:
    prices = fetch_prices(ticker)
    if prices.empty or 'Close' not in prices.columns:
        return pd.DataFrame()
    close = prices['Close']
    logret = np.log(close / close.shift(1))
    daily_vol = ewma_vol(logret, lam=lam)
    iv_series = annualize_daily_vol(daily_vol)
    return pd.DataFrame({'Close': close, 'IV': iv_series}).dropna()

# Parse tickers
raw = [t.strip().upper() for t in tickers_input.replace("\n", ",").replace(" ", ",").split(",") if t.strip()]
tickers: List[str] = sorted(list(dict.fromkeys(raw)))

st.write(f"Scanning **{len(tickers)}** tickers…")

# Aggregated table
rows = []
series_store: Dict[str, pd.DataFrame] = {}

for t in tickers:
    try:
        data = get_iv_series_with_lambda(t)
        if data.empty:
            rows.append({"Ticker": t, "Price": np.nan, "IV (proxy)": np.nan, "IV Rank (252d)": np.nan})
            continue
        iv_rank = compute_iv_rank(data['IV'], lookback=lookback)
        series_store[t] = data.join(iv_rank.rename('IV_RANK'))
        m = summarize_current_metrics(data)
        rows.append({
            "Ticker": t,
            "Price": round(m["price"], 4) if pd.notna(m["price"]) else np.nan,
            "IV (proxy)": round(m["iv"], 4) if pd.notna(m["iv"]) else np.nan,
            "IV Rank (252d)": round(m["iv_rank"], 4) if pd.notna(m["iv_rank"]) else np.nan,
        })
    except Exception as e:
        rows.append({"Ticker": t, "Price": np.nan, "IV (proxy)": np.nan, "IV Rank (252d)": np.nan})

summary_df = pd.DataFrame(rows)

st.subheader("Results (latest values)")
st.dataframe(summary_df, use_container_width=True)

# Download combined CSV
if series_store:
    combined = []
    for t, df in series_store.items():
        tmp = df.copy()
        tmp.insert(0, 'Ticker', t)
        combined.append(tmp.reset_index().rename(columns={'Date': 'Date'}))
    full_df = pd.concat(combined, axis=0, ignore_index=True)
    st.download_button(
        label="Download full time series (CSV)",
        data=df_to_csv_bytes(full_df),
        file_name="iv_proxy_ivrank_1y.csv",
        mime="text/csv",
    )

st.markdown("---")
st.subheader("Charts (per ticker)")

for t in tickers:
    df = series_store.get(t)
    if df is None or df.empty:
        st.warning(f"No data for {t}.")
        continue
    iv_rank = df['IV_RANK']
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_iv_chart(t, df), use_container_width=True)
    with c2:
        st.plotly_chart(make_ivrank_chart(t, iv_rank), use_container_width=True)

st.markdown("---")
st.caption(
    "Educational use only. This is not investment advice. Data via Yahoo Finance (yfinance).\n"
    "IV shown is a **proxy** derived from EWMA of daily returns and annualized."
)
