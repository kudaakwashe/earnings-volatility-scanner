import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="IV Rank & Volatility Scanner", layout="wide")

# Utility Functions
def get_atm_iv(stock, expiry_date, price):
    try:
        chain = stock.option_chain(expiry_date)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return None
        call_idx = (calls['strike'] - price).abs().idxmin()
        put_idx = (puts['strike'] - price).abs().idxmin()
        call_iv = calls.loc[call_idx, 'impliedVolatility']
        put_iv = puts.loc[put_idx, 'impliedVolatility']
        return (call_iv + put_iv) / 2.0
    except:
        return None

def get_iv_history(ticker, days=252):
    stock = yf.Ticker(ticker)
    today = datetime.today().date()
    expiries = stock.options
    if not expiries:
        return [], None, 'No expiries'

    expiry_date = min(expiries, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - today).days - 30))
    iv_series = []
    date_series = []

    for offset in range(days, 0, -1):
        past_date = today - timedelta(days=int(offset * 1.5))
        hist_price = stock.history(start=past_date, end=past_date + timedelta(days=1))
        if hist_price.empty:
            continue
        price_at_time = hist_price['Close'].iloc[0]
        iv = get_atm_iv(stock, expiry_date, price_at_time)
        if iv:
            iv_series.append(iv)
            date_series.append(past_date)

    current_price = stock.history(period='1d')['Close'].iloc[-1]
    current_iv = get_atm_iv(stock, expiry_date, current_price)

    return pd.Series(iv_series, index=date_series), current_iv, None

def compute_hv(price_series, window=30):
    log_returns = np.log(price_series / price_series.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)

# Streamlit UI
st.title("📈 Batch IV Rank & Volatility Visualizer")
ticker_input = st.text_area("Enter tickers (comma-separated):", "AAPL,MSFT,TSLA")
run_button = st.button("Run Scan")

if run_button:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    summary_rows = []

    for ticker in tickers:
        st.subheader(f"🔍 {ticker}")
        iv_series, current_iv, err = get_iv_history(ticker)

        if err:
            st.error(f"{ticker}: {err}")
            continue

        if len(iv_series) < 30:
            st.warning(f"{ticker}: Not enough IV data.")
            continue

        iv_rank = round(100 * (current_iv - iv_series.min()) / (iv_series.max() - iv_series.min()), 2)
        summary_rows.append({
            "Ticker": ticker,
            "IV Rank": iv_rank,
            "Current IV": round(current_iv, 4),
            "IV Max": round(iv_series.max(), 4),
            "IV Min": round(iv_series.min(), 4)
        })

        # Plot IV Rank
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(x=iv_series.index, y=iv_series.values, mode='lines', name='ATM IV'))
        fig_iv.add_trace(go.Scatter(x=[iv_series.index[-1]], y=[current_iv], mode='markers', name='Current IV', marker=dict(size=10)))
        fig_iv.update_layout(title=f"ATM IV over Time for {ticker}", xaxis_title="Date", yaxis_title="Implied Volatility")
        st.plotly_chart(fig_iv, use_container_width=True)

        # Plot Historical Volatility
        hist_data = yf.download(ticker, period='1y')
        hv_series = compute_hv(hist_data['Close'])
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(x=hv_series.index, y=hv_series.values, mode='lines', name='Historical Volatility'))
        fig_hv.update_layout(title=f"30-Day Historical Volatility for {ticker}", xaxis_title="Date", yaxis_title="HV")
        st.plotly_chart(fig_hv, use_container_width=True)

    if summary_rows:
        st.markdown("---")
        st.markdown("### 📊 Summary Table")
        st.dataframe(pd.DataFrame(summary_rows))
