import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Earnings Volatility Scanner")

def compute_metrics(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")

    if hist['Volume'].rolling(30).mean().iloc[-1] < 1_000_000:
        return None  # Filter out illiquid stocks

    price = stock.history(period="1d")['Close'].iloc[-1]

    ivs, dtes = [], []
    for exp in stock.options:
        d = pd.to_datetime(exp).date()
        days = (d - datetime.today().date()).days
        if days < 0 or days > 60: continue
        chain = stock.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty: continue
        c = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]].iloc[0]
        p = puts.iloc[(puts['strike'] - price).abs().argsort()[:1]].iloc[0]
        ivs.append((c['impliedVolatility'] + p['impliedVolatility']) / 2)
        dtes.append(days)

    if len(ivs) < 2:
        return None

    ts = interp1d(dtes, ivs, fill_value="extrapolate")
    iv30 = float(ts(30))
    slope = (ts(45) - ts(min(dtes))) / (45 - min(dtes))
    exp_move = ((c['bid'] + c['ask'])/2 + (p['bid'] + p['ask'])/2) / price * 100

    return dict(Ticker=ticker, Price=price, IV30=iv30, Slope=slope, ExpMove=exp_move, dtes=dtes, ivs=ivs)

def plot_term_structure(ticker, dtes, ivs):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(dtes, ivs, marker='o')
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"{ticker} - IV Term Structure")
    return fig

# Streamlit UI
st.title("Earnings Volatility Scanner")

tickers_input = st.text_area("Enter stock symbols (comma-separated)", "").strip()

if st.button("Analyze"):
    tickers_list = [t.strip().upper() for t in tickers_input.split(",")]

    results = [compute_metrics(t) for t in tickers_list]
    results = [r for r in results if r]

    if not results:
        st.warning("No qualifying stocks found.")
    else:
        df = pd.DataFrame(results).sort_values("ExpMove", ascending=False)
        st.dataframe(df[['Ticker','Price','IV30','Slope','ExpMove']])

        st.subheader("IV Term Structure for Selected Stocks")

        cols = st.columns(4)  # Arrange in 4 columns
        for i, row in enumerate(results):
            with cols[i % 4]:  # Cycle through columns
                st.pyplot(plot_term_structure(row['Ticker'], row['dtes'], row['ivs']))
