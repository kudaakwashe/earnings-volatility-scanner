import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Earnings Volatility Scanner")

# Function to fetch tickers based on earnings calendar within the next 21 days
def fetch_universe():
    today = pd.Timestamp.today().normalize()
    end_date = today + pd.Timedelta(days=21)

    # Load S&P 500 tickers from Wikipedia
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sp500['Symbol'].tolist()

    earnings_list = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            earnings_dates = stock.calendar
            if not earnings_dates.empty:
                next_earnings = earnings_dates.loc['Earnings Date'][0]
                if isinstance(next_earnings, pd.Timestamp) and today <= next_earnings <= end_date:
                    earnings_list.append((ticker, next_earnings.date()))
        except:
            continue

    return earnings_list

def yang_zhang(hist):
    w = 30
    log_oc = np.log(hist['Open'] / hist['Close'].shift(1))
    log_cc = np.log(hist['Close'] / hist['Close'].shift(1))
    log_ho = np.log(hist['High'] / hist['Open'])
    log_lo = np.log(hist['Low'] / hist['Open'])
    log_co = np.log(hist['Close'] / hist['Open'])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    co2 = (log_cc ** 2).rolling(w).sum() / (w - 1)
    oo2 = (log_oc ** 2).rolling(w).sum() / (w - 1)
    rsum = rs.rolling(w).sum() / (w - 1)
    k = 0.34 / (1.34 + (w + 1)/(w - 1))
    vol = (oo2 + k * co2 + (1 - k) * rsum).apply(np.sqrt) * np.sqrt(252)
    return vol.dropna().iloc[-1]

def plot_iv_vs_rv(iv30, rv30):
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(['IV30', 'RV30'], [iv30, rv30], color=['skyblue', 'orange'])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')
    ax.set_title("IV30 vs RV30")
    return fig

def plot_term_structure(dtes, ivs):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dtes, ivs, marker='o')
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Implied Volatility")
    ax.set_title("IV Term Structure")
    return fig

def compute_metrics(ticker, earn_date):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")['Close'].iloc[-1]
    hist = stock.history(period="3mo")
    rv = yang_zhang(hist)

    ivs, dtes, total_vol = [], [], 0
    for exp in stock.options:
        d = pd.to_datetime(exp).date()
        days = (d - datetime.today().date()).days
        if days < 0 or days > 60: continue
        chain = stock.option_chain(exp)
        pr = price
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty: continue
        c = calls.iloc[(calls['strike'] - pr).abs().argsort()[:1]].iloc[0]
        p = puts.iloc[(puts['strike'] - pr).abs().argsort()[:1]].iloc[0]
        total_vol += c['volume'] + p['volume']
        ivs.append((c['impliedVolatility'] + p['impliedVolatility']) / 2)
        dtes.append(days)
    if total_vol < 500 or len(ivs) < 2:
        return None

    ts = interp1d(dtes, ivs, fill_value="extrapolate")
    iv30 = float(ts(30))
    slope = (ts(45) - ts(min(dtes))) / (45 - min(dtes))
    exp_move = ((c['bid']+c['ask'])/2 + (p['bid']+p['ask'])/2) / price * 100

    return dict(Ticker=ticker, EarningsDate=earn_date, Price=price,
                IV30=iv30, RV30=rv, IVRV=iv30/rv,
                Slope=slope, Volume=total_vol,
                ExpMove=exp_move, dtes=dtes, ivs=ivs)

def main():
    st.title("Earnings Volatility Scanner")
    universe = fetch_universe()
    results = [compute_metrics(t, d) for t, d in universe]
    results = [r for r in results if r]
    if not results:
        st.warning("No qualifying stocks found.")
        return
    df = pd.DataFrame(results).sort_values("ExpMove", ascending=False)
    st.dataframe(df[['Ticker','EarningsDate','Price','Volume','IV30','RV30','IVRV','Slope','ExpMove']])

    ticker = st.selectbox("Chart Ticker", df['Ticker'])
    row = next(r for r in results if r['Ticker']==ticker)
    st.pyplot(plot_iv_vs_rv(row['IV30'], row['RV30']))
    st.pyplot(plot_term_structure(row['dtes'], row['ivs']))

if __name__=="__main__":
    main()
