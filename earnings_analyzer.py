
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    return [d.strftime("%Y-%m-%d") for d in sorted_dates if d >= cutoff_date]


def yang_zhang(price_data, window=30, trading_periods=252):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = (log_cc**2).rolling(window=window).sum() / (window - 1)
    open_vol = (log_oc**2).rolling(window=window).sum() / (window - 1)
    window_rs = rs.rolling(window=window).sum() / (window - 1)

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    return ((open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)).dropna()


def build_term_structure(days, ivs):
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))


def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        exp_dates = filter_dates(stock.options)
        options_chains = {d: stock.option_chain(d) for d in exp_dates}
        underlying_price = stock.history(period='1d')['Close'].iloc[-1]

        atm_iv = {}
        for d in exp_dates:
            chain = options_chains[d]
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                continue
            call_iv = calls.iloc[(calls['strike'] - underlying_price).abs().argmin()]['impliedVolatility']
            put_iv = puts.iloc[(puts['strike'] - underlying_price).abs().argmin()]['impliedVolatility']
            atm_iv[d] = (call_iv + put_iv) / 2.0

        dtes = [(datetime.strptime(d, "%Y-%m-%d").date() - datetime.today().date()).days for d in atm_iv]
        ivs = list(atm_iv.values())
        ts = build_term_structure(dtes, ivs)

        price_history = stock.history(period='3mo')
        rv30_series = yang_zhang(price_history)
        rv30 = rv30_series.iloc[-1] if not rv30_series.empty else None
        iv30 = ts(30)

        ts_slope = (ts(45) - ts(min(dtes))) / (45 - min(dtes))
        iv30_rv30 = iv30 / rv30 if rv30 else None

        return {
            'ticker': ticker,
            'avg_volume': price_history['Volume'].rolling(30).mean().iloc[-1],
            'iv30_rv30': iv30_rv30,
            'ts_slope': ts_slope,
            'term_structure': (dtes, ivs),
            'iv30': iv30,
            'rv30': rv30
        }

    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}


st.title("Multi-Ticker Earnings Options Analyzer")
tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL,MSFT,TSLA")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Analyze"):
    results = [analyze_ticker(t) for t in tickers]
    df = pd.DataFrame([{
        'Ticker': r['ticker'],
        'Avg Volume': round(r['avg_volume'], 0) if 'avg_volume' in r else 'Error',
        'IV30 / RV30': round(r['iv30_rv30'], 2) if r.get('iv30_rv30') else 'Error',
        'TS Slope (0-45)': round(r['ts_slope'], 4) if r.get('ts_slope') else 'Error'
    } for r in results])

    st.dataframe(df)

    for r in results:
        if 'error' in r:
            st.error(f"{r['ticker']}: {r['error']}")
            continue

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Term Structure - {r['ticker']}")
            fig, ax = plt.subplots()
            ax.plot(r['term_structure'][0], r['term_structure'][1], marker='o')
            ax.set_title("Implied Volatility Term Structure")
            ax.set_xlabel("Days to Expiry")
            ax.set_ylabel("IV")
            st.pyplot(fig)

        with col2:
            st.subheader(f"IV vs RV - {r['ticker']}")
            fig2, ax2 = plt.subplots()
            ax2.bar(['IV30', 'RV30'], [r['iv30'], r['rv30']])
            ax2.set_title("IV30 vs Realized Vol (30d)")
            st.pyplot(fig2)
