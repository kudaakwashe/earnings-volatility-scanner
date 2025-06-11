import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# --- Yang-Zhang Volatility Calculation ---
def yang_zhang(price_data, window=30, trading_periods=252):
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    open_vol = log_oc.pow(2).rolling(window).sum() / (window - 1)
    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1)
    window_rs = rs.rolling(window).sum() / (window - 1)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = np.sqrt((open_vol + k * close_vol + (1 - k) * window_rs) * trading_periods)
    return result.dropna().iloc[-1]


# --- Build Term Structure ---
def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        if dte > days[-1]: return ivs[-1]
        return float(spline(dte))

    return term_spline


# --- Filter Expiry Dates ---
def filter_expiries(dates):
    today = datetime.today().date()
    cutoff = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = [d.strftime("%Y-%m-%d") for d in sorted_dates if d >= cutoff]
    return arr[:3]  # Limit to 3 expiries for efficiency


# --- Process Single Ticker ---
def process_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        if len(stock.options) == 0:
            return None

        exp_dates = filter_expiries(stock.options)
        if not exp_dates:
            return None

        current_price = stock.history(period="1d")["Close"].iloc[-1]
        atm_iv = {}
        straddle_price = None

        for i, exp_date in enumerate(exp_dates):
            chain = stock.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                continue

            call_idx = (calls['strike'] - current_price).abs().idxmin()
            put_idx = (puts['strike'] - current_price).abs().idxmin()

            call_iv = calls.loc[call_idx, 'impliedVolatility']
            put_iv = puts.loc[put_idx, 'impliedVolatility']
            atm_iv[exp_date] = (call_iv + put_iv) / 2

            if i == 0:
                call_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2
                put_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2
                straddle_price = call_mid + put_mid

        if not atm_iv:
            return None

        today = datetime.today().date()
        dtes = [(datetime.strptime(k, "%Y-%m-%d").date() - today).days for k in atm_iv.keys()]
        iv_values = list(atm_iv.values())

        term_func = build_term_structure(dtes, iv_values)
        slope = (term_func(45) - term_func(min(dtes))) / (45 - min(dtes))

        history = stock.history(period="3mo")
        if history.empty:
            return None
        rv = yang_zhang(history)
        iv30 = term_func(30)
        iv_rv_ratio = iv30 / rv

        avg_volume = history['Volume'].rolling(30).mean().iloc[-1]
        expected_move = round(straddle_price / current_price * 100, 2) if straddle_price else None

        return {
            'ticker': ticker,
            'current_price': current_price,
            'avg_volume': avg_volume,
            'iv30': iv30,
            'rv30': rv,
            'iv30/rv30': iv_rv_ratio,
            'term_slope_0_45': slope,
            'expected_move_%': expected_move,
            'term_structure': (dtes, iv_values)
        }
    except Exception:
        return None


# --- Streamlit UI ---
st.title("Earnings IV/RV Scanner")
st.write("Enter comma-separated tickers:")

tickers_input = st.text_input("Tickers", "AAPL,MSFT,NVDA,GOOGL,AMZN")

if st.button("Run Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []

    for t in tickers:
        with st.spinner(f"Processing {t}..."):
            data = process_ticker(t)
            if data:
                results.append(data)

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df[['ticker', 'current_price', 'avg_volume', 'iv30', 'rv30', 'iv30/rv30', 'term_slope_0_45', 'expected_move_%']])

        selected_ticker = st.selectbox("Select ticker for visuals", df['ticker'])
        selected_data = next(item for item in results if item["ticker"] == selected_ticker)

        # IV vs RV
        fig1, ax1 = plt.subplots()
        ax1.bar(['IV30', 'RV30'], [selected_data['iv30'], selected_data['rv30']], color=['#1f77b4', '#ff7f0e'])
        ax1.set_title(f"IV30 vs RV30: {selected_ticker}")
        st.pyplot(fig1)

        # Term Structure
        dtes, ivs = selected_data['term_structure']
        fig2, ax2 = plt.subplots()
        ax2.plot(dtes, ivs, marker='o')
        ax2.set_title(f"Term Structure: {selected_ticker}")
        ax2.set_xlabel("DTE (Days to Expiry)")
        ax2.set_ylabel("IV")
        st.pyplot(fig2)
    else:
        st.error("No data returned for any ticker.")
