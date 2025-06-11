import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i + 1]]
            if arr and arr[0] == today.strftime("%Y-%m-%d"):
                return arr[1:]
            return arr
    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252):
    price_data = price_data.dropna(subset=['Open', 'Close', 'High', 'Low'])
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1)
    open_vol = log_oc.pow(2).rolling(window).sum() / (window - 1)
    window_rs = rs.rolling(window).sum() / (window - 1)

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).pow(0.5) * np.sqrt(trading_periods)
    return result.dropna().iloc[-1]


def build_term_structure(days, ivs):
    spline = interp1d(sorted(days), [ivs[i] for i in np.argsort(days)], kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte)), days, ivs


def get_current_price(ticker_obj):
    todays_data = ticker_obj.history(period='1d')
    return todays_data['Close'].iloc[0] if not todays_data.empty else None


def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        if not stock.options:
            return {'Ticker': ticker, 'Error': 'No options found'}

        exp_dates = filter_dates(stock.options)
        options_chains = {d: stock.option_chain(d) for d in exp_dates}

        price = get_current_price(stock)
        if price is None:
            return {'Ticker': ticker, 'Error': 'No price data'}

        atm_iv = {}
        straddle = None
        call_iv_list, put_iv_list = pd.DataFrame(), pd.DataFrame()
        for i, (exp, chain) in enumerate(options_chains.items()):
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                continue
            call_iv = calls.iloc[(calls['strike'] - price).abs().idxmin()]['impliedVolatility']
            put_iv = puts.iloc[(puts['strike'] - price).abs().idxmin()]['impliedVolatility']
            atm_iv[exp] = (call_iv + put_iv) / 2

            if i == 0:
                call = calls.iloc[(calls['strike'] - price).abs().idxmin()]
                put = puts.iloc[(puts['strike'] - price).abs().idxmin()]
                straddle = ((call['bid'] + call['ask']) / 2) + ((put['bid'] + put['ask']) / 2)

                call_iv_list = calls[['strike', 'impliedVolatility']].dropna()
                put_iv_list = puts[['strike', 'impliedVolatility']].dropna()

        if not atm_iv:
            return {'Ticker': ticker, 'Error': 'Could not determine IV'}

        dtes = [(datetime.strptime(d, "%Y-%m-%d").date() - datetime.today().date()).days for d in atm_iv]
        term_curve, raw_days, raw_ivs = build_term_structure(dtes, list(atm_iv.values()))

        ts_slope = (term_curve(45) - term_curve(min(dtes))) / (45 - min(dtes))
        iv30 = term_curve(30)
        rv30 = yang_zhang(stock.history(period='3mo'))

        iv30_rv30 = iv30 / rv30
        avg_vol = stock.history(period='3mo')['Volume'].rolling(30).mean().dropna().iloc[-1]

        result = {
            'Ticker': ticker,
            'avg_volume': avg_vol,
            'iv30_rv30': round(iv30_rv30, 2),
            'ts_slope_0_45': round(ts_slope, 5),
            'Expected Move': f"{round((straddle / price) * 100, 2)}%" if straddle else 'N/A',
            'Error': '',
            'Term_Days': raw_days,
            'Term_IVs': raw_ivs,
            'Call_IVs': call_iv_list,
            'Put_IVs': put_iv_list
        }

        return result

    except Exception as e:
        return {'Ticker': ticker, 'Error': str(e)}


# --- STREAMLIT APP UI ---

st.set_page_config(page_title="IV Screener", layout="wide")
st.title("📈 IV Term Structure and Skew Viewer")

tickers_input = st.text_input("Enter stock symbols (comma separated)", value="AAPL, MSFT, AMZN")

if st.button("Analyze"):
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
    results = [compute_recommendation(ticker) for ticker in tickers]
    df = pd.DataFrame([r for r in results if r['Error'] == ''])

    st.subheader("📊 Summary Table")
    st.dataframe(df[['Ticker', 'avg_volume', 'iv30_rv30', 'ts_slope_0_45', 'Expected Move']], use_container_width=True)

    for res in results:
        if res['Error'] != '':
            st.warning(f"{res['Ticker']} - {res['Error']}")
            continue

        st.markdown(f"---\n### {res['Ticker']}")

        # Plot IV Term Structure
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=res['Term_Days'],
            y=res['Term_IVs'],
            mode='lines+markers',
            name=f"{res['Ticker']} IV Term Structure"
        ))
        fig_ts.update_layout(
            title=f"{res['Ticker']} - IV Term Structure",
            xaxis_title="Days to Expiration",
            yaxis_title="Implied Volatility",
            height=300
        )

        # Plot IV Skew
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Scatter(
            x=res['Call_IVs']['strike'],
            y=res['Call_IVs']['impliedVolatility'],
            mode='lines+markers',
            name='Calls IV',
            line=dict(color='blue')
        ))
        fig_skew.add_trace(go.Scatter(
            x=res['Put_IVs']['strike'],
            y=res['Put_IVs']['impliedVolatility'],
            mode='lines+markers',
            name='Puts IV',
            line=dict(color='red')
        ))
        fig_skew.update_layout(
            title=f"{res['Ticker']} - IV Skew",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility",
            height=300
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_ts, use_container_width=True)
        col2.plotly_chart(fig_skew, use_container_width=True)
