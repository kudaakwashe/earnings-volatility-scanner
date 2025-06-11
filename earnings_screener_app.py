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
            'avg_volume': 'PASS' if avg_vol >= 1_500_000 else 'FAIL',
            'iv30_rv30': 'PASS' if iv30_rv30 >= 1.25 else 'FAIL',
            'ts_slope_0_45': 'PASS' if ts_slope <= -0.00406 else 'FAIL',
            'Expected Move': f"{round((straddle / price) * 100, 2)}%" if straddle else 'N/A',
            'Error': '',
            'Term_Days': raw_days,
            'Term_IVs': raw_ivs,
        }

        av = result['avg_volume'] == 'PASS'
        ivr = result['iv30_rv30'] == 'PASS'
        ts = result['ts_slope_0_45'] == 'PASS'

        if av and ivr and ts:
            result['Recommendation'] = 'Recommended'
        elif ts and (av or ivr):
            result['Recommendation'] = 'Consider'
        else:
            result['Recommendation'] = 'Avoid'

        return result

    except Exception as e:
        return {'Ticker': ticker, 'Error': str(e)}


# ------------------ STREAMLIT APP ------------------ #

st.title("📈 Earnings Position Screener")

tickers_input = st.text_input("Enter one or more stock symbols (comma separated)", value="AAPL, MSFT, AMZN")

if st.button("Analyze"):
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
    results = [compute_recommendation(ticker) for ticker in tickers]
    df = pd.DataFrame(results)

    if 'Error' in df.columns:
        error_df = df[df['Error'] != '']
        if not error_df.empty:
            st.warning("Some tickers failed:")
            st.dataframe(error_df[['Ticker', 'Error']])

        df = df[df['Error'] == '']

    if not df.empty:
        st.success("✅ Analysis complete.")

        # Filter
        selected_filters = st.multiselect(
            "Filter by Recommendation",
            options=['Recommended', 'Consider', 'Avoid'],
            default=['Recommended', 'Consider', 'Avoid']
        )
        filtered_df = df[df['Recommendation'].isin(selected_filters)]

        st.dataframe(
            filtered_df[['Ticker', 'avg_volume', 'iv30_rv30', 'ts_slope_0_45', 'Expected Move', 'Recommendation']]
            .reset_index(drop=True),
            use_container_width=True
        )

        # IV Term Structure
        selected_row = st.selectbox("Select a ticker to view details", filtered_df['Ticker'].tolist())
        row = filtered_df[filtered_df['Ticker'] == selected_row].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=row['Term_Days'],
            y=row['Term_IVs'],
            mode='lines+markers',
            name=f'{selected_row} IV Term Structure'
        ))
        fig.update_layout(
            title=f"IV Term Structure for {selected_row}",
            xaxis_title="Days to Expiration",
            yaxis_title="Implied Volatility",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Option Chain with slider & IV skew chart
        try:
            stock = yf.Ticker(selected_row)
            available_expiries = stock.options
            selected_expiry = st.selectbox(
                f"Select Expiration Date for {selected_row}", available_expiries
            )

            chain = stock.option_chain(selected_expiry)
            calls_df = chain.calls[['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
            puts_df = chain.puts[['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

            # Slider range for strike selection
            min_strike = min(calls_df['strike'].min(), puts_df['strike'].min())
            max_strike = max(calls_df['strike'].max(), puts_df['strike'].max())
            strike_range = st.slider("Select Strike Price Range", float(min_strike), float(max_strike),
                                     (float(min_strike), float(max_strike)))

            calls_df = calls_df[(calls_df['strike'] >= strike_range[0]) & (calls_df['strike'] <= strike_range[1])]
            puts_df = puts_df[(puts_df['strike'] >= strike_range[0]) & (puts_df['strike'] <= strike_range[1])]

            # Display option tables
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🟦 Calls")
                st.dataframe(calls_df.reset_index(drop=True), use_container_width=True)
            with col2:
                st.markdown("### 🟥 Puts")
                st.dataframe(puts_df.reset_index(drop=True), use_container_width=True)

            # IV skew curve
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['impliedVolatility'],
                mode='lines+markers',
                name='Calls IV',
                line=dict(color='blue')
            ))
            fig_iv.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['impliedVolatility'],
                mode='lines+markers',
                name='Puts IV',
                line=dict(color='red')
            ))
            fig_iv.update_layout(
                title=f"IV Skew for {selected_row} - Expiry {selected_expiry}",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility",
                height=400
            )
            st.plotly_chart(fig_iv, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load option chain for {selected_row}: {e}")
