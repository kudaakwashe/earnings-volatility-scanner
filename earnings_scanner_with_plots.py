
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc_sq.rolling(window=window).sum() / (window - 1.0)
    open_vol = log_oc_sq.rolling(window=window).sum() / (window - 1.0)
    window_rs = rs.rolling(window=window).sum() / (window - 1.0)
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        elif dte > days[-1]: return ivs[-1]
        else: return float(spline(dte))
    return term_spline

def plot_iv_term_structure(ticker, dtes, ivs):
    fig, ax = plt.subplots()
    ax.plot(dtes, ivs, marker='o', linestyle='-')
    ax.set_title(f"IV Term Structure - {ticker}")
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True)
    return fig

def compute_metrics_with_plot(ticker):
    try:
        stock = yf.Ticker(ticker)
        if len(stock.options) == 0:
            return None, None
        exp_dates = filter_dates(stock.options)
        options_chains = {date: stock.option_chain(date) for date in exp_dates}
        price = stock.history(period='1d')['Close'][0]
        atm_iv = {}
        straddle = None
        for i, (exp_date, chain) in enumerate(options_chains.items()):
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                continue
            call_idx = (calls['strike'] - price).abs().idxmin()
            put_idx = (puts['strike'] - price).abs().idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']
            put_iv = puts.loc[put_idx, 'impliedVolatility']
            atm_iv[exp_date] = (call_iv + put_iv) / 2.0
            if i == 0:
                call_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2
                put_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2
                straddle = call_mid + put_mid
        if not atm_iv:
            return None, None
        today = datetime.today().date()
        dtes, ivs = [], []
        for exp, iv in atm_iv.items():
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            dtes.append(dte)
            ivs.append(iv)
        term_spline = build_term_structure(dtes, ivs)
        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
        price_history = stock.history(period='3mo')
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        expected_move = round(straddle / price * 100, 2) if straddle else None

        metrics = {
            'Ticker': ticker,
            'Current Price': round(price, 2),
            'Expected Move': f"{expected_move}%" if expected_move else None,
            'IV30/RV30': round(iv30_rv30, 2),
            'TS Slope 0-45': round(ts_slope_0_45, 4),
            'Avg Volume Pass': avg_volume >= 1500000,
            'IV30/RV30 Pass': iv30_rv30 >= 1.25,
            'TS Slope Pass': ts_slope_0_45 <= -0.00406,
        }
        plot = plot_iv_term_structure(ticker, dtes, ivs)
        return metrics, plot
    except:
        return None, None

def run_app_with_plots():
    st.title("Earnings Option Strategy Scanner (Recommended Only)")
    tickers_input = st.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, AMZN")
    if st.button("Run Scan"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        all_metrics = []
        all_plots = []
        for ticker in tickers:
            metrics, plot = compute_metrics_with_plot(ticker)
            if metrics:
                all_metrics.append(metrics)
                all_plots.append((ticker, plot))
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            def get_recommendation(row):
                if row['Avg Volume Pass'] and row['IV30/RV30 Pass'] and row['TS Slope Pass']:
                    return 'Recommended (Bullish)'
                elif row['TS Slope Pass'] and (row['Avg Volume Pass'] or row['IV30/RV30 Pass']):
                    return 'Consider (Bullish)'
                elif row['IV30/RV30'] <= 0.75 and row['TS Slope 0-45'] >= 0.00406:
                    return 'Recommended (Bearish)'
                elif row['IV30/RV30'] <= 0.75 or row['TS Slope 0-45'] >= 0.00406:
                    return 'Consider (Bearish)'
                else:
                    return 'Avoid'
            df['Recommendation'] = df.apply(get_recommendation, axis=1)
            recommended_df = df[df['Recommendation'].str.contains('Recommended')]
            st.dataframe(recommended_df)
            for ticker, fig in all_plots:
                if ticker in recommended_df['Ticker'].values:
                    st.pyplot(fig)
        else:
            st.warning("No valid data could be retrieved for the tickers provided.")

if __name__ == "__main__":
    run_app_with_plots()
