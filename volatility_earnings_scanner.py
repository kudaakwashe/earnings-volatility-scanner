
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


def get_term_structure(ticker):
    try:
        tk = yf.Ticker(ticker)
        options = tk.options
        filtered = filter_dates(options)

        ivs, days = [], []
        for date in filtered:
            opt = tk.option_chain(date)
            calls = opt.calls
            iv = calls.impliedVolatility.mean()
            days_to_exp = (datetime.strptime(date, "%Y-%m-%d").date() - datetime.today().date()).days
            if not np.isnan(iv):
                ivs.append(iv)
                days.append(days_to_exp)

        if len(ivs) >= 2:
            slope = (ivs[-1] - ivs[0]) / (days[-1] - days[0])
        else:
            slope = np.nan

        return days, ivs, slope
    except Exception as e:
        st.warning(f"Error processing {ticker}: {e}")
        return [], [], np.nan


def get_iv_rv_metrics(ticker):
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        data.dropna(inplace=True)

        data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
        rv_series = data['Return'].rolling(window=30).std() * np.sqrt(252)
        iv_series = data['Close'].rolling(window=30).std() * np.sqrt(252) * 1.2  # proxy

        iv30 = iv_series.iloc[-1]
        rv30 = rv_series.iloc[-1]
        ratio = iv30 / rv30 if rv30 else np.nan

        return iv30, rv30, ratio, iv_series, rv_series, data.index
    except Exception as e:
        st.warning(f"Error calculating IV/RV for {ticker}: {e}")
        return np.nan, np.nan, np.nan, pd.Series(), pd.Series(), pd.Index([])


def main():
    st.title("Volatility Earnings Scanner")

    tickers = st.text_input("Enter tickers separated by commas:", value="AAPL,MSFT,NVDA,ADBE,GOOGL")
    tickers = [t.strip().upper() for t in tickers.split(",")]

    results = []

    for ticker in tickers:
        days, ivs, ts_slope = get_term_structure(ticker)
        iv30, rv30, iv30_rv30, iv_series, rv_series, dates = get_iv_rv_metrics(ticker)
        price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "IV30": round(iv30, 4),
            "RV30": round(rv30, 4),
            "IV30/RV30": round(iv30_rv30, 4),
            "TS Slope (0-45d)": round(ts_slope, 5),
            "Days": days,
            "IVs": ivs,
            "IV_Series": iv_series,
            "RV_Series": rv_series,
            "Date_Index": dates
        })

    df = pd.DataFrame(results)
    st.dataframe(df.drop(columns=["Days", "IVs", "IV_Series", "RV_Series", "Date_Index"]))

    for row in results:
        st.markdown(f"### {row['Ticker']} - Volatility Plots")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=row["Days"], y=row["IVs"], mode='lines+markers', name='IV Term Structure'))
            fig.update_layout(title="IV Term Structure (0–45d)", xaxis_title="Days to Expiry", yaxis_title="Implied Volatility")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=row["Date_Index"], y=row["IV_Series"], name="IV30"))
            fig2.add_trace(go.Scatter(x=row["Date_Index"], y=row["RV_Series"], name="RV30"))
            fig2.update_layout(title="IV30 vs RV30 (6 months)", xaxis_title="Date", yaxis_title="Volatility")
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
