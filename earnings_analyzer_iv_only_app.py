
import streamlit as st
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

st.title("📈 ATM Implied Volatility (~30D) Viewer")

tickers_input = st.text_input("Enter stock symbols (comma-separated)", "AAPL, MSFT, AMZN")

if st.button("Analyze IV"):
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

    for ticker in tickers:
        st.subheader(f"{ticker} - IV (~30D ATM)")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            if hist.empty:
                st.warning(f"No price history for {ticker}. Skipping.")
                continue

            iv_vals = []
            iv_dates = []
            for exp in stock.options:
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d")
                    dte = (exp_date.date() - datetime.today().date()).days
                    if not (20 <= dte <= 40):
                        continue

                    chain = stock.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    if calls.empty or puts.empty:
                        continue

                    price = hist['Close'].iloc[-1]
                    call_iv = calls.iloc[(calls['strike'] - price).abs().idxmin()]['impliedVolatility']
                    put_iv = puts.iloc[(puts['strike'] - price).abs().idxmin()]['impliedVolatility']
                    iv_vals.append((call_iv + put_iv) / 2)
                    iv_dates.append(exp_date)
                except:
                    continue

            if iv_vals:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(iv_dates, iv_vals, marker='o', linestyle='-')
                ax.set_title(f"{ticker} - ATM IV (30D Expiries)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Implied Volatility")
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.info(f"No valid IV data for {ticker} in the 20–40 day expiry range.")
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
