
import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

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

def yang_zhang(price_data, window=30, trading_periods=252):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))
    return term_spline

def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def compute_iv_rank(iv_series):
    iv_30 = iv_series[-1]
    iv_rank = (iv_30 - iv_series.min()) / (iv_series.max() - iv_series.min()) * 100 if iv_series.max() > iv_series.min() else 0
    return iv_rank

def compute_recommendation(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    if len(ticker.options) == 0:
        return None, "No options found for this ticker."

    try:
        exp_dates = filter_dates(ticker.options)
    except:
        return None, "Not enough option data (need expirations 45+ days out)."

    options_chains = {date: ticker.option_chain(date) for date in exp_dates}
    try:
        underlying_price = get_current_price(ticker)
    except:
        return None, "Unable to fetch current price."

    atm_iv = {}
    straddle = None
    for i, (exp_date, chain) in enumerate(options_chains.items()):
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            continue
        call_diffs = (calls['strike'] - underlying_price).abs()
        call_idx = call_diffs.idxmin()
        put_diffs = (puts['strike'] - underlying_price).abs()
        put_idx = put_diffs.idxmin()
        call_iv = calls.loc[call_idx, 'impliedVolatility']
        put_iv = puts.loc[put_idx, 'impliedVolatility']
        atm_iv_value = (call_iv + put_iv) / 2.0
        atm_iv[exp_date] = atm_iv_value

        if i == 0:
            call_bid, call_ask = calls.loc[call_idx, ['bid', 'ask']]
            put_bid, put_ask = puts.loc[put_idx, ['bid', 'ask']]
            call_mid = (call_bid + call_ask) / 2.0 if call_bid and call_ask else None
            put_mid = (put_bid + put_ask) / 2.0 if put_bid and put_ask else None
            if call_mid and put_mid:
                straddle = call_mid + put_mid

    if not atm_iv:
        return None, "Could not determine ATM IVs."

    today = datetime.today().date()
    dtes, ivs = [], []
    for exp_date, iv in atm_iv.items():
        dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
        dtes.append(dte)
        ivs.append(iv)

    term_spline = build_term_structure(dtes, ivs)
    ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
    price_history = ticker.history(period='1y')
    rv30_series = yang_zhang(price_history)
    rv30 = rv30_series.iloc[-1] if not rv30_series.empty else 0
    iv30 = term_spline(30)
    iv30_rv30 = iv30 / rv30 if rv30 > 0 else 0
    iv_rank = compute_iv_rank(rv30_series)
    avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
    expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else "N/A"

    return {
        'ticker': ticker_symbol,
        'avg_volume_pass': avg_volume >= 1500000,
        'iv30_rv30_pass': iv30_rv30 >= 1.25,
        'ts_slope_pass': ts_slope_0_45 <= -0.00406,
        'iv_rank': round(iv_rank, 2),
        'expected_move': expected_move
    }, None


# Streamlit App Interface
def fetch_iv_hv_series(ticker_symbol):
    try:
        price_data = yf.Ticker(ticker_symbol).history(period="1y")
        hv_series = yang_zhang(price_data)
        iv_series = hv_series * 1.25  # simulate IV using multiplier (placeholder)
        return iv_series[-252:], hv_series[-252:]
    except:
        return pd.Series(dtype=float), pd.Series(dtype=float)

st.title("📊 Multi-Ticker Earnings Options Scanner")
tickers = st.text_area("Enter comma-separated stock tickers", "AAPL,MSFT,GOOGL")

if st.button("Run Analysis"):

    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    
    all_results = []
    for symbol in tickers_list:

        with st.spinner(f"Processing {symbol}..."):
            result, error = compute_recommendation(symbol)
            if error:
                st.error(f"{symbol}: {error}")
            else:
                if all_results:
                    st.subheader("📋 Summary Table")
                    st.dataframe(pd.DataFrame(all_results))
        
        
    for ticker in [res["Ticker"] for res in all_results]:
        iv_series, hv_series = fetch_iv_hv_series(ticker)
        if not iv_series.empty and not hv_series.empty:
            st.subheader(f"📈 IV vs HV (Last 1 Year) - {ticker}")
            chart_data = pd.DataFrame({
                "IV (simulated)": iv_series.values,
                "HV (realized)": hv_series.values
            }, index=iv_series.index)
            st.line_chart(chart_data)
    

            if result['avg_volume_pass'] and result['iv30_rv30_pass'] and result['ts_slope_pass']:
                st.success("✅ Recommended")
            elif result['ts_slope_pass'] and (result['avg_volume_pass'] or result['iv30_rv30_pass']):
                st.warning("⚠️ Consider")
            else:
                st.error("❌ Avoid")
    
            st.markdown(f"- **Expected Move**: {result['expected_move']}")
            st.markdown(f"- **IV Rank (1Y)**: {result['iv_rank']}")
            st.markdown(f"- **Avg Volume**: {'✅ PASS' if result['avg_volume_pass'] else '❌ FAIL'}")
            st.markdown(f"- **IV30 / RV30**: {'✅ PASS' if result['iv30_rv30_pass'] else '❌ FAIL'}")
    
            st.markdown(f"- **Term Structure Slope**: {'✅ PASS' if result['ts_slope_pass'] else '❌ FAIL'}")
            all_results.append({
                "Ticker": symbol,
                "Expected Move": result['expected_move'],
                "IV Rank (1Y)": result['iv_rank'],
                "Avg Volume": "PASS" if result['avg_volume_pass'] else "FAIL",
                "IV30/RV30": "PASS" if result['iv30_rv30_pass'] else "FAIL",
                "Slope": "PASS" if result['ts_slope_pass'] else "FAIL",
                "Recommendation": "Recommended" if result['avg_volume_pass'] and result['iv30_rv30_pass'] and result['ts_slope_pass']
                                 else "Consider" if result['ts_slope_pass'] and (result['avg_volume_pass'] or result['iv30_rv30_pass'])
                                 else "Avoid"
            })

