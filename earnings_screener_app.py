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
            'iv30_rv30_val': round(iv30_rv30, 3),
            'ts_slope_val': round(ts_slope, 5),
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

        filtered_df = df[df['Recommendation'] == 'Recommended']

        st.dataframe(
            filtered_df[['Ticker', 'avg_volume', 'iv30_rv30', 'iv30_rv30_val',
                         'ts_slope_0_45', 'ts_slope_val', 'Expected Move', 'Recommendation']]
            .reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("📊 IV Term Structure and Skew")
        for _, row in filtered_df.iterrows():
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=row['Term_Days'],
                    y=row['Term_IVs'],
                    mode='lines+markers',
                    name=f"{row['Ticker']} IV Term Structure"
                ))
                fig.update_layout(
                    title=f"IV Term Structure: {row['Ticker']}",
                    xaxis_title="Days to Expiration",
                    yaxis_title="Implied Volatility",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                try:
                    stock = yf.Ticker(row['Ticker'])
                    expiry = stock.options[0]
                    chain = stock.option_chain(expiry)
                    calls = chain.calls
                    puts = chain.puts

                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Scatter(
                        x=calls['strike'],
                        y=calls['impliedVolatility'],
                        mode='lines+markers',
                        name='Calls IV'
                    ))
                    fig_skew.add_trace(go.Scatter(
                        x=puts['strike'],
                        y=puts['impliedVolatility'],
                        mode='lines+markers',
                        name='Puts IV'
                    ))
                    fig_skew.update_layout(
                        title=f"IV Skew: {row['Ticker']} ({expiry})",
                        xaxis_title="Strike",
                        yaxis_title="Implied Volatility",
                        height=350
                    )
                    st.plotly_chart(fig_skew, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting IV skew for {row['Ticker']}: {e}")
