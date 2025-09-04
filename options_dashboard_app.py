"""
DISCLAIMER:
This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The authors are not financial advisors and accept no responsibility for financial decisions or losses.
Always consult a professional financial advisor before making any investment decisions.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Literal, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ============================
# Streamlit page configuration
# ============================
st.set_page_config(page_title="Options Dashboard: Skew, Timeseries, Term Structure, VRP", layout="wide")
st.title("Options Analytics Dashboard")
st.caption("Vertical skew, skew timeseries, IV term structure with RV, and VRP plots. Data source: yfinance.")

# ============================
# Black-Scholes utilities
# ============================
def _norm_cdf(x: float) -> float: return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def _norm_pdf(x: float) -> float: return math.exp(-0.5*x*x)/math.sqrt(2.0*math.pi)
def _bs_d1(S,K,T,r,q,s): return (math.log(S/K)+(r-q+0.5*s*s)*T)/(s*math.sqrt(T))
def bs_delta(S,K,T,r,q,s,kind):
    if T<=0 or s<=0 or S<=0 or K<=0: return 0.0
    d1=_bs_d1(S,K,T,r,q,s)
    return math.exp(-q*T)*_norm_cdf(d1) if kind=="call" else -math.exp(-q*T)*_norm_cdf(-d1)

# Simplified implied vol solver
def implied_vol(price,S,K,T,r,q,kind,tol=1e-6,max_iter=100):
    if price<=0 or S<=0 or K<=0 or T<=0: return None
    sigma=0.3
    for _ in range(max_iter):
        d1=_bs_d1(S,K,T,r,q,sigma)
        d2=d1-sigma*math.sqrt(T)
        if kind=="call":
            model=S*math.exp(-q*T)*_norm_cdf(d1)-K*math.exp(-r*T)*_norm_cdf(d2)
        else:
            model=K*math.exp(-r*T)*_norm_cdf(-d2)-S*math.exp(-q*T)*_norm_cdf(-d1)
        diff=model-price
        if abs(diff)<tol: return sigma
        vega=S*math.exp(-q*T)*_norm_pdf(d1)*math.sqrt(T)
        if vega<1e-8: break
        sigma=max(1e-4,sigma-diff/vega)
    return None

# ============================
# Realized volatility (Yang-Zhang)
# ============================
def realized_vol_yz_series(hist: pd.DataFrame, window:int=30)->pd.Series:
    if hist is None or hist.empty: return pd.Series(dtype=float)
    O,H,L,C=[hist[c].astype(float) for c in["Open","High","Low","Close"]]
    ro=np.log(O/C.shift(1)); rc=np.log(C/O)
    rs=(np.log(H/O)*np.log(H/C)+np.log(L/O)*np.log(L/C))
    k=0.34/(1.34+(window+1.0)/max(window-1.0,1.0))
    var=ro.rolling(window).var()+ (1-k)*rc.rolling(window).var()+k*rs.rolling(window).mean()
    return np.sqrt(np.maximum(var,0))*np.sqrt(252.0)

def realized_vol_single(hist: pd.DataFrame, window:int=30)->float:
    s=realized_vol_yz_series(hist,window)
    return float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")

# ============================
# Data loaders
# ============================
@st.cache_data(ttl=300)
def get_spot(ticker:str)->float:
    t=yf.Ticker(ticker)
    try: return float(t.fast_info["last_price"])
    except: return float(t.history(period="1d")["Close"].iloc[-1])

@st.cache_data(ttl=300)
def get_expirations(ticker:str)->List[str]: return list(yf.Ticker(ticker).options or [])
@st.cache_data(ttl=300)
def get_chain(ticker:str,expiry:str): ch=yf.Ticker(ticker).option_chain(expiry);return ch.calls.copy(),ch.puts.copy()
@st.cache_data(ttl=300)
def get_hist(ticker:str,days:int): return yf.Ticker(ticker).history(period=f"{days}d",auto_adjust=False)

# ============================
# Sidebar
# ============================
with st.sidebar:
    ticker=st.text_input("Ticker","AAPL").upper()
    lookback=st.slider("History lookback",20,90,30)
    rv_window=st.slider("RV window",10,60,30)

if not ticker: st.stop()
spot=get_spot(ticker); expiries=get_expirations(ticker)
if not expiries: st.warning("No expiries"); st.stop()

expiry_sel=st.selectbox("Select expiry",expiries)

# ============================
# Layout 2x2
# ============================
c1,c2=st.columns(2);c3,c4=st.columns(2)

# Tile 1: Skew snapshot
with c1:
    st.subheader("Vertical Skew")
    calls,puts=get_chain(ticker,expiry_sel)
    df=pd.concat([calls.assign(side="Call"),puts.assign(side="Put")])
    df=df[(df["impliedVolatility"]>0)&(df["impliedVolatility"]<5)]
    if df.empty: st.warning("No data")
    else:
        fig=px.line(df,x="strike",y="impliedVolatility",color="side",title=f"{ticker} {expiry_sel} Skew")
        st.plotly_chart(fig,use_container_width=True)

# Tile 2: Skew slope timeseries (simple)
with c2:
    st.subheader("Skew Timeseries (slope)")
    # simple slope from moneyness vs IV
    calls,puts=get_chain(ticker,expiry_sel)
    calls=calls[(calls["impliedVolatility"]>0)&(calls["impliedVolatility"]<5)]
    if calls.empty: st.warning("No data")
    else:
        hist=get_hist(ticker,lookback)
        slopes=[]
        for date in hist.index:
            S=float(hist.loc[date,"Close"])
            ks=calls["strike"].astype(float)
            ivs=calls["impliedVolatility"].astype(float)
            if len(ks)<3: continue
            m=ks/S-1.0
            coeffs=np.polyfit(m,ivs,1)
            slopes.append({"date":date,"slope":coeffs[0]})
        if slopes:
            df=pd.DataFrame(slopes)
            fig=px.line(df,x="date",y="slope",title="IV Skew Slope")
            st.plotly_chart(fig,use_container_width=True)

# Tile 3: Term structure with RV
with c3:
    st.subheader("Term Structure vs RV")
    rows=[]
    for exp in expiries[:10]:
        dte=(datetime.strptime(exp,"%Y-%m-%d")-datetime.now()).days
        if dte<=0: continue
        calls,puts=get_chain(ticker,exp)
        for df in [calls,puts]:
            df=df[(df["impliedVolatility"]>0)&(df["impliedVolatility"]<5)]
        if calls.empty or puts.empty: continue
        atm_iv=np.mean([calls.iloc[(calls["strike"]-spot).abs().argsort()[:1]]["impliedVolatility"].values[0],
                        puts.iloc[(puts["strike"]-spot).abs().argsort()[:1]]["impliedVolatility"].values[0]])
        rows.append({"dte":dte,"iv":atm_iv})
    ts=pd.DataFrame(rows).sort_values("dte")
    rv_val=realized_vol_single(get_hist(ticker,lookback+rv_window),rv_window)
    if not ts.empty:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=ts["dte"],y=ts["iv"],mode="lines+markers",name="IV"))
        if np.isfinite(rv_val): fig.add_hline(y=rv_val,line_dash="dot",annotation_text=f"RV {rv_window}d {rv_val:.2f}")
        st.plotly_chart(fig,use_container_width=True)

# Tile 4: VRP timeseries
with c4:
    st.subheader("Variance Risk Premium")
    hist=get_hist(ticker,lookback+rv_window+10)
    rv_series=realized_vol_yz_series(hist,rv_window)
    iv_series=pd.Series(index=hist.index,dtype=float)
    calls,puts=get_chain(ticker,expiry_sel)
    calls=calls[(calls["impliedVolatility"]>0)&(calls["impliedVolatility"]<5)]
    if not calls.empty:
        K=float(calls.iloc[(calls["strike"]-spot).abs().argsort()[:1]]["strike"])
        iv_series[:]=calls["impliedVolatility"].median()
    df=pd.concat([rv_series.rename("rv"),iv_series.rename("iv")],axis=1).dropna()
    if not df.empty:
        df["vrp"]=df["iv"]-df["rv"]
        fig=px.line(df,y="vrp",title="VRP (IV - RV)")
        st.plotly_chart(fig,use_container_width=True)
