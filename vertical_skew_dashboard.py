"""
DISCLAIMER:
This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The authors are not financial advisors and accept no responsibility for financial decisions or losses.
Always consult a professional financial advisor before making any investment decisions.
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from typing import Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --------------------------
# Streamlit config
# --------------------------
st.set_page_config(page_title="Vertical Skew — Snapshot, 10Δ/25Δ, History & Skewness", layout="wide")
st.title("Vertical Skew — Snapshot, 10Δ/25Δ Smile Points, History & Skewness")

st.caption(
    "Left panel: snapshot vertical skew (IV vs Strike) with highlighted **10Δ** and **25Δ** smile points "
    "for puts & calls (plus Put–Call diffs). "
    "Right panel: historical smile reconstruction (date × strike) and a **skewness time series** "
    "(slope of IV vs moneyness)."
)

# ==========================================================
# Black–Scholes pricing, Greeks, and IV inversion utilities
# ==========================================================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def black_scholes_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: Literal["call","put"]) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S*math.exp(-q*T) - K*math.exp(-r*T)) if kind == "call" else (K*math.exp(-r*T) - S*math.exp(-q*T)))
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def black_scholes_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)

def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: Literal["call","put"]) -> float:
    """Black–Scholes delta with continuous dividend yield q."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    if kind == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:
        return -math.exp(-q * T) * _norm_cdf(-d1)

def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, kind: Literal["call","put"],
                tol: float = 1e-6, max_iter: int = 100) -> float | None:
    """Newton–Raphson with bisection fallback."""
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    # Seed by moneyness
    m = abs(K / S - 1.0)
    sigma = min(max(0.05 + 2.0*m, 0.05), 2.5)
    for _ in range(max_iter):
        model = black_scholes_price(S, K, T, r, q, sigma, kind)
        diff = model - price
        if abs(diff) < tol:
            return float(sigma)
        vega = black_scholes_vega(S, K, T, r, q, sigma)
        if vega < 1e-8:
            break
        sigma = max(1e-4, sigma - diff / vega)
    # Bisection
    low, high = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        pmid = black_scholes_price(S, K, T, r, q, mid, kind)
        if abs(pmid - price) < tol:
            return float(mid)
        if pmid > price:
            high = mid
        else:
            low = mid
    return None

# ===================
# Cached data loaders
# ===================
@st.cache_data(show_spinner=False, ttl=300)
def get_spot_and_divyield(ticker: str) -> Tuple[float, float]:
    t = yf.Ticker(ticker)
    price = None
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        hist = t.history(period="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
    if price is None:
        raise RuntimeError("Unable to get spot price.")

    div_yield = 0.0
    try:
        dy = t.fast_info.get("dividend_yield", 0.0) or 0.0
        if dy is None:
            dy = 0.0
        div_yield = float(dy)
    except Exception:
        pass
    return price, div_yield

@st.cache_data(show_spinner=False, ttl=300)
def get_expirations(ticker: str) -> list[str]:
    try:
        return list(yf.Ticker(ticker).options or [])
    except Exception:
        return []

@st.cache_data(show_spinner=True, ttl=300)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ch = yf.Ticker(ticker).option_chain(expiry)
    calls = ch.calls.copy()
    puts = ch.puts.copy()
    for df, side in ((calls, "Call"), (puts, "Put")):
        df["side"] = side
    return calls, puts

@st.cache_data(show_spinner=True, ttl=1200)
def get_option_history(contract_symbol: str, lookback_days: int) -> pd.DataFrame:
    hist = yf.Ticker(contract_symbol).history(period=f"{lookback_days}d", auto_adjust=False)
    if hist.empty:
        return hist
    return hist[["Close"]].rename(columns={"Close": "opt_close"})

@st.cache_data(show_spinner=True, ttl=1200)
def get_underlying_history(ticker: str, lookback_days: int) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=f"{lookback_days}d", auto_adjust=False)
    return hist[["Close"]].rename(columns={"Close": "spot_close"})

# ===================
# Sidebar controls
# ===================
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    expiries = get_expirations(ticker) if ticker else []
    expiry = st.selectbox("Expiry", options=expiries, index=0 if expiries else None)
    side_hist = st.selectbox("Historical side", options=["Put", "Call"], index=0)
    lookback_days = st.slider("Lookback (trading days approx.)", 10, 90, 30, help="Historical reconstruction window.")
    strikes_window_pct = st.slider("Strikes around spot (±%)", 5, 80, 30, help="Keeps strikes within ±X% of spot.")
    max_strikes = st.slider("Max strikes per side (history)", 3, 25, 10, help="Performance guardrail for history.")
    rf_user = st.number_input("Risk-free rate (annual, decimal)", value=0.045, step=0.005, format="%.3f")
    override_div = st.checkbox("Override dividend yield", value=False)
    div_user = st.number_input("Dividend yield (annual, decimal)", value=0.000, step=0.005, format="%.3f", disabled=not override_div)
    st.caption("Tip: Narrow the strike window and lower max strikes if it feels slow.")

if not ticker or not expiry:
    st.stop()

# ===================
# Load basics
# ===================
try:
    spot_now, dy_guess = get_spot_and_divyield(ticker)
except Exception as e:
    st.error(f"Failed to load spot/dividend: {e}")
    st.stop()

q = div_user if override_div else dy_guess
calls, puts = get_chain(ticker, expiry)

def _clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["contractSymbol","strike","lastPrice","bid","ask","impliedVolatility","volume","openInterest","inTheMoney","side"] if c in df.columns]
    df = df[keep].dropna(subset=["strike"]).copy()
    df = df.drop_duplicates(subset=["contractSymbol"]) if "contractSymbol" in df.columns else df
    # keep sensible IVs for snapshot calcs
    if "impliedVolatility" in df.columns:
        df = df[(df["impliedVolatility"] > 0) & (df["impliedVolatility"] < 5.0)]
    return df.sort_values("strike")

calls = _clean_chain(calls)
puts  = _clean_chain(puts)

# Filter by moneyness window around current spot
k_low, k_high = spot_now*(1 - strikes_window_pct/100.0), spot_now*(1 + strikes_window_pct/100.0)
calls = calls[(calls["strike"]>=k_low) & (calls["strike"]<=k_high)]
puts  = puts[(puts["strike"] >=k_low) & (puts["strike"] <=k_high)]

# Trim to max_strikes nearest ATM for history (snapshot uses full window)
def _nearest_atm(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if df.empty: return df
    df = df.assign(dist=(df["strike"] - spot_now).abs()).sort_values("dist").head(max_n)
    return df.drop(columns=["dist"]).sort_values("strike")

calls_hist = _nearest_atm(calls, max_strikes)
puts_hist  = _nearest_atm(puts,  max_strikes)

# ===================
# Time to expiry (years)
# ===================
def _expiry_to_datetime_utc(exp_str: str) -> datetime:
    dt = datetime.strptime(exp_str, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, 20, 0, 0, tzinfo=timezone.utc)

exp_dt_utc = _expiry_to_datetime_utc(expiry)
now_utc = datetime.now(timezone.utc)
T_snapshot = max((exp_dt_utc - now_utc).total_seconds() / (365.0*24*3600), 1e-6)

# ==========================================================
# 10Δ & 25Δ smile points (snapshot)
# ==========================================================
def compute_delta_for_df(df: pd.DataFrame, S: float, T: float, r: float, q: float, kind: Literal["call","put"]) -> pd.DataFrame:
    """Add columns: 'delta' computed from each row's IV."""
    out = df.copy()
    if "impliedVolatility" not in out.columns or out.empty:
        out["delta"] = np.nan
        return out
    deltas = []
    for _, row in out.iterrows():
        K = float(row["strike"])
        iv = float(row["impliedVolatility"])
        if not (0 < iv < 5.0):
            deltas.append(np.nan)
            continue
        deltas.append(bs_delta(S=S, K=K, T=T, r=r, q=q, sigma=iv, kind=kind))
    out["delta"] = deltas
    return out

def find_delta_point(df_with_delta: pd.DataFrame, target_abs_delta: float, kind: Literal["call","put"]) -> Dict[str, Any] | None:
    """Find contract with delta closest to +target (call) or -target (put)."""
    if df_with_delta.empty or "delta" not in df_with_delta.columns:
        return None
    d = df_with_delta.dropna(subset=["delta"]).copy()
    if d.empty:
        return None
    if kind == "call":
        d["delta_diff"] = (d["delta"] - target_abs_delta).abs()
    else:
        d["delta_diff"] = (d["delta"] + target_abs_delta).abs()  # put deltas are negative
    row = d.nsmallest(1, "delta_diff")
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "contractSymbol": r.get("contractSymbol"),
        "strike": float(r["strike"]),
        "iv": float(r["impliedVolatility"]) if "impliedVolatility" in r else np.nan,
        "delta": float(r["delta"]),
    }

# Prepare deltas for snapshot
calls_d = compute_delta_for_df(calls, spot_now, T_snapshot, rf_user, q, "call")
puts_d  = compute_delta_for_df(puts,  spot_now, T_snapshot, rf_user, q, "put")

targets = [0.25, 0.10]
smile_points = []
for tgt in targets:
    c_pt = find_delta_point(calls_d, tgt, "call")
    p_pt = find_delta_point(puts_d,  tgt, "put")
    if c_pt or p_pt:
        smile_points.append({
            "delta_abs": tgt,
            "call_iv": c_pt["iv"] if c_pt else np.nan,
            "call_strike": c_pt["strike"] if c_pt else np.nan,
            "put_iv": p_pt["iv"] if p_pt else np.nan,
            "put_strike": p_pt["strike"] if p_pt else np.nan,
            "put_minus_call": (p_pt["iv"] - c_pt["iv"]) if (c_pt and p_pt) else np.nan
        })

smile_df = pd.DataFrame(smile_points, columns=["delta_abs","call_iv","call_strike","put_iv","put_strike","put_minus_call"])

# ==========================================================
# Historical reconstruction helpers
# ==========================================================
@st.cache_data(show_spinner=True, ttl=900)
def build_hist_iv_table(ticker: str, selection: pd.DataFrame, lookback_days: int, rf: float, q: float,
                        kind: Literal["call","put"], exp_dt_utc: datetime) -> pd.DataFrame:
    """Date-indexed DataFrame with columns=strike, values=IV (decimal)."""
    if selection.empty:
        return pd.DataFrame()
    u_hist = get_underlying_history(ticker, lookback_days)
    if u_hist.empty:
        return pd.DataFrame()
    u_hist = u_hist.dropna()
    u_hist.index = pd.to_datetime(u_hist.index).tz_localize(None)

    frames = []
    for _, row in selection.iterrows():
        sym = row["contractSymbol"]; K = float(row["strike"])
        h = get_option_history(sym, lookback_days)
        if h.empty: 
            continue
        h = h.dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None)
        df = pd.concat([u_hist, h], axis=1, join="inner").dropna()

        dates = pd.to_datetime(df.index)
        T = (pd.to_datetime(exp_dt_utc).tz_convert(None) - dates).dt.total_seconds() / (365.0*24*3600)
        df = df.assign(T=T.values)
        df = df[df["T"] > 0].copy()
        if df.empty: 
            continue

        ivs = []
        for (S, price, ttm) in zip(df["spot_close"].values, df["opt_close"].values, df["T"].values):
            iv = implied_vol(price=float(price), S=float(S), K=float(K), T=float(ttm), r=float(rf), q=float(q), kind=kind)
            ivs.append(iv if iv is not None and 0 < iv < 5.0 else np.nan)
        df[f"IV_{kind}_{K:g}"] = ivs
        frames.append(df[[f"IV_{kind}_{K:g}"]])

    if not frames:
        return pd.DataFrame()
    joined = pd.concat(frames, axis=1)
    joined.columns = [col.split("_")[-1] for col in joined.columns]  # strike only
    joined.index.name = "date"
    try:
        joined.columns = [float(c) for c in joined.columns]
    except Exception:
        pass
    joined = joined.sort_index().sort_index(axis=1)
    return joined

# ===================
# Layout: two-panel dashboard
# ===================
left_col, right_col = st.columns([1.05, 1.35], vertical_alignment="top")

# -------------------
# LEFT: Snapshot + 10Δ/25Δ
# -------------------
with left_col:
    st.subheader("Snapshot: Vertical Skew (IV vs Strike) + 10Δ / 25Δ")
    m1, m2, m3 = st.columns(3)
    m1.metric("Spot", f"{spot_now:,.2f}")
    m2.write(f"**Expiry:** `{expiry}`")
    m3.write(f"**T (yrs):** `{T_snapshot:.3f}`")

    # Prepare snapshot plotting data
    snap_df = []
    if not calls.empty: 
        c_df = calls.copy(); c_df["Series"] = "Call"; snap_df.append(c_df)
    if not puts.empty:  
        p_df = puts.copy();  p_df["Series"] = "Put";  snap_df.append(p_df)
    snap_df = pd.concat(snap_df, ignore_index=True) if snap_df else pd.DataFrame()

    if snap_df.empty:
        st.warning("No contracts found in the selected window.")
    else:
        fig_snap = px.line(
            snap_df, x="strike", y="impliedVolatility", color="Series", markers=True,
            labels={"strike":"Strike", "impliedVolatility":"IV (decimal)"},
            title=f"{ticker} — {expiry} — Vertical Skew (Snapshot)"
        )

        # Overlay 10Δ and 25Δ markers if available
        markers = []
        for _, row in smile_df.iterrows():
            da = row["delta_abs"]
            # Call marker
            if np.isfinite(row["call_iv"]) and np.isfinite(row["call_strike"]):
                markers.append({"strike": row["call_strike"], "iv": row["call_iv"], "Series": f"Call {int(da*100)}Δ"})
            # Put marker
            if np.isfinite(row["put_iv"]) and np.isfinite(row["put_strike"]):
                markers.append({"strike": row["put_strike"], "iv": row["put_iv"], "Series": f"Put {int(da*100)}Δ"})

        if markers:
            mk_df = pd.DataFrame(markers)
            fig_snap.add_trace(
                go.Scatter(
                    x=mk_df["strike"], y=mk_df["iv"], mode="markers+text",
                    text=mk_df["Series"], textposition="top center",
                    marker=dict(size=10, symbol="x"),
                    name="Δ markers"
                )
            )

        st.plotly_chart(fig_snap, use_container_width=True)

        # Smile points table (Put/Call IVs and Put–Call diff)
        if not smile_df.empty:
            st.markdown("**10Δ & 25Δ Smile Points (snapshot)**")
            display = smile_df.copy()
            display["delta_abs"] = display["delta_abs"].map(lambda x: f"{int(x*100)}Δ")
            display = display.rename(columns={
                "delta_abs": "Δ",
                "call_iv": "Call IV",
                "call_strike": "Call Strike",
                "put_iv": "Put IV",
                "put_strike": "Put Strike",
                "put_minus_call": "Put − Call (IV)"
            })
            st.dataframe(display, use_container_width=True, height=160)

# -------------------
# RIGHT: History (top) + Skewness series (bottom)
# -------------------
with right_col:
    st.subheader("Historical Vertical Skew & Skewness")

    # --- Historical vertical skew (top) ---
    st.markdown("**Historical smile (reconstructed IV by strike over time)**")
    sel_hist = puts_hist if side_hist == "Put" else calls_hist
    hist_ivs = build_hist_iv_table(
        ticker=ticker, selection=sel_hist, lookback_days=lookback_days,
        rf=rf_user, q=q, kind="put" if side_hist == "Put" else "call", exp_dt_utc=exp_dt_utc
    )

    if hist_ivs.empty:
        st.warning("No historical data could be built (option histories may be missing). Try fewer strikes or another expiry.")
    else:
        tidy = hist_ivs.reset_index().melt(id_vars="date", var_name="strike", value_name="iv").dropna().sort_values(["date","strike"])
        fig_lines = px.line(
            tidy, x="date", y="iv", color="strike",
            title=f"{ticker} — {expiry} — Historical IV by Strike ({side_hist}s)",
            labels={"iv":"IV (decimal)", "date":"Date", "strike":"Strike"}
        )
        st.plotly_chart(fig_lines, use_container_width=True)

    # --- Skewness time series (bottom) ---
    st.markdown("**Skewness time series (slope of IV vs moneyness)**")
    # If we already have hist_ivs above for a side, reuse; otherwise compute for the other side.
    def get_hist_for_side(which: str) -> pd.DataFrame:
        if which == side_hist:
            return hist_ivs
        sel = puts_hist if which == "Put" else calls_hist
        return build_hist_iv_table(
            ticker=ticker, selection=sel, lookback_days=lookback_days,
            rf=rf_user, q=q, kind="put" if which == "Put" else "call", exp_dt_utc=exp_dt_utc
        )

    skew_side = st.radio("Skewness side", ["Put", "Call"], horizontal=True, index=0, key="skew_side_radio")
    hist_for_skew = get_hist_for_side(skew_side)

    if hist_for_skew.empty:
        st.warning("Could not reconstruct IV history for skewness calculation.")
    else:
        u_hist = get_underlying_history(ticker, lookback_days)
        if u_hist.empty:
            st.warning("No underlying history available.")
        else:
            u_hist = u_hist.dropna()
            u_hist.index = pd.to_datetime(u_hist.index).tz_localize(None)
            df = hist_for_skew.join(u_hist, how="inner").dropna(subset=["spot_close"])
            if df.empty:
                st.warning("No overlapping dates between option and underlying history.")
            else:
                slopes, dates = [], []
                for dt_idx, row in df.iterrows():
                    S = float(row["spot_close"])
                    ivs = row.drop(labels=["spot_close"]).astype(float)
                    strikes = ivs.index.astype(float)
                    iv_vals = ivs.values
                    mask = np.isfinite(iv_vals)
                    strikes = strikes[mask]; iv_vals = iv_vals[mask]
                    if len(iv_vals) < 3:
                        continue
                    moneyness = strikes / S - 1.0
                    mm = np.clip(moneyness, -1.0, 1.0)
                    coeffs = np.polyfit(mm, iv_vals, 1)  # slope, intercept
                    slopes.append(float(coeffs[0])); dates.append(dt_idx)

                if not slopes:
                    st.warning("Insufficient data to compute skewness time series.")
                else:
                    skew_df = pd.DataFrame({"date": dates, "skew_slope": slopes}).sort_values("date")
                    fig_skew = px.line(
                        skew_df, x="date", y="skew_slope",
                        title=f"{ticker} — {expiry} — Skewness (slope IV~moneyness) over time [{skew_side}s]",
                        labels={"date":"Date", "skew_slope":"Slope (per unit moneyness)"}
                    )
                    fig_skew.add_hline(y=0.0, line_dash="dash")
                    st.plotly_chart(fig_skew, use_container_width=True)

# -------------------
# Footer tips
# -------------------
st.info(
    "Performance tips:\n"
    "- Reduce **Lookback**, **Max strikes**, or **Strikes ±%** if it’s slow.\n"
    "- If a contract has no history, it’s skipped (data gaps are normal).\n"
    "- Adjust **risk-free** and **dividend** yields if pricing looks off.\n"
    "- Δ points are chosen from *listed* contracts by closest delta; they are approximations."
)
