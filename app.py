"""Gold Strategy Lab - Streamlit app: backtest gold data and discover strategies with Gemini."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from goldlab.backtester import BTConfig, equity_curve, monte_carlo_drawdown
from goldlab.data import (DataError, fetch_mt5, fetch_yfinance, infer_bar_minutes, load_csv,
                       make_synthetic, resample, summarize_market)
from goldlab.dsl import describe, validate_spec
from goldlab.gemini_ai import DEFAULT_MODEL, MODEL_CHOICES, GeminiClient
from goldlab.search import OBJECTIVES, Lab, SearchConfig, run_search
from goldlab.strategies import TEMPLATES

st.set_page_config(page_title="Gold Strategy Lab", page_icon="🥇", layout="wide")


# ----------------------------------------------------------------------------- helpers
def get_api_key() -> str:
    key = st.session_state.get("api_key_input", "").strip()
    if key:
        return key
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def fmt_metrics(m: dict) -> pd.DataFrame:
    rows = [("Trades", m["trades"]), ("Win rate %", m["win_rate"]), ("Profit factor", m["profit_factor"]),
            ("Expectancy (R)", m["expectancy_r"]), ("Avg win (R)", m["avg_win_r"]),
            ("Avg loss (R)", m["avg_loss_r"]), ("Return %", m["total_return_pct"]),
            ("Max drawdown %", m["max_dd_pct"]), ("Return / DD", m["return_dd"]), ("SQN", m["sqn"]),
            ("Max consecutive losses", m["max_consec_losses"]),
            ("Long / Short", f"{m['long_trades']} / {m['short_trades']}"),
            ("Avg bars held", m["avg_bars"]), ("Final balance", m["final_balance"])]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).astype({"Value": str})


def plot_equity(trades: pd.DataFrame, cfg: BTConfig, split_time=None) -> go.Figure:
    eq = equity_curve(trades["r"].to_numpy(float), cfg)
    x = [trades["entry_time"].iloc[0]] + list(trades["exit_time"])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=x, y=eq, mode="lines", name="Equity", line=dict(color="#d4a017", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=dd, fill="tozeroy", name="Drawdown %", line=dict(color="#c0392b", width=1)), 2, 1)
    if split_time is not None:
        fig.add_vline(x=split_time, line_dash="dash", line_color="gray")
        fig.add_annotation(x=split_time, y=1, yref="paper", text="train | holdout", showarrow=False,
                           xanchor="left", font=dict(size=11, color="gray"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig


def plot_price_trades(df: pd.DataFrame, trades: pd.DataFrame, limit: int = 3000) -> go.Figure:
    sub = df.iloc[-limit:]
    t = trades[trades["entry_time"] >= sub.index[0]]
    fig = go.Figure(go.Candlestick(x=sub.index, open=sub["open"], high=sub["high"], low=sub["low"],
                                   close=sub["close"], name="XAUUSD"))
    for d, color, sym in ((1, "#27ae60", "triangle-up"), (-1, "#c0392b", "triangle-down")):
        s = t[t["dir"] == d]
        fig.add_trace(go.Scatter(x=s["entry_time"], y=s["entry"], mode="markers", name="Long" if d == 1 else "Short",
                                 marker=dict(symbol=sym, size=9, color=color)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def load_dataset(source: str):
    if source == "Upload CSV":
        up = st.file_uploader("CSV with time, open, high, low, close", type=["csv", "txt"])
        return load_csv(up) if up else None
    if source == "Yahoo Finance (GC=F futures)":
        c1, c2 = st.columns(2)
        interval = c1.selectbox("Interval", ["1h", "30m", "15m", "5m", "1d"], index=0)
        period = c2.selectbox("Period", ["730d", "365d", "180d", "60d", "30d"], index=0)
        if st.button("Download"):
            with st.spinner("Downloading..."):
                st.session_state["raw_df"] = fetch_yfinance("GC=F", interval, period)
        return st.session_state.get("raw_df")
    if source == "MetaTrader 5 (local, Windows)":
        c1, c2, c3 = st.columns(3)
        sym = c1.text_input("Symbol", "XAUUSDz")
        tf = c2.selectbox("Timeframe", ["M5", "M15", "M30", "H1", "H4"], index=1)
        bars = c3.number_input("Bars", 1000, 200_000, 50_000, 5000)
        if st.button("Fetch from MT5"):
            st.session_state["raw_df"] = fetch_mt5(sym, tf, int(bars))
        return st.session_state.get("raw_df")
    n = st.slider("Synthetic bars", 5000, 60_000, 20_000, 1000)
    st.warning("Synthetic data is random and only for trying the app. Never trust results from it.")
    return make_synthetic(n)


# ----------------------------------------------------------------------------- sidebar
st.title("🥇 Gold Strategy Lab")
st.caption("Backtest gold strategies and let Gemini propose new ones. Everything is validated on unseen data.")

with st.sidebar:
    st.header("Settings")
    st.text_input("Gemini API key", type="password", key="api_key_input",
                  help="Or set GEMINI_API_KEY in Streamlit secrets / environment. Never commit keys.")
    model = st.selectbox("Gemini model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
    custom_model = st.text_input("...or custom model id", "", help="Overrides the list above.")
    st.divider()
    st.subheader("Trading costs (price units)")
    spread = st.number_input("Spread ($/oz)", 0.0, 5.0, 0.30, 0.05,
                             help="Exness Zero-type accounts are tight but charge commission; adjust to your broker.")
    slip = st.number_input("Slippage per side ($/oz)", 0.0, 2.0, 0.05, 0.01)
    comm = st.number_input("Commission per lot, round turn ($)", 0.0, 50.0, 0.0, 0.5)
    risk = st.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)
    bal = st.number_input("Starting balance ($)", 100.0, 10_000_000.0, 10_000.0, 500.0)
    holdout = st.slider("Holdout (unseen) fraction", 0.15, 0.5, 0.30, 0.05,
                        help="Final slice of data hidden from the search. It judges robustness.")

cfg = BTConfig(spread=spread, slippage=slip, commission_per_lot=comm, risk_pct=risk, initial_balance=bal)

tab_data, tab_bt, tab_ai, tab_res = st.tabs(["1. Data", "2. Backtest", "3. AI strategy search", "4. Results"])

# ----------------------------------------------------------------------------- data tab
with tab_data:
    source = st.radio("Source", ["Upload CSV", "Yahoo Finance (GC=F futures)", "MetaTrader 5 (local, Windows)",
                                 "Synthetic demo"], horizontal=True)
    try:
        raw = load_dataset(source)
    except (DataError, Exception) as exc:  # noqa: BLE001 - show any loader problem to the user
        st.error(str(exc))
        raw = None
    if raw is not None:
        base_min = infer_bar_minutes(raw)
        tfs = [tf for tf, m in {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.items()
               if m >= base_min - 1e-6]
        tf = st.selectbox("Timeframe (resample up)", tfs, index=0)
        try:
            df = raw if tf == tfs[0] else resample(raw, tf)
        except DataError as exc:
            st.error(str(exc))
            df = None
        if df is not None:
            st.session_state["df"] = df
            summ = summarize_market(df)
            c = st.columns(4)
            c[0].metric("Bars", f"{summ['bars']:,}")
            c[1].metric("From", summ["start"][:10])
            c[2].metric("To", summ["end"][:10])
            c[3].metric("Median ATR(14)", summ["median_atr14"])
            if summ["bars"] < 3000:
                st.warning("Under ~3,000 bars: results will be statistically weak.")
            st.plotly_chart(plot_price_trades(df, pd.DataFrame(columns=["entry_time", "dir", "entry"]), 2000),
                            use_container_width=True)
    else:
        st.info("Load some gold OHLC data to begin. MetaTrader 5 CSV exports work as-is.")

df = st.session_state.get("df")

# ----------------------------------------------------------------------------- backtest tab
with tab_bt:
    if df is None:
        st.info("Load data first.")
    else:
        lab = Lab(df, cfg, holdout)
        names = [t["name"] for t in TEMPLATES] + ["From AI results", "Custom JSON"]
        pick = st.selectbox("Strategy", names)
        spec = None
        if pick == "Custom JSON":
            txt = st.text_area("Strategy JSON (see README for the DSL)", json.dumps(TEMPLATES[0], indent=2), height=300)
            try:
                spec = validate_spec(json.loads(txt))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid spec: {exc}")
        elif pick == "From AI results":
            found = st.session_state.get("found", [])
            if not found:
                st.info("Run the AI search first.")
            else:
                idx = st.selectbox("Candidate", range(len(found)), format_func=lambda i: found[i].spec["name"] + f" [{found[i].id}]")
                spec = found[idx].spec
        else:
            spec = next(t for t in TEMPLATES if t["name"] == pick)

        if spec:
            st.code(describe(spec))
            scope = st.radio("Period", ["Full history", "Train only", "Holdout only"], horizontal=True)
            lo, hi = {"Full history": (0, None), "Train only": (0, lab.split), "Holdout only": (lab.split, None)}[scope]
            trades, m = lab.evaluate(spec, lo, hi)
            if m["trades"] == 0:
                st.warning("This strategy produced no trades on the selected period.")
            else:
                left, right = st.columns([1, 2])
                left.dataframe(fmt_metrics(m), hide_index=True, use_container_width=True)
                with right:
                    st.plotly_chart(plot_equity(trades, cfg, df.index[lab.split] if lo == 0 and hi is None else None),
                                    use_container_width=True)
                mc = monte_carlo_drawdown(trades["r"].to_numpy(float), cfg)
                st.caption(f"Monte Carlo (1,000 reshuffles): median max DD {mc['dd_median']}%, "
                           f"95th pct {mc['dd_p95']}%, chance of ending below start {mc['prob_loss']}%.")
                with st.expander("Trade list & price chart"):
                    st.plotly_chart(plot_price_trades(df, trades), use_container_width=True)
                    st.dataframe(trades.tail(500), use_container_width=True)
                    st.download_button("Download trades CSV", trades.to_csv(index=False), "trades.csv")

# ----------------------------------------------------------------------------- AI tab
with tab_ai:
    if df is None:
        st.info("Load data first.")
    else:
        c1, c2, c3 = st.columns(3)
        objective = c1.selectbox("Objective", list(OBJECTIVES), format_func=OBJECTIVES.get)
        rounds = c2.slider("AI rounds", 1, 10, 4)
        per_round = c3.slider("Strategies per round", 2, 15, 8)
        c4, c5, c6 = st.columns(3)
        mut = c4.slider("Local mutations per round", 0, 40, 12)
        min_tr = c5.number_input("Min trades (train)", 10, 1000, 40, 5)
        topk = c6.slider("Finalists to validate", 3, 20, 10)
        key = get_api_key()
        use_ai = st.checkbox("Use Gemini", value=bool(key), help="Untick to search with templates + mutation only.")
        if use_ai and not key:
            st.warning("Enter a Gemini API key in the sidebar (get one at aistudio.google.com).")
        st.info(
            "Search only sees the training window. The finalists are then scored on the hidden holdout and on "
            "5 chronological folds. A strategy is flagged **robust** only if it stays profitable on data it never saw. "
            "Pushing for a very high win rate often just means tiny targets with wide stops - check profit factor.")
        if st.button("🚀 Run search", type="primary", disabled=(use_ai and not key)):
            lab = Lab(df, cfg, holdout)
            gem = GeminiClient(key, custom_model.strip() or model) if use_ai else None
            bar = st.progress(0.0, text="Starting...")
            try:
                found, log = run_search(
                    lab, SearchConfig(objective=objective, rounds=rounds, per_round=per_round,
                                      mutations_per_round=mut, min_trades=int(min_tr), holdout_frac=holdout, top_k=topk),
                    gemini=gem, progress=lambda f, msg: bar.progress(f, text=msg))
                st.session_state["found"], st.session_state["log"] = found, log
                st.session_state["found_cfg"] = cfg.to_dict()
                bar.empty()
                st.success("Search finished - see the Results tab.")
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(f"Search failed: {exc}")
        if st.session_state.get("log"):
            with st.expander("Search log"):
                st.text("\n".join(st.session_state["log"]))

# ----------------------------------------------------------------------------- results tab
with tab_res:
    found = st.session_state.get("found")
    if not found or df is None:
        st.info("No search results yet.")
    else:
        if st.session_state.get("found_cfg") != cfg.to_dict():
            st.warning("Cost/risk settings changed since the search ran. Numbers below are from the old settings; "
                       "re-run for consistency (the Backtest tab uses your current settings).")
        table = pd.DataFrame([c.row() for c in found])
        n_ok = int(table["robust"].fillna(False).sum())
        if n_ok == 0:
            st.error("No finalist passed the holdout robustness test. That is a normal and honest outcome - "
                     "try more data, more rounds, or a different objective rather than trusting the top row.")
        else:
            st.success(f"{n_ok} of {len(found)} finalists are profitable on the unseen holdout.")
        st.dataframe(table, hide_index=True, use_container_width=True)
        i = st.selectbox("Inspect", range(len(found)), format_func=lambda i: f"{found[i].spec['name']} [{found[i].id}]")
        c = found[i]
        a, b = st.columns(2)
        a.markdown("**Train**"); a.dataframe(fmt_metrics(c.train), hide_index=True, use_container_width=True)
        b.markdown("**Holdout (unseen)**"); b.dataframe(fmt_metrics(c.holdout), hide_index=True, use_container_width=True)
        st.code(describe(c.spec))
        if c.spec.get("rationale"):
            st.caption(c.spec["rationale"])
        st.download_button("Download strategy JSON", json.dumps(c.spec, indent=2), f"{c.id}.json")
        key = get_api_key()
        if key and st.button("Ask Gemini to critique this strategy"):
            with st.spinner("Reviewing..."):
                try:
                    g = GeminiClient(key, custom_model.strip() or model)
                    st.markdown(g.review(c.spec, c.train, c.holdout, c.fold_profitable))
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        st.caption("Backtests are not predictions. Past results, even on holdout data, do not guarantee future "
                   "performance. Forward-test on a demo account before risking capital.")
