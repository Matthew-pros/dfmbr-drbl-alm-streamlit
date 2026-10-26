import streamlit as st
import pandas as pd
from temacd_engine import (
    CORE_ASSETS,
    AGGRESSIVE_ASSETS,
    fetch_data,
    backtest,
    period_metrics,
    walk_forward_metrics,
    monte_carlo,
    export_results,
)

st.set_page_config(page_title="TEMACD Volatility-Sized Rotation", layout="wide")
st.title("TEMACD Volatility-Sized Sector & Beta Rotation")
st.caption("Daily yfinance data, TEMACD momentum selection, ATR risk-parity sizing, walk-forward validation and Monte Carlo.")

tab_backtest, tab_validation, tab_export = st.tabs(["Backtest", "Validation", "Export"])

with st.sidebar:
    st.header("Strategy settings")
    universe_name = st.selectbox("Asset universe", ["Core ETFs", "Aggressive", "Combined"])
    if universe_name == "Core ETFs":
        universe = CORE_ASSETS
    elif universe_name == "Aggressive":
        universe = AGGRESSIVE_ASSETS
    else:
        universe = CORE_ASSETS + AGGRESSIVE_ASSETS

    start = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    initial_capital = st.number_input("Initial capital", 10000, 100000000, 100000, 10000)
    top_n = st.slider("Top N assets", 1, 6, 3)
    frequency = st.selectbox("Rebalance frequency", ["Weekly", "Monthly"])
    risk_unit = st.slider("Risk unit per position", 0.25, 3.0, 1.0, 0.25) / 100
    max_weight = st.slider("Maximum weight per asset", 0.10, 1.00, 0.35, 0.05)
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0.0, 100.0, 5.0, 0.5)
    mc_simulations = st.slider("Monte Carlo simulations", 1000, 20000, 5000, 1000)
    run = st.button("Run backtest", type="primary")

params = {
    "universe": universe,
    "initial_capital": initial_capital,
    "top_n": top_n,
    "frequency": frequency,
    "risk_unit": risk_unit,
    "max_weight": max_weight,
    "cost_bps": cost_bps,
}

if run:
    with st.spinner("Downloading daily yfinance data and running backtest..."):
        close, high, low = fetch_data(str(start), pd.Timestamp.today().strftime("%Y-%m-%d"), universe)
        result, rebalance_log, indicators = backtest(close, high, low, params)

    with tab_backtest:
        st.subheader("Equity curve vs SPY Buy & Hold")
        st.line_chart(result[["equity", "benchmark"]])

        st.subheader("Performance by period")
        metrics = period_metrics(result)
        if metrics.empty:
            st.warning("Not enough history for the requested periods.")
        else:
            st.dataframe(
                metrics.style.format({
                    "Strategy CAGR": "{:.2%}",
                    "Strategy MaxDD": "{:.2%}",
                    "Strategy Sharpe": "{:.2f}",
                    "Strategy Sortino": "{:.2f}",
                    "Strategy Profit Factor": "{:.2f}",
                    "Strategy Net Profit": "{:.2%}",
                    "SPY CAGR": "{:.2%}",
                    "SPY MaxDD": "{:.2%}",
                    "SPY Net Profit": "{:.2%}",
                }),
                use_container_width=True,
            )

        st.subheader("Current eligible signals")
        signal_rows = []
        for ticker in universe:
            if ticker not in indicators:
                continue
            ind = indicators[ticker]
            signal_rows.append({
                "Ticker": ticker,
                "Trend OK": bool(ind["trend_ok"].iloc[-1]),
                "TEMACD Bullish": bool(ind["macd"].iloc[-1] > ind["signal"].iloc[-1]),
                "Momentum Score": float(ind["score"].iloc[-1]),
                "ATR %": float(ind["atr_pct"].iloc[-1]),
            })
        st.dataframe(pd.DataFrame(signal_rows), use_container_width=True)

        st.subheader("Rebalance log")
        st.dataframe(rebalance_log, use_container_width=True)

    with tab_validation:
        st.subheader("Walk-forward validation")
        wf = walk_forward_metrics(result)
        if wf.empty:
            st.warning("Not enough history for walk-forward validation.")
        else:
            st.dataframe(wf, use_container_width=True)

        st.subheader("Monte Carlo bootstrap")
        mc = monte_carlo(result, mc_simulations)
        if mc.empty:
            st.warning("Not enough history for Monte Carlo simulation.")
        else:
            st.dataframe(mc, use_container_width=True)

    with tab_export:
        st.subheader("Download results")
        equity_csv = result.to_csv().encode("utf-8")
        rebalance_csv = rebalance_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download equity CSV", equity_csv, "temacd_equity.csv", "text/csv")
        st.download_button("Download rebalance CSV", rebalance_csv, "temacd_rebalances.csv", "text/csv")
else:
    st.info("Set strategy settings in the sidebar and run the backtest.")
