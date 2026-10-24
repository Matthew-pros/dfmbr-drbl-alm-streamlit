import streamlit as st
import pandas as pd
from strategy_engine import fetch_data, backtest, performance

st.set_page_config(page_title="DFMBR + DRBL + ALM", layout="wide")
st.title("DFMBR + DRBL + ALM Strategy")
st.caption("Monthly rotation across QQQ / SPY / GLD / BTC with dynamic leverage and ALM entry.")

with st.sidebar:
    st.header("Parameters")
    start = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    initial_capital = st.number_input("Initial capital", 10000, 100000000, 100000, 10000)
    commission_bps = st.number_input("Commission (bps)", 0.0, 100.0, 5.0, 0.5)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 0.5)
    adx_threshold = st.slider("ADX threshold for 3x", 15, 40, 25)
    rvol_threshold = st.slider("Rvol threshold for 3x", 1.0, 2.5, 1.4, 0.1)
    atr_percentile = st.slider("ATR percentile limit for 3x", 50, 99, 90)
    momentum_buffer = st.slider("Momentum buffer", 0.0, 0.25, 0.08, 0.01)
    lrs_window = st.slider("LRS window", 50, 200, 100, 10)
    cash_reserve = st.slider("Permanent cash reserve", 0.0, 0.30, 0.15, 0.01)
    run = st.button("Run backtest", type="primary")

params = {
    "initial_capital": initial_capital,
    "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
    "adx_threshold": adx_threshold,
    "rvol_threshold": rvol_threshold,
    "atr_percentile": atr_percentile / 100,
    "momentum_buffer": momentum_buffer,
    "lrs_window": lrs_window,
    "cash_reserve": cash_reserve,
    "alm_tier_1": -0.80,
    "alm_tier_2": -0.50,
    "alm_tier_3": -0.30,
    "alm_alloc_1": 0.50,
    "alm_alloc_2": 0.20,
    "alm_alloc_3": 0.15,
}

if run:
    with st.spinner("Downloading data and running backtest..."):
        close, volume, ohlcv = fetch_data(str(start), pd.Timestamp.today().strftime("%Y-%m-%d"))
        result, trades = backtest(close, volume, ohlcv, params)
    metrics = performance(result["equity"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{metrics['CAGR'] * 100:.2f}%")
    c2.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{metrics['MaxDrawdown'] * 100:.2f}%")
    c4.metric("Final equity", f"{metrics['FinalEquity']:,.0f}")
    st.subheader("Equity curve")
    st.line_chart(result["equity"])
    st.subheader("Current allocation")
    last = result.iloc[-1]
    st.write(f"**Leader:** {last['leader']}")
    st.write(f"**Position:** {last['position']}")
    st.write(f"**Leverage:** {last['leverage']}x")
    st.subheader("Trades")
    st.dataframe(trades, use_container_width=True)
else:
    st.info("Set parameters in the sidebar and run the backtest.")
