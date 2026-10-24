import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = {
    "QQQ": {"leveraged": "TQQQ", "method": "TSDB"},
    "SPY": {"leveraged": "UPRO", "method": "TSDB"},
    "GLD": {"leveraged": "UGL", "method": "LRS"},
    "BTC-USD": {"leveraged": "BTC-USD", "method": "LRS"},
}


def fetch_data(start, end):
    tickers = sorted({cfg["leveraged"] for cfg in UNIVERSE.values()})
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Yahoo Finance did not return multi-index data.")
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    return close.dropna(how="all"), volume.reindex(close.index).fillna(0), ohlcv


def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def rvol(volume, fast=20, slow=200):
    return volume.rolling(fast).mean() / volume.rolling(slow).mean()


def atr_pct(df, n=20):
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / df["Close"]


def tsdb_signals(close):
    entry = close > close.rolling(40).max().shift(1)
    exit_signal = close < close.rolling(20).min().shift(1)
    return entry, exit_signal


def lrs_signals(close, window=100):
    x = np.arange(window)
    slope = pd.Series(index=close.index, dtype=float)
    fitted = pd.Series(index=close.index, dtype=float)
    resid = pd.Series(index=close.index, dtype=float)
    values = close.values
    for i in range(window, len(close)):
        y = values[i - window:i]
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        slope.iloc[i] = b
        fitted.iloc[i] = pred[-1]
        resid.iloc[i] = np.std(y - pred)
    entry = (slope > 0) & (close > fitted + 2 * resid)
    exit_signal = slope < 0
    return entry, exit_signal


def rotation_scores(close):
    momentum = close.shift(21) / close.shift(252) - 1
    vol = close.pct_change().rolling(60).std()
    return momentum / vol


def select_leader(close, current=None, buffer=0.08):
    scores = rotation_scores(close)
    if scores.empty:
        return current
    latest = scores.iloc[-1].dropna()
    if latest.empty:
        return current
    best = latest.idxmax()
    if current is None or current not in latest:
        return best
    return best if latest[best] > latest[current] * (1 + buffer) else current


def build_signals(ohlcv, volume, params):
    signals = {}
    for asset, cfg in UNIVERSE.items():
        ticker = cfg["leveraged"]
        if ticker not in ohlcv.columns.get_level_values(1):
            continue
        df = ohlcv.xs(ticker, axis=1, level=1)
        close = df["Close"]
        if cfg["method"] == "TSDB":
            entry, exit_signal = tsdb_signals(close)
        else:
            entry, exit_signal = lrs_signals(close, params["lrs_window"])
        a = adx(df)
        rv = rvol(volume[ticker])
        ap = atr_pct(df)
        limit = ap.quantile(params["atr_percentile"])
        lev_ok = (a >= params["adx_threshold"]) & (rv >= params["rvol_threshold"]) & (ap <= limit)
        signals[asset] = {"entry": entry, "exit": exit_signal, "leverage_ok": lev_ok}
    return signals


def drawdown_from_ath(close):
    return close / close.cummax() - 1


def alm_allocation(dd, params):
    if dd <= params["alm_tier_1"]:
        return params["alm_alloc_1"]
    if dd <= params["alm_tier_2"]:
        return params["alm_alloc_2"]
    if dd <= params["alm_tier_3"]:
        return params["alm_alloc_3"]
    return 0.0


def backtest(close, volume, ohlcv, params):
    dates = close.index
    signals = build_signals(ohlcv, volume, params)
    equity = params["initial_capital"]
    reserve = equity * params["cash_reserve"]
    deployable = equity - reserve
    position = None
    leverage = 1.0
    leader = None
    initialized = False
    rows = []
    trades = []
    prev = close.iloc[0]
    for i, date in enumerate(dates[1:], start=1):
        prices = close.loc[date]
        if position is not None and position in prices.index:
            equity *= 1 + (prices[position] / prev[position] - 1) * leverage
        if i == 1 or date.month != dates[i - 1].month or leader is None:
            leader = select_leader(close.loc[:date], leader, params["momentum_buffer"])
        sig = signals.get(leader)
        if sig is not None:
            entry = bool(sig["entry"].loc[date]) if date in sig["entry"].index else False
            exit_signal = bool(sig["exit"].loc[date]) if date in sig["exit"].index else False
            lev_ok = bool(sig["leverage_ok"].loc[date]) if date in sig["leverage_ok"].index else False
            cost_rate = (params["commission_bps"] + params["slippage_bps"]) / 10000
            if position is None and entry:
                if not initialized and leader in ["QQQ", "SPY"]:
                    dd = float(drawdown_from_ath(close[leader].loc[:date]).iloc[-1])
                    alloc = alm_allocation(dd, params)
                    if alloc > 0:
                        value = deployable * alloc
                        equity -= value * cost_rate
                        position, leverage, initialized = leader, 3.0 if lev_ok else 1.0, True
                        trades.append({"date": date, "action": "ALM BUY", "asset": leader, "leverage": leverage, "value": value})
                else:
                    value = equity - reserve
                    equity -= value * cost_rate
                    position, leverage = leader, 3.0 if lev_ok else 1.0
                    trades.append({"date": date, "action": "BUY", "asset": leader, "leverage": leverage, "value": value})
            elif position is not None and exit_signal:
                value = equity - reserve
                equity -= value * cost_rate
                trades.append({"date": date, "action": "SELL", "asset": position, "leverage": leverage, "value": value})
                position, leverage = None, 1.0
        rows.append({"date": date, "equity": equity, "position": position or "CASH", "leverage": leverage, "leader": leader})
        prev = prices
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades)


def performance(equity):
    returns = equity.pct_change().dropna()
    years = len(returns) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    dd = equity / equity.cummax() - 1
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": dd.min(), "FinalEquity": equity.iloc[-1]}
