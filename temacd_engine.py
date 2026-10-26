import numpy as np
import pandas as pd
import yfinance as yf

CORE_ASSETS = ["SPY", "QQQ", "XLK", "XLY", "XLI", "XLV", "XLF", "XLE", "XLP", "XLU"]
AGGRESSIVE_ASSETS = ["TQQQ", "SOXL", "SMH", "BTC-USD", "ETH-USD", "SOL-USD"]
BENCHMARK = "SPY"

DEFAULT_TEMACD = {
    "ema_fast": 10,
    "ema_mid": 30,
    "ema_slow": 120,
    "macd_fast": 30,
    "macd_slow": 80,
    "macd_signal": 15,
}


def fetch_data(start, end, tickers):
    tickers = sorted(set(tickers + [BENCHMARK]))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Yahoo Finance did not return multi-index data.")
    close = raw["Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    return close.dropna(how="all"), high, low


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def tema(series, span):
    e1 = ema(series, span)
    e2 = ema(e1, span)
    e3 = ema(e2, span)
    return 3 * e1 - 3 * e2 + e3


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def atr_percent(high, low, close, n=20):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean() / close


def temacd(close, params=None):
    p = {**DEFAULT_TEMACD, **(params or {})}
    fast = tema(close, p["ema_fast"])
    slow = tema(close, p["ema_slow"])
    macd_line = fast - slow
    signal = ema(macd_line, p["macd_signal"])
    histogram = macd_line - signal
    trend_filter = close > ema(close, p["ema_slow"])
    return {
        "macd": macd_line,
        "signal": signal,
        "histogram": histogram,
        "trend_ok": trend_filter,
    }


def momentum_score(close, indicators):
    hist_z = (indicators["histogram"] - indicators["histogram"].rolling(60).mean()) / (
        indicators["histogram"].rolling(60).std() + 1e-12
    )
    rsi_score = (rsi(close) - 50) / 50
    return 0.6 * hist_z + 0.4 * rsi_score


def compute_indicators(close, high, low, params=None):
    indicators = {}
    for ticker in close.columns:
        ind = temacd(close[ticker], params)
        ind["rsi"] = rsi(close[ticker])
        ind["atr_pct"] = atr_percent(high[ticker], low[ticker], close[ticker])
        ind["score"] = momentum_score(close[ticker], ind)
        indicators[ticker] = ind
    return indicators


def rebalance_dates(index, frequency):
    if frequency == "Weekly":
        dates = index[index.dayofweek == 4]
    else:
        dates = index.to_series().groupby([index.year, index.month]).last()
    return pd.DatetimeIndex(dates)


def select_portfolio(date, close, indicators, top_n, universe):
    eligible = []
    for ticker in universe:
        if ticker not in close.columns:
            continue
        ind = indicators[ticker]
        if date not in ind["trend_ok"].index:
            continue
        trend_ok = bool(ind["trend_ok"].loc[date])
        bullish = bool(ind["macd"].loc[date] > ind["signal"].loc[date])
        score = float(ind["score"].loc[date]) if pd.notna(ind["score"].loc[date]) else -np.inf
        if trend_ok and bullish:
            eligible.append((ticker, score))
    eligible.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in eligible[:top_n]]


def atr_weights(selected, date, indicators, risk_unit, max_weight):
    raw_weights = {}
    for ticker in selected:
        atrp = float(indicators[ticker]["atr_pct"].loc[date])
        if atrp <= 0 or np.isnan(atrp):
            raw_weights[ticker] = 0.0
        else:
            raw_weights[ticker] = min(risk_unit / atrp, max_weight)
    total = sum(raw_weights.values())
    if total > 1:
        raw_weights = {k: v / total for k, v in raw_weights.items()}
    return raw_weights


def backtest(close, high, low, params):
    universe = params["universe"]
    indicators = compute_indicators(close, high, low, params.get("temacd"))
    dates = close.index
    rebalances = rebalance_dates(dates, params["frequency"])
    weights = {}
    equity = params["initial_capital"]
    equity_curve = []
    rebalance_log = []
    cost_rate = params["cost_bps"] / 10000
    returns = close.pct_change().fillna(0)
    prev_weights = {}
    for i, date in enumerate(dates[1:], start=1):
        daily_return = sum(weights.get(t, 0) * returns[t].loc[date] for t in weights)
        equity *= 1 + daily_return
        if date in rebalances:
            selected = select_portfolio(date, close, indicators, params["top_n"], universe)
            new_weights = atr_weights(selected, date, indicators, params["risk_unit"], params["max_weight"])
            turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(new_weights) | set(prev_weights))
            equity *= 1 - turnover * cost_rate
            rebalance_log.append({
                "date": date,
                "selected": ", ".join(selected) if selected else "CASH",
                "weights": ", ".join(f"{k}:{v:.1%}" for k, v in new_weights.items()) if new_weights else "CASH",
                "turnover": turnover,
            })
            weights = new_weights
            prev_weights = new_weights
        equity_curve.append({"date": date, "equity": equity})
    result = pd.DataFrame(equity_curve).set_index("date")
    benchmark_equity = params["initial_capital"] * (1 + returns[BENCHMARK]).cumprod()
    result["benchmark"] = benchmark_equity.reindex(result.index).ffill()
    return result, pd.DataFrame(rebalance_log), indicators


def performance(equity):
    returns = equity.pct_change().dropna()
    years = len(returns) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = returns.mean() * 252 / downside if downside > 0 else 0
    dd = equity / equity.cummax() - 1
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = (returns > 0).mean()
    return {
        "CAGR": cagr,
        "MaxDD": dd.min(),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "ProfitFactor": profit_factor,
        "WinRate": win_rate,
        "NetProfit": equity.iloc[-1] / equity.iloc[0] - 1,
    }


def period_metrics(result):
    periods = {"1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520, "20Y": 5040, "30Y": 7560}
    rows = []
    for label, days in periods.items():
        if len(result) < days:
            continue
        window = result.iloc[-days:]
        strategy = performance(window["equity"])
        benchmark = performance(window["benchmark"])
        rows.append({
            "Period": label,
            "Strategy CAGR": strategy["CAGR"],
            "Strategy MaxDD": strategy["MaxDD"],
            "Strategy Sharpe": strategy["Sharpe"],
            "Strategy Sortino": strategy["Sortino"],
            "Strategy Profit Factor": strategy["ProfitFactor"],
            "Strategy Net Profit": strategy["NetProfit"],
            "SPY CAGR": benchmark["CAGR"],
            "SPY MaxDD": benchmark["MaxDD"],
            "SPY Net Profit": benchmark["NetProfit"],
        })
    return pd.DataFrame(rows)
