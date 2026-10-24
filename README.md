# DFMBR + DRBL + ALM Streamlit Strategy

Streamlit app and backtest engine for the hybrid trend strategy:

- DFMBR: monthly rotation across QQQ, SPY, GLD and BTC using 12-1 momentum divided by 60-day volatility, with an 8 percent buffer.
- DRBL: switching between 3x leveraged ETF, 1x ETF and cash based on trend, volume and volatility.
- ALM: staged entry based on drawdown from ATH, applied only to QQQ and SPY.
- Backtest: daily engine including commission and slippage.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rules

| Asset | Signal method | Leveraged ticker |
|---|---|---|
| QQQ | Turtle Soup / Darvas Box: 40-day high entry, 20-day low exit | TQQQ |
| SPY | Turtle Soup / Darvas Box: 40-day high entry, 20-day low exit | UPRO |
| GLD | Linear Regression Slope: 100-day | UGL |
| BTC-USD | Linear Regression Slope: 100-day | BTC-USD |

Yahoo Finance does not provide a stable 3x gold or 3x Bitcoin ticker. UGL is a 2x gold ETF and BTC-USD is used as a 1x proxy.

## Regimes

- 3x: price in entry zone, ADX >= 25, Rvol >= 1.4, ATR percent below the 90th percentile.
- 1x: price in entry zone but weaker trend or volume.
- Cash: exit signal or price below the selling line.

## ALM

Applies only to QQQ and SPY:

- Drawdown 80 percent or more from ATH: buy 50 percent of equity.
- Drawdown 50-79 percent: buy 20 percent of equity.
- Drawdown 30-49 percent: buy 15 percent of equity.
- Permanent cash reserve: 15 percent.
