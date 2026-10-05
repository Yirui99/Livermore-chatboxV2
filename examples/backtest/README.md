# Backtest example

The original course project's technical-indicator backtest, kept as an example.
It is not part of the `livermore` package and is not covered by the eval gate.

- `livermore_strategy.py` — 50/200-day MA trend filter + 20-day breakout (yfinance data)
- `enhanced_strategy.py` — adds RSI / MACD / Bollinger / volume confirmation and ATR trailing stops
- `test_strategy.py` — prints strategy vs buy-and-hold for AAPL, TSLA, NVDA, MSFT (2020–2024)
- `app.py` + `page.py` — Streamlit UI
- `notebooks/` — the original strategy notebook

```bash
pip install -e ".[app,examples]"
python examples/backtest/test_strategy.py
streamlit run examples/backtest/app.py
```
