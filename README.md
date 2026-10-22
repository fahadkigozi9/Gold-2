# 🥇 Gold Strategy Lab

A Streamlit app that backtests gold (XAUUSD) strategies and uses the **Gemini API** to propose new ones, then judges them on data the search never saw.

## What it does

- **Data**: upload a CSV (MetaTrader 5 / TradingView / Dukascopy exports work as-is), download from Yahoo Finance (`GC=F`), pull candles from a local MT5 terminal (Windows), or use synthetic demo data.
- **Backtester**: bar-by-bar, next-bar-open entries, ATR-based SL/TP, spread + slippage + commission, stop-first when SL and TP hit in the same bar, gap-aware fills, results in R-multiples with compounding equity, drawdown, profit factor, SQN, and Monte Carlo drawdown.
- **AI search**: Gemini proposes strategies as **JSON (never code)**. Each spec is validated against a whitelist DSL, so nothing the model writes can execute. Results feed back to Gemini round by round, and a local mutation step adds variants.
- **Honest validation**: the search sees only the training window. Finalists are re-tested on a hidden holdout and 5 chronological folds, and flagged **robust** only if they pass a strict, significance-based gate.

## Quick start

```bash
git clone <your-repo-url> && cd gold-strategy-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free key at <https://aistudio.google.com/apikey>, then either paste it in the sidebar or set it once:

```bash
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit
# or: export GEMINI_API_KEY=...
```

`secrets.toml` is git-ignored. **Never commit your key.**

### Deploy on Streamlit Community Cloud
Push to GitHub, create an app pointing at `app.py`, and add `GEMINI_API_KEY` under *Settings → Secrets*. MT5 fetching only works locally on Windows; use CSV upload or Yahoo there.

## CSV format
Any delimiter (comma, tab, semicolon). Needs a time column (`time`, `datetime`, `timestamp`, or separate `date` + `time`, epoch seconds/ms also fine) and `open, high, low, close`. MT5 headers like `<DATE> <TIME> <OPEN>` are handled.

## Strategy DSL
```json
{
  "name": "EMA pullback",
  "long":  [{"a": {"ind": "ema", "n": 50}, "op": ">", "b": {"ind": "ema", "n": 200}},
            {"a": {"ind": "rsi", "n": 14}, "op": "crosses_above", "b": 40}],
  "short": [],
  "sl_atr": 1.5, "tp_atr": 2.5, "atr_n": 14, "max_bars": 80
}
```
Conditions on a side are AND-ed. Operators: `> < >= <= crosses_above crosses_below`. Indicators: `close open high low hour dow ema sma rsi atr adx stoch_k roc zscore macd_hist bb_upper bb_lower donchian_high donchian_low dist_ema_atr body_atr range_atr`. Paste custom JSON on the **Backtest** tab.

## How to read the results (please read)

- **Win rate is the easiest metric to fake.** A tiny take-profit with a wide stop gives 80%+ wins and still loses money. The "High win rate" objective therefore requires profit factor ≥ 1.2 and positive expectancy first.
- **Searching many strategies guarantees some look great by luck.** That is why only the unseen holdout counts. The robust gate requires ≥ 30 holdout trades, PF ≥ 1.25, SQN ≥ 2.5, ≥ 4 of 5 profitable folds, and no collapse versus training.
- Tested here on 20 pure random-walk datasets, the gate produced **0 false positives**, and it still detected a planted trend edge. Seeing **"no robust strategy"** on real data is a normal, honest result, not a bug.
- The search reuses the holdout every time you re-run it. If you tweak and re-run repeatedly, you are slowly fitting to it. Keep a final slice of newer data aside, and **forward-test on a demo account** before real money. This matters even more if the goal is passing a prop-firm challenge, where drawdown limits are strict (check the Monte Carlo drawdown line).
- Costs are inputs. Set spread, slippage and commission to your actual broker. Yahoo `GC=F` is futures data, so levels differ slightly from spot XAUUSD. Intraday history there is limited (1h ≈ 730 days, 15m ≈ 60 days).

*Not financial advice. Backtests do not predict future performance.*

## Project layout
```
app.py               Streamlit UI
goldlab/data.py         CSV / Yahoo / MT5 loaders, cleaning, resampling
goldlab/indicators.py   causal indicators
goldlab/dsl.py          safe JSON strategy language + validation
goldlab/backtester.py   simulator + metrics + Monte Carlo
goldlab/strategies.py   seed templates + mutation
goldlab/gemini_ai.py    Gemini client (retries, JSON parsing, critique)
goldlab/search.py       search loop + holdout/fold validation
tests/               28 tests (no-lookahead, fill logic, noise rejection, UI smoke)
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Gemini models
Defaults to `gemini-3.8-flash`. Model names change often, so use the *custom model id* box in the sidebar if one is retired. See <https://ai.google.dev/gemini-api/docs/models>.
