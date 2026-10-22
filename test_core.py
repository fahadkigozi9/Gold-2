import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldlab import indicators as ind
from goldlab.backtester import BTConfig, MarketData, compute_metrics, simulate
from goldlab.data import load_csv, make_synthetic
from goldlab.dsl import SeriesCache, SpecError, signals, validate_spec
from goldlab.gemini_ai import GeminiClient, parse_strategies
from goldlab.search import Lab, SearchConfig, run_search
from goldlab.strategies import TEMPLATES, mutate

ZERO_COST = BTConfig(spread=0.0, slippage=0.0, min_stop=0.0)


def frame(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def run_one(rows, direction, sl=1.0, tp=2.0, max_bars=10, cfg=ZERO_COST, atr_val=1.0):
    df = frame(rows)
    md = MarketData(df)
    sig = np.zeros(len(df), dtype=np.int8)
    sig[0] = direction
    atr = np.full(len(df), atr_val)
    return simulate(md, sig, atr, sl, tp, max_bars, cfg)


# ------------------------------------------------------------------ backtester
def test_long_take_profit():
    t = run_one([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)], 1)
    assert len(t) == 1 and t.iloc[0]["reason"] == "tp"
    assert t.iloc[0]["r"] == pytest.approx(2.0)
    assert t.iloc[0]["entry"] == 100  # next bar open, not signal bar close


def test_long_stop_loss():
    t = run_one([(100, 100, 100, 100), (100, 100.5, 98.5, 99), (99, 99, 99, 99)], 1)
    assert t.iloc[0]["reason"] == "sl" and t.iloc[0]["r"] == pytest.approx(-1.0)


def test_short_take_profit_and_stop():
    t = run_one([(100, 100, 100, 100), (100, 100.2, 97.5, 98), (98, 98, 98, 98)], -1)
    assert t.iloc[0]["reason"] == "tp" and t.iloc[0]["r"] == pytest.approx(2.0)
    t = run_one([(100, 100, 100, 100), (100, 101.5, 99.8, 101), (101, 101, 101, 101)], -1)
    assert t.iloc[0]["reason"] == "sl" and t.iloc[0]["r"] == pytest.approx(-1.0)


def test_both_hit_same_bar_is_stop():
    t = run_one([(100, 100, 100, 100), (100, 103, 98, 101), (101, 101, 101, 101)], 1)
    assert t.iloc[0]["reason"] == "sl"


def test_gap_through_stop_fills_at_open():
    t = run_one([(100, 100, 100, 100), (100, 100.1, 99.9, 100), (97, 97.5, 96.5, 97), (97, 97, 97, 97)], 1)
    # bar1 doesn't hit; bar2 gaps to 97 -> exits at 97 (R=-3), not at stop 99
    assert t.iloc[0]["exit"] == 97 and t.iloc[0]["r"] == pytest.approx(-3.0)


def test_costs_reduce_r():
    cfg = BTConfig(spread=0.5, slippage=0.0, min_stop=0.0)
    t = run_one([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)], 1, cfg=cfg)
    assert t.iloc[0]["r"] == pytest.approx(1.5)


def test_time_exit():
    rows = [(100, 100, 100, 100)] + [(100, 100.3, 99.7, 100.1)] * 5
    t = run_one(rows, 1, max_bars=3)
    assert t.iloc[0]["reason"] == "time" and t.iloc[0]["bars"] == 3


def test_no_overlapping_trades():
    df = frame([(100, 100.2, 99.8, 100)] * 30)
    md = MarketData(df)
    sig = np.ones(len(df), dtype=np.int8)
    t = simulate(md, sig, np.full(len(df), 1.0), 1.0, 2.0, 5, ZERO_COST)
    assert (t["entry_time"].iloc[1:].values >= t["exit_time"].iloc[:-1].values).all()


def test_metrics():
    t = pd.DataFrame({"r": [2.0, -1.0, 2.0, -1.0], "dir": [1, 1, -1, -1], "bars": [3, 4, 5, 6]})
    m = compute_metrics(t, BTConfig())
    assert m["trades"] == 4 and m["win_rate"] == 50.0
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["expectancy_r"] == pytest.approx(0.5)
    assert m["max_consec_losses"] == 1
    assert compute_metrics(t.iloc[0:0], BTConfig())["trades"] == 0


# ------------------------------------------------------------------ indicators: no lookahead
def test_indicators_are_causal():
    df = make_synthetic(1500, seed=3)
    cut = 1000
    full, part = SeriesCache(df), SeriesCache(df.iloc[:cut])
    probes = [("ema", {"n": 50}), ("rsi", {"n": 14}), ("atr", {"n": 14}), ("adx", {"n": 14}),
              ("macd_hist", {"fast": 12, "slow": 26, "signal": 9}), ("donchian_high", {"n": 20}),
              ("bb_upper", {"n": 20, "k": 2.0}), ("zscore", {"n": 30}), ("stoch_k", {"n": 14}),
              ("dist_ema_atr", {"n": 20}), ("body_atr", {}), ("hour", {})]
    for name, p in probes:
        a = full.get(name, **p).iloc[:cut].to_numpy()
        b = part.get(name, **p).to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{name} uses future data"


def test_signals_causal():
    df = make_synthetic(1500, seed=4)
    cut = 1000
    for spec in TEMPLATES:
        a = signals(spec, SeriesCache(df))[:cut]
        b = signals(spec, SeriesCache(df.iloc[:cut]))
        assert (a == b).all(), spec["name"]


def test_rsi_bounds():
    r = ind.rsi(make_synthetic(800)["close"]).dropna()
    assert r.between(0, 100).all()


# ------------------------------------------------------------------ DSL
def test_validate_clamps_and_repairs():
    s = validate_spec({"long": [{"a": {"ind": "EMA", "n": 99999}, "operator": "above", "b": {"ind": "close"}}],
                       "sl_atr": 999, "tp_atr": -4})
    assert s["long"][0]["a"]["n"] == 400 and s["long"][0]["op"] == ">"
    assert s["sl_atr"] == 8.0 and s["tp_atr"] == 0.2


@pytest.mark.parametrize("bad", [
    {"long": [{"a": {"ind": "__import__('os')"}, "op": ">", "b": 1}]},
    {"long": [{"a": 1, "op": ">", "b": 2}]},
    {"long": [{"a": {"ind": "rsi"}, "op": "~=", "b": 2}]},
    {"long": [], "short": []},
    "not a dict",
])
def test_validate_rejects(bad):
    with pytest.raises(SpecError):
        validate_spec(bad)


def test_templates_and_mutation_valid():
    rng = np.random.default_rng(0)
    for t in TEMPLATES:
        for _ in range(20):
            m = mutate(t, rng)
            if m is not None:
                validate_spec(m)


# ------------------------------------------------------------------ data
def test_load_csv_mt5_style(tmp_path):
    df = make_synthetic(400, seed=1)
    p = tmp_path / "x.csv"
    out = pd.DataFrame({"<DATE>": df.index.strftime("%Y.%m.%d"), "<TIME>": df.index.strftime("%H:%M:%S"),
                        "<OPEN>": df["open"], "<HIGH>": df["high"], "<LOW>": df["low"], "<CLOSE>": df["close"]})
    out.to_csv(p, sep="\t", index=False)
    loaded = load_csv(str(p))
    assert len(loaded) == len(df) and np.allclose(loaded["close"], df["close"])


def test_load_csv_epoch(tmp_path):
    df = make_synthetic(300, seed=2)
    p = tmp_path / "e.csv"
    pd.DataFrame({"time": df.index.as_unit("s").astype("int64"), "open": df["open"], "high": df["high"],
                  "low": df["low"], "close": df["close"]}).to_csv(p, index=False)
    assert len(load_csv(str(p))) == len(df)


# ------------------------------------------------------------------ Gemini plumbing (mocked)
class FakeResp:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def generate_content(self, **kw):
        self.calls += 1
        t = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(t, Exception):
            raise t
        return FakeResp(t)


class FakeClient:
    def __init__(self, texts):
        self.models = FakeModels(texts)


GOOD = {"strategies": [{"name": "t1", "long": [{"a": {"ind": "rsi", "n": 14}, "op": "<", "b": 30}],
                        "short": [{"a": {"ind": "rsi", "n": 14}, "op": ">", "b": 70}],
                        "sl_atr": 1.5, "tp_atr": 1.5},
                       {"name": "bad", "long": [{"a": {"ind": "magic"}, "op": ">", "b": 1}]}]}


def test_parse_handles_fenced_json_and_rejects_bad():
    specs, errs = parse_strategies("Sure!\n```json\n" + json.dumps(GOOD) + "\n```")
    assert len(specs) == 1 and len(errs) == 1


def test_gemini_retries_then_succeeds():
    fc = FakeClient([RuntimeError("503 overloaded"), json.dumps(GOOD)])
    g = GeminiClient("k", client=fc, sleep=lambda s: None)
    specs, errs = g.propose("go")
    assert len(specs) == 1 and fc.models.calls == 2


def test_gemini_bad_key_not_retried():
    class E(Exception):
        code = 403
    fc = FakeClient([E("denied")])
    g = GeminiClient("k", client=fc, sleep=lambda s: None)
    with pytest.raises(Exception):
        g.propose("go")
    assert fc.models.calls == 1


# ------------------------------------------------------------------ end-to-end search
def test_search_end_to_end_with_mock_gemini():
    df = make_synthetic(6000, seed=11)
    lab = Lab(df, BTConfig(), 0.3)
    g = GeminiClient("k", client=FakeClient([json.dumps(GOOD)]), sleep=lambda s: None)
    res, log = run_search(lab, SearchConfig(rounds=2, per_round=4, mutations_per_round=6, min_trades=20, top_k=5),
                          gemini=g)
    assert 1 <= len(res) <= 5
    assert all(c.holdout is not None for c in res)
    assert any("Gemini" in line for line in log)


def test_search_survives_gemini_failure():
    df = make_synthetic(5000, seed=12)
    lab = Lab(df, BTConfig(), 0.3)
    g = GeminiClient("k", client=FakeClient([RuntimeError("boom")]), sleep=lambda s: None)
    res, log = run_search(lab, SearchConfig(rounds=1, mutations_per_round=4, min_trades=20, top_k=3), gemini=g)
    assert len(res) >= 1 and any("failed" in line for line in log)


def _random_walk(seed, n=12000):
    from goldlab.data import clean_ohlc
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="15min")
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
    op = np.r_[2000, close[:-1]]
    w = np.abs(rng.normal(0, 0.0004, (n, 2))) * close[:, None]
    return clean_ohlc(pd.DataFrame({"open": op, "high": np.maximum(op, close) + w[:, 0],
                                    "low": np.minimum(op, close) - w[:, 1], "close": close}, index=idx))


def test_robust_gate_rejects_noise_but_finds_real_edge():
    cfg = SearchConfig(rounds=3, mutations_per_round=10, min_trades=40)
    for seed in (100, 101, 102, 103):
        res, _ = run_search(Lab(_random_walk(seed), BTConfig(), 0.3), cfg)
        assert not any(c.robust for c in res), f"false positive on pure noise (seed {seed})"
    res, _ = run_search(Lab(make_synthetic(20000), BTConfig(), 0.3), cfg)
    assert any(c.robust for c in res), "failed to detect a planted trend edge"
