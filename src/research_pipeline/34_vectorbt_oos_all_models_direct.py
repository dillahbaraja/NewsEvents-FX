from common import *
from pathlib import Path
import json
import math
import shutil

import joblib
import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(
        "vectorbt is not installed or failed to import. Run: pip install vectorbt\n"
        f"Import error: {exc}"
    )

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"
OUT_DIR = ARTIFACTS / "vectorbt_oos_model_comparison"
CATBOOST_V2_DIR = ARTIFACTS / "catboost_v2_pair_specific_loop"
JOBLIB_DIR = ARTIFACTS / "joblib"
MT5_SIGNAL_TABLE = ARTIFACTS / "mt5_signal_table.csv"
EVENT_STUDY_SIGNAL_TABLE = ARTIFACTS / "event_study_v3_signal_table.csv"

INITIAL_CASH = 10000.0
TRADE_SIZE = 1.0
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-08-31 23:59:59")
DEFAULT_HOLDING_BARS = 1
TOLERANCE_HOURS = 2
FREQ = "1h"

GLOBAL_MODELS = ["elastic_net", "random_forest", "svm_rbf", "ebm"]
SIGNAL_TABLE_MODELS = {
    "manual_best": {
        "path": MT5_SIGNAL_TABLE,
        "model_filter": "manual_block_inverse_validation_gated_conservative",
        "runtime": "signal_table",
        "holding_bars": 1,
    },
    "pair_specific_event_selection": {
        "path": EVENT_STUDY_SIGNAL_TABLE,
        "model_filter": "event_study_v3",
        "runtime": "signal_table",
        "holding_bars": None,
    },
}


def load_metadata():
    path = ARTIFACTS / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds():
    path = ROOT / "model_pair_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model pair thresholds: {path}")
    return pd.read_csv(path)


def read_price(pair):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        return None
    d = read_prices(path).set_index("time")
    d = d[(d.index >= START) & (d.index <= END)].copy()
    return d if not d.empty else None


def prepare_pair_events(pair, holding_bars=DEFAULT_HOLDING_BARS):
    mapping = available_currency_pair_mapping()
    pair_map = mapping[mapping["pair"].eq(pair)].copy()
    if pair_map.empty:
        return pd.DataFrame()

    px = read_price(pair)
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.copy()
    px["close_fwd"] = px["Close"].shift(-holding_bars)
    px["exit_time"] = px.index.to_series().shift(-holding_bars)
    px = px.reset_index()

    news = read_news(NEWS_TEST)
    news = news[(news["event_time"] >= START) & (news["event_time"] <= END)].copy()
    news = news[news["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
    if news.empty:
        return pd.DataFrame()

    news["pair"] = pair
    news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
    news = news[news["orientation"].isin(["base", "quote"])].copy()
    news["category"] = news["Name"].map(event_category)
    news["release_hour"] = news["event_time"].dt.hour
    news["surprise_fc"] = news["Actual"] - news["Forecast"]
    news["surprise_prev"] = news["Actual"] - news["Previous"]
    news["surprise_pct_fc"] = news["surprise_fc"] / (news["Forecast"].abs() + 1e-9)
    news = add_context_features(news)

    joined = pd.merge_asof(
        news.sort_values("event_time"),
        px.sort_values("time"),
        left_on="event_time",
        right_on="time",
        direction="forward",
        tolerance=pd.Timedelta(hours=TOLERANCE_HOURS),
    )
    joined = joined.dropna(subset=["Close", "close_fwd", "exit_time"]).copy()
    if joined.empty:
        return joined

    raw_return = np.log(joined["close_fwd"] / joined["Close"])
    sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
    joined["domestic_return_1h"] = raw_return * sign
    joined["pip_size"] = joined["pair"].map(pip_size)
    joined["domestic_return_pips_1h"] = ((joined["close_fwd"] - joined["Close"]) / joined["pip_size"]) * sign
    joined["domestic_return_pct_1h"] = np.expm1(joined["domestic_return_1h"])
    return joined


def svm_decision_probability_frame(model, X):
    if not hasattr(model, "named_steps"):
        return None
    final_step = model.named_steps.get("svc")
    if final_step is None:
        return None
    if not hasattr(final_step, "classes_"):
        return None
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    proba = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    for idx, cls in enumerate(final_step.classes_):
        if int(cls) == -1:
            out["prob_sell"] = proba[:, idx]
        elif int(cls) == 0:
            out["prob_no_trade"] = proba[:, idx]
        elif int(cls) == 1:
            out["prob_buy"] = proba[:, idx]
    return out


def class_probability_frame(model, X):
    svm_probs = svm_decision_probability_frame(model, X)
    if svm_probs is not None:
        return svm_probs
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            last_step = model.named_steps[list(model.named_steps.keys())[-1]]
            classes = getattr(last_step, "classes_", [])
        out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
        for idx, cls in enumerate(classes):
            if int(cls) == -1:
                out["prob_sell"] = proba[:, idx]
            elif int(cls) == 0:
                out["prob_no_trade"] = proba[:, idx]
            elif int(cls) == 1:
                out["prob_buy"] = proba[:, idx]
        return out

    pred = pd.Series(np.asarray(model.predict(X)).reshape(-1), index=X.index)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    out.loc[pred.eq(-1), "prob_sell"] = 1.0
    out.loc[pred.eq(0), "prob_no_trade"] = 1.0
    out.loc[pred.eq(1), "prob_buy"] = 1.0
    return out


def apply_thresholds(events, probs, threshold_row):
    threshold = float(threshold_row["threshold"])
    signal_mode = str(threshold_row.get("signal_mode", "strict") or "strict")
    collapse = bool(threshold_row.get("collapse_flag", False))
    final_signal = []
    reason = []
    raw_pred = []
    confidence = []
    for _, row in probs.iterrows():
        if signal_mode == "directional":
            raw = "BUY" if row.prob_buy >= row.prob_sell else "SELL"
            conf = float(max(row.prob_buy, row.prob_sell))
        else:
            values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
            raw = max(values, key=values.get)
            conf = float(values[raw])
        raw_pred.append(raw)
        confidence.append(conf)
        if collapse and signal_mode != "directional":
            final_signal.append("NO_TRADE")
            reason.append("COLLAPSE_FILTERED")
        elif raw == "NO_TRADE":
            final_signal.append("NO_TRADE")
            reason.append("NO_TRADE")
        elif conf < threshold:
            final_signal.append("NO_TRADE")
            reason.append("LOW_CONFIDENCE")
        else:
            final_signal.append(raw)
            reason.append(raw)
    out = events.reset_index(drop=True).copy()
    out[["prob_sell", "prob_no_trade", "prob_buy"]] = probs.reset_index(drop=True)[["prob_sell", "prob_no_trade", "prob_buy"]]
    out["raw_predicted_class"] = raw_pred
    out["confidence"] = confidence
    out["threshold"] = threshold
    out["signal_mode"] = signal_mode
    out["signal"] = final_signal
    out["signal_reason"] = reason
    return out


def catboost_v2_probability_to_signal(model, X, cfg):
    classes = list(model.classes_)
    proba = np.asarray(model.predict_proba(X))
    probs = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    for idx, cls in enumerate(classes):
        cls = int(cls)
        if cls == -1:
            probs["prob_sell"] = proba[:, idx]
        elif cls == 0:
            probs["prob_no_trade"] = proba[:, idx]
        elif cls == 1:
            probs["prob_buy"] = proba[:, idx]

    raw, confidence, signals, reasons = [], [], [], []
    for _, row in probs.iterrows():
        if cfg["target_mode"] == "binary_directional":
            pred = "BUY" if row.prob_buy >= row.prob_sell else "SELL"
            conf = float(max(row.prob_buy, row.prob_sell))
        else:
            values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
            pred = max(values, key=values.get)
            conf = float(values[pred])
        raw.append(pred)
        confidence.append(conf)
        if pred == "NO_TRADE":
            signals.append("NO_TRADE")
            reasons.append("NO_TRADE")
        elif conf < float(cfg["threshold"]):
            signals.append("NO_TRADE")
            reasons.append("LOW_CONFIDENCE")
        else:
            signals.append(pred)
            reasons.append(pred)
    return probs, raw, confidence, signals, reasons


def catboost_v2_artifact(pair):
    best_dir = CATBOOST_V2_DIR / pair / "best_model"
    if not best_dir.exists():
        return None, None
    cfg_path = best_dir / "config.json"
    model_files = list(best_dir.glob("*.joblib"))
    if not cfg_path.exists() or not model_files:
        return None, None
    return json.loads(cfg_path.read_text(encoding="utf-8")), model_files[0]


def build_direct_global_predictions(pair, model_name, metadata, thresholds):
    model_path = JOBLIB_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        return pd.DataFrame(), f"missing_model:{model_path}"
    events = prepare_pair_events(pair, DEFAULT_HOLDING_BARS)
    if events.empty:
        return pd.DataFrame(), "no_events_or_price_coverage"
    threshold_match = thresholds[(thresholds["model"].eq(model_name)) & (thresholds["pair"].eq(pair))]
    if threshold_match.empty:
        return pd.DataFrame(), "missing_threshold"

    feature_names = metadata["features"]
    for col in feature_names:
        if col not in events.columns:
            events[col] = 0
    X = events[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = joblib.load(model_path)
    probs = class_probability_frame(model, X)
    pred = apply_thresholds(events, probs, threshold_match.iloc[0])
    pred["model"] = model_name
    pred["runtime"] = "direct_joblib"
    pred["source_model"] = model_name
    pred["holding_bars"] = DEFAULT_HOLDING_BARS
    return pred, "ok"


def build_catboost_v2_predictions(pair):
    cfg, model_path = catboost_v2_artifact(pair)
    if cfg is None or model_path is None:
        return pd.DataFrame(), "missing_catboost_v2_best_model"
    events = prepare_pair_events(pair, DEFAULT_HOLDING_BARS)
    if events.empty:
        return pd.DataFrame(), "no_events_or_price_coverage"
    events = events[
        pd.to_numeric(events["Importance"], errors="coerce").fillna(0).ge(float(cfg["importance_min"])) &
        pd.to_numeric(events["Impact"], errors="coerce").fillna(0).ge(float(cfg["impact_min"]))
    ].copy()
    if events.empty:
        return pd.DataFrame(), "filtered_empty"

    features = model_feature_names()
    for col in features:
        if col not in events.columns:
            events[col] = 0
    X = events[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = joblib.load(model_path)
    probs, raw, confidence, signals, reasons = catboost_v2_probability_to_signal(model, X, cfg)
    pred = events.reset_index(drop=True).copy()
    pred[["prob_sell", "prob_no_trade", "prob_buy"]] = probs.reset_index(drop=True)
    pred["raw_predicted_class"] = raw
    pred["confidence"] = confidence
    pred["threshold"] = float(cfg["threshold"])
    pred["signal_mode"] = cfg["target_mode"]
    pred["signal"] = signals
    pred["signal_reason"] = reasons
    pred["model"] = "catboost_v2_pair_direct"
    pred["runtime"] = "direct_pair_joblib"
    pred["source_model"] = cfg["name"]
    pred["holding_bars"] = DEFAULT_HOLDING_BARS
    return pred, "ok"


def load_signal_table_predictions(pair, model_name, spec):
    path = spec["path"]
    if not path.exists():
        return pd.DataFrame(), f"missing_signal_table:{path}"
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), "missing_price"
    signals = pd.read_csv(path)
    signals["EVENT_TIME"] = pd.to_datetime(signals["EVENT_TIME"], errors="coerce")
    signals = signals.dropna(subset=["EVENT_TIME"]).copy()
    signals["SYMBOL"] = signals["SYMBOL"].astype(str).str.upper()
    signals = signals[(signals["EVENT_TIME"] >= START) & (signals["EVENT_TIME"] <= END)]
    signals = signals[signals["SYMBOL"].eq(pair)].copy()
    if "MODEL" in signals.columns:
        signals = signals[signals["MODEL"].astype(str).eq(spec["model_filter"])].copy()
    if signals.empty:
        return pd.DataFrame(), "no_signal_rows"

    rows = []
    idx = price.index
    for _, row in signals.sort_values("EVENT_TIME").iterrows():
        signal_value = int(row["SIGNAL"])
        if signal_value == 0:
            continue
        entry_pos = idx.searchsorted(pd.Timestamp(row["EVENT_TIME"]), side="left")
        if entry_pos >= len(idx):
            continue
        holding = int(row.get("HORIZON_HOURS", spec.get("holding_bars") or DEFAULT_HOLDING_BARS))
        if not holding or holding < 1:
            holding = DEFAULT_HOLDING_BARS
        exit_pos = entry_pos + holding
        if exit_pos >= len(idx):
            continue
        entry_time = idx[entry_pos]
        exit_time = idx[exit_pos]
        entry_price = float(price["Close"].iloc[entry_pos])
        exit_price = float(price["Close"].iloc[exit_pos])
        side = "BUY" if signal_value > 0 else "SELL"
        raw_direction = 1.0 if side == "BUY" else -1.0
        pnl_pips = ((exit_price - entry_price) / pip_size(pair)) * raw_direction
        rows.append({
            "event_time": row["EVENT_TIME"],
            "time": entry_time,
            "exit_time": exit_time,
            "pair": pair,
            "CountryCode": row.get("COUNTRY", ""),
            "currency": "",
            "Name": row.get("EVENT_NAME", ""),
            "Actual": np.nan,
            "Forecast": np.nan,
            "Previous": np.nan,
            "Importance": np.nan,
            "Impact": np.nan,
            "orientation": "",
            "Close": entry_price,
            "close_fwd": exit_price,
            "prob_sell": np.nan,
            "prob_no_trade": np.nan,
            "prob_buy": np.nan,
            "raw_predicted_class": side,
            "confidence": row.get("PROBABILITY", np.nan),
            "threshold": np.nan,
            "signal_mode": "signal_table",
            "signal": side,
            "signal_reason": "SIGNAL_TABLE",
            "model": model_name,
            "runtime": spec["runtime"],
            "source_model": row.get("SOURCE_MODEL", spec["model_filter"]),
            "holding_bars": holding,
            "pnl_pips_precomputed": pnl_pips,
        })
    return pd.DataFrame(rows), "ok"


def max_drawdown_pips(values):
    equity = np.cumsum(pd.Series(values).fillna(0).to_numpy())
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return float(worst)


def summarize_orders(orders, model, pair, runtime, status, vectorbt_stats=None):
    if orders.empty:
        return {
            "model": model,
            "pair": pair,
            "runtime": runtime,
            "status": status,
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "avg_pips_per_trade": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
        }
    wins = int((orders["pnl_pips"] > 0).sum())
    gross_profit = float(orders.loc[orders["pnl_pips"] > 0, "pnl_pips"].sum())
    gross_loss = float(-orders.loc[orders["pnl_pips"] < 0, "pnl_pips"].sum())
    net_pips = float(orders["pnl_pips"].sum())
    out = {
        "model": model,
        "pair": pair,
        "runtime": runtime,
        "status": status,
        "trades": int(len(orders)),
        "buy_trades": int(orders["signal"].eq("BUY").sum()),
        "sell_trades": int(orders["signal"].eq("SELL").sum()),
        "win_rate": wins / len(orders),
        "net_pips": net_pips,
        "avg_pips_per_trade": net_pips / len(orders),
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
        "max_drawdown_pips": max_drawdown_pips(orders["pnl_pips"]),
    }
    if vectorbt_stats is not None:
        out.update(vectorbt_stats)
    return out


def scalar_stat(stats, key, default=np.nan):
    try:
        value = stats.loc[key]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return value
    except Exception:
        return default


def run_vectorbt(pair, predictions):
    price = read_price(pair)
    if price is None or predictions.empty:
        return {}, pd.DataFrame()
    idx = price.index
    long_entries = pd.Series(False, index=idx)
    long_exits = pd.Series(False, index=idx)
    short_entries = pd.Series(False, index=idx)
    short_exits = pd.Series(False, index=idx)

    orders = []
    for _, row in predictions[predictions["signal"].isin(["BUY", "SELL"])].sort_values("time").iterrows():
        entry_pos = idx.searchsorted(pd.Timestamp(row["time"]), side="left")
        if entry_pos >= len(idx):
            continue
        exit_pos = idx.searchsorted(pd.Timestamp(row["exit_time"]), side="left")
        if exit_pos >= len(idx) or exit_pos <= entry_pos:
            continue
        side = row["signal"]
        entry_price = float(price["Close"].iloc[entry_pos])
        exit_price = float(price["Close"].iloc[exit_pos])

        if "pnl_pips_precomputed" in row and pd.notna(row["pnl_pips_precomputed"]):
            pnl_pips = float(row["pnl_pips_precomputed"])
        else:
            domestic_direction = 1.0 if side == "BUY" else -1.0
            if row.get("orientation", "") == "base":
                raw_direction = domestic_direction
            elif row.get("orientation", "") == "quote":
                raw_direction = -domestic_direction
            else:
                raw_direction = 1.0 if side == "BUY" else -1.0
            pnl_pips = ((exit_price - entry_price) / pip_size(pair)) * raw_direction

        if side == "BUY":
            long_entries.iloc[entry_pos] = True
            long_exits.iloc[exit_pos] = True
        else:
            short_entries.iloc[entry_pos] = True
            short_exits.iloc[exit_pos] = True
        orders.append({
            "model": row["model"],
            "runtime": row["runtime"],
            "source_model": row.get("source_model", ""),
            "pair": pair,
            "event_time": row["event_time"],
            "entry_time": idx[entry_pos],
            "exit_time": idx[exit_pos],
            "country": row.get("CountryCode", ""),
            "event_name": row.get("Name", ""),
            "signal": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "confidence": row.get("confidence", np.nan),
            "holding_bars": row.get("holding_bars", DEFAULT_HOLDING_BARS),
            "pnl_pips": pnl_pips,
        })

    try:
        pf = vbt.Portfolio.from_signals(
            close=price["Close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=INITIAL_CASH,
            fees=0.0,
            slippage=0.0,
            freq=FREQ,
        )
        stats = pf.stats()
        vectorbt_stats = {
            "vectorbt_total_return_pct": scalar_stat(stats, "Total Return [%]"),
            "vectorbt_max_drawdown_pct": scalar_stat(stats, "Max Drawdown [%]"),
            "vectorbt_total_trades": scalar_stat(stats, "Total Trades"),
            "vectorbt_win_rate_pct": scalar_stat(stats, "Win Rate [%]"),
            "vectorbt_profit_factor": scalar_stat(stats, "Profit Factor"),
            "vectorbt_expectancy": scalar_stat(stats, "Expectancy"),
        }
    except Exception as exc:
        vectorbt_stats = {"vectorbt_error": str(exc)}
    return vectorbt_stats, pd.DataFrame(orders)


def build_all_model_specs():
    specs = [{"model": "catboost_v2_pair_direct", "runtime": "direct_pair_joblib"}]
    for model in GLOBAL_MODELS:
        specs.append({"model": model, "runtime": "direct_joblib"})
    specs.append({"model": "manual_best", "runtime": "signal_table"})
    specs.append({"model": "pair_specific_event_selection", "runtime": "signal_table"})
    return specs


def build_predictions(pair, model, metadata, thresholds):
    if model == "catboost_v2_pair_direct":
        return build_catboost_v2_predictions(pair)
    if model in GLOBAL_MODELS:
        return build_direct_global_predictions(pair, model, metadata, thresholds)
    if model in SIGNAL_TABLE_MODELS:
        return load_signal_table_predictions(pair, model, SIGNAL_TABLE_MODELS[model])
    return pd.DataFrame(), "unknown_model"


def format_float(value, digits=4):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_charts(monthly):
    if plt is None or monthly.empty:
        return []
    chart_dir = OUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, d in monthly.groupby("model"):
        pivot = d.pivot(index="month", columns="pair", values="cumulative_pips").sort_index()
        ax = pivot.plot(figsize=(12, 6), title=f"Cumulative OOS pips per pair - {model}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative pips")
        ax.grid(True, alpha=0.3)
        fig = ax.get_figure()
        path = chart_dir / f"{model}_cumulative_pips_by_pair.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(summary, by_model, by_pair, monthly, chart_paths):
    lines = [
        "# VectorBT OOS Model Comparison - Macro News FX",
        "",
        f"Periode OOS: `{START.date()}` sampai `{END.date()}`.",
        f"Data test: `{DATATEST}`",
        "",
        "## Metode",
        "",
        "- `catboost_v2_pair_direct` diuji langsung dari model pair-specific `.joblib` di folder `catboost_v2_pair_specific_loop`.",
        "- `elastic_net`, `random_forest`, `svm_rbf`, dan `ebm` diuji langsung dari model `.joblib`.",
        "- `manual_best` memakai signal table manual terbaik `manual_block_inverse_validation_gated_conservative`.",
        "- `pair_specific_event_selection` memakai signal table Event Study V3, yaitu seleksi model terbaik per pair.",
        "- Metric `net_pips` dihitung dari audit event entry/exit. Metric `vectorbt_*` berasal dari simulasi portfolio vectorbt dan bisa berbeda karena handling overlap posisi.",
        "",
        "## Ranking Model Agregat",
        "",
    ]
    cols = ["model", "runtime", "pairs", "positive_pairs", "trades", "win_rate", "net_pips", "avg_pips_per_trade", "profit_factor", "max_drawdown_pips"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in by_model.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if c in {"win_rate", "avg_pips_per_trade", "profit_factor", "max_drawdown_pips"}:
                vals.append(format_float(v, 4))
            elif c == "net_pips":
                vals.append(format_float(v, 1))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += ["", "## Ranking Pair Per Model", ""]
    pair_cols = ["model", "pair", "runtime", "trades", "win_rate", "net_pips", "profit_factor", "max_drawdown_pips", "status"]
    view = summary.sort_values(["model", "net_pips"], ascending=[True, False])
    lines.append("| " + " | ".join(pair_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(pair_cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for c in pair_cols:
            v = row.get(c, "")
            if c in {"win_rate", "profit_factor", "max_drawdown_pips"}:
                vals.append(format_float(v, 4))
            elif c == "net_pips":
                vals.append(format_float(v, 1))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += ["", "## Ranking Pair Agregat", ""]
    pair_agg_cols = ["pair", "models_tested", "positive_models", "total_trades", "total_net_pips", "best_model", "best_model_net_pips"]
    lines.append("| " + " | ".join(pair_agg_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(pair_agg_cols)) + " |")
    for _, row in by_pair.iterrows():
        vals = []
        for c in pair_agg_cols:
            v = row.get(c, "")
            if c in {"total_net_pips", "best_model_net_pips"}:
                vals.append(format_float(v, 1))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    if chart_paths:
        lines += ["", "## Grafik", ""]
        for path in chart_paths:
            lines.append(f"- `{path}`")

    lines += [
        "",
        "## Catatan Interpretasi",
        "",
        "Hasil ini adalah OOS screening dari Python/vectorbt. Untuk klaim deployment, hasil tetap perlu dikonfirmasi dengan Strategy Tester MT5.",
        "Data OOS tidak dipakai untuk training ulang, threshold tuning, atau seleksi model dalam script ini.",
    ]
    (OUT_DIR / "LAPORAN_VECTORBT_OOS_MODEL_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[PROGRESS] START 34_vectorbt_oos_all_models_direct.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    thresholds = load_thresholds()
    pairs = available_pairs()
    specs = build_all_model_specs()
    print(f"[DATA] pairs={pairs}", flush=True)
    print(f"[DATA] models={[s['model'] for s in specs]}", flush=True)

    summary_rows = []
    prediction_rows = []
    order_rows = []

    for model_idx, spec in enumerate(specs, start=1):
        model = spec["model"]
        print("=" * 72, flush=True)
        print(f"[MODEL] START {model_idx}/{len(specs)} {model}", flush=True)
        for pair_idx, pair in enumerate(pairs, start=1):
            print(f"[PAIR] START model={model} pair={pair_idx}/{len(pairs)} {pair}", flush=True)
            try:
                predictions, status = build_predictions(pair, model, metadata, thresholds)
                runtime = spec["runtime"]
                if predictions.empty:
                    summary = summarize_orders(pd.DataFrame(), model, pair, runtime, status)
                else:
                    pred_path = OUT_DIR / "predictions" / model / f"{pair}_predictions.csv"
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(pred_path, index=False)
                    prediction_rows.append(predictions)
                    vectorbt_stats, orders = run_vectorbt(pair, predictions)
                    if not orders.empty:
                        order_rows.append(orders)
                        order_path = OUT_DIR / "orders" / model / f"{pair}_orders.csv"
                        order_path.parent.mkdir(parents=True, exist_ok=True)
                        orders.to_csv(order_path, index=False)
                    summary = summarize_orders(orders, model, pair, runtime, status, vectorbt_stats)
            except Exception as exc:
                summary = summarize_orders(pd.DataFrame(), model, pair, spec["runtime"], f"failed:{exc}")
                print(f"[PAIR] FAILED model={model} pair={pair} error={exc}", flush=True)
            summary_rows.append(summary)
            print(
                f"[PAIR] DONE model={model} pair={pair} status={summary['status']} "
                f"trades={summary['trades']} net_pips={summary['net_pips']:.1f} winrate={summary['win_rate']:.4f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "vectorbt_oos_summary_by_model_pair.csv", index=False)
    predictions_all = pd.concat(prediction_rows, ignore_index=True, sort=False) if prediction_rows else pd.DataFrame()
    orders_all = pd.concat(order_rows, ignore_index=True, sort=False) if order_rows else pd.DataFrame()
    predictions_all.to_csv(OUT_DIR / "vectorbt_oos_predictions_all.csv", index=False)
    orders_all.to_csv(OUT_DIR / "vectorbt_oos_orders_all.csv", index=False)

    if not orders_all.empty:
        monthly = orders_all.copy()
        monthly["month"] = pd.to_datetime(monthly["exit_time"]).dt.to_period("M").astype(str)
        monthly = (
            monthly.groupby(["model", "pair", "month"], as_index=False)["pnl_pips"]
            .sum()
            .sort_values(["model", "pair", "month"])
        )
        monthly["cumulative_pips"] = monthly.groupby(["model", "pair"])["pnl_pips"].cumsum()
    else:
        monthly = pd.DataFrame(columns=["model", "pair", "month", "pnl_pips", "cumulative_pips"])
    monthly.to_csv(OUT_DIR / "vectorbt_oos_monthly_cumulative.csv", index=False)

    def aggregate_model(group):
        trades = int(group["trades"].sum())
        gross_profit = float(group["gross_profit_pips"].sum())
        gross_loss = float(group["gross_loss_pips"].sum())
        return pd.Series({
            "runtime": ",".join(sorted(group["runtime"].dropna().astype(str).unique())),
            "pairs": int(group["pair"].nunique()),
            "positive_pairs": int((group["net_pips"] > 0).sum()),
            "trades": trades,
            "win_rate": float((group["win_rate"] * group["trades"]).sum() / trades) if trades else 0.0,
            "net_pips": float(group["net_pips"].sum()),
            "avg_pips_per_trade": float(group["net_pips"].sum() / trades) if trades else 0.0,
            "gross_profit_pips": gross_profit,
            "gross_loss_pips": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
            "max_drawdown_pips": float(group["max_drawdown_pips"].min()) if len(group) else 0.0,
        })

    by_model = summary.groupby("model", as_index=False).apply(aggregate_model, include_groups=False)
    if isinstance(by_model.index, pd.MultiIndex):
        by_model = by_model.reset_index(drop=True)
    by_model = by_model.sort_values("net_pips", ascending=False)
    by_model.to_csv(OUT_DIR / "vectorbt_oos_summary_by_model.csv", index=False)

    best_by_pair = summary.sort_values("net_pips", ascending=False).drop_duplicates("pair")
    by_pair = summary.groupby("pair", as_index=False).agg(
        models_tested=("model", "nunique"),
        positive_models=("net_pips", lambda s: int((s > 0).sum())),
        total_trades=("trades", "sum"),
        total_net_pips=("net_pips", "sum"),
    )
    by_pair = by_pair.merge(
        best_by_pair[["pair", "model", "net_pips"]].rename(columns={"model": "best_model", "net_pips": "best_model_net_pips"}),
        on="pair",
        how="left",
    ).sort_values("total_net_pips", ascending=False)
    by_pair.to_csv(OUT_DIR / "vectorbt_oos_summary_by_pair.csv", index=False)

    chart_paths = write_charts(monthly)
    write_report(summary, by_model, by_pair, monthly, chart_paths)

    metadata_out = {
        "start": str(START),
        "end": str(END),
        "pairs": pairs,
        "models": [s["model"] for s in specs],
        "out_dir": str(OUT_DIR),
        "note": "Direct model inference for ML models where possible; signal tables for manual/event-selection methods.",
    }
    (OUT_DIR / "vectorbt_oos_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'vectorbt_oos_summary_by_model_pair.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'vectorbt_oos_summary_by_model.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_VECTORBT_OOS_MODEL_COMPARISON.md'}", flush=True)


if __name__ == "__main__":
    main()
