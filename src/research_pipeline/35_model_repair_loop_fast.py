from common import *
from pathlib import Path
import json
import math
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


OUT_DIR = ARTIFACTS / "model_repair_loop_fast"
MODEL_DIR = OUT_DIR / "joblib"
DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"

TRAIN_START_YEAR = 2010
TRAIN_END_YEAR = 2020
VALID_START_YEAR = 2021
VALID_END_YEAR = 2023
OOS_START = pd.Timestamp("2024-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")
MIN_VALID_TRADES = 30
MIN_OOS_TRADES_FOR_DEPLOY = 30
RANDOM_STATE = 42
MAX_TRAIN_ROWS_PER_PAIR = int(os.environ.get("REPAIR_MAX_TRAIN_ROWS_PER_PAIR", "12000"))

BASE_FEATURES = model_feature_names()
EXTRA_FEATURES = [
    "surprise_abs_fc",
    "surprise_abs_prev",
    "surprise_sign_fc",
    "surprise_sign_prev",
    "has_actual",
    "has_forecast",
    "has_previous",
    "event_minute",
    "event_weekday",
]
FEATURE_SETS = {
    "base42": BASE_FEATURES,
    "event_plus": BASE_FEATURES + EXTRA_FEATURES,
}


def safe_profit_factor(values):
    s = pd.Series(values).dropna()
    gross_profit = float(s[s > 0].sum())
    gross_loss = float(-s[s < 0].sum())
    if gross_loss <= 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


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


def add_extra_features(df):
    out = df.copy()
    out["surprise_abs_fc"] = pd.to_numeric(out.get("surprise_fc"), errors="coerce").abs()
    out["surprise_abs_prev"] = pd.to_numeric(out.get("surprise_prev"), errors="coerce").abs()
    out["surprise_sign_fc"] = np.sign(pd.to_numeric(out.get("surprise_fc"), errors="coerce").fillna(0)).astype("int8")
    out["surprise_sign_prev"] = np.sign(pd.to_numeric(out.get("surprise_prev"), errors="coerce").fillna(0)).astype("int8")
    out["has_actual"] = pd.to_numeric(out.get("Actual"), errors="coerce").notna().astype("int8")
    out["has_forecast"] = pd.to_numeric(out.get("Forecast"), errors="coerce").notna().astype("int8")
    out["has_previous"] = pd.to_numeric(out.get("Previous"), errors="coerce").notna().astype("int8")
    if "event_time" in out.columns:
        t = pd.to_datetime(out["event_time"], errors="coerce")
    else:
        t = pd.to_datetime(out["Datetime"], errors="coerce")
    out["event_minute"] = t.dt.minute.fillna(0).astype("int16")
    out["event_weekday"] = t.dt.weekday.fillna(0).astype("int16")
    return out


def load_modeling_frame():
    path = ROOT / "modeling_dataset_multi_currency.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling dataset: {path}")
    usecols = None
    d = pd.read_csv(path, low_memory=False, usecols=usecols)
    d = add_extra_features(d)
    d = d.dropna(subset=["event_time", "pair", "trade_label_1h", "domestic_return_pips_1h", "event_year"]).copy()
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce").astype("Int64")
    d["trade_label_1h"] = pd.to_numeric(d["trade_label_1h"], errors="coerce").fillna(0).astype(int)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce")
    d = d.dropna(subset=["event_time", "event_year", "domestic_return_pips_1h"]).copy()
    for col in set(sum(FEATURE_SETS.values(), [])):
        if col not in d.columns:
            d[col] = 0
    d[list(set(sum(FEATURE_SETS.values(), [])))] = d[list(set(sum(FEATURE_SETS.values(), [])))].replace([np.inf, -np.inf], np.nan).fillna(0)
    return d


def read_test_price(pair, holding_bars):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        return None
    px = read_prices(path).set_index("time")
    px = px[(px.index >= OOS_START) & (px.index <= OOS_END)].copy()
    if px.empty:
        return None
    px["close_fwd"] = px["Close"].shift(-holding_bars)
    px["exit_time"] = px.index.to_series().shift(-holding_bars)
    return px.reset_index()


def prepare_oos_events(pair, holding_bars, min_importance, min_impact):
    mapping = available_currency_pair_mapping()
    pair_map = mapping[mapping["pair"].eq(pair)].copy()
    if pair_map.empty:
        return pd.DataFrame()
    px = read_test_price(pair, holding_bars)
    if px is None:
        return pd.DataFrame()
    news = read_news(NEWS_TEST)
    news = news[(news["event_time"] >= OOS_START) & (news["event_time"] <= OOS_END)].copy()
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
    news = add_context_features(add_extra_features(news))
    news = news[
        pd.to_numeric(news["Importance"], errors="coerce").fillna(0).ge(min_importance) &
        pd.to_numeric(news["Impact"], errors="coerce").fillna(0).ge(min_impact)
    ].copy()
    if news.empty:
        return pd.DataFrame()
    joined = pd.merge_asof(
        news.sort_values("event_time"),
        px.sort_values("time"),
        left_on="event_time",
        right_on="time",
        direction="forward",
        tolerance=pd.Timedelta(hours=2),
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


def make_model(model_name, params):
    if model_name == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                class_weight=params["class_weight"],
                max_iter=1500,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),
        )
    if model_name == "logistic_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=params["C"],
                class_weight=params["class_weight"],
                max_iter=2500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "svm_rbf_fast":
        return make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=params["gamma"], n_components=params["n_components"], random_state=RANDOM_STATE),
            LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=2500, random_state=RANDOM_STATE, n_jobs=-1),
        )
    if model_name == "hist_gradient":
        return HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            random_state=RANDOM_STATE,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def probability_frame(model, X):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
        out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
        for idx, cls in enumerate(classes):
            cls = int(cls)
            if cls == -1:
                out["prob_sell"] = proba[:, idx]
            elif cls == 0:
                out["prob_no_trade"] = proba[:, idx]
            elif cls == 1:
                out["prob_buy"] = proba[:, idx]
        return out
    pred = pd.Series(np.asarray(model.predict(X)).reshape(-1), index=X.index)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    out.loc[pred.eq(-1), "prob_sell"] = 1.0
    out.loc[pred.eq(0), "prob_no_trade"] = 1.0
    out.loc[pred.eq(1), "prob_buy"] = 1.0
    return out


def apply_signal(probs, mode, threshold):
    signals = []
    raw = []
    confidence = []
    for _, row in probs.iterrows():
        if mode == "directional":
            pred = "BUY" if row.prob_buy >= row.prob_sell else "SELL"
            conf = float(max(row.prob_buy, row.prob_sell))
        else:
            values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
            pred = max(values, key=values.get)
            conf = float(values[pred])
        raw.append(pred)
        confidence.append(conf)
        if pred == "NO_TRADE" or conf < threshold:
            signals.append("NO_TRADE")
        else:
            signals.append(pred)
    return raw, confidence, signals


def evaluate_events(events, model_name, pair, probs, mode, threshold):
    d = events.reset_index(drop=True).copy()
    if "time" not in d.columns:
        if "Time_y" in d.columns:
            d["time"] = pd.to_datetime(d["Time_y"], errors="coerce")
        else:
            d["time"] = pd.to_datetime(d["event_time"], errors="coerce")
    if "exit_time" not in d.columns:
        d["exit_time"] = pd.to_datetime(d["time"], errors="coerce") + pd.Timedelta(hours=1)
    if "close_fwd" not in d.columns:
        if "close_fwd_1h" in d.columns:
            d["close_fwd"] = d["close_fwd_1h"]
        else:
            d["close_fwd"] = np.nan
    raw, confidence, signals = apply_signal(probs.reset_index(drop=True), mode, threshold)
    d[["prob_sell", "prob_no_trade", "prob_buy"]] = probs.reset_index(drop=True)
    d["raw_predicted_class"] = raw
    d["confidence"] = confidence
    d["signal"] = signals
    d["model"] = model_name
    d["pair"] = pair
    trades = d[d["signal"].isin(["BUY", "SELL"])].copy()
    if trades.empty:
        return summary_from_orders(pd.DataFrame(), model_name, pair, "no_trades"), d, pd.DataFrame()
    domestic_direction = np.where(trades["signal"].eq("BUY"), 1.0, -1.0)
    trades["pnl_pips"] = trades["domestic_return_pips_1h"] * domestic_direction
    trades["entry_time"] = trades["time"]
    trades["exit_time"] = trades["exit_time"]
    trades["event_name"] = trades["Name"]
    trades["country"] = trades["CountryCode"]
    keep = [
        "event_time", "entry_time", "exit_time", "pair", "model", "country", "event_name",
        "signal", "Close", "close_fwd", "confidence", "pnl_pips",
    ]
    return summary_from_orders(trades[keep], model_name, pair, "ok"), d, trades[keep]


def summary_from_orders(orders, model_name, pair, status):
    if orders.empty:
        return {
            "model": model_name,
            "pair": pair,
            "status": status,
            "trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "avg_pips_per_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
            "buy_trades": 0,
            "sell_trades": 0,
        }
    gross_profit = float(orders.loc[orders["pnl_pips"] > 0, "pnl_pips"].sum())
    gross_loss = float(-orders.loc[orders["pnl_pips"] < 0, "pnl_pips"].sum())
    net_pips = float(orders["pnl_pips"].sum())
    return {
        "model": model_name,
        "pair": pair,
        "status": status,
        "trades": int(len(orders)),
        "win_rate": float((orders["pnl_pips"] > 0).mean()),
        "net_pips": net_pips,
        "avg_pips_per_trade": net_pips / len(orders),
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
        "max_drawdown_pips": max_drawdown_pips(orders["pnl_pips"]),
        "buy_trades": int(orders["signal"].eq("BUY").sum()),
        "sell_trades": int(orders["signal"].eq("SELL").sum()),
    }


def candidate_configs():
    common = []
    for feature_set, importance_min, impact_min in [
        ("base42", 1, 0),
        ("event_plus", 1, 0),
        ("event_plus", 2, 1),
        ("event_plus", 2, 2),
    ]:
        common.append({
            "horizon": 1,
            "feature_set": feature_set,
            "importance_min": importance_min,
            "impact_min": impact_min,
        })
    configs = []
    for base in common:
        for params in [
            {"alpha": 0.0005, "l1_ratio": 0.35, "class_weight": "balanced"},
            {"alpha": 0.0010, "l1_ratio": 0.50, "class_weight": "balanced"},
        ]:
            configs.append({"model_family": "elastic_net", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
        for params in [
            {"C": 1.0, "class_weight": "balanced"},
        ]:
            configs.append({"model_family": "logistic_l2", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
        for params in [
            {"n_estimators": 220, "max_depth": 8, "min_samples_leaf": 12, "class_weight": "balanced_subsample"},
            {"n_estimators": 320, "max_depth": None, "min_samples_leaf": 25, "class_weight": "balanced_subsample"},
        ]:
            configs.append({"model_family": "random_forest", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
        for params in [
            {"gamma": 0.05, "n_components": 350, "C": 1.0, "class_weight": "balanced"},
        ]:
            configs.append({"model_family": "svm_rbf_fast", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
        for params in [
            {"max_iter": 400, "learning_rate": 0.035, "max_leaf_nodes": 31, "l2_regularization": 0.3},
        ]:
            configs.append({"model_family": "hist_gradient", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
        if LGBMClassifier is not None:
            for params in [
                {"n_estimators": 450, "num_leaves": 31, "learning_rate": 0.035, "subsample": 0.85, "colsample_bytree": 0.85, "class_weight": "balanced"},
            ]:
                configs.append({"model_family": "lightgbm", "signal_modes": ["strict", "directional"], "thresholds": [0.50, 0.60], **base, "params": params})
    return configs


def score_summary(summary):
    trades = int(summary["trades"])
    if trades < MIN_VALID_TRADES:
        return -1e9 + trades
    pf = float(summary["profit_factor"])
    pf_term = 0.0 if not np.isfinite(pf) else pf
    low_trade_penalty = max(0, MIN_VALID_TRADES * 2 - trades) * 5.0
    return (
        float(summary["net_pips"]) +
        1000.0 * pf_term +
        300.0 * float(summary["win_rate"]) -
        0.5 * abs(float(summary["max_drawdown_pips"])) -
        low_trade_penalty
    )


def run_trial(train_pair, valid_pair, pair, cfg, trial_index):
    train = train_pair.copy()
    valid = valid_pair.copy()
    train = train[
        pd.to_numeric(train["Importance"], errors="coerce").fillna(0).ge(cfg["importance_min"]) &
        pd.to_numeric(train["Impact"], errors="coerce").fillna(0).ge(cfg["impact_min"])
    ].copy()
    valid = valid[
        pd.to_numeric(valid["Importance"], errors="coerce").fillna(0).ge(cfg["importance_min"]) &
        pd.to_numeric(valid["Impact"], errors="coerce").fillna(0).ge(cfg["impact_min"])
    ].copy()
    label_col = f"trade_label_{cfg['horizon']}h" if f"trade_label_{cfg['horizon']}h" in train.columns else "trade_label_1h"
    return_col = f"domestic_return_pips_{cfg['horizon']}h" if f"domestic_return_pips_{cfg['horizon']}h" in train.columns else "domestic_return_pips_1h"
    if cfg["horizon"] != 1 and return_col not in train.columns:
        # Build pips from log return if pips column for horizon is unavailable.
        return_col = "domestic_return_pips_1h"
    if train.empty or valid.empty or train[label_col].nunique() < 2:
        return None
    if MAX_TRAIN_ROWS_PER_PAIR > 0 and len(train) > MAX_TRAIN_ROWS_PER_PAIR:
        train = train.sample(MAX_TRAIN_ROWS_PER_PAIR, random_state=RANDOM_STATE + trial_index).sort_values("event_time")
    features = FEATURE_SETS[cfg["feature_set"]]
    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[label_col].astype(int)
    X_valid = valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = make_model(cfg["model_family"], cfg["params"])
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    probs = probability_frame(model, X_valid)
    best = None
    for mode in cfg["signal_modes"]:
        for threshold in cfg["thresholds"]:
            summary, pred, orders = evaluate_events(valid.assign(domestic_return_pips_1h=valid[return_col]), cfg["model_family"], pair, probs, mode, threshold)
            row = {
                **summary,
                "trial": trial_index,
                "model_family": cfg["model_family"],
                "horizon": cfg["horizon"],
                "feature_set": cfg["feature_set"],
                "importance_min": cfg["importance_min"],
                "impact_min": cfg["impact_min"],
                "signal_mode": mode,
                "threshold": threshold,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "fit_seconds": round(fit_seconds, 2),
                "params_json": json.dumps(cfg["params"], sort_keys=True),
            }
            row["score"] = score_summary(row)
            if best is None or row["score"] > best["score"]:
                best = row
    return best, model, cfg


def run_oos(pair, model, cfg, selected_row):
    events = prepare_oos_events(pair, cfg["horizon"], cfg["importance_min"], cfg["impact_min"])
    if events.empty:
        return summary_from_orders(pd.DataFrame(), selected_row["selected_model_name"], pair, "no_oos_events"), pd.DataFrame(), pd.DataFrame()
    features = FEATURE_SETS[cfg["feature_set"]]
    for col in features:
        if col not in events.columns:
            events[col] = 0
    X = events[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = probability_frame(model, X)
    summary, pred, orders = evaluate_events(events, selected_row["selected_model_name"], pair, probs, selected_row["signal_mode"], float(selected_row["threshold"]))
    summary.update({
        "selected_from_validation_score": selected_row["score"],
        "horizon": cfg["horizon"],
        "feature_set": cfg["feature_set"],
        "importance_min": cfg["importance_min"],
        "impact_min": cfg["impact_min"],
        "signal_mode": selected_row["signal_mode"],
        "threshold": selected_row["threshold"],
    })
    return summary, pred, orders


def write_report(best, oos):
    best_rank = best.sort_values(["model_family", "score"], ascending=[True, False])
    oos_rank = oos.sort_values("net_pips", ascending=False)
    by_model = oos.groupby("model_family", as_index=False).agg(
        pairs=("pair", "nunique"),
        positive_pairs=("net_pips", lambda s: int((s > 0).sum())),
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        avg_win_rate=("win_rate", "mean"),
        avg_profit_factor=("profit_factor", "mean"),
    ).sort_values("net_pips", ascending=False)
    lines = [
        "# Model Repair Loop Fast - Macro News FX",
        "",
        "Seleksi model dilakukan memakai training `2010-2020` dan validation `2021-2023`.",
        "OOS `2024-08.2026` hanya dipakai setelah kandidat terbaik per pair dipilih dari validation.",
        "",
        "## OOS Ranking By Model Family",
        "",
    ]
    cols = ["model_family", "pairs", "positive_pairs", "trades", "net_pips", "avg_win_rate", "avg_profit_factor"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in by_model.iterrows():
        lines.append("| " + " | ".join([
            str(row[c]) if c not in {"net_pips", "avg_win_rate", "avg_profit_factor"} else f"{float(row[c]):.4f}"
            for c in cols
        ]) + " |")
    lines += ["", "## OOS Best Per Pair", ""]
    cols = ["pair", "model_family", "selected_model_name", "trades", "win_rate", "net_pips", "profit_factor", "horizon", "feature_set", "signal_mode", "threshold"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in oos_rank.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if c in {"win_rate", "net_pips", "profit_factor", "threshold"}:
                vals.append(f"{float(v):.4f}" if pd.notna(v) and np.isfinite(float(v)) else str(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## Validation Selected Configs", ""]
    cols = ["pair", "model_family", "selected_model_name", "trades", "win_rate", "net_pips", "profit_factor", "score", "horizon", "feature_set", "importance_min", "impact_min", "signal_mode", "threshold"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in best_rank.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if c in {"win_rate", "net_pips", "profit_factor", "score", "threshold"}:
                vals.append(f"{float(v):.4f}" if pd.notna(v) and np.isfinite(float(v)) else str(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Catatan",
        "",
        "- EBM tidak dimasukkan ke loop cepat karena waktu training panjang; gunakan batch EBM terpisah.",
        "- Kandidat dipilih dari validation, bukan dari OOS.",
        "- Jika hasil OOS bagus, tahap berikutnya adalah export signal table untuk MT5 dan Strategy Tester.",
    ]
    (OUT_DIR / "LAPORAN_MODEL_REPAIR_LOOP_FAST.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[PROGRESS] START 35_model_repair_loop_fast.py", flush=True)
    if OUT_DIR.exists():
        backup = OUT_DIR.parent / f"{OUT_DIR.name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(OUT_DIR), str(backup))
        print(f"[BACKUP] previous output moved to {backup}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_modeling_frame()
    pairs = available_pairs()
    configs = candidate_configs()
    print(f"[DATA] rows={len(data)} pairs={pairs} configs={len(configs)} lightgbm={'yes' if LGBMClassifier is not None else 'no'}", flush=True)

    trial_rows = []
    best_rows = []
    oos_rows = []
    pred_frames = []
    order_frames = []

    for pair_idx, pair in enumerate(pairs, start=1):
        pair_data = data[data["pair"].astype(str).str.upper().eq(pair)].copy()
        train_pair = pair_data[(pair_data["event_year"] >= TRAIN_START_YEAR) & (pair_data["event_year"] <= TRAIN_END_YEAR)].copy()
        valid_pair = pair_data[(pair_data["event_year"] >= VALID_START_YEAR) & (pair_data["event_year"] <= VALID_END_YEAR)].copy()
        print("=" * 80, flush=True)
        print(f"[PAIR] START {pair_idx}/{len(pairs)} {pair} train={len(train_pair)} valid={len(valid_pair)}", flush=True)
        best_by_family = {}
        for trial_idx, cfg in enumerate(configs, start=1):
            print(
                f"[TRIAL] START pair={pair} {trial_idx}/{len(configs)} "
                f"model={cfg.get('model_family')} features={cfg.get('feature_set')} "
                f"imp={cfg.get('importance_min')} impact={cfg.get('impact_min')}",
                flush=True,
            )
            try:
                result = run_trial(train_pair, valid_pair, pair, cfg, trial_idx)
                if result is None:
                    print(f"[TRIAL] SKIP pair={pair} trial={trial_idx} reason=empty_or_single_class", flush=True)
                    continue
                row, model, used_cfg = result
                trial_rows.append(row)
                family = row["model_family"]
                if family not in best_by_family or row["score"] > best_by_family[family]["row"]["score"]:
                    best_by_family[family] = {"row": row, "model": model, "cfg": used_cfg}
                print(
                    f"[TRIAL] DONE pair={pair} trial={trial_idx} model={family} "
                    f"valid_trades={row.get('trades', 0)} net_pips={float(row.get('net_pips', 0.0)):.1f} "
                    f"pf={float(row.get('profit_factor', 0.0)):.4f} score={float(row.get('score', 0.0)):.1f}",
                    flush=True,
                )
            except Exception as exc:
                trial_rows.append({
                    "pair": pair,
                    "trial": trial_idx,
                    "model_family": cfg.get("model_family"),
                    "status": f"failed:{exc}",
                    "score": -1e9,
                })
                print(f"[TRIAL] FAILED pair={pair} trial={trial_idx} model={cfg.get('model_family')} error={exc}", flush=True)
            if trial_idx % 10 == 0:
                pd.DataFrame(trial_rows).to_csv(OUT_DIR / "repair_loop_all_trials.csv", index=False)

        for family, pack in best_by_family.items():
            row = dict(pack["row"])
            selected_name = f"{family}_repair_{pair}"
            row["selected_model_name"] = selected_name
            best_rows.append(row)
            model_path = MODEL_DIR / f"{selected_name}.joblib"
            joblib.dump(pack["model"], model_path)
            (MODEL_DIR / f"{selected_name}_config.json").write_text(json.dumps(pack["cfg"], indent=2, default=str), encoding="utf-8")
            print(
                f"[SELECT] pair={pair} family={family} valid_trades={row.get('trades', 0)} "
                f"valid_net_pips={float(row.get('net_pips', 0.0)):.1f} valid_pf={float(row.get('profit_factor', 0.0)):.4f}",
                flush=True,
            )
            try:
                oos_summary, pred, orders = run_oos(pair, pack["model"], pack["cfg"], row)
                oos_summary["model_family"] = family
                oos_summary["selected_model_name"] = selected_name
                oos_rows.append(oos_summary)
                if not pred.empty:
                    pred["model_family"] = family
                    pred_frames.append(pred)
                if not orders.empty:
                    orders["model_family"] = family
                    order_frames.append(orders)
                print(
                    f"[OOS] pair={pair} family={family} trades={oos_summary.get('trades', 0)} "
                    f"net_pips={float(oos_summary.get('net_pips', 0.0)):.1f} pf={float(oos_summary.get('profit_factor', 0.0)):.4f}",
                    flush=True,
                )
            except Exception as exc:
                oos_rows.append({"pair": pair, "model_family": family, "selected_model_name": selected_name, "status": f"failed:{exc}"})
                print(f"[OOS] FAILED pair={pair} family={family} error={exc}", flush=True)

        pd.DataFrame(trial_rows).to_csv(OUT_DIR / "repair_loop_all_trials.csv", index=False)
        pd.DataFrame(best_rows).to_csv(OUT_DIR / "repair_loop_best_by_pair_model.csv", index=False)
        pd.DataFrame(oos_rows).to_csv(OUT_DIR / "repair_loop_oos_results.csv", index=False)

    trials = pd.DataFrame(trial_rows)
    best = pd.DataFrame(best_rows)
    oos = pd.DataFrame(oos_rows)
    trials.to_csv(OUT_DIR / "repair_loop_all_trials.csv", index=False)
    best.to_csv(OUT_DIR / "repair_loop_best_by_pair_model.csv", index=False)
    oos.to_csv(OUT_DIR / "repair_loop_oos_results.csv", index=False)
    (pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()).to_csv(OUT_DIR / "repair_loop_oos_predictions.csv", index=False)
    (pd.concat(order_frames, ignore_index=True, sort=False) if order_frames else pd.DataFrame()).to_csv(OUT_DIR / "repair_loop_oos_orders.csv", index=False)
    write_report(best, oos)
    print(f"[OUTPUT] {OUT_DIR / 'repair_loop_oos_results.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_MODEL_REPAIR_LOOP_FAST.md'}", flush=True)


if __name__ == "__main__":
    main()
