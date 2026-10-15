from common import *
from pathlib import Path
import json
import math
import os
import shutil
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


class XGBEventClassifier:
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.classes_ = np.array([-1, 0, 1])

    def fit(self, X, y):
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        y_fit = pd.Series(y).map({-1: 0, 0: 1, 1: 2}).astype(int)
        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y_fit)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("XGBEventClassifier is not fitted")
        return self.model.predict_proba(X)


OUT_DIR = ARTIFACTS / "event_study_v3"
DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"
PAIRS = [p.strip().upper() for p in os.environ.get("EVENT_STUDY_V3_PAIRS", ",".join(available_pairs())).split(",") if p.strip()]
HORIZONS = [int(x) for x in os.environ.get("EVENT_STUDY_V3_HORIZONS", "1,2").split(",") if x.strip()]
MIN_VALID_TRADES = int(os.environ.get("EVENT_STUDY_V3_MIN_VALID_TRADES", "25"))
MAX_TRAIN_ROWS_PER_PAIR = int(os.environ.get("EVENT_STUDY_V3_MAX_TRAIN_ROWS_PER_PAIR", "0"))

BASE_FEATURES = model_feature_names()
EXTRA_FEATURES = [
    "surprise_abs_fc",
    "surprise_abs_prev",
    "surprise_sign_fc",
    "surprise_sign_prev",
    "has_actual",
    "has_forecast",
    "has_previous",
    "pre_return_1h",
    "pre_return_2h",
    "pre_range_1h",
    "pre_body_1h",
    "pre_volume_log",
    "event_minute",
    "event_weekday",
]
FEATURES = BASE_FEATURES + EXTRA_FEATURES


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


def build_pair_events(pair, data_dir, news_path, start, end, horizon):
    mapping = available_currency_pair_mapping()
    pair_map = mapping[mapping["pair"].eq(pair)].copy()
    if pair_map.empty:
        return pd.DataFrame()

    price_path = Path(data_dir) / pair_file(pair)
    if not price_path.exists():
        return pd.DataFrame()

    px = read_prices(price_path).set_index("time")
    px["close_fwd"] = px["Close"].shift(-horizon)
    px["exit_time"] = px.index.to_series().shift(-horizon)
    px["prev_close_1h"] = px["Close"].shift(1)
    px["prev_close_2h"] = px["Close"].shift(2)
    px["prev_high_1h"] = px["High"].shift(1)
    px["prev_low_1h"] = px["Low"].shift(1)
    px["prev_open_1h"] = px["Open"].shift(1)
    px["prev_volume_1h"] = px["Volume"].shift(1)
    px = px.reset_index()

    news = read_news(news_path)
    news = news[(news["event_time"] >= pd.Timestamp(start)) & (news["event_time"] <= pd.Timestamp(end))]
    news = news[news["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
    if news.empty:
        return pd.DataFrame()

    news["pair"] = pair
    news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
    news = news[news["orientation"].isin(["base", "quote"])].copy()
    news["category"] = news["Name"].map(event_category)
    news["release_hour"] = news["event_time"].dt.hour
    news["event_minute"] = news["event_time"].dt.minute
    news["event_weekday"] = news["event_time"].dt.weekday
    news["has_actual"] = news["Actual"].notna().astype("int8")
    news["has_forecast"] = news["Forecast"].notna().astype("int8")
    news["has_previous"] = news["Previous"].notna().astype("int8")
    news["surprise_fc"] = news["Actual"] - news["Forecast"]
    news["surprise_prev"] = news["Actual"] - news["Previous"]
    news["surprise_pct_fc"] = news["surprise_fc"] / (news["Forecast"].abs() + 1e-9)
    news["surprise_abs_fc"] = news["surprise_fc"].abs()
    news["surprise_abs_prev"] = news["surprise_prev"].abs()
    news["surprise_sign_fc"] = np.sign(news["surprise_fc"]).fillna(0)
    news["surprise_sign_prev"] = np.sign(news["surprise_prev"]).fillna(0)
    news = add_context_features(news)

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

    sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
    joined["pip_size"] = joined["pair"].map(pip_size)
    joined["domestic_return_pips"] = ((joined["close_fwd"] - joined["Close"]) / joined["pip_size"]) * sign
    joined["domestic_return"] = np.log(joined["close_fwd"] / joined["Close"]) * sign
    joined["pre_return_1h"] = np.log(joined["Close"] / joined["prev_close_1h"]).replace([np.inf, -np.inf], np.nan)
    joined["pre_return_2h"] = np.log(joined["Close"] / joined["prev_close_2h"]).replace([np.inf, -np.inf], np.nan)
    joined["pre_range_1h"] = (joined["prev_high_1h"] - joined["prev_low_1h"]) / joined["pip_size"]
    joined["pre_body_1h"] = (joined["Close"] - joined["prev_open_1h"]) / joined["pip_size"]
    joined["pre_volume_log"] = np.log1p(joined["prev_volume_1h"].clip(lower=0))
    joined["event_year"] = joined["event_time"].dt.year
    joined["horizon_hours"] = horizon
    return joined


def add_labels(frame, train_threshold=None):
    d = frame.copy()
    if train_threshold is None:
        train_threshold = float(d.loc[d["event_year"].le(2020), "domestic_return_pips"].abs().quantile(0.45))
        if not np.isfinite(train_threshold) or train_threshold <= 0:
            train_threshold = float(d["domestic_return_pips"].abs().quantile(0.45))
    d["trade_label"] = np.select(
        [d["domestic_return_pips"] >= train_threshold, d["domestic_return_pips"] <= -train_threshold],
        [1, -1],
        default=0,
    )
    d["direction_label"] = np.where(d["domestic_return_pips"] >= 0, 1, -1)
    return d, train_threshold


def candidate_models():
    models = [
        ("logistic_l2_baseline", LogisticRegression(max_iter=1200, C=0.05, penalty="l2", solver="lbfgs", class_weight="balanced")),
        ("random_forest_event", RandomForestClassifier(n_estimators=350, max_depth=9, min_samples_leaf=15, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ("hist_gradient_event", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.1, random_state=42)),
    ]
    if CatBoostClassifier is not None:
        models += [
            ("catboost_event_balanced", CatBoostClassifier(iterations=700, depth=5, learning_rate=0.035, l2_leaf_reg=12, loss_function="MultiClass", auto_class_weights="Balanced", random_seed=42, verbose=False, allow_writing_files=False)),
            ("catboost_event_deep", CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03, l2_leaf_reg=16, loss_function="MultiClass", auto_class_weights="Balanced", random_seed=43, verbose=False, allow_writing_files=False)),
        ]
    if XGBClassifier is not None:
        models.append(("xgboost_event", XGBEventClassifier(n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1)))
    if LGBMClassifier is not None:
        models.append(("lightgbm_event", LGBMClassifier(n_estimators=600, max_depth=-1, num_leaves=31, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)))
    return models


def probability_frame(model, X):
    proba = np.asarray(model.predict_proba(X))
    classes = getattr(model, "classes_", None)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    for idx, cls in enumerate(classes):
        cls_int = int(cls)
        if cls_int == -1:
            out["prob_sell"] = proba[:, idx]
        elif cls_int == 0:
            out["prob_no_trade"] = proba[:, idx]
        elif cls_int == 1:
            out["prob_buy"] = proba[:, idx]
    return out


def apply_event_signals(events, probs, threshold):
    raw = []
    conf = []
    signal = []
    for _, row in probs.iterrows():
        values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
        selected = max(values, key=values.get)
        confidence = float(values[selected])
        raw.append(selected)
        conf.append(confidence)
        if selected == "NO_TRADE" or confidence < threshold:
            signal.append("NO_TRADE")
        else:
            signal.append(selected)
    out = events.copy()
    out[["prob_sell", "prob_no_trade", "prob_buy"]] = probs[["prob_sell", "prob_no_trade", "prob_buy"]].to_numpy()
    out["raw_predicted_class"] = raw
    out["confidence"] = conf
    out["threshold"] = threshold
    out["signal"] = signal
    return out


def evaluate_signals(events):
    trades = events[events["signal"].isin(["BUY", "SELL"])].copy()
    if trades.empty:
        return {"trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "avg_pips": 0.0, "max_dd_pips": 0.0, "buy_count": 0, "sell_count": 0}
    direction = np.where(trades["signal"].eq("BUY"), 1.0, -1.0)
    trades["pnl_pips"] = trades["domestic_return_pips"] * direction
    wins = int((trades["pnl_pips"] > 0).sum())
    return {
        "trades": int(len(trades)),
        "win_rate": wins / len(trades),
        "net_pips": float(trades["pnl_pips"].sum()),
        "profit_factor": safe_profit_factor(trades["pnl_pips"]),
        "avg_pips": float(trades["pnl_pips"].mean()),
        "max_dd_pips": max_drawdown_pips(trades["pnl_pips"]),
        "buy_count": int(trades["signal"].eq("BUY").sum()),
        "sell_count": int(trades["signal"].eq("SELL").sum()),
    }


def score_result(row):
    if row["trades"] < MIN_VALID_TRADES:
        return -1e9 + row["net_pips"]
    balance_penalty = abs(row["buy_count"] - row["sell_count"]) / max(1, row["trades"]) * 50.0
    dd_penalty = abs(min(0.0, row["max_dd_pips"])) * 0.25
    return row["net_pips"] + 250.0 * row["profit_factor"] + 100.0 * row["win_rate"] - balance_penalty - dd_penalty


def prepare_features(df):
    out = df.copy()
    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_and_select_pair(pair, horizon):
    pair_dir = OUT_DIR / pair / f"h{horizon}"
    if pair_dir.exists():
        backup = pair_dir.parent / f"h{horizon}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(pair_dir), str(backup))
    pair_dir.mkdir(parents=True, exist_ok=True)

    trainval = build_pair_events(pair, DATA, NEWS, "2010-01-01", "2023-12-31", horizon)
    test = build_pair_events(pair, DATATEST, NEWS_TEST, "2024-01-01", "2026-08-31", horizon)
    if trainval.empty:
        return {"pair": pair, "horizon": horizon, "status": "no_train_events"}
    trainval, threshold_pips = add_labels(trainval)
    test, _ = add_labels(test, threshold_pips) if not test.empty else (test, threshold_pips)

    train = trainval[trainval["event_year"].le(2020)].copy()
    valid = trainval[(trainval["event_year"] >= 2021) & (trainval["event_year"] <= 2023)].copy()
    if MAX_TRAIN_ROWS_PER_PAIR > 0 and len(train) > MAX_TRAIN_ROWS_PER_PAIR:
        train = train.sample(MAX_TRAIN_ROWS_PER_PAIR, random_state=42).sort_values("event_time")
    if train.empty or valid.empty:
        return {"pair": pair, "horizon": horizon, "status": "empty_split"}

    X_train = prepare_features(train)
    y_train = train["trade_label"].astype(int)
    X_valid = prepare_features(valid)
    X_test = prepare_features(test) if not test.empty else pd.DataFrame(columns=FEATURES)

    rows = []
    best = None
    models = candidate_models()
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    print(f"[PAIR] {pair} H{horizon} train={len(train)} valid={len(valid)} test={len(test)} threshold_pips={threshold_pips:.2f}", flush=True)
    print(f"[LABEL] train={dict(pd.Series(y_train).value_counts().sort_index())}", flush=True)

    for model_index, (name, model) in enumerate(models, start=1):
        started = time.perf_counter()
        print(f"[MODEL] START pair={pair} H{horizon} {model_index}/{len(models)} {name}", flush=True)
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            rows.append({"pair": pair, "horizon": horizon, "model": name, "status": "fit_failed", "error": str(exc)})
            print(f"[MODEL] FAILED pair={pair} H{horizon} model={name} error={exc}", flush=True)
            continue

        probs_valid = probability_frame(model, X_valid)

        valid_pred = np.select(
            [probs_valid["prob_buy"] >= probs_valid[["prob_sell", "prob_no_trade"]].max(axis=1),
             probs_valid["prob_sell"] >= probs_valid[["prob_buy", "prob_no_trade"]].max(axis=1)],
            [1, -1],
            default=0,
        )
        bal_acc = balanced_accuracy_score(valid["trade_label"].astype(int), valid_pred)
        macro_f1 = f1_score(valid["trade_label"].astype(int), valid_pred, labels=[-1, 0, 1], average="macro", zero_division=0)

        for threshold in thresholds:
            valid_signals = apply_event_signals(valid.reset_index(drop=True), probs_valid.reset_index(drop=True), threshold)
            metrics = evaluate_signals(valid_signals)
            row = {
                "pair": pair,
                "horizon": horizon,
                "model": name,
                "threshold": threshold,
                "threshold_pips": threshold_pips,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 2),
                **metrics,
            }
            row["score"] = score_result(row)
            rows.append(row)
            if best is None or row["score"] > best["row"]["score"]:
                best = {"row": row, "model": model, "probs_valid": probs_valid}
        print(f"[MODEL] DONE pair={pair} H{horizon} model={name}", flush=True)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(pair_dir / "validation_results.csv", index=False)
    if best is None:
        return {"pair": pair, "horizon": horizon, "status": "no_model_selected"}

    best_row = dict(best["row"])
    best_name = best_row["model"]
    joblib_path = pair_dir / f"{best_name}_{pair}_h{horizon}.joblib"
    joblib.dump(best["model"], joblib_path)
    (pair_dir / "selected_config.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    # Recompute selected model probabilities for validation and final test.
    probs_valid = probability_frame(best["model"], X_valid)
    probs_test = probability_frame(best["model"], X_test) if not test.empty else pd.DataFrame(columns=["prob_sell", "prob_no_trade", "prob_buy"])

    valid_signals = apply_event_signals(valid.reset_index(drop=True), probs_valid.reset_index(drop=True), best_row["threshold"])
    test_signals = apply_event_signals(test.reset_index(drop=True), probs_test.reset_index(drop=True), best_row["threshold"]) if not test.empty else pd.DataFrame()
    valid_signals.to_csv(pair_dir / "validation_predictions.csv", index=False)
    test_signals.to_csv(pair_dir / "test_predictions_2024_2026.csv", index=False)

    test_metrics = evaluate_signals(test_signals) if not test_signals.empty else {"trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "avg_pips": 0.0, "max_dd_pips": 0.0, "buy_count": 0, "sell_count": 0}
    summary = {
        **best_row,
        "joblib": str(joblib_path),
        "test_trades": test_metrics["trades"],
        "test_win_rate": test_metrics["win_rate"],
        "test_net_pips": test_metrics["net_pips"],
        "test_profit_factor": test_metrics["profit_factor"],
        "test_avg_pips": test_metrics["avg_pips"],
        "test_max_dd_pips": test_metrics["max_dd_pips"],
        "test_buy_count": test_metrics["buy_count"],
        "test_sell_count": test_metrics["sell_count"],
    }
    pd.DataFrame([summary]).to_csv(pair_dir / "selected_summary.csv", index=False)
    print(
        f"[SELECTED] pair={pair} H{horizon} model={best_name} valid_trades={best_row['trades']} "
        f"valid_wr={best_row['win_rate']:.4f} valid_pips={best_row['net_pips']:.1f} "
        f"test_trades={test_metrics['trades']} test_wr={test_metrics['win_rate']:.4f} test_pips={test_metrics['net_pips']:.1f}",
        flush=True,
    )
    return summary


def make_report(summary):
    d = pd.DataFrame(summary)
    report = ["# Event Study V3 Training and Test Report", ""]
    report.append("Model belajar dari event news yang ditempel ke OHLCV H1, dengan label reaksi harga setelah horizon 1H/2H.")
    report.append("")
    if d.empty:
        report.append("Tidak ada hasil.")
    else:
        cols = ["pair", "horizon", "model", "threshold", "trades", "win_rate", "net_pips", "profit_factor", "test_trades", "test_win_rate", "test_net_pips", "test_profit_factor"]
        view = d[[c for c in cols if c in d.columns]].copy()
        for c in ["win_rate", "profit_factor", "test_win_rate", "test_profit_factor"]:
            if c in view.columns:
                view[c] = view[c].map(lambda x: f"{float(x):.4f}" if pd.notna(x) and np.isfinite(float(x)) else str(x))
        for c in ["net_pips", "test_net_pips"]:
            if c in view.columns:
                view[c] = view[c].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
        report.append("| " + " | ".join(view.columns) + " |")
        report.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
        for _, row in view.iterrows():
            report.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
        report.append("")
        report.append(f"Validation total net pips: {float(pd.to_numeric(d.get('net_pips', 0), errors='coerce').sum()):.1f}")
        report.append(f"Final test total net pips: {float(pd.to_numeric(d.get('test_net_pips', 0), errors='coerce').sum()):.1f}")
    (OUT_DIR / "LAPORAN_EVENT_STUDY_V3.md").write_text("\n".join(report), encoding="utf-8")


def main():
    print("[PROGRESS] START 31_event_study_v3_train_test.py", flush=True)
    print(f"[CONFIG] pairs={PAIRS} horizons={HORIZONS}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "features.json").write_text(json.dumps({"features": FEATURES, "base_features": BASE_FEATURES, "extra_features": EXTRA_FEATURES}, indent=2), encoding="utf-8")
    summaries = []
    for pair_index, pair in enumerate(PAIRS, start=1):
        for horizon in HORIZONS:
            print("=" * 80, flush=True)
            print(f"[RUN] pair={pair_index}/{len(PAIRS)} {pair} horizon={horizon}H", flush=True)
            try:
                summaries.append(train_and_select_pair(pair, horizon))
            except Exception as exc:
                print(f"[RUN] FAILED pair={pair} horizon={horizon} error={exc}", flush=True)
                summaries.append({"pair": pair, "horizon": horizon, "status": "failed", "error": str(exc)})
            pd.DataFrame(summaries).to_csv(OUT_DIR / "event_study_v3_summary.csv", index=False)
    make_report(summaries)
    print(f"[OUTPUT] {OUT_DIR / 'event_study_v3_summary.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_EVENT_STUDY_V3.md'}", flush=True)


if __name__ == "__main__":
    main()
