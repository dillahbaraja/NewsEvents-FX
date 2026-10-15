"""Train fast event-character-aware two-stage models.

Fast models: Elastic Net Logistic, Random Forest, CatBoost if installed.
SVM_RBF and EBM are intentionally left for separate long-running batches.
"""
from pathlib import Path
import json
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from common import ARTIFACTS, model_feature_names, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
DATA = OUT / "data"
PROFILE = OUT / "profiles"
CLUSTERS = OUT / "clusters"
MODELS = OUT / "models"
REPORTS = OUT / "reports"

INPUT = DATA / "event_aware_modeling_dataset.csv"
PROFILE_INPUT = PROFILE / "event_profiles_best_horizon_train_2010_2020.csv"
CLUSTER_INPUT = CLUSTERS / "event_clusters.csv"
MAX_TRAIN_ROWS = int(os.environ.get("EVENT_AWARE_FAST_MAX_TRAIN_ROWS", "250000"))
MAX_VALIDATION_ROWS = int(os.environ.get("EVENT_AWARE_FAST_MAX_VALIDATION_ROWS", "140000"))

EXTRA_FEATURES = [
    "surprise_abs_fc", "surprise_abs_prev", "surprise_sign_fc", "surprise_sign_prev",
    "has_actual", "has_forecast", "has_previous", "pre_return_1h", "pre_return_2h",
    "pre_range_1h", "pre_body_1h", "pre_volume_log", "event_minute", "event_weekday",
]
PROFILE_FEATURES = [
    "profile_trade_count", "profile_original_net_pips", "profile_inverse_net_pips",
    "profile_original_profit_factor", "profile_inverse_profit_factor",
    "profile_avg_pips_original", "profile_avg_pips_inverse",
    "profile_winrate_original", "profile_winrate_inverse",
    "profile_volatility_after_event", "profile_positive_year_ratio_original",
    "profile_positive_year_ratio_inverse", "profile_dominant_year_share_original",
    "profile_original_reliability_score", "profile_inverse_reliability_score",
    "profile_best_event_score", "profile_best_horizon",
]


def safe_pf(values):
    return float(profit_factor(values))


def prepare_dataset():
    base_features = model_feature_names()
    usecols = set([
        "event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours",
        "split", "event_year", "domestic_return_pips", "trade_target", "direction_label",
    ] + base_features + EXTRA_FEATURES)
    d = pd.read_csv(INPUT, usecols=lambda c: c in usecols, low_memory=False)
    profiles = pd.read_csv(PROFILE_INPUT, low_memory=False)
    clusters = pd.read_csv(CLUSTER_INPUT, usecols=["pair", "CountryCode", "event_name_norm", "event_cluster"], low_memory=False)
    profile_cols = {
        "trade_count": "profile_trade_count",
        "original_net_pips": "profile_original_net_pips",
        "inverse_net_pips": "profile_inverse_net_pips",
        "original_profit_factor": "profile_original_profit_factor",
        "inverse_profit_factor": "profile_inverse_profit_factor",
        "avg_pips_original": "profile_avg_pips_original",
        "avg_pips_inverse": "profile_avg_pips_inverse",
        "winrate_original": "profile_winrate_original",
        "winrate_inverse": "profile_winrate_inverse",
        "volatility_after_event": "profile_volatility_after_event",
        "positive_year_ratio_original": "profile_positive_year_ratio_original",
        "positive_year_ratio_inverse": "profile_positive_year_ratio_inverse",
        "dominant_year_share_original": "profile_dominant_year_share_original",
        "original_reliability_score": "profile_original_reliability_score",
        "inverse_reliability_score": "profile_inverse_reliability_score",
        "best_event_score": "profile_best_event_score",
        "horizon_hours": "profile_best_horizon",
    }
    profiles = profiles.rename(columns=profile_cols)
    profile_keep = ["pair", "CountryCode", "event_name_norm"] + list(profile_cols.values())
    d = d.merge(profiles[profile_keep], on=["pair", "CountryCode", "event_name_norm"], how="left")
    d = d.merge(clusters, on=["pair", "CountryCode", "event_name_norm"], how="left")
    d["event_cluster"] = d["event_cluster"].fillna(-1).astype(int)
    for cluster in sorted(d["event_cluster"].unique()):
        d[f"event_cluster_{cluster}"] = (d["event_cluster"] == cluster).astype("int8")
    cluster_features = [c for c in d.columns if c.startswith("event_cluster_")]
    features = base_features + EXTRA_FEATURES + PROFILE_FEATURES + cluster_features
    for c in features:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d, features


def stratified_cap(frame, cap, label):
    if cap <= 0 or len(frame) <= cap:
        progress_step("SAMPLE_KEEP_FULL", label=label, rows=len(frame), cap=cap)
        return frame
    keys = frame["pair"].astype(str) + "|" + frame["horizon_hours"].astype(str) + "|" + frame["trade_target"].astype(str)
    sample_parts = []
    keyed = frame.assign(_sample_key=keys)
    for _, g in keyed.groupby("_sample_key", sort=False):
        n = max(1, int(round(len(g) / len(frame) * cap)))
        sample_parts.append(g.sample(n=min(n, len(g)), random_state=42))
    sampled = pd.concat(sample_parts, ignore_index=True).drop(columns=["_sample_key"])
    if len(sampled) > cap:
        sampled = sampled.sample(n=cap, random_state=42)
    progress_step("SAMPLE_APPLIED", label=label, before=len(frame), after=len(sampled), cap=cap)
    return sampled.reset_index(drop=True)


def candidate_models():
    models = {
        "elastic_net_event_character": {
            "trade": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0005, l1_ratio=0.20, max_iter=35, tol=1e-3, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
            "direction": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0005, l1_ratio=0.20, max_iter=35, tol=1e-3, class_weight="balanced", random_state=43, n_jobs=-1)),
            ]),
        },
        "random_forest_event_character": {
            "trade": RandomForestClassifier(n_estimators=450, max_depth=10, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            "direction": RandomForestClassifier(n_estimators=450, max_depth=10, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", random_state=43, n_jobs=-1),
        },
    }
    if CatBoostClassifier is not None:
        models["catboost_event_character"] = {
            "trade": CatBoostClassifier(iterations=650, depth=5, learning_rate=0.035, l2_leaf_reg=14, loss_function="Logloss", auto_class_weights="Balanced", random_seed=42, verbose=False, allow_writing_files=False),
            "direction": CatBoostClassifier(iterations=650, depth=5, learning_rate=0.035, l2_leaf_reg=14, loss_function="Logloss", auto_class_weights="Balanced", random_seed=43, verbose=False, allow_writing_files=False),
        }
    return models


def proba_positive(model, x):
    p = np.asarray(model.predict_proba(x))
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return p[:, list(classes).index(1)]
    return p[:, -1]


def evaluate_signals(frame):
    if frame.empty:
        return {"trades": 0, "net_pips": 0.0, "profit_factor": 0.0, "winrate": 0.0, "avg_pips": 0.0}
    pips = frame["strategy_pips"]
    return {
        "trades": int(len(frame)),
        "net_pips": float(pips.sum()),
        "profit_factor": safe_pf(pips),
        "winrate": float((pips > 0).mean()),
        "avg_pips": float(pips.mean()),
    }


def make_predictions(model_name, models, d, features):
    train = stratified_cap(d[d["split"].eq("train")].copy(), MAX_TRAIN_ROWS, f"{model_name}_train")
    valid = stratified_cap(d[d["split"].eq("validation")].copy(), MAX_VALIDATION_ROWS, f"{model_name}_validation")
    x_train = train[features]
    x_valid = valid[features]
    y_trade = train["trade_target"].astype(int)
    y_dir = train.loc[train["trade_target"].eq(1), "direction_label"].map({-1: 0, 1: 1}).astype(int)

    progress_step("FIT_TRADE_START", model=model_name, rows=len(x_train), features=len(features))
    t0 = time.time()
    trade_model = models["trade"]
    trade_model.fit(x_train, y_trade)
    progress_step("FIT_TRADE_DONE", model=model_name, seconds=f"{time.time() - t0:.1f}")

    progress_step("FIT_DIRECTION_START", model=model_name, rows=len(y_dir))
    t0 = time.time()
    direction_model = models["direction"]
    direction_model.fit(train.loc[train["trade_target"].eq(1), features], y_dir)
    progress_step("FIT_DIRECTION_DONE", model=model_name, seconds=f"{time.time() - t0:.1f}")

    pred = valid[["event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours", "domestic_return_pips"]].copy()
    pred["model"] = model_name
    pred["trade_probability"] = proba_positive(trade_model, x_valid)
    pred["buy_probability"] = proba_positive(direction_model, x_valid)
    pred["raw_direction"] = np.where(pred["buy_probability"] >= 0.5, 1, -1)

    best = None
    grid = []
    for trade_th in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for dir_margin in [0.00, 0.05, 0.10, 0.15]:
            tmp = pred.copy()
            confidence = (tmp["buy_probability"] - 0.5).abs()
            tmp["signal"] = np.where((tmp["trade_probability"] >= trade_th) & (confidence >= dir_margin), tmp["raw_direction"], 0)
            trades = tmp[tmp["signal"].ne(0)].copy()
            trades["strategy_pips"] = trades["domestic_return_pips"] * trades["signal"]
            metrics = evaluate_signals(trades)
            metrics.update({"model": model_name, "trade_threshold": trade_th, "direction_margin": dir_margin})
            metrics["score"] = metrics["net_pips"] + 500.0 * min(metrics["profit_factor"], 3.0) + 25.0 * math.log1p(metrics["trades"])
            grid.append(metrics)
            if best is None or metrics["score"] > best["score"]:
                best = metrics

    out_pred = pred.copy()
    out_pred["signal"] = np.where(
        (out_pred["trade_probability"] >= best["trade_threshold"]) & ((out_pred["buy_probability"] - 0.5).abs() >= best["direction_margin"]),
        out_pred["raw_direction"],
        0,
    )
    out_pred["strategy_pips"] = out_pred["domestic_return_pips"] * out_pred["signal"]
    bundle = {"trade_model": trade_model, "direction_model": direction_model, "features": features, "threshold": best}
    joblib.dump(bundle, MODELS / f"{model_name}.joblib")
    return out_pred, pd.DataFrame(grid), best


def main():
    started = progress_start(SCRIPT)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    d, features = prepare_dataset()
    progress_step(
        "DATA_READY",
        rows=len(d),
        features=len(features),
        train=int(d["split"].eq("train").sum()),
        validation=int(d["split"].eq("validation").sum()),
        max_train_rows=MAX_TRAIN_ROWS,
        max_validation_rows=MAX_VALIDATION_ROWS,
    )

    all_predictions = []
    all_grid = []
    summary_rows = []
    models = candidate_models()
    for i, (name, model_pair) in enumerate(models.items(), 1):
        progress_step("MODEL_START", i, len(models), model=name)
        pred, grid, best = make_predictions(name, model_pair, d, features)
        all_predictions.append(pred)
        all_grid.append(grid)
        pair_summary = pred[pred["signal"].ne(0)].copy()
        pair_summary["strategy_pips"] = pair_summary["domestic_return_pips"] * pair_summary["signal"]
        by_pair = pair_summary.groupby("pair").agg(
            trades=("strategy_pips", "size"),
            net_pips=("strategy_pips", "sum"),
            profit_factor=("strategy_pips", safe_pf),
            winrate=("strategy_pips", lambda s: float((s > 0).mean())),
        ).reset_index()
        by_pair["model"] = name
        by_pair["trade_threshold"] = best["trade_threshold"]
        by_pair["direction_margin"] = best["direction_margin"]
        summary_rows.append(by_pair)
        progress_step("MODEL_DONE", i, len(models), model=name, trades=best["trades"], net_pips=f"{best['net_pips']:.1f}", pf=f"{best['profit_factor']:.4f}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(REPORTS / "event_aware_validation_predictions.csv", index=False)
    pd.concat(all_grid, ignore_index=True).to_csv(REPORTS / "event_aware_threshold_grid.csv", index=False)
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary.to_csv(REPORTS / "event_aware_validation_pair_summary.csv", index=False)
    metadata = {
        "features": features,
        "models": list(models.keys()),
        "training_period": "2010-2020",
        "validation_period": "2021-2023",
        "profile_source": "2010-2020 only",
        "target": "two-stage trade/no-trade then buy/sell",
    }
    (REPORTS / "event_aware_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
