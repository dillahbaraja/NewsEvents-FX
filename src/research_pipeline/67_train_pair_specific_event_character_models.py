"""Train event-character-aware models separately for each FX pair."""
from pathlib import Path
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from common import ARTIFACTS, model_feature_names, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific"
SOURCE = ARTIFACTS / "event_character_aware"
DATA = SOURCE / "data"
PROFILE = SOURCE / "profiles"
CLUSTERS = SOURCE / "clusters"
MODELS = OUT / "models"
REPORTS = OUT / "reports"

INPUT = DATA / "event_aware_modeling_dataset.csv"
PROFILE_INPUT = PROFILE / "event_profiles_best_horizon_train_2010_2020.csv"
CLUSTER_INPUT = CLUSTERS / "event_clusters.csv"
MAX_TRAIN_ROWS_PER_PAIR = int(os.environ.get("PAIR_EVENT_AWARE_MAX_TRAIN_ROWS", "120000"))
MAX_VALID_ROWS_PER_PAIR = int(os.environ.get("PAIR_EVENT_AWARE_MAX_VALID_ROWS", "50000"))

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


def stratified_cap(frame, cap, label):
    if cap <= 0 or len(frame) <= cap:
        progress_step("SAMPLE_KEEP_FULL", label=label, rows=len(frame))
        return frame.reset_index(drop=True)
    keys = frame["horizon_hours"].astype(str) + "|" + frame["trade_target"].astype(str)
    parts = []
    keyed = frame.assign(_sample_key=keys)
    for _, g in keyed.groupby("_sample_key", sort=False):
        n = max(1, int(round(len(g) / len(frame) * cap)))
        parts.append(g.sample(n=min(n, len(g)), random_state=42))
    out = pd.concat(parts, ignore_index=True).drop(columns=["_sample_key"])
    if len(out) > cap:
        out = out.sample(n=cap, random_state=42)
    progress_step("SAMPLE_APPLIED", label=label, before=len(frame), after=len(out), cap=cap)
    return out.reset_index(drop=True)


def load_dataset():
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
    d = d.merge(profiles[["pair", "CountryCode", "event_name_norm"] + list(profile_cols.values())], on=["pair", "CountryCode", "event_name_norm"], how="left")
    d = d.merge(clusters, on=["pair", "CountryCode", "event_name_norm"], how="left")
    d["event_cluster"] = d["event_cluster"].fillna(-1).astype(int)
    for cluster in sorted(d["event_cluster"].unique()):
        d[f"event_cluster_{cluster}"] = (d["event_cluster"] == cluster).astype("int8")
    cluster_features = [c for c in d.columns if c.startswith("event_cluster_")]
    # Pair one-hot features are redundant inside pair-specific models.
    features = [c for c in base_features if not c.startswith("pair_")] + EXTRA_FEATURES + PROFILE_FEATURES + cluster_features
    for c in features:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d, features


def candidate_models():
    models = {
        "elastic_net_pair_event_character": lambda seed: {
            "trade": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0004, l1_ratio=0.18, max_iter=45, tol=1e-3, class_weight="balanced", random_state=seed, n_jobs=-1)),
            ]),
            "direction": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0004, l1_ratio=0.18, max_iter=45, tol=1e-3, class_weight="balanced", random_state=seed + 1, n_jobs=-1)),
            ]),
        },
        "random_forest_pair_event_character": lambda seed: {
            "trade": RandomForestClassifier(n_estimators=500, max_depth=11, min_samples_leaf=15, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
            "direction": RandomForestClassifier(n_estimators=500, max_depth=11, min_samples_leaf=15, max_features="sqrt", class_weight="balanced_subsample", random_state=seed + 1, n_jobs=-1),
        },
    }
    if CatBoostClassifier is not None:
        models["catboost_pair_event_character"] = lambda seed: {
            "trade": CatBoostClassifier(iterations=750, depth=5, learning_rate=0.035, l2_leaf_reg=14, loss_function="Logloss", auto_class_weights="Balanced", random_seed=seed, verbose=False, allow_writing_files=False),
            "direction": CatBoostClassifier(iterations=750, depth=5, learning_rate=0.035, l2_leaf_reg=14, loss_function="Logloss", auto_class_weights="Balanced", random_seed=seed + 1, verbose=False, allow_writing_files=False),
        }
    return models


def proba_positive(model, x):
    p = np.asarray(model.predict_proba(x))
    classes = getattr(model, "classes_", np.array([0, 1]))
    return p[:, list(classes).index(1)] if 1 in classes else p[:, -1]


def eval_thresholds(pred, model_name, pair):
    rows = []
    best = None
    for trade_th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for margin in [0.0, 0.03, 0.06, 0.10, 0.15]:
            tmp = pred.copy()
            tmp["signal"] = np.where((tmp["trade_probability"] >= trade_th) & ((tmp["buy_probability"] - 0.5).abs() >= margin), tmp["raw_direction"], 0)
            trades = tmp[tmp["signal"].ne(0)].copy()
            trades["strategy_pips"] = trades["domestic_return_pips"] * trades["signal"]
            if trades.empty:
                metrics = {"trades": 0, "net_pips": 0.0, "profit_factor": 0.0, "winrate": 0.0}
            else:
                metrics = {
                    "trades": int(len(trades)),
                    "net_pips": float(trades["strategy_pips"].sum()),
                    "profit_factor": safe_pf(trades["strategy_pips"]),
                    "winrate": float((trades["strategy_pips"] > 0).mean()),
                }
            metrics.update({"model": model_name, "pair": pair, "trade_threshold": trade_th, "direction_margin": margin})
            metrics["score"] = metrics["net_pips"] + 400.0 * min(metrics["profit_factor"], 3.0) + 30.0 * math.log1p(metrics["trades"])
            rows.append(metrics)
            if best is None or metrics["score"] > best["score"]:
                best = metrics
    return pd.DataFrame(rows), best


def fit_pair_model(pair, model_name, maker, pair_data, features, seed):
    train = stratified_cap(pair_data[pair_data["split"].eq("train")].copy(), MAX_TRAIN_ROWS_PER_PAIR, f"{pair}_{model_name}_train")
    valid = stratified_cap(pair_data[pair_data["split"].eq("validation")].copy(), MAX_VALID_ROWS_PER_PAIR, f"{pair}_{model_name}_validation")
    models = maker(seed)
    x_train = train[features]
    y_trade = train["trade_target"].astype(int)
    trade_rows = train["trade_target"].eq(1)
    if y_trade.nunique() < 2 or train.loc[trade_rows, "direction_label"].nunique() < 2:
        progress_step("SKIP_MODEL_INSUFFICIENT_CLASS", pair=pair, model=model_name)
        return None, None, None
    progress_step("FIT_TRADE_START", pair=pair, model=model_name, rows=len(train), features=len(features))
    models["trade"].fit(x_train, y_trade)
    progress_step("FIT_DIRECTION_START", pair=pair, model=model_name, rows=int(trade_rows.sum()))
    models["direction"].fit(train.loc[trade_rows, features], train.loc[trade_rows, "direction_label"].map({-1: 0, 1: 1}).astype(int))

    pred = valid[["event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours", "domestic_return_pips"]].copy()
    pred["model"] = model_name
    pred["trade_probability"] = proba_positive(models["trade"], valid[features])
    pred["buy_probability"] = proba_positive(models["direction"], valid[features])
    pred["raw_direction"] = np.where(pred["buy_probability"] >= 0.5, 1, -1)
    grid, best = eval_thresholds(pred, model_name, pair)
    pred["signal"] = np.where(
        (pred["trade_probability"] >= best["trade_threshold"]) & ((pred["buy_probability"] - 0.5).abs() >= best["direction_margin"]),
        pred["raw_direction"],
        0,
    )
    pred["strategy_pips"] = pred["domestic_return_pips"] * pred["signal"]
    bundle = {"pair": pair, "model": model_name, "trade_model": models["trade"], "direction_model": models["direction"], "features": features, "threshold": best}
    joblib.dump(bundle, MODELS / f"{model_name}_{pair}.joblib")
    progress_step("FIT_PAIR_MODEL_DONE", pair=pair, model=model_name, trades=best["trades"], net_pips=f"{best['net_pips']:.1f}", pf=f"{best['profit_factor']:.4f}")
    return pred, grid, best


def main():
    started = progress_start(SCRIPT)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    d, features = load_dataset()
    pairs = sorted(d["pair"].dropna().unique())
    makers = candidate_models()
    progress_step("PAIR_SPECIFIC_CONFIG", pairs=",".join(pairs), models=",".join(makers.keys()), features=len(features))

    predictions = []
    grids = []
    summaries = []
    total_jobs = len(pairs) * len(makers)
    job = 0
    for pair in pairs:
        pair_data = d[d["pair"].eq(pair)].copy()
        for model_name, maker in makers.items():
            job += 1
            progress_step("PAIR_MODEL_START", job, total_jobs, pair=pair, model=model_name)
            pred, grid, best = fit_pair_model(pair, model_name, maker, pair_data, features, 1000 + job)
            if pred is None:
                continue
            predictions.append(pred)
            grids.append(grid)
            summaries.append(best)

    pd.concat(predictions, ignore_index=True).to_csv(REPORTS / "pair_specific_validation_predictions.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(REPORTS / "pair_specific_threshold_grid.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS / "pair_specific_validation_summary.csv", index=False)
    (REPORTS / "pair_specific_metadata.json").write_text(json.dumps({
        "features": features,
        "models": list(makers.keys()),
        "train": "2010-2020",
        "validation": "2021-2023",
        "unit": "one model per pair",
    }, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
