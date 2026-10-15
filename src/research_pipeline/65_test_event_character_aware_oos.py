"""Evaluate event-character-aware models on datatest OOS data."""
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

from common import (
    ARTIFACTS,
    DATA,
    add_context_features,
    available_currency_pair_mapping,
    event_category,
    model_feature_names,
    pair_file,
    pip_size,
    progress_done,
    progress_start,
    progress_step,
    profit_factor,
    read_news,
    read_prices,
)

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"
PROFILE_INPUT = OUT / "profiles" / "event_profiles_best_horizon_train_2010_2020.csv"
CLUSTER_INPUT = OUT / "clusters" / "event_clusters.csv"
MODELS = OUT / "models"
REPORTS = OUT / "reports"
VALIDATION_PREDICTIONS = REPORTS / "event_aware_validation_predictions.csv"
HORIZONS = [1, 2, 4, 24]

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


def normalize_event_name(value):
    return " ".join(str(value).strip().lower().split())


def safe_pf(values):
    return float(profit_factor(values))


def build_pair_horizon(pair, pair_map, horizon):
    price_path = DATATEST / pair_file(pair)
    if not price_path.exists():
        return pd.DataFrame()

    px = read_prices(price_path).set_index("time")
    px["entry_time"] = px.index
    px["entry_close"] = px["Close"]
    px["exit_close"] = px["Close"].shift(-horizon)
    px["exit_time"] = px.index.to_series().shift(-horizon)
    px["pre_close_1h"] = px["Close"].shift(1)
    px["pre_close_2h"] = px["Close"].shift(2)
    px["pre_high_1h"] = px["High"].shift(1)
    px["pre_low_1h"] = px["Low"].shift(1)
    px["pre_open_1h"] = px["Open"].shift(1)
    px["pre_volume_1h"] = px["Volume"].shift(1)
    px = px.reset_index(drop=True)

    news = read_news(NEWS_TEST)
    news = news[news["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
    if news.empty:
        return pd.DataFrame()
    news["pair"] = pair
    news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
    news = news[news["orientation"].isin(["base", "quote"])].copy()
    news["event_name_norm"] = news["Name"].map(normalize_event_name)
    news["category"] = news["Name"].map(event_category)
    news["release_hour"] = news["event_time"].dt.hour
    news["event_minute"] = news["event_time"].dt.minute
    news["event_weekday"] = news["event_time"].dt.weekday
    news["surprise_fc"] = news["Actual"] - news["Forecast"]
    news["surprise_prev"] = news["Actual"] - news["Previous"]
    news["surprise_pct_fc"] = news["surprise_fc"] / (news["Forecast"].abs() + 1e-9)
    news["surprise_abs_fc"] = news["surprise_fc"].abs()
    news["surprise_abs_prev"] = news["surprise_prev"].abs()
    news["surprise_sign_fc"] = np.sign(news["surprise_fc"]).fillna(0)
    news["surprise_sign_prev"] = np.sign(news["surprise_prev"]).fillna(0)
    news["has_actual"] = news["Actual"].notna().astype("int8")
    news["has_forecast"] = news["Forecast"].notna().astype("int8")
    news["has_previous"] = news["Previous"].notna().astype("int8")
    news = add_context_features(news)

    joined = pd.merge_asof(
        news.sort_values("event_time"),
        px.sort_values("entry_time"),
        left_on="event_time",
        right_on="entry_time",
        direction="forward",
        tolerance=pd.Timedelta(hours=2),
    )
    joined = joined.dropna(subset=["entry_close", "exit_close", "exit_time"]).copy()
    if joined.empty:
        return joined
    sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
    joined["horizon_hours"] = horizon
    joined["pip_size"] = pip_size(pair)
    joined["domestic_return_pips"] = ((joined["exit_close"] - joined["entry_close"]) / joined["pip_size"]) * sign
    return joined


def build_oos_dataset():
    mapping = available_currency_pair_mapping()
    parts = []
    pairs = mapping["pair"].drop_duplicates().tolist()
    total = len(pairs) * len(HORIZONS)
    job = 0
    for pair in pairs:
        pair_map = mapping[mapping["pair"].eq(pair)].copy()
        for horizon in HORIZONS:
            job += 1
            progress_step("OOS_BUILD_START", job, total, pair=pair, horizon=f"{horizon}H")
            part = build_pair_horizon(pair, pair_map, horizon)
            if not part.empty:
                parts.append(part)
            progress_step("OOS_BUILD_DONE", job, total, pair=pair, horizon=f"{horizon}H", rows=len(part))
    if not parts:
        raise RuntimeError("No OOS rows built.")
    return pd.concat(parts, ignore_index=True)


def attach_profiles(d):
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
    out = d.merge(profiles[profile_keep], on=["pair", "CountryCode", "event_name_norm"], how="left")
    out = out.merge(clusters, on=["pair", "CountryCode", "event_name_norm"], how="left")
    out["event_cluster"] = out["event_cluster"].fillna(-1).astype(int)
    for cluster in sorted(out["event_cluster"].unique()):
        out[f"event_cluster_{cluster}"] = (out["event_cluster"] == cluster).astype("int8")
    return out


def proba_positive(model, x):
    p = np.asarray(model.predict_proba(x))
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return p[:, list(classes).index(1)]
    return p[:, -1]


def main():
    started = progress_start(SCRIPT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    d = attach_profiles(build_oos_dataset())
    base_features = model_feature_names()
    cluster_features = [c for c in d.columns if c.startswith("event_cluster_")]
    available_feature_pool = set(d.columns)
    progress_step("OOS_DATA_READY", rows=len(d), pairs=d["pair"].nunique(), cluster_features=len(cluster_features))

    all_rows = []
    summaries = []
    validation_rates = pd.DataFrame()
    if VALIDATION_PREDICTIONS.exists():
        vp = pd.read_csv(VALIDATION_PREDICTIONS, usecols=["model", "pair", "signal"], low_memory=False)
        validation_rates = vp.groupby(["model", "pair"]).agg(
            validation_rows=("signal", "size"),
            validation_trade_rows=("signal", lambda s: int((s != 0).sum())),
        ).reset_index()
        validation_rates["validation_trade_rate"] = validation_rates["validation_trade_rows"] / validation_rates["validation_rows"].clip(lower=1)
    for path in sorted(MODELS.glob("*_event_character.joblib")):
        model_name = path.stem
        bundle = joblib.load(path)
        for missing in [c for c in bundle["features"] if c not in available_feature_pool]:
            d[missing] = 0.0
            available_feature_pool.add(missing)
        features = list(bundle["features"])
        for c in features:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        threshold = bundle["threshold"]
        progress_step("OOS_MODEL_START", model=model_name, rows=len(d), features=len(features))
        pred = d[["event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours", "domestic_return_pips"]].copy()
        pred["model"] = model_name
        pred["trade_probability"] = proba_positive(bundle["trade_model"], d[features])
        pred["buy_probability"] = proba_positive(bundle["direction_model"], d[features])
        pred["raw_direction"] = np.where(pred["buy_probability"] >= 0.5, 1, -1)
        pred["signal"] = np.where(
            (pred["trade_probability"] >= threshold["trade_threshold"]) & ((pred["buy_probability"] - 0.5).abs() >= threshold["direction_margin"]),
            pred["raw_direction"],
            0,
        )
        pred["deployment_policy"] = "absolute_validation_threshold"
        if int((pred["signal"] != 0).sum()) == 0 and not validation_rates.empty:
            pred["score_for_rank"] = pred["trade_probability"] * ((pred["buy_probability"] - 0.5).abs() * 2.0)
            pred["signal"] = 0
            for pair, g in pred.groupby("pair", sort=False):
                rate_row = validation_rates[(validation_rates["model"].eq(model_name)) & (validation_rates["pair"].eq(pair))]
                if rate_row.empty:
                    continue
                rate = float(rate_row["validation_trade_rate"].iloc[0])
                if rate <= 0:
                    continue
                n = max(1, int(round(len(g) * min(rate, 0.35))))
                selected = g.nlargest(n, "score_for_rank").index
                pred.loc[selected, "signal"] = pred.loc[selected, "raw_direction"]
            pred["deployment_policy"] = "validation_trade_rate_rank_fallback"
        pred["strategy_pips"] = pred["domestic_return_pips"] * pred["signal"]
        trades = pred[pred["signal"].ne(0)].copy()
        all_rows.append(pred)
        by_pair = trades.groupby("pair").agg(
            trades=("strategy_pips", "size"),
            net_pips=("strategy_pips", "sum"),
            profit_factor=("strategy_pips", safe_pf),
            winrate=("strategy_pips", lambda s: float((s > 0).mean())),
        ).reset_index()
        by_pair["model"] = model_name
        summaries.append(by_pair)
        total = {
            "trades": int(len(trades)),
            "net_pips": float(trades["strategy_pips"].sum()) if len(trades) else 0.0,
            "profit_factor": safe_pf(trades["strategy_pips"]) if len(trades) else 0.0,
        }
        progress_step("OOS_MODEL_DONE", model=model_name, trades=total["trades"], net_pips=f"{total['net_pips']:.1f}", pf=f"{total['profit_factor']:.4f}")

    pd.concat(all_rows, ignore_index=True).to_csv(REPORTS / "event_aware_oos_predictions.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(REPORTS / "event_aware_oos_pair_summary.csv", index=False)
    total_summary = summary.groupby("model").agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
    ).reset_index()
    total_summary.to_csv(REPORTS / "event_aware_oos_model_summary.csv", index=False)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
