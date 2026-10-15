"""Build event-character-aware dataset for macro news FX research.

This script creates one row per relevant pair-country-event-horizon and keeps
the dataset independent from older Phase 1-8 artifacts.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from common import (
    ARTIFACTS,
    DATA,
    NEWS,
    add_context_features,
    available_currency_pair_mapping,
    event_category,
    model_feature_names,
    pair_file,
    pip_size,
    progress_done,
    progress_start,
    progress_step,
    read_news,
    read_prices,
    research_split,
)

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
OUT_DATA = OUT / "data"
HORIZONS = [1, 2, 4, 24]


def normalize_event_name(value):
    return " ".join(str(value).strip().lower().split())


def build_pair_horizon(pair, pair_map, horizon):
    price_path = DATA / pair_file(pair)
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

    news = read_news(NEWS)
    news = news[news["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
    if news.empty:
        return pd.DataFrame()

    news["pair"] = pair
    news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
    news = news[news["orientation"].isin(["base", "quote"])].copy()
    news["event_name_norm"] = news["Name"].map(normalize_event_name)
    news["event_key"] = news["pair"] + "|" + news["CountryCode"].astype(str) + "|" + news["event_name_norm"]
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

    direction_sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
    joined["horizon_hours"] = horizon
    joined["pip_size"] = pip_size(pair)
    joined["domestic_return"] = np.log(joined["exit_close"] / joined["entry_close"]) * direction_sign
    joined["domestic_return_pips"] = ((joined["exit_close"] - joined["entry_close"]) / joined["pip_size"]) * direction_sign
    joined["abs_domestic_return_pips"] = joined["domestic_return_pips"].abs()
    joined["pre_return_1h"] = np.log(joined["entry_close"] / joined["pre_close_1h"]).replace([np.inf, -np.inf], np.nan)
    joined["pre_return_2h"] = np.log(joined["entry_close"] / joined["pre_close_2h"]).replace([np.inf, -np.inf], np.nan)
    joined["pre_range_1h"] = (joined["pre_high_1h"] - joined["pre_low_1h"]) / joined["pip_size"]
    joined["pre_body_1h"] = (joined["entry_close"] - joined["pre_open_1h"]) / joined["pip_size"]
    joined["pre_volume_log"] = np.log1p(joined["pre_volume_1h"].clip(lower=0))
    joined["event_year"] = joined["event_time"].dt.year
    joined["split"] = research_split(joined["event_year"])
    joined["direction_label"] = np.where(joined["domestic_return_pips"] >= 0, 1, -1)
    return joined


def add_pair_threshold_labels(d):
    rows = []
    out = []
    for (pair, horizon), g in d.groupby(["pair", "horizon_hours"], sort=False):
        train = g[g["event_year"].le(2020)]
        threshold = float(train["abs_domestic_return_pips"].quantile(0.45))
        if not np.isfinite(threshold) or threshold <= 0:
            threshold = float(g["abs_domestic_return_pips"].quantile(0.45))
        part = g.copy()
        part["trade_label"] = np.select(
            [part["domestic_return_pips"] >= threshold, part["domestic_return_pips"] <= -threshold],
            [1, -1],
            default=0,
        )
        part["trade_target"] = (part["trade_label"] != 0).astype("int8")
        rows.append({"pair": pair, "horizon_hours": horizon, "threshold_pips": threshold, "train_rows": len(train), "rows": len(g)})
        out.append(part)
    pd.DataFrame(rows).to_csv(OUT_DATA / "event_aware_return_thresholds.csv", index=False)
    return pd.concat(out, ignore_index=True)


def main():
    started = progress_start(SCRIPT)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    mapping = available_currency_pair_mapping()
    pairs = mapping["pair"].drop_duplicates().tolist()
    progress_step("DATASET_CONFIG", pairs=",".join(pairs), horizons=",".join(map(str, HORIZONS)))

    parts = []
    total_jobs = len(pairs) * len(HORIZONS)
    job = 0
    for pair in pairs:
        pair_map = mapping[mapping["pair"].eq(pair)].copy()
        for horizon in HORIZONS:
            job += 1
            progress_step("BUILD_PAIR_HORIZON_START", job, total_jobs, pair=pair, horizon=f"{horizon}H")
            part = build_pair_horizon(pair, pair_map, horizon)
            if not part.empty:
                parts.append(part)
            progress_step("BUILD_PAIR_HORIZON_DONE", job, total_jobs, pair=pair, horizon=f"{horizon}H", rows=len(part))

    if not parts:
        raise RuntimeError("No event-character-aware rows were built.")

    d = pd.concat(parts, ignore_index=True)
    d = add_pair_threshold_labels(d)
    keep = [
        "Id", "event_time", "entry_time", "exit_time", "Name", "event_name_norm", "event_key",
        "CountryCode", "CountryName", "pair", "currency", "orientation", "category", "role",
        "Importance", "Impact", "Actual", "Forecast", "Previous", "horizon_hours",
        "entry_close", "exit_close", "domestic_return", "domestic_return_pips",
        "abs_domestic_return_pips", "trade_label", "trade_target", "direction_label",
        "event_year", "split", "surprise_abs_fc", "surprise_abs_prev", "surprise_sign_fc",
        "surprise_sign_prev", "has_actual", "has_forecast", "has_previous", "pre_return_1h",
        "pre_return_2h", "pre_range_1h", "pre_body_1h", "pre_volume_log", "event_minute",
        "event_weekday",
    ] + model_feature_names()
    keep = [c for c in keep if c in d.columns]
    d[keep].to_csv(OUT_DATA / "event_aware_modeling_dataset.csv", index=False)

    summary = d.groupby(["pair", "horizon_hours", "split"], dropna=False).agg(
        rows=("event_time", "size"),
        trade_rows=("trade_target", "sum"),
        buy_rows=("trade_label", lambda s: int((s == 1).sum())),
        sell_rows=("trade_label", lambda s: int((s == -1).sum())),
        no_trade_rows=("trade_label", lambda s: int((s == 0).sum())),
        net_pips=("domestic_return_pips", "sum"),
    ).reset_index()
    summary.to_csv(OUT_DATA / "event_aware_dataset_summary.csv", index=False)

    metadata = {
        "name": "event_character_aware",
        "dataset": str(OUT_DATA / "event_aware_modeling_dataset.csv"),
        "horizons": HORIZONS,
        "unit": "pair + country + event_name + horizon",
        "train": "2010-2020",
        "validation": "2021-2023",
        "final_oos": "2024+ handled by separate datatest pipeline",
        "feature_count_base": len(model_feature_names()),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT_DATA / "event_aware_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
