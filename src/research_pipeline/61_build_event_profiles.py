"""Build historical event character profiles without using OOS test data."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
DATA = OUT / "data"
PROFILE = OUT / "profiles"
INPUT = DATA / "event_aware_modeling_dataset.csv"


def safe_pf(values):
    return float(profit_factor(values))


def max_drawdown(values):
    equity = np.cumsum(pd.Series(values).fillna(0).to_numpy())
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float((equity - peaks).min())


def profile_group(g):
    pips = pd.to_numeric(g["domestic_return_pips"], errors="coerce").dropna()
    inv = -pips
    yearly = g.assign(_pips=pips).groupby("event_year")["_pips"].sum()
    active_years = int(yearly.shape[0])
    abs_sum = float(yearly.abs().sum())
    dominant = float(yearly.abs().max() / abs_sum) if abs_sum > 0 else 1.0
    positive_year_ratio = float((yearly > 0).sum() / active_years) if active_years else 0.0
    inv_yearly = -yearly
    inv_positive_year_ratio = float((inv_yearly > 0).sum() / active_years) if active_years else 0.0
    return pd.Series({
        "trade_count": int(len(pips)),
        "original_net_pips": float(pips.sum()),
        "inverse_net_pips": float(inv.sum()),
        "original_profit_factor": safe_pf(pips),
        "inverse_profit_factor": safe_pf(inv),
        "avg_pips_original": float(pips.mean()) if len(pips) else 0.0,
        "avg_pips_inverse": float(inv.mean()) if len(inv) else 0.0,
        "winrate_original": float((pips > 0).mean()) if len(pips) else 0.0,
        "winrate_inverse": float((inv > 0).mean()) if len(inv) else 0.0,
        "volatility_after_event": float(pips.std(ddof=0)) if len(pips) else 0.0,
        "max_adverse_excursion": float(pips.min()) if len(pips) else 0.0,
        "max_favorable_excursion": float(pips.max()) if len(pips) else 0.0,
        "max_drawdown_original": max_drawdown(pips),
        "active_years": active_years,
        "positive_year_ratio_original": positive_year_ratio,
        "positive_year_ratio_inverse": inv_positive_year_ratio,
        "dominant_year_share_original": dominant,
        "dominant_year_share_inverse": dominant,
    })


def reliability(row, prefix):
    net = row[f"{prefix}_net_pips"]
    pf = row[f"{prefix}_profit_factor"]
    avg = row[f"avg_pips_{prefix}"]
    pos = row[f"positive_year_ratio_{prefix}"]
    dom = row[f"dominant_year_share_{prefix}"]
    if not np.isfinite(pf):
        pf = 10.0
    stability = max(pos, 0.0) * max(1.0 - dom, 0.0)
    return float(net * math.log1p(row["trade_count"]) * stability * max(min(pf, 10.0), 0.0) * (1.0 if avg > 0 else 0.25))


def build_profiles(d, years, suffix):
    source = d[d["event_year"].between(years[0], years[1])].copy()
    rows = source.groupby(
        ["pair", "CountryCode", "event_name_norm", "horizon_hours"],
        dropna=False,
    )[["event_year", "domestic_return_pips"]].apply(profile_group).reset_index()
    rows["original_reliability_score"] = rows.apply(lambda r: reliability(r, "original"), axis=1)
    rows["inverse_reliability_score"] = rows.apply(lambda r: reliability(r, "inverse"), axis=1)
    rows["best_direction"] = np.select(
        [
            (rows["original_reliability_score"] > 0) & (rows["original_reliability_score"] >= rows["inverse_reliability_score"]),
            (rows["inverse_reliability_score"] > 0) & (rows["inverse_reliability_score"] > rows["original_reliability_score"]),
        ],
        ["ALLOW", "INVERSE"],
        default="BLOCK",
    )
    score_cols = ["original_reliability_score", "inverse_reliability_score"]
    idx = rows.groupby(["pair", "CountryCode", "event_name_norm"])[score_cols].max().max(axis=1)
    best = rows.merge(idx.rename("best_event_score").reset_index(), on=["pair", "CountryCode", "event_name_norm"], how="left")
    best = best[best[score_cols].max(axis=1).eq(best["best_event_score"])].copy()
    best = best.sort_values(["pair", "CountryCode", "event_name_norm", "horizon_hours"]).drop_duplicates(["pair", "CountryCode", "event_name_norm"])
    rows.to_csv(PROFILE / f"event_profiles_by_horizon_{suffix}.csv", index=False)
    best.to_csv(PROFILE / f"event_profiles_best_horizon_{suffix}.csv", index=False)
    return rows, best


def main():
    started = progress_start(SCRIPT)
    PROFILE.mkdir(parents=True, exist_ok=True)
    usecols = [
        "event_time", "event_year", "pair", "CountryCode", "event_name_norm",
        "horizon_hours", "domestic_return_pips",
    ]
    d = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce")
    d["domestic_return_pips"] = pd.to_numeric(d["domestic_return_pips"], errors="coerce")
    progress_step("PROFILE_INPUT", rows=len(d), pairs=d["pair"].nunique(), events=d["event_name_norm"].nunique())

    train_rows, train_best = build_profiles(d, (2010, 2020), "train_2010_2020")
    progress_step("PROFILE_TRAIN_DONE", rows=len(train_rows), best_rows=len(train_best))
    trainval_rows, trainval_best = build_profiles(d, (2010, 2023), "trainval_2010_2023")
    progress_step("PROFILE_TRAINVAL_DONE", rows=len(trainval_rows), best_rows=len(trainval_best))

    note = {
        "profile_unit": "pair + country + event_name + horizon",
        "best_horizon_unit": "pair + country + event_name",
        "train_profile": "2010-2020 for validation modeling",
        "trainval_profile": "2010-2023 for final OOS deployment",
        "anti_leakage": "No 2024+ data is used in profile construction.",
    }
    (PROFILE / "event_profile_metadata.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
