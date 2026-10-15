"""Apply yearly stability gate to pair-specific event-character models."""
from pathlib import Path
import os

import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific"
REPORTS = OUT / "reports"

MIN_POSITIVE_YEARS = int(os.environ.get("PAIR_STABILITY_MIN_POSITIVE_YEARS", "3"))
MAX_DOMINANT_YEAR_SHARE = float(os.environ.get("PAIR_STABILITY_MAX_DOMINANT_YEAR_SHARE", "0.60"))
MIN_PROFIT_FACTOR = float(os.environ.get("PAIR_STABILITY_MIN_PROFIT_FACTOR", "1.03"))
MIN_TRADES = int(os.environ.get("PAIR_STABILITY_MIN_TRADES", "500"))


def main():
    started = progress_start(SCRIPT)
    pred = pd.read_csv(
        REPORTS / "pair_specific_validation_predictions.csv",
        usecols=["model", "pair", "event_time", "signal", "strategy_pips"],
        parse_dates=["event_time"],
        low_memory=False,
    )
    pred = pred[pred["signal"].ne(0)].copy()
    pred["year"] = pred["event_time"].dt.year

    base = pd.read_csv(REPORTS / "pair_specific_validation_summary.csv")
    yearly = pred.groupby(["pair", "model", "year"]).agg(year_net_pips=("strategy_pips", "sum")).reset_index()
    stats = []
    for (pair, model), g in yearly.groupby(["pair", "model"], sort=False):
        abs_sum = float(g["year_net_pips"].abs().sum())
        stats.append({
            "pair": pair,
            "model": model,
            "positive_years": int((g["year_net_pips"] > 0).sum()),
            "active_years": int(g["year"].nunique()),
            "dominant_year_share": float(g["year_net_pips"].abs().max() / abs_sum) if abs_sum > 0 else 1.0,
            "min_year_pips": float(g["year_net_pips"].min()),
            "max_year_pips": float(g["year_net_pips"].max()),
        })
    stability = pd.DataFrame(stats)
    merged = base.merge(stability, on=["pair", "model"], how="left")
    merged["stability_pass"] = (
        (merged["trades"] >= MIN_TRADES)
        & (merged["profit_factor"] >= MIN_PROFIT_FACTOR)
        & (merged["positive_years"] >= MIN_POSITIVE_YEARS)
        & (merged["dominant_year_share"] <= MAX_DOMINANT_YEAR_SHARE)
    )
    merged["stability_score"] = (
        merged["net_pips"]
        * (merged["positive_years"].fillna(0) / merged["active_years"].fillna(1).clip(lower=1))
        * (1.0 - merged["dominant_year_share"].fillna(1.0)).clip(lower=0)
        * np.minimum(merged["profit_factor"], 2.0)
    )
    merged.to_csv(REPORTS / "pair_specific_stability_diagnostics.csv", index=False)

    selected = []
    for pair, g in merged.groupby("pair", sort=False):
        passed = g[g["stability_pass"]].copy()
        if passed.empty:
            best = g.sort_values(["stability_score", "net_pips"], ascending=[False, False]).iloc[0].to_dict()
            best["deployable"] = False
            best["gate_reason"] = "FAIL_stability_gate"
        else:
            best = passed.sort_values(["stability_score", "net_pips"], ascending=[False, False]).iloc[0].to_dict()
            best["deployable"] = True
            best["gate_reason"] = "PASS_stability_gate"
        selected.append(best)
    out = pd.DataFrame(selected).sort_values(["deployable", "stability_score"], ascending=[False, False])
    out.to_csv(REPORTS / "pair_specific_selected_models_stability.csv", index=False)
    progress_step(
        "STABILITY_GATE_DONE",
        selected=int(out["deployable"].sum()),
        blocked=int((~out["deployable"]).sum()),
        min_positive_years=MIN_POSITIVE_YEARS,
        max_dominant_year_share=MAX_DOMINANT_YEAR_SHARE,
        output=str(REPORTS / "pair_specific_selected_models_stability.csv"),
    )
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
