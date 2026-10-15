"""Test selected pair-specific event-character models on datatest OOS data."""
from pathlib import Path
import os
import sys

import joblib
import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
OUT = ARTIFACTS / "event_character_pair_specific"
GLOBAL_OUT = ARTIFACTS / "event_character_aware"
REPORTS = OUT / "reports"
MODELS = OUT / "models"
SELECTED_FILE = Path(os.environ.get("PAIR_SPECIFIC_SELECTED_FILE", str(REPORTS / "pair_specific_selected_models.csv")))

# Reuse the already tested OOS feature builder from stage 65.
sys.path.insert(0, str(PIPE))
from importlib import import_module

stage65 = import_module("65_test_event_character_aware_oos")


def safe_pf(values):
    return float(profit_factor(values))


def proba_positive(model, x):
    p = np.asarray(model.predict_proba(x))
    classes = getattr(model, "classes_", np.array([0, 1]))
    return p[:, list(classes).index(1)] if 1 in classes else p[:, -1]


def main():
    started = progress_start(SCRIPT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTED_FILE)
    selected = selected[selected["deployable"].astype(str).str.lower().isin(["true", "1"])]
    progress_step("SELECTED_MODELS", rows=len(selected), selected_file=str(SELECTED_FILE))
    d = stage65.attach_profiles(stage65.build_oos_dataset())
    progress_step("OOS_DATA_READY", rows=len(d), pairs=d["pair"].nunique())

    preds = []
    summaries = []
    for i, row in selected.reset_index(drop=True).iterrows():
        pair = row["pair"]
        model = row["model"]
        path = MODELS / f"{model}_{pair}.joblib"
        progress_step("OOS_PAIR_MODEL_START", i + 1, len(selected), pair=pair, model=model)
        if not path.exists():
            progress_step("OOS_PAIR_MODEL_MISSING", pair=pair, model=model, path=str(path))
            continue
        bundle = joblib.load(path)
        part = d[d["pair"].eq(pair)].copy()
        for missing in [c for c in bundle["features"] if c not in part.columns]:
            part[missing] = 0.0
        features = list(bundle["features"])
        for c in features:
            part[c] = pd.to_numeric(part[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = part[["event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours", "domestic_return_pips"]].copy()
        pred["model"] = model
        pred["trade_probability"] = proba_positive(bundle["trade_model"], part[features])
        pred["buy_probability"] = proba_positive(bundle["direction_model"], part[features])
        pred["raw_direction"] = np.where(pred["buy_probability"] >= 0.5, 1, -1)
        threshold = bundle["threshold"]
        pred["signal"] = np.where(
            (pred["trade_probability"] >= threshold["trade_threshold"]) & ((pred["buy_probability"] - 0.5).abs() >= threshold["direction_margin"]),
            pred["raw_direction"],
            0,
        )
        pred["deployment_policy"] = "pair_specific_validation_threshold"
        if int((pred["signal"] != 0).sum()) == 0:
            validation_rate = float(row["trades"]) / max(1.0, float(row.get("validation_rows", 1) if "validation_rows" in row else 50000))
            validation_rate = min(max(validation_rate, 0.03), 0.30)
            pred["score_for_rank"] = pred["trade_probability"] * ((pred["buy_probability"] - 0.5).abs() * 2.0)
            n = max(1, int(round(len(pred) * validation_rate)))
            idx = pred.nlargest(n, "score_for_rank").index
            pred["signal"] = 0
            pred.loc[idx, "signal"] = pred.loc[idx, "raw_direction"]
            pred["deployment_policy"] = "pair_specific_rank_fallback"
        pred["strategy_pips"] = pred["domestic_return_pips"] * pred["signal"]
        trades = pred[pred["signal"].ne(0)].copy()
        preds.append(pred)
        summaries.append({
            "pair": pair,
            "model": model,
            "trades": int(len(trades)),
            "net_pips": float(trades["strategy_pips"].sum()) if len(trades) else 0.0,
            "profit_factor": safe_pf(trades["strategy_pips"]) if len(trades) else 0.0,
            "winrate": float((trades["strategy_pips"] > 0).mean()) if len(trades) else 0.0,
            "deployment_policy": pred["deployment_policy"].iloc[0],
        })
        progress_step("OOS_PAIR_MODEL_DONE", pair=pair, model=model, trades=summaries[-1]["trades"], net_pips=f"{summaries[-1]['net_pips']:.1f}", pf=f"{summaries[-1]['profit_factor']:.4f}")

    if preds:
        pd.concat(preds, ignore_index=True).to_csv(REPORTS / "pair_specific_oos_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS / "pair_specific_oos_summary.csv", index=False)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
