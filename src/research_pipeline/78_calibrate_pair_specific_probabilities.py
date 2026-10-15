"""Validation-fitted probability calibration for the locked pair-specific models."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
ROOT = ARTIFACTS / "event_character_pair_specific"
REPORTS = ROOT / "reports"
OUT = ROOT / "calibration_experiment"

KEYS = ["event_time", "Name", "event_name_norm", "CountryCode", "pair", "horizon_hours"]


def pf(values):
    return float(profit_factor(values)) if len(values) else 0.0


def load_labels():
    source = ARTIFACTS / "event_character_aware" / "data" / "event_aware_modeling_dataset.csv"
    cols = KEYS + ["trade_target", "direction_label"]
    d = pd.read_csv(source, usecols=lambda c: c in cols, low_memory=False)
    d["horizon_hours"] = pd.to_numeric(d["horizon_hours"], errors="coerce").fillna(1).astype(int)
    return d.drop_duplicates(KEYS)


def load_pred(name):
    d = pd.read_csv(REPORTS / name, low_memory=False)
    d["horizon_hours"] = pd.to_numeric(d["horizon_hours"], errors="coerce").fillna(1).astype(int)
    return d


def fit_calibrator(x, y):
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    x = pd.to_numeric(x, errors="coerce").fillna(0.5).clip(1e-5, 1 - 1e-5)
    if y.nunique() < 2:
        return None
    z = np.log(x / (1 - x)).to_numpy().reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(z, y)
    return model


def calibrated(model, x):
    x = pd.to_numeric(x, errors="coerce").fillna(0.5).clip(1e-5, 1 - 1e-5)
    if model is None:
        return x.to_numpy(dtype=float)
    z = np.log(x / (1 - x)).to_numpy().reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def score(frame, trade_th, margin):
    signal = np.where((frame["cal_trade"] >= trade_th) & ((frame["cal_buy"] - 0.5).abs() >= margin), frame["raw_direction"], 0)
    pnl = frame["domestic_return_pips"].to_numpy(dtype=float) * signal
    active = pnl[signal != 0]
    return {
        "trades": int(len(active)),
        "net_pips": float(active.sum()) if len(active) else 0.0,
        "profit_factor": pf(active),
        "winrate": float((active > 0).mean()) if len(active) else 0.0,
        "signal": signal,
        "pnl": pnl,
    }


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(REPORTS / "pair_specific_selected_models_stability.csv", low_memory=False)
    selected = selected[selected["deployable"].astype(str).str.lower().eq("true")]
    labels = load_labels()
    valid = load_pred("pair_specific_validation_predictions.csv").merge(labels, on=KEYS, how="left", validate="many_to_one")
    oos = load_pred("pair_specific_oos_predictions.csv").merge(labels, on=KEYS, how="left", validate="many_to_one")
    valid = valid[valid["pair"].isin(selected["pair"])].copy()
    oos = oos[oos["pair"].isin(selected["pair"])].copy()
    progress_step("CALIBRATION_DATA", validation_rows=len(valid), oos_rows=len(oos), pairs=valid["pair"].nunique())
    summaries, all_oos = [], []
    for _, choice in selected.iterrows():
        pair, model_name = choice["pair"], choice["model"]
        v = valid[(valid["pair"] == pair) & (valid["model"] == model_name)].copy()
        t = oos[(oos["pair"] == pair) & (oos["model"] == model_name)].copy()
        if v.empty or t.empty:
            continue
        trade_cal = fit_calibrator(v["trade_probability"], v["trade_target"])
        direction_mask = v["trade_target"].eq(1) & v["direction_label"].isin([-1, 1])
        direction_y = (v.loc[direction_mask, "direction_label"].astype(int) == 1).astype(int)
        dir_cal = fit_calibrator(v.loc[direction_mask, "buy_probability"], direction_y)
        v["cal_trade"] = calibrated(trade_cal, v["trade_probability"])
        t["cal_trade"] = calibrated(trade_cal, t["trade_probability"])
        v["cal_buy"] = calibrated(dir_cal, v["buy_probability"])
        t["cal_buy"] = calibrated(dir_cal, t["buy_probability"])
        v["raw_direction"] = np.where(v["cal_buy"] >= 0.5, 1, -1)
        t["raw_direction"] = np.where(t["cal_buy"] >= 0.5, 1, -1)
        best, best_cfg = None, None
        for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            for margin in [0.0, 0.03, 0.06, 0.10, 0.15]:
                m = score(v, th, margin)
                if m["trades"] < 100 or m["profit_factor"] <= 0:
                    continue
                s = m["net_pips"] + 400 * min(m["profit_factor"], 3.0) + 30 * math.log1p(m["trades"])
                if best is None or s > best:
                    best, best_cfg = s, (th, margin, m)
        if best_cfg is None:
            continue
        th, margin, vm = best_cfg
        tm = score(t, th, margin)
        t["signal"] = tm["signal"]
        t["strategy_pips_calibrated"] = tm["pnl"]
        t["model_variant"] = model_name + "_calibrated"
        all_oos.append(t)
        summaries.append({"pair": pair, "model": model_name, "model_variant": model_name + "_calibrated", "validation_trades": vm["trades"], "validation_net_pips": vm["net_pips"], "validation_profit_factor": vm["profit_factor"], "oos_trades": tm["trades"], "oos_net_pips": tm["net_pips"], "oos_profit_factor": tm["profit_factor"], "oos_winrate": tm["winrate"], "trade_threshold": th, "direction_margin": margin})
        progress_step("PAIR_CALIBRATED", pair=pair, model=model_name, oos_trades=tm["trades"], oos_net_pips=f"{tm['net_pips']:.1f}", oos_pf=f"{tm['profit_factor']:.4f}")
    pd.DataFrame(summaries).to_csv(OUT / "calibrated_oos_summary.csv", index=False)
    if all_oos:
        pd.concat(all_oos, ignore_index=True).to_csv(OUT / "calibrated_oos_predictions.csv", index=False)
    (OUT / "README.md").write_text("Calibration fit on validation 2021-2023; OOS 2024-2026 is evaluation only. Baseline artifacts are unchanged.\n", encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
