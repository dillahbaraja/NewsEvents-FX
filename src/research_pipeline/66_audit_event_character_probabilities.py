"""Audit validation versus OOS probability scale for event-character models."""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
REPORTS = OUT / "reports"


def summarize(path, label):
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, usecols=["model", "pair", "trade_probability", "buy_probability", "signal"], low_memory=False)
    rows = []
    for (model, pair), g in d.groupby(["model", "pair"], sort=False):
        rows.append({
            "sample": label,
            "model": model,
            "pair": pair,
            "rows": int(len(g)),
            "trade_count": int((g["signal"] != 0).sum()),
            "trade_rate": float((g["signal"] != 0).mean()),
            "trade_prob_mean": float(g["trade_probability"].mean()),
            "trade_prob_p90": float(g["trade_probability"].quantile(0.90)),
            "trade_prob_p95": float(g["trade_probability"].quantile(0.95)),
            "trade_prob_p99": float(g["trade_probability"].quantile(0.99)),
            "trade_prob_max": float(g["trade_probability"].max()),
            "buy_prob_min": float(g["buy_probability"].min()),
            "buy_prob_max": float(g["buy_probability"].max()),
        })
    return pd.DataFrame(rows)


def main():
    started = progress_start(SCRIPT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    validation = summarize(REPORTS / "event_aware_validation_predictions.csv", "validation_2021_2023")
    oos = summarize(REPORTS / "event_aware_oos_predictions.csv", "oos_2024_2026")
    out = pd.concat([validation, oos], ignore_index=True)
    out.to_csv(REPORTS / "event_aware_probability_audit.csv", index=False)
    progress_step("AUDIT_DONE", rows=len(out), output=str(REPORTS / "event_aware_probability_audit.csv"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
