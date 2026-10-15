"""Export clean BUY/SELL signals for engine validation."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
SRC = ARTIFACTS / "event_character_pair_specific" / "reports" / "pair_specific_oos_predictions.csv"
OUT = ARTIFACTS / "event_character_pair_specific" / "engine_validation"


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC, low_memory=False, parse_dates=["event_time"])
    d = d[d["signal"].ne(0)].copy()
    d["side"] = np.where(d["signal"] > 0, "BUY", "SELL")
    d["entry_time"] = d["event_time"].dt.ceil("h")
    d["holding_bars"] = pd.to_numeric(d["horizon_hours"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    d["confidence"] = (pd.to_numeric(d["buy_probability"], errors="coerce") - 0.5).abs() * 2.0
    keep = [
        "entry_time", "event_time", "pair", "model", "side", "signal", "holding_bars",
        "horizon_hours", "CountryCode", "Name", "event_name_norm", "trade_probability",
        "buy_probability", "confidence", "domestic_return_pips", "deployment_policy",
    ]
    keep = [c for c in keep if c in d.columns]
    out = d[keep].sort_values(["pair", "entry_time", "model"]).copy()
    out.to_csv(OUT / "pair_specific_event_character_signals_oos.csv", index=False)
    progress_step("SIGNALS_EXPORTED", rows=len(out), pairs=out["pair"].nunique(), output=str(OUT / "pair_specific_event_character_signals_oos.csv"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
