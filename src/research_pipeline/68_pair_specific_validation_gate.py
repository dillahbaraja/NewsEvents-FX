"""Select deployable pair-specific event-character models from validation."""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific"
REPORTS = OUT / "reports"


def main():
    started = progress_start(SCRIPT)
    summary = pd.read_csv(REPORTS / "pair_specific_validation_summary.csv")
    rows = []
    for pair, g in summary.groupby("pair", sort=False):
        ranked = g.sort_values(["net_pips", "profit_factor", "trades"], ascending=[False, False, False]).copy()
        best = ranked.iloc[0].to_dict()
        deployable = (
            best["trades"] >= 100
            and best["net_pips"] > 0
            and best["profit_factor"] > 1.0
        )
        best["deployable"] = bool(deployable)
        best["gate_reason"] = "PASS" if deployable else "FAIL_validation_profit_or_trade_count"
        rows.append(best)
    out = pd.DataFrame(rows).sort_values("net_pips", ascending=False)
    out.to_csv(REPORTS / "pair_specific_selected_models.csv", index=False)
    progress_step("GATE_DONE", selected=int(out["deployable"].sum()), total=len(out), output=str(REPORTS / "pair_specific_selected_models.csv"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
