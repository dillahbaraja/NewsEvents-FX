"""Build an overlap-aware final signal policy from pair-specific OOS signals.

The previous engine validation showed that many signals overlap. This filter
keeps only the highest-confidence signal inside each active holding window per
pair, then re-runs engine validation on the reduced signal set.
"""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
BASE = ARTIFACTS / "event_character_pair_specific" / "engine_validation"
INFILE = BASE / "pair_specific_event_character_signals_oos.csv"
OUTFILE = BASE / "pair_specific_event_character_signals_oos_overlap_aware.csv"


def filter_pair(d):
    kept = []
    skipped = 0
    d = d.sort_values(["entry_time", "confidence", "trade_probability"], ascending=[True, False, False]).copy()
    i = 0
    rows = d.to_dict("records")
    while i < len(rows):
        current_time = pd.Timestamp(rows[i]["entry_time"])
        candidates = []
        j = i
        while j < len(rows) and pd.Timestamp(rows[j]["entry_time"]) == current_time:
            candidates.append(rows[j])
            j += 1
        best = sorted(candidates, key=lambda r: (float(r.get("confidence", 0)), float(r.get("trade_probability", 0))), reverse=True)[0]
        kept.append(best)
        exit_time = current_time + pd.Timedelta(hours=int(best.get("holding_bars", 1) or 1))
        i = j
        while i < len(rows) and pd.Timestamp(rows[i]["entry_time"]) < exit_time:
            skipped += 1
            i += 1
    out = pd.DataFrame(kept)
    out["overlap_filter_skipped_after_entry"] = skipped
    return out


def main():
    started = progress_start(SCRIPT)
    d = pd.read_csv(INFILE, parse_dates=["entry_time", "event_time"], low_memory=False)
    parts = []
    before = len(d)
    for pair, g in d.groupby("pair", sort=False):
        progress_step("OVERLAP_FILTER_PAIR_START", pair=pair, rows=len(g))
        part = filter_pair(g)
        parts.append(part)
        progress_step("OVERLAP_FILTER_PAIR_DONE", pair=pair, before=len(g), after=len(part))
    out = pd.concat(parts, ignore_index=True).sort_values(["pair", "entry_time"])
    out.to_csv(OUTFILE, index=False)
    progress_step("OVERLAP_FILTER_DONE", before=before, after=len(out), output=str(OUTFILE))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
