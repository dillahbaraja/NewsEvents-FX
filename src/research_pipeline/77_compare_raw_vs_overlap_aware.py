"""Compare raw versus overlap-aware engine validation outputs."""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific" / "engine_validation"


def md_table(df, max_rows=80):
    if df is None or df.empty:
        return "_Belum ada data._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_totals(name):
    p = OUT / f"engine_comparison_totals_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["run"] = name
    return d


def main():
    started = progress_start(SCRIPT)
    totals = pd.concat([load_totals("raw"), load_totals("overlap_aware")], ignore_index=True)
    totals.to_csv(OUT / "engine_comparison_raw_vs_overlap_totals.csv", index=False)

    raw = pd.read_csv(OUT / "engine_comparison_pair_specific_raw.csv") if (OUT / "engine_comparison_pair_specific_raw.csv").exists() else pd.DataFrame()
    filt = pd.read_csv(OUT / "engine_comparison_pair_specific_overlap_aware.csv") if (OUT / "engine_comparison_pair_specific_overlap_aware.csv").exists() else pd.DataFrame()
    if not raw.empty:
        raw["run"] = "raw"
    if not filt.empty:
        filt["run"] = "overlap_aware"
    pair = pd.concat([raw, filt], ignore_index=True)
    pair.to_csv(OUT / "engine_comparison_raw_vs_overlap_pair.csv", index=False)

    report = f"""# Raw vs Overlap-Aware Engine Validation

Tahap ini menguji apakah profit event-character-aware bertahan ketika sinyal yang bertumpuk dikurangi sejak awal.

## Total

{md_table(totals, 30)}

## Per Pair

{md_table(pair, 100)}

## Interpretasi

Jika overlap-aware memperbaiki Backtrader sequential tetapi mengurangi event-audit secara besar, berarti model sebelumnya terlalu bergantung pada sinyal bertumpuk. Jika hasil tetap lemah, tahap berikutnya harus masuk ke training overlap-aware, bukan hanya filtering setelah prediksi.
"""
    (OUT / "LAPORAN_RAW_VS_OVERLAP_AWARE_ENGINE_VALIDATION.md").write_text(report, encoding="utf-8")
    progress_step("REPORT_WRITTEN", path=str(OUT / "LAPORAN_RAW_VS_OVERLAP_AWARE_ENGINE_VALIDATION.md"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
