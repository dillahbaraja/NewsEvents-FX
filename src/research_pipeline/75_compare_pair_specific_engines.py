"""Compare event audit, VectorBT, and sequential Backtrader-style results."""
from pathlib import Path
import os

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific" / "engine_validation"
RUN_NAME = os.environ.get("PAIR_ENGINE_RUN_NAME", "raw")


def md_table(df, max_rows=50):
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


def main():
    started = progress_start(SCRIPT)
    v = pd.read_csv(OUT / f"vectorbt_pair_specific_summary_{RUN_NAME}.csv")
    b = pd.read_csv(OUT / f"backtrader_pair_specific_summary_{RUN_NAME}.csv")
    merged = v.merge(
        b,
        on=["pair", "model"],
        how="outer",
        suffixes=("_vectorbt", "_backtrader"),
    )
    merged["net_pips_gap_backtrader_minus_event"] = merged["net_pips"] - merged["event_audit_net_pips"]
    merged = merged.sort_values("net_pips", ascending=False)
    merged.to_csv(OUT / f"engine_comparison_pair_specific_{RUN_NAME}.csv", index=False)
    totals = pd.DataFrame([
        {
            "engine": "event_audit_vectorbt_input",
            "trades": int(v["event_audit_trades"].sum()),
            "net_pips": float(v["event_audit_net_pips"].sum()),
            "avg_profit_factor": float(v["event_audit_profit_factor"].mean()),
        },
        {
            "engine": "backtrader_sequential",
            "trades": int(b["trades"].sum()),
            "net_pips": float(b["net_pips"].sum()),
            "avg_profit_factor": float(b["profit_factor"].mean()),
        },
    ])
    totals.to_csv(OUT / f"engine_comparison_totals_{RUN_NAME}.csv", index=False)
    report = f"""# Engine Validation - Pair-Specific Event Character Aware

Validasi ini membandingkan tiga lapisan:

- Event-level audit dari signal table.
- VectorBT portfolio simulation.
- Backtrader-style sequential simulation, satu posisi aktif per pair.

## Total

{md_table(totals, 20)}

## Per Pair

{md_table(merged, 80)}

## Catatan

Backtrader-style sequential biasanya memiliki trade lebih sedikit karena event overlap dilewati sampai posisi aktif selesai. Jika gap terlalu besar, hasil event audit tidak boleh dipakai sebagai klaim trading final tanpa catatan.
"""
    (OUT / f"LAPORAN_ENGINE_VALIDATION_PAIR_SPECIFIC_EVENT_CHARACTER_{RUN_NAME}.md").write_text(report, encoding="utf-8")
    progress_step("REPORT_WRITTEN", path=str(OUT / f"LAPORAN_ENGINE_VALIDATION_PAIR_SPECIFIC_EVENT_CHARACTER_{RUN_NAME}.md"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
