"""Create Markdown report for pair-specific event-character-aware models."""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_pair_specific"
REPORTS = OUT / "reports"


def md_table(df, max_rows=40):
    if df is None or df.empty:
        return "_Belum ada data._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    started = progress_start(SCRIPT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(REPORTS / "pair_specific_validation_summary.csv") if (REPORTS / "pair_specific_validation_summary.csv").exists() else pd.DataFrame()
    selected = pd.read_csv(REPORTS / "pair_specific_selected_models.csv") if (REPORTS / "pair_specific_selected_models.csv").exists() else pd.DataFrame()
    selected_stability = pd.read_csv(REPORTS / "pair_specific_selected_models_stability.csv") if (REPORTS / "pair_specific_selected_models_stability.csv").exists() else pd.DataFrame()
    stability_diag = pd.read_csv(REPORTS / "pair_specific_stability_diagnostics.csv") if (REPORTS / "pair_specific_stability_diagnostics.csv").exists() else pd.DataFrame()
    oos = pd.read_csv(REPORTS / "pair_specific_oos_summary.csv") if (REPORTS / "pair_specific_oos_summary.csv").exists() else pd.DataFrame()
    if not validation.empty:
        validation = validation.sort_values(["pair", "net_pips"], ascending=[True, False])
    if not selected.empty:
        selected = selected.sort_values("net_pips", ascending=False)
    if not selected_stability.empty:
        selected_stability = selected_stability.sort_values(["deployable", "stability_score"], ascending=[False, False])
    if not oos.empty:
        oos = oos.sort_values("net_pips", ascending=False)
    total = pd.DataFrame()
    if not oos.empty:
        total = oos.groupby("model").agg(trades=("trades", "sum"), net_pips=("net_pips", "sum"), avg_pf=("profit_factor", "mean")).reset_index().sort_values("net_pips", ascending=False)

    report = f"""# Laporan Pair-Specific Event Character Aware

Tahap ini melatih model berbeda untuk setiap pair. Tujuannya agar karakter event seperti `US CPI y/y` tidak dipaksa memiliki efek sama pada `EURUSD`, `USDJPY`, `USDCAD`, dan pair lain.

## Kontrak

- Unit model: `pair-specific`.
- Unit event profile: `pair + country + event_name`.
- Training: `2010-2020`.
- Validation/gate: `2021-2023`.
- OOS: `2024-08.2026`.
- Model cepat: `Elastic Net`, `Random Forest`, `CatBoost`.
- OOS tidak dipakai untuk memilih model atau threshold.

## Selected Models Dari Validation

{md_table(selected, 30)}

## Selected Models Dengan Stability Gate

{md_table(selected_stability, 30)}

## Stability Diagnostics

{md_table(stability_diag.sort_values(["pair", "stability_score"], ascending=[True, False]) if not stability_diag.empty else stability_diag, 80)}

## Validation Semua Kandidat

{md_table(validation, 80)}

## OOS Total Per Model

{md_table(total, 20)}

## OOS Per Pair

{md_table(oos, 40)}

## Catatan

Jika suatu model-pair menghasilkan nol trade dengan threshold absolut, tester memakai fallback ranking berdasarkan frekuensi trade validation. Fallback ini tetap tidak melihat profit OOS.
"""
    (REPORTS / "LAPORAN_PAIR_SPECIFIC_EVENT_CHARACTER_AWARE.md").write_text(report, encoding="utf-8")
    progress_step("REPORT_WRITTEN", path=str(REPORTS / "LAPORAN_PAIR_SPECIFIC_EVENT_CHARACTER_AWARE.md"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
