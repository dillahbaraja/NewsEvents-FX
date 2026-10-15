"""Create a concise Markdown report for event-character-aware stage outputs."""
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
REPORTS = OUT / "reports"
DATA = OUT / "data"
PROFILE = OUT / "profiles"
CLUSTERS = OUT / "clusters"


def md_table(df, max_rows=20):
    if df is None or df.empty:
        return "_Belum ada data._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
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

    dataset_summary = pd.read_csv(DATA / "event_aware_dataset_summary.csv")
    profile_best = pd.read_csv(PROFILE / "event_profiles_best_horizon_train_2010_2020.csv")
    cluster_summary = pd.read_csv(CLUSTERS / "event_cluster_summary.csv") if (CLUSTERS / "event_cluster_summary.csv").exists() else pd.DataFrame()
    pair_summary = pd.read_csv(REPORTS / "event_aware_validation_pair_summary.csv") if (REPORTS / "event_aware_validation_pair_summary.csv").exists() else pd.DataFrame()
    oos_pair_summary = pd.read_csv(REPORTS / "event_aware_oos_pair_summary.csv") if (REPORTS / "event_aware_oos_pair_summary.csv").exists() else pd.DataFrame()
    oos_model_summary = pd.read_csv(REPORTS / "event_aware_oos_model_summary.csv") if (REPORTS / "event_aware_oos_model_summary.csv").exists() else pd.DataFrame()

    top_profiles = profile_best.sort_values("best_event_score", ascending=False)[
        ["pair", "CountryCode", "event_name_norm", "horizon_hours", "best_direction", "trade_count", "original_net_pips", "inverse_net_pips", "best_event_score"]
    ]
    if not pair_summary.empty:
        pair_summary = pair_summary.sort_values(["model", "net_pips"], ascending=[True, False])
    if not oos_pair_summary.empty:
        oos_pair_summary = oos_pair_summary.sort_values(["model", "net_pips"], ascending=[True, False])
    if not oos_model_summary.empty:
        oos_model_summary = oos_model_summary.sort_values("net_pips", ascending=False)

    report = f"""# Laporan Event Character Aware Macro News FX

Dokumen ini adalah hasil tahapan baru berbasis karakter event. Ini belum mengganti baseline lama; posisinya adalah eksperimen metodologi baru untuk memperkuat model macro news FX.

## Kontrak Metodologi

- Unit analisis: `pair + country + event_name`.
- Profil historis dibuat dari `2010-2020` untuk validasi `2021-2023`.
- Data `2024+` tidak dipakai untuk profiling, clustering, training, threshold, atau seleksi model pada tahap ini.
- Horizon yang diuji: `1H`, `2H`, `4H`, `24H`.
- Model cepat yang dilatih: `elastic_net_event_character`, `random_forest_event_character`, dan `catboost_event_character` jika CatBoost tersedia.
- SVM RBF dan EBM disiapkan sebagai tahap berat terpisah, bukan dijalankan otomatis di laporan ini.

## Ringkasan Dataset

{md_table(dataset_summary, 30)}

## Cluster Karakter Event

{md_table(cluster_summary, 20)}

## Top Event Profile Berdasarkan Reliability Score

{md_table(top_profiles, 30)}

## Validasi Model Per Pair 2021-2023

{md_table(pair_summary, 60)}

## OOS Model Summary 2024-08.2026

{md_table(oos_model_summary, 20)}

## OOS Per Pair 2024-08.2026

{md_table(oos_pair_summary, 80)}

## Interpretasi Awal

Hasil ini harus dibaca sebagai eksperimen metodologi awal. OOS memakai fallback `validation_trade_rate_rank_fallback` jika threshold probabilitas absolut dari validasi menghasilkan nol trade pada datatest. Fallback ini memakai frekuensi trade dari validation dan ranking skor probabilitas OOS, bukan profit OOS.

## Artefak

- `data/event_aware_modeling_dataset.csv`
- `profiles/event_profiles_by_horizon_train_2010_2020.csv`
- `profiles/event_profiles_best_horizon_train_2010_2020.csv`
- `clusters/event_clusters.csv`
- `reports/event_aware_validation_predictions.csv`
- `reports/event_aware_validation_pair_summary.csv`
- `reports/event_aware_oos_predictions.csv`
- `reports/event_aware_oos_pair_summary.csv`
- `reports/event_aware_oos_model_summary.csv`
"""
    (REPORTS / "LAPORAN_EVENT_CHARACTER_AWARE.md").write_text(report, encoding="utf-8")
    progress_step("REPORT_WRITTEN", path=str(REPORTS / "LAPORAN_EVENT_CHARACTER_AWARE.md"))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
