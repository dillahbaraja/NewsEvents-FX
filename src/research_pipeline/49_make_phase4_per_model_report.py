from common import *

import math


PHASE4_DIR = ARTIFACTS / "phase4_two_stage_evaluation"
PHASE7_DIR = ARTIFACTS / "phase7_selected_two_stage_policy"
OUT_DIR = ARTIFACTS / "phase4_per_model_comparison"

SUMMARY = PHASE4_DIR / "phase4_two_stage_summary.csv"
MONTHLY = PHASE4_DIR / "phase4_two_stage_monthly_cumulative.csv"


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def aggregate_by_model(df):
    totals = df[df["pair"].eq("ALL")].copy()
    pairs = df[~df["pair"].eq("ALL")].copy()
    rows = []
    for _, total in totals.iterrows():
        model = total["model"]
        g = pairs[pairs["model"].eq(model)]
        rows.append({
            "model": model,
            "pairs": int(g["pair"].nunique()),
            "positive_pairs": int((g["net_pips"] > 0).sum()),
            "trades": int(total["trade_count"]),
            "buy_count": int(total["buy_count"]),
            "sell_count": int(total["sell_count"]),
            "net_pips": float(total["net_pips"]),
            "avg_pips": float(total["avg_pips"]),
            "winrate_weighted": float(total["winrate"]),
            "profit_factor_weighted": float(total["profit_factor"]),
        })
    return pd.DataFrame(rows).sort_values("net_pips", ascending=False)


def main():
    print("[PROGRESS] START 49_make_phase4_per_model_report.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    oos = summary[summary["split"].eq("oos_2024_2026")].copy()
    validation = summary[summary["split"].eq("validation_2021_2023")].copy()

    oos_by_model = aggregate_by_model(oos)
    validation_by_model = aggregate_by_model(validation)
    oos_pair_rank = oos[~oos["pair"].eq("ALL")].sort_values(["model", "net_pips"], ascending=[True, False]).copy()
    oos_pair_rank = oos_pair_rank[[
        "model", "pair", "trade_count", "buy_count", "sell_count",
        "net_pips", "avg_pips", "winrate", "profit_factor",
    ]]

    comparison = oos_by_model.merge(
        validation_by_model[["model", "net_pips", "profit_factor_weighted", "positive_pairs"]],
        on="model",
        how="left",
        suffixes=("_oos", "_validation"),
    )
    comparison = comparison.sort_values("net_pips_oos", ascending=False)

    monthly = pd.read_csv(MONTHLY)
    monthly_oos = monthly[monthly["split"].eq("oos_2024_2026")].copy()

    oos_by_model.to_csv(OUT_DIR / "phase4_oos_summary_by_model.csv", index=False)
    validation_by_model.to_csv(OUT_DIR / "phase4_validation_summary_by_model.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase4_validation_vs_oos_by_model.csv", index=False)
    oos_pair_rank.to_csv(OUT_DIR / "phase4_oos_summary_by_model_pair.csv", index=False)
    monthly_oos.to_csv(OUT_DIR / "phase4_oos_monthly_cumulative_by_model_pair.csv", index=False)

    report_lines = [
        "# Phase 4 Per-Model Comparison - Macro News FX",
        "",
        "Laporan ini memperbarui catatan perbandingan antar model individual berdasarkan hasil fase 1-7 terbaru.",
        "",
        "Scope:",
        "",
        "- Training: `2010-2020`.",
        "- Validation/threshold/gate: `2021-2023`.",
        "- OOS test: `2024-01-01` sampai `2026-08-31`.",
        "- Model yang sudah selesai pada Phase 4: `catboost_two_stage`, `elastic_net_two_stage`, `random_forest_two_stage`.",
        "- `SVM_RBF` dan `EBM` belum dimasukkan di laporan ini sampai Phase 8 selesai.",
        "",
        "## Ranking OOS Per Model",
        "",
        md_table(oos_by_model, ".4f"),
        "",
        "## Validation vs OOS Per Model",
        "",
        md_table(comparison, ".4f"),
        "",
        "## OOS Per Model Per Pair",
        "",
        md_table(oos_pair_rank.sort_values(["model", "net_pips"], ascending=[True, False]), ".4f"),
        "",
        "## Interpretasi",
        "",
        "- `elastic_net_two_stage` adalah model individual terbaik secara agregat OOS pada Phase 4.",
        "- `elastic_net_two_stage` kuat pada pair JPY, terutama `EURJPY` dan `USDJPY`, tetapi lemah pada beberapa pair lain.",
        "- `catboost_two_stage` berada di posisi kedua secara agregat dan relatif lebih merata dibanding model lain.",
        "- `random_forest_two_stage` negatif secara agregat OOS, walaupun kuat pada `USDJPY`, `GBPJPY`, dan `USDCHF`.",
        "- Phase 7 berbeda dari tabel ini karena Phase 7 memilih model terbaik per pair memakai validation stability gate, bukan memakai satu model global untuk semua pair.",
    ]
    report = "\n".join(report_lines) + "\n"
    (OUT_DIR / "LAPORAN_PHASE4_PER_MODEL_COMPARISON.md").write_text(report, encoding="utf-8")
    (PHASE7_DIR / "LAPORAN_PHASE4_PER_MODEL_COMPARISON.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] wrote={OUT_DIR / 'LAPORAN_PHASE4_PER_MODEL_COMPARISON.md'}", flush=True)
    print("[PROGRESS] DONE 49_make_phase4_per_model_report.py", flush=True)


if __name__ == "__main__":
    main()
