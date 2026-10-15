from pathlib import Path
import pandas as pd

ART = Path(r"C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\artifacts")
OUT = ART / "paper1_final_results_and_engine_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_WISE = ART / "paper1_model_wise_enhancement_selection" / "best_model_per_pair.csv"
BT_VBT = ART / "backtrader_oos_vectorbt_signal_comparison" / "backtrader_vs_vectorbt_model_pair_comparison.csv"
BT_SUMMARY = ART / "backtrader_oos_vectorbt_signal_comparison" / "backtrader_oos_summary_by_model_pair.csv"
VBT_REPORT = ART / "vectorbt_oos_model_comparison" / "LAPORAN_VECTORBT_OOS_MODEL_COMPARISON.md"
BT_REPORT = ART / "backtrader_oos_vectorbt_signal_comparison" / "LAPORAN_BACKTRADER_VS_VECTORBT_OOS.md"
REGISTRY = Path(r"C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\NewsEvents\RESEARCH1_METHOD_REGISTRY.md")


def md_table(frame):
    if frame is None or frame.empty:
        return "_Tidak ada data._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:,.4f}" if abs(v) < 100 else f"{v:,.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    final = read(MODEL_WISE)
    final_cols = [
        "pair", "model", "variant", "pair_selection_status",
        "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor",
        "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor",
    ]
    final = final[final_cols].copy()
    final["paper_decision"] = "selected"
    final.loc[final["pair"].isin(["GBPUSD"]), "paper_decision"] = "unstable_needs_discussion"
    final.loc[final["pair"].isin(["NZDUSD", "USDCHF"]), "paper_decision"] = "selected_but_final_weak"
    final.loc[final["pair"].isin(["AUDUSD"]), "paper_decision"] = "borderline"
    final_sorted = final.sort_values("oos_dev_net_pips", ascending=False)

    totals = pd.DataFrame([
        {
            "scope": "all_pairs_model_wise_selection",
            "oos_dev_trades": int(final["oos_dev_trades"].sum()),
            "oos_dev_net_pips": float(final["oos_dev_net_pips"].sum()),
            "oos_dev_avg_pf": float(final["oos_dev_profit_factor"].mean()),
            "final_2026_trades": int(final["final_2026_trades"].sum()),
            "final_2026_net_pips": float(final["final_2026_net_pips"].sum()),
            "final_2026_avg_pf": float(final["final_2026_profit_factor"].mean()),
        },
        {
            "scope": "strong_pairs_only",
            "oos_dev_trades": int(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["oos_dev_trades"].sum()),
            "oos_dev_net_pips": float(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["oos_dev_net_pips"].sum()),
            "oos_dev_avg_pf": float(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["oos_dev_profit_factor"].mean()),
            "final_2026_trades": int(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["final_2026_trades"].sum()),
            "final_2026_net_pips": float(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["final_2026_net_pips"].sum()),
            "final_2026_avg_pf": float(final[~final["paper_decision"].isin(["unstable_needs_discussion", "selected_but_final_weak"])]["final_2026_profit_factor"].mean()),
        },
    ])

    cmp = read(BT_VBT)
    bt_summary = read(BT_SUMMARY)
    engine_model_summary = pd.DataFrame()
    engine_pair_top = pd.DataFrame()
    if not cmp.empty:
        engine_model_summary = cmp.groupby("model").agg(
            pairs=("pair", "nunique"),
            bt_trades=("bt_trades", "sum"),
            vbt_trades=("vbt_trades", "sum"),
            bt_net_pips=("bt_net_pips", "sum"),
            vbt_net_pips=("vbt_net_pips", "sum"),
            avg_bt_pf=("bt_profit_factor", "mean"),
            avg_vbt_pf=("vbt_profit_factor", "mean"),
        ).reset_index()
        engine_model_summary["net_pips_diff_bt_minus_vbt"] = engine_model_summary["bt_net_pips"] - engine_model_summary["vbt_net_pips"]
        engine_model_summary = engine_model_summary.sort_values("bt_net_pips", ascending=False)
        engine_pair_top = cmp.sort_values("net_pips_diff", ascending=False)[[
            "model", "pair", "bt_trades", "vbt_trades", "bt_net_pips", "vbt_net_pips", "net_pips_diff", "bt_profit_factor", "vbt_profit_factor"
        ]].head(15)

    final.to_csv(OUT / "paper1_final_model_wise_selected_results.csv", index=False)
    totals.to_csv(OUT / "paper1_final_totals.csv", index=False)
    if not engine_model_summary.empty:
        engine_model_summary.to_csv(OUT / "paper1_backtrader_vectorbt_model_summary.csv", index=False)
        engine_pair_top.to_csv(OUT / "paper1_backtrader_vectorbt_largest_differences.csv", index=False)

    report = [
        "# Laporan Final Sementara Paper 1 - Macro News FX",
        "",
        "Laporan ini merangkum hasil final sementara setelah pencocokan baseline/V1/V2/V3/V4/V5 serta fallback V6 dan V7.",
        "",
        "## Keputusan Metodologis",
        "",
        "- Kandidat utama memakai hasil `model-wise enhancement selection`.",
        "- V6 Two-Stage Meta-Labeling sudah diuji tetapi tidak lolos OOS.",
        "- V7 Drift-Aware Walk-Forward sudah diuji tetapi tidak lolos OOS.",
        "- Final 2026 diperlakukan sebagai audit robustness, bukan dasar tuning atau pemilihan model.",
        "",
        "## Hasil Final Per Pair",
        "",
        md_table(final_sorted),
        "",
        "## Ringkasan Total",
        "",
        md_table(totals),
        "",
        "## Interpretasi Pair",
        "",
        "- Pair paling kuat: `USDJPY`, `EURJPY`, `USDCAD`, `EURUSD`, `GBPJPY`.",
        "- Pair borderline: `AUDUSD`, karena OOS development hampir nol tetapi final 2026 positif.",
        "- Pair final lemah: `NZDUSD` dan `USDCHF`, karena OOS development positif tetapi final 2026 negatif.",
        "- Pair tidak stabil: `GBPUSD`, karena OOS development positif kecil tetapi final 2026 turun besar.",
        "",
        "## Perbandingan Backtrader vs VectorBT",
        "",
        "Perbandingan engine yang tersedia berasal dari eksperimen sebelumnya pada folder `backtrader_oos_vectorbt_signal_comparison` dan `vectorbt_oos_model_comparison`.",
        "Catatan penting: hasil ini tidak identik dengan model-wise final terbaru, karena perbandingan engine lama memakai kumpulan model/signal table sebelumnya. Namun tetap berguna untuk menjelaskan sensitivitas engine backtest.",
        "",
        "### Ringkasan Per Model",
        "",
        md_table(engine_model_summary),
        "",
        "### Selisih Terbesar Backtrader vs VectorBT",
        "",
        md_table(engine_pair_top),
        "",
        "## Kesimpulan Perbandingan Engine",
        "",
        "- VectorBT cenderung menghitung lebih banyak sinyal karena dapat memproses event lebih paralel.",
        "- Backtrader sequential membatasi posisi aktif, sehingga trade count sering lebih rendah.",
        "- Perbedaan besar pada `trade_count` dan `net_pips` terutama muncul saat banyak event berdekatan atau sinyal overlap.",
        "- Untuk paper, VectorBT cocok sebagai screening cepat, Backtrader cocok sebagai validasi sequential Python, dan MT5 Strategy Tester tetap dapat diposisikan sebagai validasi deployment tambahan bila diperlukan.",
        "",
        "## File Sumber",
        "",
        f"- Model-wise final: `{MODEL_WISE}`",
        f"- Backtrader vs VectorBT comparison: `{BT_VBT}`",
        f"- Backtrader summary: `{BT_SUMMARY}`",
        f"- VectorBT report: `{VBT_REPORT}`",
        f"- Backtrader report: `{BT_REPORT}`",
        f"- Registry: `{REGISTRY}`",
    ]
    (OUT / "LAPORAN_PAPER1_FINAL_HASIL_DAN_ENGINE_COMPARISON.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT / "LAPORAN_PAPER1_FINAL_HASIL_DAN_ENGINE_COMPARISON.md")


if __name__ == "__main__":
    main()
