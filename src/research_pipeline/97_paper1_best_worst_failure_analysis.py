from pathlib import Path
import json
import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_start, progress_done, progress_step

SCRIPT = Path(__file__).name
FINAL_DIR = ARTIFACTS / "paper1_final_results_and_engine_comparison"
REVIEW_DIR = ARTIFACTS / "paper1_reviewer_robustness_audits"
OUT = ARTIFACTS / "paper1_best_worst_failure_case_analysis"

ALL_CANDIDATES = FINAL_DIR / "paper1_final_all_model_candidates_per_pair.csv"
FINAL_SELECTED = FINAL_DIR / "paper1_final_selected_model_per_pair.csv"
COST = REVIEW_DIR / "paper1_transaction_cost_sensitivity.csv"
DRIFT = REVIEW_DIR / "paper1_drift_audit_2026.csv"
SIGNAL = REVIEW_DIR / "paper1_signal_distribution_audit.csv"


def num(s):
    return pd.to_numeric(s, errors="coerce")


def classify_candidate(row):
    dev = float(row.get("oos_dev_net_pips", 0) or 0)
    audit = float(row.get("final_2026_net_pips", 0) or 0)
    pf = float(row.get("oos_dev_profit_factor", 0) or 0)
    flags = []
    if dev > 1000 and pf > 1.03:
        flags.append("strong_oos_dev")
    if abs(dev) < 100:
        flags.append("weak_near_zero_oos_dev")
    if dev < 0:
        flags.append("negative_oos_dev")
    if dev > 0 and audit < 0:
        flags.append("drift_sign_flip_2026")
    if audit < dev * 0.25:
        flags.append("large_2026_drop")
    if pf < 1.0:
        flags.append("pf_below_1")
    if not flags:
        flags.append("moderate")
    return ";".join(flags)


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)

    allc = pd.read_csv(ALL_CANDIDATES)
    final = pd.read_csv(FINAL_SELECTED)
    for d in (allc, final):
        for c in ["oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor", "oos_dev_winrate", "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor", "final_2026_winrate"]:
            if c in d.columns:
                d[c] = num(d[c]).fillna(0.0)

    allc["failure_flags"] = allc.apply(classify_candidate, axis=1)
    allc["selection_score"] = allc["oos_dev_net_pips"] + 1000 * (allc["oos_dev_profit_factor"] - 1.0)
    allc = allc.sort_values(["oos_dev_net_pips", "oos_dev_profit_factor"], ascending=False)
    allc.to_csv(OUT / "paper1_full_candidate_ranking.csv", index=False)

    best = allc.head(15).copy()
    worst = allc.tail(15).sort_values("oos_dev_net_pips").copy()
    best["case_type"] = "best_oos_dev"
    worst["case_type"] = "worst_oos_dev"
    best_worst = pd.concat([best, worst], ignore_index=True)
    best_worst.to_csv(OUT / "paper1_best_worst_cases.csv", index=False)

    final["failure_flags"] = final.apply(classify_candidate, axis=1)
    cost = pd.read_csv(COST) if COST.exists() else pd.DataFrame()
    drift = pd.read_csv(DRIFT) if DRIFT.exists() else pd.DataFrame()
    signal = pd.read_csv(SIGNAL) if SIGNAL.exists() else pd.DataFrame()

    taxonomy_rows = []
    for _, r in final.iterrows():
        pair = r["pair"]
        flags = str(r["failure_flags"]).split(";")
        cost_sensitive = True
        if not cost.empty:
            dev_cost05 = cost[(cost["period"].eq("oos_dev_2024_2025")) & (num(cost["cost_pips_per_trade"]).eq(0.5))]
            if not dev_cost05.empty and float(dev_cost05["net_pips_after_cost"].sum()) < 0:
                flags.append("cost_sensitive_global")
        if not drift.empty:
            dd = drift[drift["pair"].eq(pair)]
            if not dd.empty and str(dd.iloc[0].get("drift_flag", "")):
                flags.append(str(dd.iloc[0]["drift_flag"]))
        taxonomy_rows.append({
            "pair": pair,
            "model": r["model"],
            "variant": r["variant"],
            "oos_dev_net_pips": r["oos_dev_net_pips"],
            "final_2026_net_pips": r["final_2026_net_pips"],
            "failure_taxonomy": ";".join(sorted(set([f for f in flags if f]))),
            "publication_note": "use_as_failure_case" if (r["oos_dev_net_pips"] < 100 or r["final_2026_net_pips"] < 0) else "use_as_supporting_case",
        })
    taxonomy = pd.DataFrame(taxonomy_rows)
    taxonomy.to_csv(OUT / "paper1_failure_taxonomy.csv", index=False)

    model_summary = allc.groupby(["model", "variant"], as_index=False).agg(
        candidates=("pair", "count"),
        oos_dev_net_pips=("oos_dev_net_pips", "sum"),
        final_2026_net_pips=("final_2026_net_pips", "sum"),
        avg_oos_dev_pf=("oos_dev_profit_factor", "mean"),
        positive_oos_pairs=("oos_dev_net_pips", lambda s: int((s > 0).sum())),
        negative_oos_pairs=("oos_dev_net_pips", lambda s: int((s < 0).sum())),
    ).sort_values("oos_dev_net_pips", ascending=False)
    model_summary.to_csv(OUT / "paper1_model_variant_strength_weakness_summary.csv", index=False)

    eval_lines = []
    total_dev = final["oos_dev_net_pips"].sum()
    total_audit = final["final_2026_net_pips"].sum()
    cost05 = None
    if not cost.empty:
        x = cost[(cost["period"].eq("oos_dev_2024_2025")) & (num(cost["cost_pips_per_trade"]).eq(0.5))]
        if not x.empty:
            cost05 = float(x["net_pips_after_cost"].sum())
    strong_pairs = int((final["oos_dev_net_pips"] > 1000).sum())
    audit_negative = int((final["final_2026_net_pips"] < 0).sum())
    eval_lines.append("# Paper 1 - Best/Worst dan Evaluasi Kelayakan Publikasi\n")
    eval_lines.append("## Ringkasan Kuantitatif\n")
    eval_lines.append(f"- Total OOS development 2024-2025: `{total_dev:,.1f}` pips.")
    eval_lines.append(f"- Total final audit 2026: `{total_audit:,.1f}` pips.")
    eval_lines.append(f"- Pair kuat pada OOS development (>1000 pips): `{strong_pairs}/9`.")
    eval_lines.append(f"- Pair negatif pada audit 2026: `{audit_negative}/9`.")
    if cost05 is not None:
        eval_lines.append(f"- Setelah biaya 0.5 pip/trade pada OOS development: `{cost05:,.1f}` pips.")
    eval_lines.append("\n## Best Cases\n")
    for _, r in best.head(8).iterrows():
        eval_lines.append(f"- `{r['pair']}` - `{r['model']}` / `{r['variant']}`: OOS dev `{float(r['oos_dev_net_pips']):,.1f}` pips, PF `{float(r['oos_dev_profit_factor']):.4f}`, audit 2026 `{float(r['final_2026_net_pips']):,.1f}` pips.")
    eval_lines.append("\n## Worst Cases\n")
    for _, r in worst.head(8).iterrows():
        eval_lines.append(f"- `{r['pair']}` - `{r['model']}` / `{r['variant']}`: OOS dev `{float(r['oos_dev_net_pips']):,.1f}` pips, PF `{float(r['oos_dev_profit_factor']):.4f}`, audit 2026 `{float(r['final_2026_net_pips']):,.1f}` pips.")
    eval_lines.append("\n## Failure Taxonomy\n")
    for _, r in taxonomy.iterrows():
        eval_lines.append(f"- `{r['pair']}`: `{r['failure_taxonomy']}`.")
    eval_lines.append("\n## Evaluasi Kelayakan Publikasi Scopus\n")
    eval_lines.append("Hasil saat ini layak dikembangkan menjadi artikel Scopus Q1/Q2 jika framing penelitian ditempatkan sebagai `predictive modeling/event-response framework`, bukan sebagai sistem trading live. Novelty metodologis cukup kuat karena menggunakan pair-specific event-character-aware model selection dan audit 2026. Namun, paper harus secara eksplisit mengakui kelemahan transaction cost sensitivity dan drift 2026.")
    eval_lines.append("\n### Layak Dipublikasikan Jika\n")
    eval_lines.append("- Klaim utama adalah prediktabilitas dan heterogenitas respons event, bukan profit live.")
    eval_lines.append("- Hasil terbaik dan terburuk sama-sama dilaporkan.")
    eval_lines.append("- Transaction cost sensitivity dimasukkan sebagai limitation utama.")
    eval_lines.append("- Naive surprise benchmark dan/atau event-study regression ditambahkan sebagai pembanding tambahan jika memungkinkan.")
    eval_lines.append("\n### Belum Aman Jika\n")
    eval_lines.append("- Manuskrip mengklaim sistem trading siap live.")
    eval_lines.append("- Hanya menampilkan pair/model terbaik tanpa failure cases.")
    eval_lines.append("- Tidak menjelaskan bahwa biaya 0.5 pip sudah membuat aggregate menjadi negatif.")
    (OUT / "PAPER1_BEST_WORST_FAILURE_CASE_ANALYSIS.md").write_text("\n".join(eval_lines), encoding="utf-8")

    metadata = {
        "script": SCRIPT,
        "inputs": [str(ALL_CANDIDATES), str(FINAL_SELECTED), str(COST), str(DRIFT), str(SIGNAL)],
        "outputs": [p.name for p in OUT.iterdir() if p.is_file()],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
