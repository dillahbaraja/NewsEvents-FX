"""Select Paper 1 best model per pair from validated candidate experiments.

Selection uses validation 2021-2023 and OOS development 2024-2025 only.
Final OOS 2026 is reported as an audit column and is not used to select.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "paper1_pair_best_model_selection"
BASE = ARTIFACTS / "paper1_baseline"
V2 = ARTIFACTS / "paper1_v2_volatility_regime"

MIN_OOS_DEV_TRADES = 100
MIN_OOS_DEV_PF = 1.0


def fmt_num(v):
    try:
        x = float(v)
    except Exception:
        return str(v)
    if np.isinf(x):
        return "inf"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    return f"{x:.4f}"


def md_table(frame):
    if frame.empty:
        return "_Tidak ada data._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = [fmt_num(row[c]) for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_baseline_candidates():
    val_path = BASE / "paper1_baseline_results.csv"
    oos_path = BASE / "paper1_oos_loop_results.csv"
    if not val_path.exists() or not oos_path.exists():
        return pd.DataFrame()
    val = pd.read_csv(val_path)
    val = val[val["split"].eq("validation")].copy()
    val = val.rename(columns={
        "trades": "validation_trades",
        "net_pips": "validation_net_pips",
        "profit_factor": "validation_profit_factor",
        "winrate": "validation_winrate",
        "balanced_accuracy": "validation_balanced_accuracy",
        "macro_f1": "validation_macro_f1",
    })
    val["variant"] = "baseline_news_only"
    val["model_key"] = val["model"]

    oos = pd.read_csv(oos_path)
    dev = oos[oos["period"].eq("oos_dev_2024_2025")].rename(columns={
        "trades": "oos_dev_trades",
        "net_pips": "oos_dev_net_pips",
        "profit_factor": "oos_dev_profit_factor",
        "winrate": "oos_dev_winrate",
    })
    final = oos[oos["period"].eq("final_oos_2026")].rename(columns={
        "trades": "final_2026_trades",
        "net_pips": "final_2026_net_pips",
        "profit_factor": "final_2026_profit_factor",
        "winrate": "final_2026_winrate",
    })
    keep_dev = ["pair", "model", "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor", "oos_dev_winrate"]
    keep_final = ["pair", "model", "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor", "final_2026_winrate"]
    out = val.merge(dev[keep_dev], on=["pair", "model"], how="left")
    out = out.merge(final[keep_final], on=["pair", "model"], how="left")
    out["label_threshold"] = np.nan
    out["source"] = "paper1_baseline"
    return out


def load_v2_candidates():
    val_path = V2 / "v2_selected_models.csv"
    oos_path = V2 / "v2_oos_results.csv"
    if not val_path.exists() or not oos_path.exists():
        return pd.DataFrame()
    val = pd.read_csv(val_path)
    val = val.rename(columns={
        "trades": "validation_trades",
        "net_pips": "validation_net_pips",
        "profit_factor": "validation_profit_factor",
        "winrate": "validation_winrate",
        "balanced_accuracy": "validation_balanced_accuracy",
        "macro_f1": "validation_macro_f1",
    })
    val["variant"] = "v2_volatility_regime"
    val["model_key"] = val["model"] + "_v2_volatility_regime"

    oos = pd.read_csv(oos_path)
    dev = oos[oos["period"].eq("oos_dev_2024_2025")].rename(columns={
        "trades": "oos_dev_trades",
        "net_pips": "oos_dev_net_pips",
        "profit_factor": "oos_dev_profit_factor",
        "winrate": "oos_dev_winrate",
    })
    final = oos[oos["period"].eq("final_oos_2026")].rename(columns={
        "trades": "final_2026_trades",
        "net_pips": "final_2026_net_pips",
        "profit_factor": "final_2026_profit_factor",
        "winrate": "final_2026_winrate",
    })
    keep_dev = ["pair", "model", "label_threshold", "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor", "oos_dev_winrate"]
    keep_final = ["pair", "model", "label_threshold", "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor", "final_2026_winrate"]
    out = val.merge(dev[keep_dev], on=["pair", "model", "label_threshold"], how="left")
    out = out.merge(final[keep_final], on=["pair", "model", "label_threshold"], how="left")
    out["source"] = "paper1_v2_volatility_regime"
    return out


def classify_candidates(candidates):
    d = candidates.copy()
    for c in [
        "validation_trades", "validation_net_pips", "validation_profit_factor", "validation_winrate",
        "validation_balanced_accuracy", "validation_macro_f1", "oos_dev_trades", "oos_dev_net_pips",
        "oos_dev_profit_factor", "oos_dev_winrate", "final_2026_trades", "final_2026_net_pips",
        "final_2026_profit_factor", "final_2026_winrate",
    ]:
        d[c] = pd.to_numeric(d.get(c, 0), errors="coerce").fillna(0.0)

    d["selection_eligible"] = (
        (d["oos_dev_trades"] >= MIN_OOS_DEV_TRADES)
        & (d["oos_dev_net_pips"] > 0)
        & (d["oos_dev_profit_factor"] >= MIN_OOS_DEV_PF)
        & (d["validation_net_pips"] > 0)
    )
    d["rejection_reason"] = ""
    d.loc[d["oos_dev_trades"] < MIN_OOS_DEV_TRADES, "rejection_reason"] = "too_few_oos_dev_trades"
    d.loc[(d["rejection_reason"].eq("")) & (d["oos_dev_net_pips"] <= 0), "rejection_reason"] = "negative_oos_dev_net_pips"
    d.loc[(d["rejection_reason"].eq("")) & (d["oos_dev_profit_factor"] < MIN_OOS_DEV_PF), "rejection_reason"] = "oos_dev_pf_below_1"
    d.loc[(d["rejection_reason"].eq("")) & (d["validation_net_pips"] <= 0), "rejection_reason"] = "negative_validation_net_pips"
    d.loc[d["selection_eligible"], "rejection_reason"] = ""

    # Selection score intentionally ignores final 2026. OOS development is dominant;
    # validation only breaks close calls and protects against weak validation fits.
    d["selection_score"] = (
        d["oos_dev_net_pips"]
        + 0.05 * d["validation_net_pips"]
        + 250.0 * d["oos_dev_profit_factor"].clip(upper=5)
        + 50.0 * d["validation_macro_f1"]
    )
    return d


def select_best(candidates):
    rows = []
    for pair, part in candidates.groupby("pair", sort=True):
        eligible = part[part["selection_eligible"]].copy()
        if eligible.empty:
            best = part.sort_values(["oos_dev_net_pips", "validation_net_pips", "oos_dev_profit_factor"], ascending=False).iloc[0].copy()
            best["selection_status"] = "not_deployable"
            best["selection_reason"] = "no_candidate_passed_oos_dev_and_validation_gate"
        else:
            best = eligible.sort_values(["oos_dev_net_pips", "selection_score", "oos_dev_profit_factor"], ascending=False).iloc[0].copy()
            best["selection_status"] = "selected"
            best["selection_reason"] = "best_oos_dev_score_among_validation_positive_candidates"
        rows.append(best)
    return pd.DataFrame(rows)


def write_report(candidates, selected):
    display_cols = [
        "pair", "model", "variant", "label_threshold", "selection_status",
        "validation_trades", "validation_net_pips", "validation_profit_factor",
        "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor",
        "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor",
        "selection_reason",
    ]
    selected_disp = selected[display_cols].sort_values("oos_dev_net_pips", ascending=False)
    eligible = candidates[candidates["selection_eligible"]].copy()
    rejected = candidates[~candidates["selection_eligible"]].copy()
    total = selected[selected["selection_status"].eq("selected")].agg({
        "oos_dev_trades": "sum",
        "oos_dev_net_pips": "sum",
        "final_2026_trades": "sum",
        "final_2026_net_pips": "sum",
    })
    lines = [
        "# Laporan Paper 1 - Pair-Specific Best Model Selection",
        "",
        "Dokumen ini memilih model terbaik per pair dari kandidat Paper 1 yang sudah tersedia.",
        "Pemilihan hanya memakai validation 2021-2023 dan OOS development 2024-2025. Final OOS 2026 hanya audit.",
        "",
        "## Ringkasan",
        "",
        f"- Total kandidat: {len(candidates)}",
        f"- Kandidat eligible: {len(eligible)}",
        f"- Kandidat rejected/not eligible: {len(rejected)}",
        f"- Pair selected deployable: {int((selected['selection_status'] == 'selected').sum())}",
        f"- Pair not deployable: {int((selected['selection_status'] != 'selected').sum())}",
        f"- Selected OOS development trades: {int(total.get('oos_dev_trades', 0))}",
        f"- Selected OOS development net pips: {float(total.get('oos_dev_net_pips', 0.0)):,.1f}",
        f"- Selected final 2026 audit trades: {int(total.get('final_2026_trades', 0))}",
        f"- Selected final 2026 audit net pips: {float(total.get('final_2026_net_pips', 0.0)):,.1f}",
        "",
        "## Model Terpilih Per Pair",
        "",
        md_table(selected_disp),
        "",
        "## Kandidat Eligible",
        "",
        md_table(eligible.sort_values(["pair", "selection_score"], ascending=[True, False])[
            ["pair", "model", "variant", "label_threshold", "validation_net_pips", "oos_dev_net_pips", "oos_dev_profit_factor", "final_2026_net_pips", "selection_score"]
        ]),
        "",
        "## Catatan Metodologis",
        "",
        "- `final_2026_net_pips` tidak digunakan untuk memilih model.",
        "- Jika sebuah pair tidak punya kandidat positif di OOS development dan validation, pair diberi status `not_deployable`.",
        "- V2 volatility-regime diterima hanya jika pair tersebut positif pada OOS development 2024-2025.",
        "- SVM-RBF dan EBM dapat ditambahkan ke file kandidat setelah hasil VPS tersedia.",
    ]
    (OUT / "LAPORAN_PAPER1_PAIR_SPECIFIC_BEST_MODEL.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for loader, name in [(load_baseline_candidates, "baseline"), (load_v2_candidates, "v2_volatility")]:
        progress_step("LOAD_CANDIDATES", source=name)
        part = loader()
        if not part.empty:
            parts.append(part)
            progress_step("LOADED", source=name, rows=len(part))
    if not parts:
        raise RuntimeError("No candidate result files found.")

    candidates = pd.concat(parts, ignore_index=True, sort=False)
    candidates = classify_candidates(candidates)
    selected = select_best(candidates)
    candidates.to_csv(OUT / "paper1_pair_model_candidates.csv", index=False)
    selected.to_csv(OUT / "paper1_pair_best_model_selection.csv", index=False)
    selected[selected["selection_status"].eq("selected")].to_csv(OUT / "paper1_pair_best_model_oos_summary.csv", index=False)
    metadata = {
        "selection_rule": "validation positive + OOS development 2024-2025 positive; final 2026 audit only",
        "min_oos_dev_trades": MIN_OOS_DEV_TRADES,
        "min_oos_dev_profit_factor": MIN_OOS_DEV_PF,
        "sources": ["paper1_baseline", "paper1_v2_volatility_regime"],
        "pending_sources": ["svm_rbf_vps", "ebm_vps"],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(candidates, selected)
    for _, r in selected.sort_values("pair").iterrows():
        progress_step("SELECTED", pair=r["pair"], model=r["model"], variant=r["variant"], status=r["selection_status"], oos_dev=f"{r['oos_dev_net_pips']:.1f}", final_2026=f"{r['final_2026_net_pips']:.1f}")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
