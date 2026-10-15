"""Paper 1: model-wise enhancement selection from available OOS results.

This script compares baseline/V1/V2/V3/V4/V5 at pair + model level.
V3/V4/V5 currently have OOS results only for their selected model per pair,
so missing model-wise OOS combinations are reported explicitly.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "paper1_model_wise_enhancement_selection"
REPORT = OUT / "LAPORAN_PAPER1_MODEL_WISE_ENHANCEMENT_SELECTION.md"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"]
MODELS = ["elastic_net", "random_forest", "catboost", "svm_rbf", "ebm"]

ROOTS = {
    "baseline_news_only": ARTIFACTS / "paper1_pair_best_model_selection",
    "v1_standardized_surprise": ARTIFACTS / "paper1_v1_standardized_surprise",
    "v2_volatility_regime": ARTIFACTS / "paper1_v2_volatility_regime",
    "v3_sparse_event_selection": ARTIFACTS / "paper1_v3_sparse_event_selection",
    "v4_jump_volatility_labeling": ARTIFACTS / "paper1_v4_jump_volatility_labeling",
    "v5_event_congestion_filter": ARTIFACTS / "paper1_v5_event_congestion_filter",
}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_model(value):
    s = str(value).strip().lower()
    aliases = {
        "elasticnet": "elastic_net",
        "elastic_net_multiclass": "elastic_net",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "catboostclassifier": "catboost",
        "svm": "svm_rbf",
        "svm_rbf_exact": "svm_rbf",
    }
    return aliases.get(s, s)


def add_score_columns(d):
    out = d.copy()
    for c in [
        "validation_net_pips", "validation_profit_factor", "validation_trades",
        "oos_dev_net_pips", "oos_dev_profit_factor", "oos_dev_trades",
        "final_2026_net_pips", "final_2026_profit_factor", "final_2026_trades",
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["validation_positive"] = (out["validation_net_pips"] > 0) & (out["validation_profit_factor"] > 1.0)
    out["oos_dev_positive"] = (out["oos_dev_net_pips"] > 0) & (out["oos_dev_profit_factor"] > 1.0)
    out["final_positive"] = (out["final_2026_net_pips"] > 0) & (out["final_2026_profit_factor"] > 1.0)
    out["unstable_final"] = out["oos_dev_positive"] & (out["final_2026_net_pips"] < -1000)
    out["accepted_from_v3_v4_v5"] = (
        out["variant"].isin(["v3_sparse_event_selection", "v4_jump_volatility_labeling", "v5_event_congestion_filter"])
        & out["validation_positive"]
        & out["oos_dev_positive"]
    )
    out["needs_v6_v7"] = ~(out["validation_positive"] & out["oos_dev_positive"])
    out["selection_score"] = (
        out["oos_dev_net_pips"].fillna(-1e9)
        + 300.0 * out["oos_dev_profit_factor"].fillna(0).clip(upper=5)
        + 0.20 * out["final_2026_net_pips"].fillna(0)
    )
    out.loc[out["unstable_final"], "selection_score"] -= 1000.0
    return out


def load_baseline_candidates():
    path = ROOTS["baseline_news_only"] / "paper1_pair_model_candidates.csv"
    d = read_csv(path)
    if d.empty:
        return d
    d = d.copy()
    d["variant"] = d["variant"].fillna("baseline_news_only")
    d["model"] = d["model"].map(normalize_model)
    d["source_file"] = str(path)
    keep = [
        "pair", "model", "variant",
        "validation_trades", "validation_net_pips", "validation_profit_factor", "validation_winrate",
        "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor", "oos_dev_winrate",
        "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor", "final_2026_winrate",
        "source", "source_file",
    ]
    return d[[c for c in keep if c in d.columns]]


def load_simple_oos(variant, selected_path=None, oos_path=None, extra_cols=None):
    selected = read_csv(selected_path) if selected_path else pd.DataFrame()
    oos = read_csv(oos_path) if oos_path else pd.DataFrame()
    if oos.empty:
        return pd.DataFrame()
    oos = oos.copy()
    if "variant" not in oos.columns:
        oos["variant"] = variant
    oos["model"] = oos["model"].map(normalize_model)
    key_cols = ["pair", "model", "variant"]
    rows = []
    for keys, g in oos.groupby(key_cols, dropna=False):
        row = {"pair": keys[0], "model": keys[1], "variant": keys[2]}
        for _, r in g.iterrows():
            period = str(r.get("period", ""))
            prefix = "final_2026" if period == "final_oos_2026" else "oos_dev" if period == "oos_dev_2024_2025" else period
            if prefix:
                row[f"{prefix}_trades"] = r.get("trades")
                row[f"{prefix}_net_pips"] = r.get("net_pips")
                row[f"{prefix}_profit_factor"] = r.get("profit_factor")
                row[f"{prefix}_winrate"] = r.get("winrate")
        rows.append(row)
    out = pd.DataFrame(rows)

    if not selected.empty:
        selected = selected.copy()
        if "variant" not in selected.columns:
            selected["variant"] = variant
        selected["model"] = selected["model"].map(normalize_model)
        selected = selected.rename(columns={
            "trades": "validation_trades",
            "net_pips": "validation_net_pips",
            "profit_factor": "validation_profit_factor",
            "winrate": "validation_winrate",
        })
        val_cols = [
            "pair", "model", "variant", "validation_trades", "validation_net_pips",
            "validation_profit_factor", "validation_winrate", "balanced_accuracy", "macro_f1",
        ]
        if extra_cols:
            val_cols += extra_cols
        selected = selected[[c for c in val_cols if c in selected.columns]]
        out = out.merge(selected, on=["pair", "model", "variant"], how="left")
    out["source"] = variant
    out["source_file"] = str(oos_path)
    return out


def load_all_available():
    frames = [load_baseline_candidates()]
    frames.append(load_simple_oos(
        "v1_standardized_surprise",
        selected_path=None,
        oos_path=ROOTS["v1_standardized_surprise"] / "v1_oos_results.csv",
    ))
    frames.append(load_simple_oos(
        "v2_volatility_regime",
        selected_path=None,
        oos_path=ROOTS["v2_volatility_regime"] / "v2_oos_results.csv",
        extra_cols=["label_threshold"],
    ))
    frames.append(load_simple_oos(
        "v3_sparse_event_selection",
        selected_path=ROOTS["v3_sparse_event_selection"] / "v3_selected_models.csv",
        oos_path=ROOTS["v3_sparse_event_selection"] / "v3_oos_results.csv",
        extra_cols=["selected_event_units", "min_events", "min_profit_factor", "min_positive_year_ratio", "min_avg_pips"],
    ))
    frames.append(load_simple_oos(
        "v4_jump_volatility_labeling",
        selected_path=ROOTS["v4_jump_volatility_labeling"] / "v4_selected_models.csv",
        oos_path=ROOTS["v4_jump_volatility_labeling"] / "v4_oos_results.csv",
        extra_cols=["barrier_multiplier"],
    ))
    frames.append(load_simple_oos(
        "v5_event_congestion_filter",
        selected_path=ROOTS["v5_event_congestion_filter"] / "v5_selected_models.csv",
        oos_path=ROOTS["v5_event_congestion_filter"] / "v5_oos_results.csv",
        extra_cols=["gate"],
    ))
    d = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True, sort=False)
    d = d[d["pair"].isin(PAIRS)].copy()
    d["model"] = d["model"].map(normalize_model)
    return add_score_columns(d)


def build_missing_matrix(available):
    variants = ["baseline_news_only", "v1_standardized_surprise", "v2_volatility_regime", "v3_sparse_event_selection", "v4_jump_volatility_labeling", "v5_event_congestion_filter"]
    existing = set(zip(available["pair"], available["model"], available["variant"]))
    rows = []
    for pair in PAIRS:
        for model in MODELS:
            for variant in variants:
                rows.append({
                    "pair": pair,
                    "model": model,
                    "variant": variant,
                    "has_oos_result": (pair, model, variant) in existing,
                    "gap_reason": "" if (pair, model, variant) in existing else "missing_oos_for_model_wise_fairness",
                })
    return pd.DataFrame(rows)


def select_best(available):
    rows_model_pair = []
    for (pair, model), g in available.groupby(["pair", "model"], dropna=False):
        gg = g.copy()
        eligible = gg[gg["validation_positive"] & gg["oos_dev_positive"]].copy()
        if eligible.empty:
            chosen = gg.sort_values(["oos_dev_net_pips", "validation_net_pips"], ascending=False).iloc[0].copy()
            status = "needs_v6_v7"
            reason = "no_variant_positive_on_validation_and_oos_dev"
        else:
            chosen = eligible.sort_values(["selection_score", "oos_dev_net_pips", "oos_dev_profit_factor"], ascending=False).iloc[0].copy()
            status = "accepted_from_v3_v4_v5" if chosen["variant"] in ["v3_sparse_event_selection", "v4_jump_volatility_labeling", "v5_event_congestion_filter"] else "accepted_baseline_or_early_variant"
            reason = "best_positive_validation_and_oos_dev"
            if bool(chosen.get("unstable_final", False)):
                status = "unstable_needs_review"
                reason = "positive_oos_dev_but_final_2026_large_drawdown"
        chosen["selection_status"] = status
        chosen["selection_reason"] = reason
        rows_model_pair.append(chosen)
    best_model_pair = pd.DataFrame(rows_model_pair)

    rows_pair = []
    for pair, g in best_model_pair.groupby("pair", dropna=False):
        eligible = g[g["selection_status"].isin(["accepted_from_v3_v4_v5", "accepted_baseline_or_early_variant"])].copy()
        if eligible.empty:
            chosen = g.sort_values(["oos_dev_net_pips", "validation_net_pips"], ascending=False).iloc[0].copy()
            chosen["pair_selection_status"] = "needs_v6_v7"
            chosen["pair_selection_reason"] = "no_model_variant_accepted"
        else:
            chosen = eligible.sort_values(["selection_score", "oos_dev_net_pips", "oos_dev_profit_factor"], ascending=False).iloc[0].copy()
            chosen["pair_selection_status"] = "selected"
            chosen["pair_selection_reason"] = "best_model_variant_for_pair"
        rows_pair.append(chosen)
    best_pair = pd.DataFrame(rows_pair)
    return best_model_pair, best_pair


def md_table(frame, max_rows=None):
    if frame.empty:
        return "_Tidak ada data._"
    show = frame if max_rows is None else frame.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(available, best_model_pair, best_pair, missing):
    summary = available.groupby("variant").agg(
        candidates=("pair", "size"),
        positive_validation=("validation_positive", "sum"),
        positive_oos_dev=("oos_dev_positive", "sum"),
        avg_oos_dev=("oos_dev_net_pips", "mean"),
        avg_final_2026=("final_2026_net_pips", "mean"),
    ).reset_index()
    pair_cols = [
        "pair", "model", "variant", "selection_status",
        "validation_net_pips", "validation_profit_factor",
        "oos_dev_net_pips", "oos_dev_profit_factor",
        "final_2026_net_pips", "final_2026_profit_factor",
    ]
    final_cols = [
        "pair", "model", "variant", "pair_selection_status",
        "oos_dev_trades", "oos_dev_net_pips", "oos_dev_profit_factor",
        "final_2026_trades", "final_2026_net_pips", "final_2026_profit_factor",
    ]
    missing_summary = missing.groupby(["variant", "has_oos_result"]).size().reset_index(name="count")
    lines = [
        "# Laporan Paper 1 - Model-Wise Enhancement Selection",
        "",
        "Tahap ini mencocokkan base model dengan enhancement yang sudah tersedia: baseline/V1/V2/V3/V4/V5.",
        "Jika V3, V4, atau V5 sudah cukup baik untuk suatu `pair + model`, kombinasi tersebut tidak perlu masuk fallback V6/V7.",
        "Jika tidak ada kandidat yang positif pada validation dan OOS development, kombinasi ditandai `needs_v6_v7`.",
        "",
        "## Ringkasan Kandidat",
        "",
        md_table(summary),
        "",
        "## Best Enhancement Per Model + Pair",
        "",
        md_table(best_model_pair[pair_cols].sort_values(["pair", "model"])),
        "",
        "## Best Model Per Pair",
        "",
        md_table(best_pair[final_cols].sort_values("oos_dev_net_pips", ascending=False)),
        "",
        "## Missing Fairness Matrix",
        "",
        "Catatan: V3/V4/V5 yang sudah dijalankan belum selalu memiliki OOS untuk semua model per pair. Kombinasi yang kosong dicatat agar tidak dianggap sudah diuji.",
        "",
        md_table(missing_summary),
        "",
        "## Keputusan",
        "",
        "- Gunakan hasil `Best Model Per Pair` sebagai kandidat sementara.",
        "- Pair/model dengan status `needs_v6_v7` menjadi prioritas untuk Two-Stage Meta-Labeling atau Drift-Aware Walk-Forward.",
        "- Final 2026 tetap audit, bukan dasar pemilihan.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    available = load_all_available()
    available.to_csv(OUT / "model_wise_enhancement_comparison_available.csv", index=False)
    progress_step("AVAILABLE_LOADED", rows=len(available), variants=available["variant"].nunique())

    missing = build_missing_matrix(available)
    missing.to_csv(OUT / "model_wise_enhancement_missing_matrix.csv", index=False)
    best_model_pair, best_pair = select_best(available)
    best_model_pair.to_csv(OUT / "best_enhancement_per_model_pair.csv", index=False)
    best_pair.to_csv(OUT / "best_model_per_pair.csv", index=False)

    metadata = {
        "stage": "model_wise_enhancement_selection",
        "inputs": {k: str(v) for k, v in ROOTS.items()},
        "selection_rule": "validation_positive and oos_dev_positive; final_2026 audit only",
        "note": "V3/V4/V5 OOS availability is not complete for every model; missing matrix records fairness gaps.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(available, best_model_pair, best_pair, missing)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
