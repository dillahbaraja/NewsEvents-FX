"""Paper 1 V7 fallback: drift-aware expanding walk-forward.

For each weak pair, train on years before Y and validate on Y for Y=2018..2023.
The final model is selected by yearly stability, then retrained on 2010-2023
and tested on OOS 2024-2026. Final 2026 remains audit-only.
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
SOURCE = PIPE / "modeling_dataset_multi_currency.csv"
OUT = ARTIFACTS / "paper1_v7_drift_aware_walkforward"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V7_DRIFT_AWARE_WALKFORWARD.md"

TARGET_PAIRS = ["GBPUSD", "AUDUSD", "NZDUSD", "USDCHF"]
WALK_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
BASE_FEATURES = [
    "surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour",
] + [f"category_{x}" for x in [
    "inflation", "labor", "growth", "rates", "sentiment", "trade", "housing", "commodities", "other",
]] + ["orientation_base", "orientation_quote"]
CONGESTION_FEATURES = [
    "same_hour_relevant_count", "same_hour_country_count", "near_30m_relevant_count",
    "near_60m_relevant_count", "minutes_since_prev_relevant", "minutes_to_next_relevant",
    "minutes_since_prev_country", "minutes_to_next_country",
    "is_clustered_30m", "is_clustered_60m", "is_first_in_cluster_60m",
]
FEATURES = BASE_FEATURES + CONGESTION_FEATURES


def safe_pf(values):
    return float(profit_factor(values))


def model_defs(seed):
    models = {
        "elastic_net": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1800,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.35,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=450,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=10,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )
    return models


def add_congestion_features(d):
    out = d.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out = out.sort_values(["pair", "event_time", "CountryCode", "Name"]).reset_index(drop=True)
    out["event_hour"] = out["event_time"].dt.floor("h")
    out["same_hour_relevant_count"] = out.groupby(["pair", "event_hour"])["event_time"].transform("size").astype(float)
    out["same_hour_country_count"] = out.groupby(["pair", "CountryCode", "event_hour"])["event_time"].transform("size").astype(float)
    parts = []
    for pair, g in out.groupby("pair", sort=False):
        g = g.sort_values("event_time").copy()
        times = g["event_time"].astype("int64").to_numpy()
        near30 = np.zeros(len(g), dtype=float)
        near60 = np.zeros(len(g), dtype=float)
        for i, t in enumerate(times):
            left30 = np.searchsorted(times, t - 30 * 60 * 1_000_000_000, side="left")
            right30 = np.searchsorted(times, t + 30 * 60 * 1_000_000_000, side="right")
            left60 = np.searchsorted(times, t - 60 * 60 * 1_000_000_000, side="left")
            right60 = np.searchsorted(times, t + 60 * 60 * 1_000_000_000, side="right")
            near30[i] = max(0, right30 - left30 - 1)
            near60[i] = max(0, right60 - left60 - 1)
        g["near_30m_relevant_count"] = near30
        g["near_60m_relevant_count"] = near60
        g["minutes_since_prev_relevant"] = g["event_time"].diff().dt.total_seconds().div(60)
        g["minutes_to_next_relevant"] = -g["event_time"].diff(-1).dt.total_seconds().div(60)
        g["minutes_since_prev_country"] = g.groupby("CountryCode")["event_time"].diff().dt.total_seconds().div(60)
        g["minutes_to_next_country"] = -g.groupby("CountryCode")["event_time"].diff(-1).dt.total_seconds().div(60)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    fill_large = 9999.0
    for c in ["minutes_since_prev_relevant", "minutes_to_next_relevant", "minutes_since_prev_country", "minutes_to_next_country"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill_large).clip(lower=0, upper=fill_large)
    out["is_clustered_30m"] = (out["near_30m_relevant_count"] > 0).astype("int8")
    out["is_clustered_60m"] = (out["near_60m_relevant_count"] > 0).astype("int8")
    out["is_first_in_cluster_60m"] = ((out["near_60m_relevant_count"] > 0) & (out["minutes_since_prev_relevant"] > 60)).astype("int8")
    return out.drop(columns=["event_hour"], errors="ignore")


def load_dataset():
    use = ["event_time", "Name", "CountryCode", "pair", "event_year", "trade_label_1h", "domestic_return_pips_1h"] + BASE_FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce").astype("Int64")
    d = d[d["pair"].isin(TARGET_PAIRS)].dropna(subset=["event_time", "pair", "Name", "CountryCode", "event_year"]).copy()
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["trade_label_1h"] = pd.to_numeric(d["trade_label_1h"], errors="coerce").fillna(0).astype(int)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    d = add_congestion_features(d)
    for c in CONGESTION_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def build_oos():
    import importlib
    stage65 = importlib.import_module("65_test_event_character_aware_oos")
    d = stage65.build_oos_dataset()
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    d = d[d["pair"].isin(TARGET_PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    d = add_congestion_features(d)
    for c in CONGESTION_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def score(pred, pnl_source):
    strategy = pnl_source * pred
    active = strategy[pred != 0]
    return {
        "trades": int(len(active)),
        "net_pips": float(active.sum()) if len(active) else 0.0,
        "profit_factor": safe_pf(active) if len(active) else 0.0,
        "winrate": float((active > 0).mean()) if len(active) else 0.0,
    }


def walkforward(d):
    rows = []
    total = len(TARGET_PAIRS) * len(model_defs(1)) * len(WALK_YEARS)
    job = 0
    for pair in TARGET_PAIRS:
        p = d[d["pair"].eq(pair)].copy()
        for model_name in model_defs(1).keys():
            for year in WALK_YEARS:
                job += 1
                tr = p[p["event_year"] < year].copy()
                va = p[p["event_year"] == year].copy()
                if len(tr) < 1000 or va.empty or tr["trade_label_1h"].nunique() < 2:
                    continue
                progress_step("WF_START", job, total, pair=pair, model=model_name, year=year, train=len(tr), validation=len(va))
                t = time.perf_counter()
                model = model_defs(700 + job)[model_name]
                model.fit(tr[FEATURES], tr["trade_label_1h"].astype(int))
                pred = np.asarray(model.predict(va[FEATURES])).reshape(-1).astype(int)
                result = score(pred, va["domestic_return_pips_1h"].to_numpy(float))
                rows.append({"pair": pair, "model": model_name, "year": year, "seconds": time.perf_counter() - t, **result})
                progress_step("WF_DONE", pair=pair, model=model_name, year=year, net_pips=f"{result['net_pips']:.1f}", pf=f"{result['profit_factor']:.4f}", trades=result["trades"])
    return pd.DataFrame(rows)


def select_models(d, wf):
    rows = []
    bundles = {}
    for (pair, model_name), g in wf.groupby(["pair", "model"], dropna=False):
        active_years = int((g["trades"] > 0).sum())
        positive_years = int((g["net_pips"] > 0).sum())
        total_net = float(g["net_pips"].sum())
        pf_avg = float(g["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).mean())
        worst_year = float(g["net_pips"].min())
        positive_ratio = positive_years / max(active_years, 1)
        score_value = total_net + 1000.0 * positive_ratio + 200.0 * pf_avg + min(worst_year, 0) * 0.5
        rows.append({
            "pair": pair,
            "model": model_name,
            "wf_years": active_years,
            "positive_years": positive_years,
            "positive_year_ratio": positive_ratio,
            "wf_net_pips": total_net,
            "wf_avg_pf": pf_avg,
            "wf_worst_year_net_pips": worst_year,
            "selection_score": score_value,
        })
    summary = pd.DataFrame(rows)
    selected_rows = []
    for pair in TARGET_PAIRS:
        part = summary[summary["pair"].eq(pair)].copy()
        if part.empty:
            continue
        eligible = part[(part["wf_net_pips"] > 0) & (part["positive_year_ratio"] >= 0.50) & (part["wf_avg_pf"] > 1.0)].copy()
        if eligible.empty:
            best = part.sort_values(["selection_score", "wf_net_pips"], ascending=False).iloc[0].copy()
            best["selection_status"] = "weak_walkforward"
        else:
            best = eligible.sort_values(["selection_score", "wf_net_pips"], ascending=False).iloc[0].copy()
            best["selection_status"] = "selected_walkforward"
        p = d[d["pair"].eq(pair) & d["event_year"].between(2010, 2023)].copy()
        model = model_defs(2027)[best["model"]]
        model.fit(p[FEATURES], p["trade_label_1h"].astype(int))
        bundles[pair] = {"model": model, "best": best.to_dict()}
        joblib.dump({"model": model, "features": FEATURES, "best": best.to_dict()}, MODELS / f"{best['model']}_{pair}_v7_drift_walkforward.joblib")
        selected_rows.append(best)
        progress_step("PAIR_SELECTED", pair=pair, model=best["model"], status=best["selection_status"], wf_net=f"{best['wf_net_pips']:.1f}", positive_ratio=f"{best['positive_year_ratio']:.2f}")
    return summary, pd.DataFrame(selected_rows), bundles


def evaluate_oos(oos, bundles):
    rows = []
    preds = []
    for pair, bundle in bundles.items():
        part = oos[oos["pair"].eq(pair)].copy()
        if part.empty:
            continue
        pred = np.asarray(bundle["model"].predict(part[FEATURES])).reshape(-1).astype(int)
        p = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year"]].copy()
        p["model"] = bundle["best"]["model"]
        p["variant"] = "v7_drift_aware_walkforward"
        p["signal"] = pred
        p["strategy_pips"] = p["domestic_return_pips_1h"].to_numpy(float) * pred
        preds.append(p)
        for period, mask in [("oos_dev_2024_2025", p["year"].between(2024, 2025)), ("final_oos_2026", p["year"].eq(2026))]:
            active = p[mask & p["signal"].ne(0)]
            x = active["strategy_pips"].to_numpy(float)
            rows.append({
                "period": period,
                "pair": pair,
                "model": bundle["best"]["model"],
                "variant": "v7_drift_aware_walkforward",
                "walkforward_status": bundle["best"]["selection_status"],
                "trades": int(len(x)),
                "net_pips": float(x.sum()) if len(x) else 0.0,
                "profit_factor": safe_pf(x) if len(x) else 0.0,
                "winrate": float((x > 0).mean()) if len(x) else 0.0,
            })
        progress_step("OOS_DONE", pair=pair, model=bundle["best"]["model"])
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def md_table(frame):
    if frame.empty:
        return "_Tidak ada data._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(selected, oos_results, wf_summary):
    total = oos_results.groupby("period").agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        avg_pf=("profit_factor", "mean"),
        positive_pairs=("net_pips", lambda s: int((s > 0).sum())),
    ).reset_index()
    lines = [
        "# Laporan Paper 1 V7 - Drift-Aware Walk-Forward",
        "",
        "Eksperimen ini memakai expanding walk-forward validation per tahun 2018-2023 untuk pair yang masih lemah.",
        "Model final dilatih ulang pada 2010-2023 dan diuji pada OOS 2024-2026. Final 2026 hanya audit.",
        "",
        "## Ringkasan OOS",
        "",
        md_table(total),
        "",
        "## Model Terpilih Berdasarkan Walk-Forward",
        "",
        md_table(selected[["pair", "model", "selection_status", "wf_years", "positive_years", "positive_year_ratio", "wf_net_pips", "wf_avg_pf", "wf_worst_year_net_pips"]]),
        "",
        "## Hasil OOS Per Pair",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
        "",
        "## Walk-Forward Summary Semua Kandidat",
        "",
        md_table(wf_summary.sort_values(["pair", "selection_score"], ascending=[True, False])),
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d = load_dataset()
    wf = walkforward(d)
    wf.to_csv(OUT / "v7_walkforward_yearly_results.csv", index=False)
    wf_summary, selected, bundles = select_models(d, wf)
    wf_summary.to_csv(OUT / "v7_walkforward_model_summary.csv", index=False)
    selected.to_csv(OUT / "v7_selected_models.csv", index=False)
    oos_results, oos_predictions = evaluate_oos(build_oos(), bundles)
    oos_results.to_csv(OUT / "v7_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "v7_oos_predictions.csv", index=False)
    metadata = {
        "variant": "v7_drift_aware_walkforward",
        "target_pairs": TARGET_PAIRS,
        "walkforward_years": WALK_YEARS,
        "final_train": "2010_2023",
        "oos_development": "2024_2025",
        "final_oos": "2026_audit_only",
        "features": FEATURES,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(selected, oos_results, wf_summary)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
