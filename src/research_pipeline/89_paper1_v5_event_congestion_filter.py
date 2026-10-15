"""Paper 1 V5: event congestion / news-cluster filter.

This experiment tests whether macro news signals improve when the model knows
whether an announcement is isolated or crowded by nearby relevant events.
Selection uses validation 2021-2023. Final 2026 remains audit-only.
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
from sklearn.metrics import balanced_accuracy_score, f1_score

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
SOURCE = PIPE / "modeling_dataset_multi_currency.csv"
OUT = ARTIFACTS / "paper1_v5_event_congestion_filter"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V5_EVENT_CONGESTION_FILTER.md"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"]
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

GATES = [
    {"gate": "no_filter", "max_same_hour": None, "min_since_prev": None},
    {"gate": "same_hour_le_3", "max_same_hour": 3, "min_since_prev": None},
    {"gate": "same_hour_le_5", "max_same_hour": 5, "min_since_prev": None},
    {"gate": "isolated_30m", "max_same_hour": None, "min_since_prev": 30},
]


def safe_pf(values):
    return float(profit_factor(values))


def model_defs(seed):
    models = {
        "elastic_net": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2500,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.35,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=14,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=650,
            depth=6,
            learning_rate=0.04,
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

    hour_pair = out.groupby(["pair", "event_hour"])["event_time"].transform("size")
    hour_country = out.groupby(["pair", "CountryCode", "event_hour"])["event_time"].transform("size")
    out["same_hour_relevant_count"] = hour_pair.astype(float)
    out["same_hour_country_count"] = hour_country.astype(float)

    result_parts = []
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
        result_parts.append(g)

    out = pd.concat(result_parts, ignore_index=True)
    fill_large = 9999.0
    for c in ["minutes_since_prev_relevant", "minutes_to_next_relevant", "minutes_since_prev_country", "minutes_to_next_country"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill_large).clip(lower=0, upper=fill_large)
    out["is_clustered_30m"] = (out["near_30m_relevant_count"] > 0).astype("int8")
    out["is_clustered_60m"] = (out["near_60m_relevant_count"] > 0).astype("int8")
    out["is_first_in_cluster_60m"] = ((out["near_60m_relevant_count"] > 0) & (out["minutes_since_prev_relevant"] > 60)).astype("int8")
    return out.drop(columns=["event_hour"], errors="ignore")


def gate_mask(d, gate):
    mask = pd.Series(True, index=d.index)
    if gate.get("max_same_hour") is not None:
        mask &= pd.to_numeric(d["same_hour_relevant_count"], errors="coerce").fillna(0) <= gate["max_same_hour"]
    if gate.get("min_since_prev") is not None:
        mins = pd.to_numeric(d["minutes_since_prev_relevant"], errors="coerce").fillna(9999)
        mask &= mins >= gate["min_since_prev"]
    return mask


def score(y_true, pred, pnl_source):
    strategy = pnl_source * pred
    active = strategy[pred != 0]
    return {
        "trades": int(len(active)),
        "net_pips": float(active.sum()) if len(active) else 0.0,
        "profit_factor": safe_pf(active) if len(active) else 0.0,
        "winrate": float((active > 0).mean()) if len(active) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def load_train_validation():
    use = ["event_time", "Name", "CountryCode", "pair", "split", "trade_label_1h", "domestic_return_pips_1h"] + BASE_FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d = d[d["pair"].isin(PAIRS)].dropna(subset=["event_time", "pair", "Name", "CountryCode"]).copy()
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
    d = d[d["pair"].isin(PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    d = add_congestion_features(d)
    for c in CONGESTION_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def validation_grid(d):
    rows = []
    total = len(PAIRS) * len(GATES) * len(model_defs(seed=1))
    job = 0
    for pair in PAIRS:
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train") & p["trade_label_1h"].ne(0)].copy()
        va = p[p["split"].eq("validation")].copy()
        for gate in GATES:
            gtr = tr[gate_mask(tr, gate)].copy()
            if len(gtr) < 200 or gtr["trade_label_1h"].nunique() < 2:
                continue
            for model_name, model in model_defs(seed=505 + job).items():
                job += 1
                progress_step("MODEL_START", job, total, pair=pair, model=model_name, gate=gate["gate"], train=len(gtr), validation=len(va))
                t = time.perf_counter()
                model.fit(gtr[FEATURES], gtr["trade_label_1h"].astype(int))
                raw_pred = np.asarray(model.predict(va[FEATURES])).reshape(-1).astype(int)
                pred = np.where(gate_mask(va, gate), raw_pred, 0)
                result = score(va["trade_label_1h"].to_numpy(int), pred, va["domestic_return_pips_1h"].to_numpy(float))
                rows.append({
                    "pair": pair,
                    "model": model_name,
                    "gate": gate["gate"],
                    "max_same_hour": gate.get("max_same_hour"),
                    "min_since_prev": gate.get("min_since_prev"),
                    "train_rows": len(gtr),
                    "seconds": time.perf_counter() - t,
                    **result,
                })
                progress_step("MODEL_DONE", pair=pair, model=model_name, gate=gate["gate"], net_pips=f"{result['net_pips']:.1f}", pf=f"{result['profit_factor']:.4f}", trades=result["trades"])
    return pd.DataFrame(rows)


def choose_and_retrain(d, grid):
    selected_rows = []
    bundles = {}
    for pair in PAIRS:
        part = grid[(grid["pair"].eq(pair)) & (grid["trades"] >= 50)].copy()
        if part.empty:
            part = grid[grid["pair"].eq(pair)].copy()
        if part.empty:
            continue
        part["selection_score"] = part["net_pips"] + 250.0 * part["profit_factor"].clip(upper=5) + 50.0 * part["macro_f1"]
        best = part.sort_values(["net_pips", "selection_score", "profit_factor"], ascending=False).iloc[0].to_dict()
        gate = next(g for g in GATES if g["gate"] == best["gate"])
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train") & p["trade_label_1h"].ne(0)].copy()
        tr = tr[gate_mask(tr, gate)].copy()
        model = model_defs(seed=888)[best["model"]]
        model.fit(tr[FEATURES], tr["trade_label_1h"].astype(int))
        selected_rows.append(best)
        bundles[pair] = {"model": model, "model_name": best["model"], "gate": gate, "best": best}
        joblib.dump(
            {"model": model, "features": FEATURES, "gate": gate, "best": best},
            MODELS / f"{best['model']}_{pair}_v5_event_congestion_filter.joblib",
        )
        progress_step("PAIR_SELECTED", pair=pair, model=best["model"], gate=best["gate"], validation_net_pips=f"{best['net_pips']:.1f}")
    return pd.DataFrame(selected_rows), bundles


def evaluate_oos(oos, bundles):
    rows = []
    preds = []
    for pair, bundle in bundles.items():
        part = oos[oos["pair"].eq(pair)].copy()
        if part.empty:
            continue
        raw_pred = np.asarray(bundle["model"].predict(part[FEATURES])).reshape(-1).astype(int)
        pred = np.where(gate_mask(part, bundle["gate"]), raw_pred, 0)
        p = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year"] + CONGESTION_FEATURES].copy()
        p["model"] = bundle["model_name"]
        p["variant"] = "v5_event_congestion_filter"
        p["gate"] = bundle["gate"]["gate"]
        p["signal"] = pred
        p["strategy_pips"] = p["domestic_return_pips_1h"].to_numpy(float) * pred
        preds.append(p)
        for period, mask in [("oos_dev_2024_2025", p["year"].between(2024, 2025)), ("final_oos_2026", p["year"].eq(2026))]:
            active = p[mask & p["signal"].ne(0)]
            x = active["strategy_pips"].to_numpy(float)
            rows.append({
                "period": period,
                "pair": pair,
                "model": bundle["model_name"],
                "variant": "v5_event_congestion_filter",
                "gate": bundle["gate"]["gate"],
                "trades": int(len(x)),
                "net_pips": float(x.sum()) if len(x) else 0.0,
                "profit_factor": safe_pf(x) if len(x) else 0.0,
                "winrate": float((x > 0).mean()) if len(x) else 0.0,
            })
        progress_step("OOS_DONE", pair=pair, model=bundle["model_name"], gate=bundle["gate"]["gate"])
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


def write_report(selected, oos_results):
    total = oos_results.groupby("period").agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        avg_pf=("profit_factor", "mean"),
        positive_pairs=("net_pips", lambda s: int((s > 0).sum())),
    ).reset_index()
    lines = [
        "# Laporan Paper 1 V5 - Event Congestion Filter",
        "",
        "Eksperimen ini menambahkan fitur kepadatan berita dan filter event yang terlalu berdekatan.",
        "Pemilihan model memakai validation 2021-2023. Final 2026 hanya audit.",
        "",
        "## Ringkasan OOS",
        "",
        md_table(total),
        "",
        "## Model Terpilih Berdasarkan Validation",
        "",
        md_table(selected[["pair", "model", "gate", "trades", "net_pips", "profit_factor", "macro_f1"]]),
        "",
        "## Hasil OOS Per Pair",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
        "",
        "## Keputusan",
        "",
        "Bandingkan hasil ini dengan baseline aktif, V3, dan V4. Jika OOS development tidak lebih baik, V5 tidak mengganti baseline utama.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d = load_train_validation()
    d[["pair", "event_time", "CountryCode", "Name"] + CONGESTION_FEATURES].to_csv(OUT / "v5_congestion_features_train_validation.csv", index=False)
    grid = validation_grid(d)
    grid.to_csv(OUT / "v5_validation_grid_results.csv", index=False)
    selected, bundles = choose_and_retrain(d, grid)
    selected.to_csv(OUT / "v5_selected_models.csv", index=False)
    oos_results, oos_predictions = evaluate_oos(build_oos(), bundles)
    oos_results.to_csv(OUT / "v5_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "v5_oos_predictions.csv", index=False)
    metadata = {
        "variant": "v5_event_congestion_filter",
        "training": "2010_2020",
        "validation": "2021_2023",
        "oos_development": "2024_2025",
        "final_oos": "2026_audit_only",
        "features": FEATURES,
        "gates": GATES,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(selected, oos_results)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
