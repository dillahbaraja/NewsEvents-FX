"""Paper 1 V6 fallback: two-stage meta-labeling for weak pairs.

Stage 1 predicts TRADE vs NO_TRADE.
Stage 2 predicts BUY vs SELL only when Stage 1 allows a trade.

This fallback is run only for pairs that remained weak after baseline/V1/V2/V3/V4/V5.
Selection uses validation 2021-2023. Final 2026 is audit-only.
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
OUT = ARTIFACTS / "paper1_v6_two_stage_fallback"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V6_TWO_STAGE_FALLBACK.md"

TARGET_PAIRS = ["GBPUSD", "AUDUSD", "NZDUSD", "USDCHF"]
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
TRADE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
DIRECTION_MARGINS = [0.00, 0.05, 0.10, 0.15]


def safe_pf(values):
    return float(profit_factor(values))


def normalize_model(value):
    return str(value).strip().lower()


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
            n_estimators=500,
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
            iterations=700,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=10,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )
    return models


def proba_positive(model, x):
    p = np.asarray(model.predict_proba(x))
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return p[:, list(classes).index(1)]
    return p[:, -1]


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


def load_train_validation():
    use = ["event_time", "Name", "CountryCode", "pair", "split", "trade_label_1h", "domestic_return_pips_1h"] + BASE_FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d = d[d["pair"].isin(TARGET_PAIRS)].dropna(subset=["event_time", "pair", "Name", "CountryCode"]).copy()
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


def score_signal(pred, pnl_source):
    strategy = pnl_source * pred
    active = strategy[pred != 0]
    return {
        "trades": int(len(active)),
        "net_pips": float(active.sum()) if len(active) else 0.0,
        "profit_factor": safe_pf(active) if len(active) else 0.0,
        "winrate": float((active > 0).mean()) if len(active) else 0.0,
    }


def make_signal(trade_prob, buy_prob, trade_threshold, direction_margin):
    direction = np.where(buy_prob >= 0.5, 1, -1)
    allow = (trade_prob >= trade_threshold) & (np.abs(buy_prob - 0.5) >= direction_margin)
    return np.where(allow, direction, 0).astype(int)


def fit_two_stage(x_train, y_train, model_name, seed):
    y_trade = (y_train != 0).astype(int)
    direction_mask = y_train != 0
    y_dir = (y_train[direction_mask] == 1).astype(int)
    defs = model_defs(seed)
    trade_model = defs[model_name]
    direction_model = model_defs(seed + 1000)[model_name]
    trade_model.fit(x_train, y_trade)
    direction_model.fit(x_train.loc[direction_mask], y_dir)
    return trade_model, direction_model


def validation_grid(d):
    rows = []
    bundles = {}
    total = len(TARGET_PAIRS) * len(model_defs(1))
    job = 0
    for pair in TARGET_PAIRS:
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train")].copy()
        va = p[p["split"].eq("validation")].copy()
        if tr.empty or va.empty:
            continue
        for model_name in model_defs(seed=1).keys():
            job += 1
            progress_step("MODEL_FIT_START", job, total, pair=pair, model=model_name, train=len(tr), validation=len(va))
            t = time.perf_counter()
            trade_model, direction_model = fit_two_stage(tr[FEATURES], tr["trade_label_1h"].astype(int), model_name, seed=900 + job)
            trade_prob = proba_positive(trade_model, va[FEATURES])
            buy_prob = proba_positive(direction_model, va[FEATURES])
            for tt in TRADE_THRESHOLDS:
                for dm in DIRECTION_MARGINS:
                    pred = make_signal(trade_prob, buy_prob, tt, dm)
                    result = score_signal(pred, va["domestic_return_pips_1h"].to_numpy(float))
                    rows.append({
                        "pair": pair,
                        "model": model_name,
                        "trade_threshold": tt,
                        "direction_margin": dm,
                        "seconds": time.perf_counter() - t,
                        **result,
                    })
            bundles[(pair, model_name)] = {"trade_model": trade_model, "direction_model": direction_model}
            progress_step("MODEL_FIT_DONE", pair=pair, model=model_name, seconds=f"{time.perf_counter() - t:.1f}")
    return pd.DataFrame(rows), bundles


def choose_and_retrain(d, grid):
    selected_rows = []
    bundles = {}
    for pair in TARGET_PAIRS:
        part = grid[(grid["pair"].eq(pair)) & (grid["trades"] >= 50)].copy()
        if part.empty:
            part = grid[grid["pair"].eq(pair)].copy()
        if part.empty:
            continue
        part["selection_score"] = part["net_pips"] + 300.0 * part["profit_factor"].clip(upper=5)
        best = part.sort_values(["net_pips", "selection_score", "profit_factor"], ascending=False).iloc[0].to_dict()
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train")].copy()
        trade_model, direction_model = fit_two_stage(tr[FEATURES], tr["trade_label_1h"].astype(int), best["model"], seed=1550)
        selected_rows.append(best)
        bundles[pair] = {"trade_model": trade_model, "direction_model": direction_model, "best": best}
        joblib.dump(
            {"trade_model": trade_model, "direction_model": direction_model, "features": FEATURES, "best": best},
            MODELS / f"{best['model']}_{pair}_v6_two_stage_fallback.joblib",
        )
        progress_step("PAIR_SELECTED", pair=pair, model=best["model"], validation_net_pips=f"{best['net_pips']:.1f}", threshold=best["trade_threshold"], margin=best["direction_margin"])
    return pd.DataFrame(selected_rows), bundles


def evaluate_oos(oos, bundles):
    rows = []
    preds = []
    for pair, bundle in bundles.items():
        part = oos[oos["pair"].eq(pair)].copy()
        if part.empty:
            continue
        best = bundle["best"]
        trade_prob = proba_positive(bundle["trade_model"], part[FEATURES])
        buy_prob = proba_positive(bundle["direction_model"], part[FEATURES])
        pred = make_signal(trade_prob, buy_prob, float(best["trade_threshold"]), float(best["direction_margin"]))
        p = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year"]].copy()
        p["model"] = best["model"]
        p["variant"] = "v6_two_stage_fallback"
        p["trade_probability"] = trade_prob
        p["buy_probability"] = buy_prob
        p["signal"] = pred
        p["strategy_pips"] = p["domestic_return_pips_1h"].to_numpy(float) * pred
        preds.append(p)
        for period, mask in [("oos_dev_2024_2025", p["year"].between(2024, 2025)), ("final_oos_2026", p["year"].eq(2026))]:
            active = p[mask & p["signal"].ne(0)]
            x = active["strategy_pips"].to_numpy(float)
            rows.append({
                "period": period,
                "pair": pair,
                "model": best["model"],
                "variant": "v6_two_stage_fallback",
                "trade_threshold": best["trade_threshold"],
                "direction_margin": best["direction_margin"],
                "trades": int(len(x)),
                "net_pips": float(x.sum()) if len(x) else 0.0,
                "profit_factor": safe_pf(x) if len(x) else 0.0,
                "winrate": float((x > 0).mean()) if len(x) else 0.0,
            })
        progress_step("OOS_DONE", pair=pair, model=best["model"])
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
        "# Laporan Paper 1 V6 - Two-Stage Fallback",
        "",
        "Eksperimen ini hanya dijalankan untuk pair yang belum kuat setelah baseline/V1/V2/V3/V4/V5.",
        "Stage 1 memprediksi TRADE/NO_TRADE, Stage 2 memprediksi BUY/SELL.",
        "Pemilihan memakai validation 2021-2023. Final 2026 hanya audit.",
        "",
        "## Ringkasan OOS",
        "",
        md_table(total),
        "",
        "## Model Terpilih Berdasarkan Validation",
        "",
        md_table(selected[["pair", "model", "trade_threshold", "direction_margin", "trades", "net_pips", "profit_factor", "winrate"]]),
        "",
        "## Hasil OOS Per Pair",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d = load_train_validation()
    grid, _ = validation_grid(d)
    grid.to_csv(OUT / "v6_validation_grid_results.csv", index=False)
    selected, bundles = choose_and_retrain(d, grid)
    selected.to_csv(OUT / "v6_selected_models.csv", index=False)
    oos_results, oos_predictions = evaluate_oos(build_oos(), bundles)
    oos_results.to_csv(OUT / "v6_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "v6_oos_predictions.csv", index=False)
    metadata = {
        "variant": "v6_two_stage_fallback",
        "target_pairs": TARGET_PAIRS,
        "training": "2010_2020",
        "validation": "2021_2023",
        "oos_development": "2024_2025",
        "final_oos": "2026_audit_only",
        "features": FEATURES,
        "trade_thresholds": TRADE_THRESHOLDS,
        "direction_margins": DIRECTION_MARGINS,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(selected, oos_results)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
