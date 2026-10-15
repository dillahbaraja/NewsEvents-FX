"""Paper 1 V4: jump / volatility labeling for macro news FX events.

This experiment labels a news event as JUMP_UP, JUMP_DOWN, or NO_JUMP.
The jump barrier is based on pre-event H1 volatility fitted without using OOS.
Final 2026 remains audit-only and is not used for model selection.
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

from common import (
    ARTIFACTS,
    DATA,
    pair_file,
    pip_size,
    progress_done,
    progress_start,
    progress_step,
    profit_factor,
    read_prices,
)

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
SOURCE = PIPE / "modeling_dataset_multi_currency.csv"
OUT = ARTIFACTS / "paper1_v4_jump_volatility_labeling"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V4_JUMP_VOLATILITY_LABELING.md"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"]
BASE_FEATURES = [
    "surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour",
] + [f"category_{x}" for x in [
    "inflation", "labor", "growth", "rates", "sentiment", "trade", "housing", "commodities", "other",
]] + ["orientation_base", "orientation_quote"]
VOL_FEATURES = [
    "pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h",
    "pre_vol_pips_6h", "pre_vol_pips_24h", "pre_vol_z_24h",
    "jump_barrier_pips", "surprise_fc_x_pre_vol", "surprise_prev_x_pre_vol",
    "vol_regime_low", "vol_regime_normal", "vol_regime_high",
]
FEATURES = BASE_FEATURES + VOL_FEATURES

BARRIER_MULTIPLIERS = [0.75, 1.00, 1.25, 1.50]
MIN_JUMP_PIPS = 3.0


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
            iterations=700,
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


def build_price_context(pair, datatest=False):
    base = DATA / "datatest" if datatest else DATA
    px = read_prices(base / pair_file(pair)).copy()
    pips = pip_size(pair)
    px["price_time"] = px["time"]
    px["pre_range_pips_1h"] = ((px["High"].shift(1) - px["Low"].shift(1)).abs() / pips)
    px["pre_body_pips_1h"] = ((px["Close"].shift(1) - px["Open"].shift(1)).abs() / pips)
    px["pre_ret_pips_1h"] = ((px["Close"].shift(1) - px["Close"].shift(2)) / pips)
    px["bar_ret_pips"] = px["Close"].diff() / pips
    px["pre_vol_pips_6h"] = px["bar_ret_pips"].shift(1).rolling(6, min_periods=3).std()
    px["pre_vol_pips_24h"] = px["bar_ret_pips"].shift(1).rolling(24, min_periods=8).std()
    return px[["price_time", "pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h", "pre_vol_pips_6h", "pre_vol_pips_24h"]].sort_values("price_time")


def attach_price_context(events, pair, datatest=False):
    ctx = build_price_context(pair, datatest=datatest)
    return pd.merge_asof(
        events.sort_values("event_time"),
        ctx,
        left_on="event_time",
        right_on="price_time",
        direction="forward",
        tolerance=pd.Timedelta(hours=2),
    ).drop(columns=["price_time"], errors="ignore")


def add_vol_features(d, train_stats=None, fit=False, barrier_multiplier=1.0):
    d = d.copy()
    derived_cols = [
        "vol_median", "vol_std", "vol_q33", "vol_q66",
        "pre_vol_z_24h", "jump_barrier_pips",
        "surprise_fc_x_pre_vol", "surprise_prev_x_pre_vol",
        "vol_regime_low", "vol_regime_normal", "vol_regime_high",
    ]
    d = d.drop(columns=derived_cols, errors="ignore")
    for c in ["pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h", "pre_vol_pips_6h", "pre_vol_pips_24h", "surprise_fc", "surprise_prev"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d["pre_vol_pips_24h"] = d["pre_vol_pips_24h"].fillna(d["pre_vol_pips_6h"]).fillna(d["pre_range_pips_1h"]).fillna(0.0)
    d["pre_vol_pips_6h"] = d["pre_vol_pips_6h"].fillna(d["pre_vol_pips_24h"]).fillna(0.0)
    for c in ["pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h"]:
        d[c] = d[c].fillna(0.0)

    if fit:
        train = d[d["split"].eq("train")].copy()
        train_stats = train.groupby("pair")["pre_vol_pips_24h"].agg(
            vol_median="median",
            vol_std="std",
            vol_q33=lambda s: s.quantile(0.33),
            vol_q66=lambda s: s.quantile(0.66),
        ).reset_index()
    d = d.merge(train_stats, on="pair", how="left")
    d["vol_median"] = pd.to_numeric(d["vol_median"], errors="coerce").replace(0, np.nan).fillna(1.0)
    d["vol_std"] = pd.to_numeric(d["vol_std"], errors="coerce").replace(0, np.nan).fillna(1.0)
    d["vol_q33"] = pd.to_numeric(d["vol_q33"], errors="coerce").fillna(d["vol_median"])
    d["vol_q66"] = pd.to_numeric(d["vol_q66"], errors="coerce").fillna(d["vol_median"])
    d["pre_vol_z_24h"] = (d["pre_vol_pips_24h"] - d["vol_median"]) / d["vol_std"]
    d["jump_barrier_pips"] = np.maximum(MIN_JUMP_PIPS, d["pre_vol_pips_24h"] * barrier_multiplier)
    d["surprise_fc_x_pre_vol"] = d["surprise_fc"].fillna(0.0) * d["pre_vol_z_24h"]
    d["surprise_prev_x_pre_vol"] = d["surprise_prev"].fillna(0.0) * d["pre_vol_z_24h"]
    d["vol_regime_low"] = (d["pre_vol_pips_24h"] <= d["vol_q33"]).astype("int8")
    d["vol_regime_high"] = (d["pre_vol_pips_24h"] >= d["vol_q66"]).astype("int8")
    d["vol_regime_normal"] = ((d["vol_regime_low"] == 0) & (d["vol_regime_high"] == 0)).astype("int8")
    return d, train_stats


def make_jump_label(d, barrier_multiplier):
    ret = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    vol = pd.to_numeric(d["pre_vol_pips_24h"], errors="coerce").replace(0, np.nan).fillna(d["pre_vol_pips_6h"]).replace(0, np.nan).fillna(1.0)
    barrier = np.maximum(MIN_JUMP_PIPS, vol * barrier_multiplier)
    return np.where(ret >= barrier, 1, np.where(ret <= -barrier, -1, 0)).astype(int)


def score_predictions(y_true, pred, pnl_source):
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


def load_training_data():
    use = ["event_time", "Name", "CountryCode", "pair", "split", "domestic_return_pips_1h"] + BASE_FEATURES
    raw = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    raw["event_time"] = pd.to_datetime(raw["event_time"], errors="coerce")
    raw = raw[raw["pair"].isin(PAIRS)].dropna(subset=["event_time", "pair"]).copy()
    for c in BASE_FEATURES:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw["domestic_return_pips_1h"] = pd.to_numeric(raw["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    parts = []
    for i, pair in enumerate(PAIRS, 1):
        part = raw[raw["pair"].eq(pair)].copy()
        if part.empty:
            continue
        progress_step("TRAIN_CONTEXT", i, len(PAIRS), pair=pair, rows=len(part))
        parts.append(attach_price_context(part, pair, datatest=False))
    out = pd.concat(parts, ignore_index=True)
    out, stats = add_vol_features(out, fit=True, barrier_multiplier=1.0)
    return out, stats


def build_oos(train_stats):
    import importlib
    stage65 = importlib.import_module("65_test_event_character_aware_oos")
    d = stage65.build_oos_dataset()
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    d = d[d["pair"].isin(PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts = []
    for i, pair in enumerate(PAIRS, 1):
        part = d[d["pair"].eq(pair)].copy()
        if part.empty:
            continue
        progress_step("OOS_CONTEXT", i, len(PAIRS), pair=pair, rows=len(part))
        parts.append(attach_price_context(part, pair, datatest=True))
    out = pd.concat(parts, ignore_index=True)
    out["split"] = "oos"
    out["year"] = pd.to_datetime(out["event_time"], errors="coerce").dt.year
    out, _ = add_vol_features(out, train_stats=train_stats, fit=False, barrier_multiplier=1.0)
    return out


def choose_best(rows):
    frame = pd.DataFrame(rows)
    eligible = frame[(frame["split"].eq("validation")) & (frame["trades"] >= 50)].copy()
    if eligible.empty:
        eligible = frame[frame["split"].eq("validation")].copy()
    eligible["selection_score"] = (
        eligible["net_pips"]
        + 250.0 * eligible["profit_factor"].replace(np.inf, 10.0).clip(upper=5)
        + 50.0 * eligible["macro_f1"]
    )
    return eligible.sort_values(["net_pips", "selection_score", "profit_factor"], ascending=False).iloc[0]


def write_report(selected, oos_results):
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

    total = oos_results.groupby("period").agg(trades=("trades", "sum"), net_pips=("net_pips", "sum"), avg_pf=("profit_factor", "mean")).reset_index()
    lines = [
        "# Laporan Paper 1 V4 - Jump / Volatility Labeling",
        "",
        "Eksperimen ini menggunakan label `jump_up`, `jump_down`, dan `no_jump` berdasarkan return 1H yang melewati barrier volatilitas pra-event.",
        "Pemilihan model memakai validation 2021-2023. Final 2026 hanya audit.",
        "",
        "## Ringkasan OOS",
        "",
        md_table(total),
        "",
        "## Model Terpilih Berdasarkan Validation",
        "",
        md_table(selected[["pair", "model", "barrier_multiplier", "trades", "net_pips", "profit_factor", "macro_f1"]]),
        "",
        "## Hasil OOS Per Pair",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
        "",
        "## Keputusan",
        "",
        "Bandingkan hasil ini dengan baseline aktif dan V3. Jika OOS development lebih rendah, V4 tidak mengganti baseline utama.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d, stats = load_training_data()
    stats.to_csv(OUT / "v4_volatility_train_statistics.csv", index=False)
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    validation_rows = []
    selected_rows = []
    selected_models = {}
    total_jobs = len(PAIRS) * len(BARRIER_MULTIPLIERS) * 3
    job = 0
    for pair in PAIRS:
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train")].copy()
        va = p[p["split"].eq("validation")].copy()
        candidate_rows = []
        candidate_models = {}
        for barrier_multiplier in BARRIER_MULTIPLIERS:
            tr2, _ = add_vol_features(tr, train_stats=stats, fit=False, barrier_multiplier=barrier_multiplier)
            va2, _ = add_vol_features(va, train_stats=stats, fit=False, barrier_multiplier=barrier_multiplier)
            ytr = make_jump_label(tr2, barrier_multiplier)
            yva = make_jump_label(va2, barrier_multiplier)
            for model_name, model in model_defs(seed=404 + int(barrier_multiplier * 100)).items():
                job += 1
                progress_step("MODEL_START", job, total_jobs, pair=pair, model=model_name, barrier=barrier_multiplier, train=len(tr2), validation=len(va2))
                t = time.perf_counter()
                model.fit(tr2[FEATURES], ytr)
                pred = np.asarray(model.predict(va2[FEATURES])).reshape(-1).astype(int)
                result = score_predictions(yva, pred, va2["domestic_return_pips_1h"].to_numpy(float))
                row = {"pair": pair, "model": model_name, "barrier_multiplier": barrier_multiplier, "split": "validation", "seconds": time.perf_counter() - t, **result}
                validation_rows.append(row)
                candidate_rows.append(row)
                candidate_models[(model_name, barrier_multiplier)] = model
                progress_step("MODEL_DONE", pair=pair, model=model_name, barrier=barrier_multiplier, net_pips=f"{result['net_pips']:.1f}", pf=f"{result['profit_factor']:.4f}", trades=result["trades"])
        best = choose_best(candidate_rows)
        key = (best["model"], float(best["barrier_multiplier"]))
        selected_rows.append(dict(best))
        selected_models[pair] = {"model": candidate_models[key], "model_name": key[0], "barrier_multiplier": key[1]}
        joblib.dump({"model": candidate_models[key], "features": FEATURES, "barrier_multiplier": key[1], "train_stats": stats}, MODELS / f"{key[0]}_{pair}_v4_jump_volatility.joblib")
        progress_step("PAIR_SELECTED", pair=pair, model=key[0], barrier=key[1], validation_net_pips=f"{best['net_pips']:.1f}")

    validation = pd.DataFrame(validation_rows)
    selected = pd.DataFrame(selected_rows)
    validation.to_csv(OUT / "v4_validation_grid_results.csv", index=False)
    selected.to_csv(OUT / "v4_selected_models.csv", index=False)

    oos = build_oos(stats)
    predictions = []
    oos_rows = []
    for c in FEATURES:
        oos[c] = pd.to_numeric(oos[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for pair, bundle in selected_models.items():
        part = oos[oos["pair"].eq(pair)].copy()
        part, _ = add_vol_features(part, train_stats=stats, fit=False, barrier_multiplier=bundle["barrier_multiplier"])
        for c in FEATURES:
            part[c] = pd.to_numeric(part[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = make_jump_label(part, bundle["barrier_multiplier"])
        pred = np.asarray(bundle["model"].predict(part[FEATURES])).reshape(-1).astype(int)
        p = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year"]].copy()
        p["model"] = bundle["model_name"]
        p["variant"] = "v4_jump_volatility_labeling"
        p["barrier_multiplier"] = bundle["barrier_multiplier"]
        p["jump_label"] = y
        p["signal"] = pred
        p["strategy_pips"] = p["domestic_return_pips_1h"].to_numpy(float) * pred
        predictions.append(p)
        for period, mask in [("oos_dev_2024_2025", p["year"].between(2024, 2025)), ("final_oos_2026", p["year"].eq(2026))]:
            active = p[mask & p["signal"].ne(0)]
            x = active["strategy_pips"].to_numpy(float)
            y_slice = y[mask.to_numpy()]
            pred_slice = pred[mask.to_numpy()]
            score = score_predictions(y_slice, pred_slice, p.loc[mask, "domestic_return_pips_1h"].to_numpy(float))
            oos_rows.append({"period": period, "pair": pair, "model": bundle["model_name"], "variant": "v4_jump_volatility_labeling", "barrier_multiplier": bundle["barrier_multiplier"], **score})
        progress_step("OOS_DONE", pair=pair, model=bundle["model_name"])

    oos_results = pd.DataFrame(oos_rows)
    oos_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    oos_results.to_csv(OUT / "v4_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "v4_oos_predictions.csv", index=False)
    metadata = {
        "variant": "v4_jump_volatility_labeling",
        "train": "2010-2020",
        "validation": "2021-2023",
        "oos_development": "2024-2025",
        "final_oos": "2026_audit_only",
        "label": "jump_up/jump_down/no_jump based on abs(return_1h_pips) >= max(3 pips, barrier_multiplier * pre_event_24h_volatility)",
        "features": FEATURES,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(selected, oos_results)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
