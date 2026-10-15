"""Paper 1 V2: volatility-regime event classification.

This experiment classifies macro news events after conditioning the event on
pre-release market volatility. It keeps the standard Paper 1 scope: news-only
macro features plus market-state variables used to describe the event context,
not technical-entry indicators.
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
    available_currency_pair_mapping,
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
OUT = ARTIFACTS / "paper1_v2_volatility_regime"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V2_VOLATILITY_REGIME.md"

BASE_FEATURES = [
    "surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour",
] + [f"category_{x}" for x in [
    "inflation", "labor", "growth", "rates", "sentiment", "trade", "housing", "commodities", "other",
]] + ["orientation_base", "orientation_quote"]

VOL_FEATURES = [
    "pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h",
    "pre_vol_pips_6h", "pre_vol_pips_24h", "pre_vol_z_24h",
    "surprise_fc_x_pre_vol", "surprise_prev_x_pre_vol",
    "vol_regime_low", "vol_regime_normal", "vol_regime_high",
]

FEATURES = BASE_FEATURES + VOL_FEATURES
LABEL_THRESHOLDS = [0.60, 0.80, 1.00, 1.25]
MIN_ABS_PIPS = 2.0
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"]


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
    keep = ["price_time", "pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h", "pre_vol_pips_6h", "pre_vol_pips_24h"]
    return px[keep].sort_values("price_time")


def attach_price_context(events, pair, datatest=False):
    ctx = build_price_context(pair, datatest=datatest)
    out = pd.merge_asof(
        events.sort_values("event_time"),
        ctx,
        left_on="event_time",
        right_on="price_time",
        direction="forward",
        tolerance=pd.Timedelta(hours=2),
    )
    return out.drop(columns=["price_time"], errors="ignore")


def add_vol_features(d, train_stats=None, fit=False):
    d = d.copy()
    numeric = ["surprise_fc", "surprise_prev"] + [c for c in VOL_FEATURES if c.startswith("pre_")]
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h", "pre_vol_pips_6h", "pre_vol_pips_24h"]:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    d["pre_vol_pips_24h"] = d["pre_vol_pips_24h"].fillna(d["pre_vol_pips_6h"]).fillna(d["pre_range_pips_1h"]).fillna(0.0)
    d["pre_vol_pips_6h"] = d["pre_vol_pips_6h"].fillna(d["pre_vol_pips_24h"]).fillna(0.0)
    d["pre_range_pips_1h"] = d["pre_range_pips_1h"].fillna(0.0)
    d["pre_body_pips_1h"] = d["pre_body_pips_1h"].fillna(0.0)
    d["pre_ret_pips_1h"] = d["pre_ret_pips_1h"].fillna(0.0)

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
    d["surprise_fc_x_pre_vol"] = pd.to_numeric(d["surprise_fc"], errors="coerce").fillna(0.0) * d["pre_vol_z_24h"]
    d["surprise_prev_x_pre_vol"] = pd.to_numeric(d["surprise_prev"], errors="coerce").fillna(0.0) * d["pre_vol_z_24h"]
    d["vol_regime_low"] = (d["pre_vol_pips_24h"] <= d["vol_q33"]).astype("int8")
    d["vol_regime_high"] = (d["pre_vol_pips_24h"] >= d["vol_q66"]).astype("int8")
    d["vol_regime_normal"] = ((d["vol_regime_low"] == 0) & (d["vol_regime_high"] == 0)).astype("int8")
    return d, train_stats


def make_vol_label(d, threshold):
    pips = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    vol = pd.to_numeric(d["pre_vol_pips_24h"], errors="coerce").replace(0, np.nan).fillna(d["pre_vol_pips_6h"]).replace(0, np.nan).fillna(1.0)
    active = (pips.abs() >= np.maximum(MIN_ABS_PIPS, vol * threshold))
    return np.where(active, np.sign(pips).astype(int), 0)


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
    use = ["event_time", "Name", "CountryCode", "pair", "split", "trade_label_1h", "domestic_return_pips_1h"] + BASE_FEATURES
    raw = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    raw["event_time"] = pd.to_datetime(raw["event_time"], errors="coerce")
    raw = raw[raw["pair"].isin(PAIRS)].dropna(subset=["event_time", "pair"]).copy()
    parts = []
    for i, pair in enumerate(PAIRS, 1):
        part = raw[raw["pair"].eq(pair)].copy()
        if part.empty:
            continue
        progress_step("TRAIN_CONTEXT", i, len(PAIRS), pair=pair, rows=len(part))
        parts.append(attach_price_context(part, pair, datatest=False))
    out = pd.concat(parts, ignore_index=True)
    out, stats = add_vol_features(out, fit=True)
    return out, stats


def build_oos_data(train_stats):
    import importlib
    stage65 = importlib.import_module("65_test_event_character_aware_oos")
    d = stage65.build_oos_dataset()
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    d = d[d["pair"].isin(PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    parts = []
    for i, pair in enumerate(PAIRS, 1):
        part = d[d["pair"].eq(pair)].copy()
        if part.empty:
            continue
        progress_step("OOS_CONTEXT", i, len(PAIRS), pair=pair, rows=len(part))
        parts.append(attach_price_context(part, pair, datatest=True))
    out = pd.concat(parts, ignore_index=True)
    out["split"] = "oos"
    out, _ = add_vol_features(out, train_stats=train_stats, fit=False)
    out["year"] = pd.to_datetime(out["event_time"], errors="coerce").dt.year
    return out


def choose_best(rows):
    frame = pd.DataFrame(rows)
    eligible = frame[(frame["split"].eq("validation")) & (frame["trades"] >= 50)].copy()
    if eligible.empty:
        eligible = frame[frame["split"].eq("validation")].copy()
    eligible["rank_score"] = eligible["net_pips"] + 250.0 * eligible["profit_factor"].replace(np.inf, 10.0) + 100.0 * eligible["macro_f1"]
    return eligible.sort_values(["rank_score", "net_pips", "profit_factor"], ascending=False).iloc[0]


def write_report(validation_results, oos_results, selected):
    def md_table(frame):
        if frame.empty:
            return "_Tidak ada data._"
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            values = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    values.append(f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}")
                else:
                    values.append(str(v))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    total_dev = oos_results[oos_results["period"].eq("oos_dev_2024_2025")]["net_pips"].sum()
    total_final = oos_results[oos_results["period"].eq("final_oos_2026")]["net_pips"].sum()
    lines = [
        "# Laporan Paper 1 V2 - Volatility Regime Event Classification",
        "",
        "Eksperimen ini menguji apakah klasifikasi event saat volatilitas pasar berubah dapat memperbaiki baseline Paper 1.",
        "Label trade dibuat dari return 1H yang harus cukup besar relatif terhadap volatilitas pra-event 24H.",
        "",
        "## Ringkasan OOS",
        "",
        f"- OOS development 2024-2025 net pips: {total_dev:,.1f}",
        f"- Final OOS 2026 net pips: {total_final:,.1f}",
        "- Final OOS 2026 hanya dicatat sebagai audit, bukan untuk tuning.",
        "",
        "## Model Terpilih Per Pair",
        "",
        md_table(selected[["pair", "model", "label_threshold", "trades", "net_pips", "profit_factor", "macro_f1"]]),
        "",
        "## Hasil OOS Per Pair/Model",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
        "",
        "## Keputusan",
        "",
        "Bandingkan file ini dengan baseline aktif sebelum mengganti model. Jika OOS development lebih buruk, eksperimen ditolak dan baseline tetap dipakai.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d, stats = load_training_data()
    stats.to_csv(OUT / "volatility_train_statistics.csv", index=False)
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)

    validation_rows = []
    selected_rows = []
    selected_models = {}
    total_jobs = len(PAIRS) * len(LABEL_THRESHOLDS) * 3
    job = 0
    for pair in PAIRS:
        part = d[d["pair"].eq(pair)].copy()
        tr = part[part["split"].eq("train")].copy()
        va = part[part["split"].eq("validation")].copy()
        candidate_rows = []
        candidate_models = {}
        for threshold in LABEL_THRESHOLDS:
            ytr = make_vol_label(tr, threshold)
            yva = make_vol_label(va, threshold)
            for model_name, model in model_defs(seed=202 + int(threshold * 100)).items():
                job += 1
                progress_step("MODEL_START", job, total_jobs, pair=pair, model=model_name, label_threshold=threshold, train=len(tr), validation=len(va))
                t = time.perf_counter()
                model.fit(tr[FEATURES], ytr)
                pred = np.asarray(model.predict(va[FEATURES])).reshape(-1).astype(int)
                score = score_predictions(yva, pred, va["domestic_return_pips_1h"].to_numpy(float))
                row = {"pair": pair, "model": model_name, "label_threshold": threshold, "split": "validation", "seconds": time.perf_counter() - t, **score}
                validation_rows.append(row)
                candidate_rows.append(row)
                candidate_models[(model_name, threshold)] = model
                progress_step("MODEL_DONE", pair=pair, model=model_name, label_threshold=threshold, net_pips=f"{score['net_pips']:.1f}", pf=f"{score['profit_factor']:.4f}", trades=score["trades"])
        best = choose_best(candidate_rows)
        key = (best["model"], float(best["label_threshold"]))
        selected = dict(best)
        selected_rows.append(selected)
        selected_models[pair] = {"model": candidate_models[key], "model_name": key[0], "label_threshold": key[1]}
        joblib.dump({"model": candidate_models[key], "features": FEATURES, "label_threshold": key[1], "train_stats": stats}, MODELS / f"{key[0]}_{pair}_volatility_regime.joblib")
        progress_step("PAIR_SELECTED", pair=pair, model=key[0], label_threshold=key[1], validation_net_pips=f"{best['net_pips']:.1f}")

    validation = pd.DataFrame(validation_rows)
    selected = pd.DataFrame(selected_rows)
    validation.to_csv(OUT / "v2_validation_grid_results.csv", index=False)
    selected.to_csv(OUT / "v2_selected_models.csv", index=False)

    oos = build_oos_data(stats)
    for c in FEATURES:
        oos[c] = pd.to_numeric(oos[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    predictions = []
    oos_rows = []
    for pair, bundle in selected_models.items():
        part = oos[oos["pair"].eq(pair)].copy()
        if part.empty:
            continue
        threshold = bundle["label_threshold"]
        y = make_vol_label(part, threshold)
        pred = np.asarray(bundle["model"].predict(part[FEATURES])).reshape(-1).astype(int)
        pred_frame = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year"]].copy()
        pred_frame["model"] = bundle["model_name"]
        pred_frame["label_threshold"] = threshold
        pred_frame["signal"] = pred
        pred_frame["strategy_pips"] = pred_frame["domestic_return_pips_1h"].to_numpy(float) * pred
        predictions.append(pred_frame)
        for period, mask in [("oos_dev_2024_2025", pred_frame["year"].between(2024, 2025)), ("final_oos_2026", pred_frame["year"].eq(2026))]:
            active = pred_frame[mask & pred_frame["signal"].ne(0)].copy()
            x = active["strategy_pips"].to_numpy(float)
            y_slice = y[mask.to_numpy()]
            pred_slice = pred[mask.to_numpy()]
            score = score_predictions(y_slice, pred_slice, pred_frame.loc[mask, "domestic_return_pips_1h"].to_numpy(float))
            oos_rows.append({"period": period, "pair": pair, "model": bundle["model_name"], "label_threshold": threshold, **score})
        progress_step("OOS_DONE", pair=pair, model=bundle["model_name"])
    oos_results = pd.DataFrame(oos_rows)
    oos_results.to_csv(OUT / "v2_oos_results.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(OUT / "v2_oos_predictions.csv", index=False)
    metadata = {
        "variant": "volatility_regime_event_classification",
        "train": "2010-2020",
        "validation": "2021-2023",
        "oos_development": "2024-2025",
        "final_oos": "2026",
        "label": "sign(domestic_return_1h) only when abs(return_pips) exceeds max(2 pips, threshold * pre_event_24h_volatility)",
        "models": ["elastic_net", "random_forest", "catboost" if CatBoostClassifier is not None else "catboost_unavailable"],
        "features": FEATURES,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(validation, oos_results, selected)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
