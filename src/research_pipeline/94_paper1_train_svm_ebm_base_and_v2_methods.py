"""Paper 1 SVM/EBM pipeline for VPS.

This script prepares Version 1 and Version 2 candidates for SVM-RBF and EBM.

Version 1:
    Base model only.

Version 2:
    Base model + additional method:
    - Sparse Event Selection
    - Jump / Volatility Labeling
    - Event Congestion Filter

The script is intentionally self-contained so it can be uploaded to a VPS with
the existing research_pipeline folder.
"""
from pathlib import Path
import json
import os
import time
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:
    ExplainableBoostingClassifier = None

from common import ARTIFACTS, DATA, pair_file, pip_size, progress_done, progress_start, progress_step, profit_factor, read_prices

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
SOURCE = PIPE / "modeling_dataset_multi_currency.csv"
OUT = ARTIFACTS / os.environ.get("PAPER1_OUTPUT_NAME", "paper1_svm_ebm_base_and_v2_methods")
MODELS_DIR = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_SVM_EBM_BASE_DAN_V2_METHODS.md"

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
VOL_FEATURES = [
    "pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h",
    "pre_vol_pips_6h", "pre_vol_pips_24h", "pre_vol_z_24h",
    "jump_barrier_pips", "surprise_fc_x_pre_vol", "surprise_prev_x_pre_vol",
    "vol_regime_low", "vol_regime_normal", "vol_regime_high",
]

TARGET_MODELS = [m.strip().lower() for m in os.environ.get("PAPER1_SVM_EBM_MODELS", "svm_rbf,ebm").split(",") if m.strip()]
TARGET_VARIANTS = [v.strip() for v in os.environ.get("PAPER1_VARIANTS", "").split(",") if v.strip()]
SVM_MODE = os.environ.get("PAPER1_SVM_MODE", "approx").lower()  # approx or exact
HEARTBEAT_SECONDS = int(os.environ.get("PAPER1_HEARTBEAT_SECONDS", "60"))
EBM_MAX_ROUNDS = int(os.environ.get("PAPER1_EBM_MAX_ROUNDS", "5000"))
EBM_OUTER_BAGS = int(os.environ.get("PAPER1_EBM_OUTER_BAGS", "4"))
RBF_COMPONENTS = int(os.environ.get("PAPER1_RBF_COMPONENTS", "800"))
RBF_GAMMA = float(os.environ.get("PAPER1_RBF_GAMMA", "0.05"))
MIN_SPARSE_EVENTS = int(os.environ.get("PAPER1_SPARSE_MIN_EVENTS", "15"))
MIN_SPARSE_PF = float(os.environ.get("PAPER1_SPARSE_MIN_PF", "1.05"))
MIN_SPARSE_POS_YEAR_RATIO = float(os.environ.get("PAPER1_SPARSE_MIN_POS_YEAR_RATIO", "0.35"))
JUMP_BARRIER_MULTIPLIER = float(os.environ.get("PAPER1_JUMP_BARRIER_MULTIPLIER", "1.0"))
MIN_JUMP_PIPS = float(os.environ.get("PAPER1_MIN_JUMP_PIPS", "3.0"))


def normalize_event_name(value):
    return " ".join(str(value).strip().lower().split())


def safe_pf(values):
    return float(profit_factor(values))


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


def build_model(name, seed):
    if name == "svm_rbf":
        if SVM_MODE == "exact":
            return make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    cache_size=4000,
                    random_state=seed,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=RBF_GAMMA, n_components=RBF_COMPONENTS, random_state=seed),
            LogisticRegression(max_iter=4000, class_weight="balanced", solver="saga", n_jobs=-1, random_state=seed),
        )
    if name == "ebm":
        if ExplainableBoostingClassifier is None:
            raise RuntimeError("Package 'interpret' is not installed; cannot train EBM.")
        return ExplainableBoostingClassifier(
            interactions=0,
            outer_bags=EBM_OUTER_BAGS,
            max_bins=256,
            learning_rate=0.015,
            max_rounds=EBM_MAX_ROUNDS,
            early_stopping_rounds=100,
            validation_size=0.15,
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_with_heartbeat(model, x, y, label):
    stop = threading.Event()
    started = time.perf_counter()

    def beat():
        while not stop.wait(HEARTBEAT_SECONDS):
            counts = pd.Series(y).value_counts().sort_index().to_dict()
            print(
                f"[HEARTBEAT] {label} fitting elapsed={time.perf_counter() - started:.1f}s "
                f"rows={len(x)} features={x.shape[1]} class_counts={counts}",
                flush=True,
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x, y)
    finally:
        stop.set()
        thread.join(timeout=1)
    print(f"[MODEL] FIT_DONE {label} seconds={time.perf_counter() - started:.1f}", flush=True)
    return model


def add_event_key(d):
    out = d.copy()
    out["event_name_norm"] = out["Name"].map(normalize_event_name)
    out["event_key"] = out["pair"].astype(str) + "|" + out["CountryCode"].astype(str) + "|" + out["event_name_norm"].astype(str)
    return out


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
    out = d.copy()
    derived = [
        "vol_median", "vol_std", "vol_q33", "vol_q66", "pre_vol_z_24h", "jump_barrier_pips",
        "surprise_fc_x_pre_vol", "surprise_prev_x_pre_vol", "vol_regime_low", "vol_regime_normal", "vol_regime_high",
    ]
    out = out.drop(columns=derived, errors="ignore")
    for c in ["pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h", "pre_vol_pips_6h", "pre_vol_pips_24h", "surprise_fc", "surprise_prev"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["pre_vol_pips_24h"] = out["pre_vol_pips_24h"].fillna(out["pre_vol_pips_6h"]).fillna(out["pre_range_pips_1h"]).fillna(0.0)
    out["pre_vol_pips_6h"] = out["pre_vol_pips_6h"].fillna(out["pre_vol_pips_24h"]).fillna(0.0)
    for c in ["pre_range_pips_1h", "pre_body_pips_1h", "pre_ret_pips_1h"]:
        out[c] = out[c].fillna(0.0)
    if fit:
        train = out[out["split"].eq("train")].copy()
        train_stats = train.groupby("pair")["pre_vol_pips_24h"].agg(
            vol_median="median",
            vol_std="std",
            vol_q33=lambda s: s.quantile(0.33),
            vol_q66=lambda s: s.quantile(0.66),
        ).reset_index()
    out = out.merge(train_stats, on="pair", how="left")
    out["vol_median"] = pd.to_numeric(out["vol_median"], errors="coerce").replace(0, np.nan).fillna(1.0)
    out["vol_std"] = pd.to_numeric(out["vol_std"], errors="coerce").replace(0, np.nan).fillna(1.0)
    out["vol_q33"] = pd.to_numeric(out["vol_q33"], errors="coerce").fillna(out["vol_median"])
    out["vol_q66"] = pd.to_numeric(out["vol_q66"], errors="coerce").fillna(out["vol_median"])
    out["pre_vol_z_24h"] = (out["pre_vol_pips_24h"] - out["vol_median"]) / out["vol_std"]
    out["jump_barrier_pips"] = np.maximum(MIN_JUMP_PIPS, out["pre_vol_pips_24h"] * barrier_multiplier)
    out["surprise_fc_x_pre_vol"] = out["surprise_fc"].fillna(0.0) * out["pre_vol_z_24h"]
    out["surprise_prev_x_pre_vol"] = out["surprise_prev"].fillna(0.0) * out["pre_vol_z_24h"]
    out["vol_regime_low"] = (out["pre_vol_pips_24h"] <= out["vol_q33"]).astype("int8")
    out["vol_regime_high"] = (out["pre_vol_pips_24h"] >= out["vol_q66"]).astype("int8")
    out["vol_regime_normal"] = ((out["vol_regime_low"] == 0) & (out["vol_regime_high"] == 0)).astype("int8")
    return out, train_stats


def make_jump_label(d):
    ret = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    vol = pd.to_numeric(d["pre_vol_pips_24h"], errors="coerce").replace(0, np.nan).fillna(d["pre_vol_pips_6h"]).replace(0, np.nan).fillna(1.0)
    barrier = np.maximum(MIN_JUMP_PIPS, vol * JUMP_BARRIER_MULTIPLIER)
    return np.where(ret >= barrier, 1, np.where(ret <= -barrier, -1, 0)).astype(int)


def score(pred, pnl_source):
    strategy = pnl_source * pred
    active = strategy[pred != 0]
    return {
        "trades": int(len(active)),
        "net_pips": float(active.sum()) if len(active) else 0.0,
        "profit_factor": safe_pf(active) if len(active) else 0.0,
        "winrate": float((active > 0).mean()) if len(active) else 0.0,
    }


def load_base_dataset():
    use = ["event_time", "Name", "CountryCode", "pair", "split", "event_year", "trade_label_1h", "domestic_return_pips_1h"] + BASE_FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d = d[d["pair"].isin(PAIRS)].dropna(subset=["event_time", "pair", "Name", "CountryCode"]).copy()
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["trade_label_1h"] = pd.to_numeric(d["trade_label_1h"], errors="coerce").fillna(0).astype(int)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    return add_event_key(d)


def prepare_variant_data(base):
    variants = {}
    variants["base_model_only"] = {"data": base.copy(), "features": BASE_FEATURES, "target": "trade_label_1h", "event_keys": {}}

    sparse = base.copy()
    train = sparse[sparse["split"].eq("train")].copy()
    train["strategy_pips_label"] = train["domestic_return_pips_1h"] * train["trade_label_1h"]
    train["year"] = pd.to_datetime(train["event_time"], errors="coerce").dt.year
    stats = []
    for keys, g in train.groupby(["pair", "CountryCode", "event_name_norm"], dropna=False):
        x = g["strategy_pips_label"].dropna()
        years = g.groupby("year")["strategy_pips_label"].sum()
        stats.append({
            "pair": keys[0],
            "CountryCode": keys[1],
            "event_name_norm": keys[2],
            "trade_count": int(len(x)),
            "net_pips": float(x.sum()) if len(x) else 0.0,
            "profit_factor": safe_pf(x),
            "avg_pips": float(x.mean()) if len(x) else 0.0,
            "positive_year_ratio": float((years > 0).sum() / max(len(years), 1)),
        })
    stats = pd.DataFrame(stats)
    selected = stats[
        (stats["trade_count"] >= MIN_SPARSE_EVENTS)
        & (stats["net_pips"] > 0)
        & (stats["profit_factor"] >= MIN_SPARSE_PF)
        & (stats["positive_year_ratio"] >= MIN_SPARSE_POS_YEAR_RATIO)
    ].copy()
    event_keys = {}
    for pair, g in selected.groupby("pair"):
        event_keys[pair] = set(g["pair"].astype(str) + "|" + g["CountryCode"].astype(str) + "|" + g["event_name_norm"].astype(str))
    variants["sparse_event_selection"] = {"data": sparse, "features": BASE_FEATURES, "target": "trade_label_1h", "event_keys": event_keys}
    selected.to_csv(OUT / "sparse_event_selection_units.csv", index=False)

    congestion = add_congestion_features(base)
    for c in CONGESTION_FEATURES:
        congestion[c] = pd.to_numeric(congestion[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variants["event_congestion_filter"] = {"data": congestion, "features": BASE_FEATURES + CONGESTION_FEATURES, "target": "trade_label_1h", "event_keys": {}}

    parts = []
    for i, pair in enumerate(PAIRS, 1):
        part = base[base["pair"].eq(pair)].copy()
        if part.empty:
            continue
        progress_step("VOL_CONTEXT", i, len(PAIRS), pair=pair, rows=len(part))
        parts.append(attach_price_context(part, pair, datatest=False))
    vol = pd.concat(parts, ignore_index=True)
    vol, train_stats = add_vol_features(vol, fit=True, barrier_multiplier=JUMP_BARRIER_MULTIPLIER)
    vol["jump_label_1h"] = make_jump_label(vol)
    train_stats.to_csv(OUT / "jump_volatility_train_statistics.csv", index=False)
    variants["jump_volatility_labeling"] = {"data": vol, "features": BASE_FEATURES + VOL_FEATURES, "target": "jump_label_1h", "event_keys": {}}
    return variants


def train_eval_variant(variant_name, spec, model_name, pair, seed):
    d = spec["data"]
    features = spec["features"]
    target = spec["target"]
    event_keys = spec.get("event_keys", {}).get(pair, set())
    part = d[d["pair"].eq(pair)].copy()
    tr = part[part["split"].eq("train")].copy()
    va = part[part["split"].eq("validation")].copy()
    if variant_name == "sparse_event_selection":
        tr = tr[tr["event_key"].isin(event_keys)].copy()
    tr = tr[tr[target].ne(0)].copy()
    if len(tr) < 200 or tr[target].nunique() < 2:
        return None, None, {"status": "skipped", "reason": "too_few_training_rows_or_classes"}
    for c in features:
        tr[c] = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        va[c] = pd.to_numeric(va[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = build_model(model_name, seed)
    label = f"{model_name}|{variant_name}|{pair}"
    fit_with_heartbeat(model, tr[features], tr[target].astype(int), label)
    raw_pred = np.asarray(model.predict(va[features])).reshape(-1).astype(int)
    pred = raw_pred
    if variant_name == "sparse_event_selection":
        pred = np.where(va["event_key"].isin(event_keys), raw_pred, 0)
    result = score(pred, va["domestic_return_pips_1h"].to_numpy(float))
    result.update({
        "pair": pair,
        "model": model_name,
        "variant": variant_name,
        "version": "version_1_base" if variant_name == "base_model_only" else "version_2_base_plus_method",
        "method_name": {
            "base_model_only": "Base Model Only",
            "sparse_event_selection": "Sparse Event Selection",
            "jump_volatility_labeling": "Jump/Volatility Labeling",
            "event_congestion_filter": "Event Congestion Filter",
        }[variant_name],
        "train_rows": len(tr),
        "validation_rows": len(va),
        "target": target,
        "features": len(features),
        "status": "trained",
    })
    return model, features, result


def evaluate_oos(oos_base, bundles):
    rows = []
    preds = []
    for bundle in bundles:
        pair = bundle["pair"]
        variant_name = bundle["variant"]
        model = bundle["model_object"]
        features = bundle["features"]
        event_keys = bundle.get("event_keys", set())
        target_rows = oos_base[oos_base["pair"].eq(pair)].copy()
        if target_rows.empty:
            continue
        d = target_rows
        if variant_name == "event_congestion_filter":
            d = add_congestion_features(d)
        elif variant_name == "jump_volatility_labeling":
            d = attach_price_context(d, pair, datatest=True)
            d, _ = add_vol_features(d, train_stats=bundle["train_stats"], fit=False, barrier_multiplier=JUMP_BARRIER_MULTIPLIER)
        for c in features:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_pred = np.asarray(model.predict(d[features])).reshape(-1).astype(int)
        pred = raw_pred
        if variant_name == "sparse_event_selection":
            pred = np.where(d["event_key"].isin(event_keys), raw_pred, 0)
        p = d[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year", "event_key"]].copy()
        p["model"] = bundle["model_name"]
        p["variant"] = variant_name
        p["version"] = bundle["version"]
        p["method_name"] = bundle["method_name"]
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
                "variant": variant_name,
                "version": bundle["version"],
                "method_name": bundle["method_name"],
                "trades": int(len(x)),
                "net_pips": float(x.sum()) if len(x) else 0.0,
                "profit_factor": safe_pf(x) if len(x) else 0.0,
                "winrate": float((x > 0).mean()) if len(x) else 0.0,
            })
        progress_step("OOS_DONE", pair=pair, model=bundle["model_name"], variant=variant_name)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def build_oos_base():
    import importlib
    stage65 = importlib.import_module("65_test_event_character_aware_oos")
    d = stage65.build_oos_dataset()
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    d = d[d["pair"].isin(PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    for c in BASE_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    return add_event_key(d)


def write_report(validation, oos):
    selected = []
    for (pair, model), g in oos[oos["period"].eq("oos_dev_2024_2025")].groupby(["pair", "model"]):
        best = g.sort_values(["net_pips", "profit_factor"], ascending=False).iloc[0].copy()
        selected.append(best)
    selected = pd.DataFrame(selected)
    totals = oos.groupby(["period", "version", "method_name", "model"]).agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        avg_pf=("profit_factor", "mean"),
        positive_pairs=("net_pips", lambda s: int((s > 0).sum())),
    ).reset_index()
    lines = [
        "# Laporan Paper 1 - SVM/EBM Base dan Versi 2",
        "",
        "Pipeline ini disiapkan untuk VPS. Versi 1 adalah base model murni; Versi 2 adalah base model + metode tambahan V3/V4/V5.",
        "",
        "## Konfigurasi",
        "",
        f"- Models: `{', '.join(TARGET_MODELS)}`",
        f"- SVM mode: `{SVM_MODE}`",
        f"- EBM max rounds: `{EBM_MAX_ROUNDS}`",
        f"- RBF components: `{RBF_COMPONENTS}`",
        "",
        "## Ringkasan OOS",
        "",
        md_table(totals),
        "",
        "## Kandidat Terbaik Per Pair + Model Berdasarkan OOS Development",
        "",
        md_table(selected[["pair", "model", "version", "method_name", "variant", "trades", "net_pips", "profit_factor", "winrate"]] if not selected.empty else selected),
        "",
        "## Validation Results",
        "",
        md_table(validation[["pair", "model", "version", "method_name", "variant", "trades", "net_pips", "profit_factor", "winrate", "status"]]),
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_dataset()
    variants = prepare_variant_data(base)
    if TARGET_VARIANTS:
        missing = sorted(set(TARGET_VARIANTS) - set(variants))
        if missing:
            raise ValueError(f"PAPER1_VARIANTS contains unknown variants: {missing}. Available: {sorted(variants)}")
        variants = {name: variants[name] for name in TARGET_VARIANTS}
        progress_step("VARIANT_FILTER", variants=",".join(variants.keys()))
    validation_rows = []
    bundles = []
    train_stats = None
    if (OUT / "jump_volatility_train_statistics.csv").exists():
        train_stats = pd.read_csv(OUT / "jump_volatility_train_statistics.csv")
    total = len(PAIRS) * len(TARGET_MODELS) * len(variants)
    job = 0
    for pair in PAIRS:
        for model_name in TARGET_MODELS:
            for variant_name, spec in variants.items():
                job += 1
                progress_step("TRAIN_START", job, total, pair=pair, model=model_name, variant=variant_name)
                try:
                    model, features, result = train_eval_variant(variant_name, spec, model_name, pair, seed=2026 + job)
                except Exception as exc:
                    result = {
                        "pair": pair, "model": model_name, "variant": variant_name,
                        "version": "version_1_base" if variant_name == "base_model_only" else "version_2_base_plus_method",
                        "method_name": variant_name, "trades": 0, "net_pips": 0.0, "profit_factor": 0.0, "winrate": 0.0,
                        "train_rows": 0, "validation_rows": 0, "target": "", "features": 0, "status": "failed", "reason": str(exc),
                    }
                    print(f"[ERROR] pair={pair} model={model_name} variant={variant_name} error={exc}", flush=True)
                    model = None
                    features = []
                validation_rows.append(result)
                if model is not None:
                    out_name = f"{model_name}_{pair}_{variant_name}.joblib"
                    event_keys = spec.get("event_keys", {}).get(pair, set())
                    bundle = {
                        "pair": pair,
                        "model_name": model_name,
                        "variant": variant_name,
                        "version": result["version"],
                        "method_name": result["method_name"],
                        "features": features,
                        "event_keys": sorted(event_keys),
                        "train_stats": train_stats,
                    }
                    joblib.dump({"model": model, **bundle}, MODELS_DIR / out_name)
                    bundles.append({"model_object": model, **bundle, "event_keys": set(event_keys)})
                progress_step("TRAIN_DONE", pair=pair, model=model_name, variant=variant_name, status=result.get("status"), net_pips=f"{result.get('net_pips', 0):.1f}")
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "svm_ebm_base_v2_validation_results.csv", index=False)
    oos_results, oos_predictions = evaluate_oos(build_oos_base(), bundles)
    oos_results.to_csv(OUT / "svm_ebm_base_v2_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "svm_ebm_base_v2_oos_predictions.csv", index=False)
    metadata = {
        "models": TARGET_MODELS,
        "svm_mode": SVM_MODE,
        "version_1": "base_model_only",
        "version_2_methods": ["sparse_event_selection", "jump_volatility_labeling", "event_congestion_filter"],
        "train": "2010_2020",
        "validation": "2021_2023",
        "oos_development": "2024_2025",
        "final_oos": "2026_audit_only",
        "note": "Prepared for VPS. Exact SVM can be enabled with PAPER1_SVM_MODE=exact.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(validation, oos_results)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
