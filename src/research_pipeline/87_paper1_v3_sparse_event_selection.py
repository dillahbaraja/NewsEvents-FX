"""Paper 1 V3: sparse event selection before direction modeling.

The selection rule is fit on 2010-2020 only. During validation/OOS, events that
were not selected at pair + country + event-name level are forced to NO_TRADE.
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
OUT = ARTIFACTS / "paper1_v3_sparse_event_selection"
MODELS = OUT / "models"
REPORT = OUT / "LAPORAN_PAPER1_V3_SPARSE_EVENT_SELECTION.md"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"]
FEATURES = [
    "surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour",
] + [f"category_{x}" for x in [
    "inflation", "labor", "growth", "rates", "sentiment", "trade", "housing", "commodities", "other",
]] + ["orientation_base", "orientation_quote"]

MIN_EVENTS_GRID = [8, 15, 25]
MIN_PF_GRID = [1.00, 1.05, 1.10]
MIN_POS_YEAR_RATIO_GRID = [0.35, 0.45]
MIN_AVG_PIPS_GRID = [0.0, 0.10]


def normalize_event_name(value):
    return " ".join(str(value).strip().lower().split())


def safe_pf(values):
    return float(profit_factor(values))


def model_defs(seed=303):
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


def event_stats(train):
    d = train.copy()
    d["event_name_norm"] = d["Name"].map(normalize_event_name)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    d["strategy_pips_label"] = d["domestic_return_pips_1h"] * d["trade_label_1h"]
    rows = []
    for keys, g in d.groupby(["pair", "CountryCode", "event_name_norm"], dropna=False):
        x = g["strategy_pips_label"].dropna()
        years = g.groupby("year")["strategy_pips_label"].sum()
        active_years = int((years.index.notna()).sum())
        positive_years = int((years > 0).sum())
        rows.append({
            "pair": keys[0],
            "CountryCode": keys[1],
            "event_name_norm": keys[2],
            "trade_count": int(len(x)),
            "net_pips": float(x.sum()) if len(x) else 0.0,
            "gross_profit": float(x[x > 0].sum()) if len(x) else 0.0,
            "gross_loss": float(-x[x < 0].sum()) if len(x) else 0.0,
            "profit_factor": safe_pf(x),
            "avg_pips": float(x.mean()) if len(x) else 0.0,
            "winrate": float((x > 0).mean()) if len(x) else 0.0,
            "active_years": active_years,
            "positive_years": positive_years,
            "positive_year_ratio": float(positive_years / max(active_years, 1)),
        })
    return pd.DataFrame(rows)


def select_events(stats, min_events, min_pf, min_pos_year_ratio, min_avg_pips):
    selected = stats[
        (stats["trade_count"] >= min_events)
        & (stats["net_pips"] > 0)
        & (stats["profit_factor"] >= min_pf)
        & (stats["avg_pips"] >= min_avg_pips)
        & (stats["positive_year_ratio"] >= min_pos_year_ratio)
    ].copy()
    selected["selection_rule"] = f"min_events={min_events};min_pf={min_pf};min_pos_year_ratio={min_pos_year_ratio};min_avg_pips={min_avg_pips}"
    return selected


def add_event_key(d):
    out = d.copy()
    out["event_name_norm"] = out["Name"].map(normalize_event_name)
    out["event_key"] = out["pair"].astype(str) + "|" + out["CountryCode"].astype(str) + "|" + out["event_name_norm"].astype(str)
    return out


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
    use = ["event_time", "Name", "CountryCode", "pair", "split", "trade_label_1h", "domestic_return_pips_1h"] + FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d = d[d["pair"].isin(PAIRS)].dropna(subset=["event_time", "pair", "Name", "CountryCode"]).copy()
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["trade_label_1h"] = pd.to_numeric(d["trade_label_1h"], errors="coerce").fillna(0).astype(int)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    return add_event_key(d)


def build_oos():
    import importlib
    stage65 = importlib.import_module("65_test_event_character_aware_oos")
    d = stage65.build_oos_dataset()
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    d = d[d["pair"].isin(PAIRS)].copy()
    d = d.rename(columns={"domestic_return_pips": "domestic_return_pips_1h"})
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    return add_event_key(d)


def validation_grid(d, stats):
    rows = []
    selected_cache = {}
    total = len(PAIRS) * len(MIN_EVENTS_GRID) * len(MIN_PF_GRID) * len(MIN_POS_YEAR_RATIO_GRID) * len(MIN_AVG_PIPS_GRID)
    job = 0
    for pair in PAIRS:
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train")].copy()
        va = p[p["split"].eq("validation")].copy()
        for min_events in MIN_EVENTS_GRID:
            for min_pf in MIN_PF_GRID:
                for min_pos_year_ratio in MIN_POS_YEAR_RATIO_GRID:
                    for min_avg_pips in MIN_AVG_PIPS_GRID:
                        job += 1
                        selected = select_events(
                            stats[stats["pair"].eq(pair)],
                            min_events,
                            min_pf,
                            min_pos_year_ratio,
                            min_avg_pips,
                        )
                        keys = set(selected["pair"].astype(str) + "|" + selected["CountryCode"].astype(str) + "|" + selected["event_name_norm"].astype(str))
                        selected_cache[(pair, min_events, min_pf, min_pos_year_ratio, min_avg_pips)] = selected
                        trf = tr[tr["event_key"].isin(keys) & tr["trade_label_1h"].ne(0)].copy()
                        if len(trf) < 200 or trf["trade_label_1h"].nunique() < 2:
                            continue
                        ytr = trf["trade_label_1h"].astype(int)
                        for model_name, model in model_defs(seed=303 + job).items():
                            progress_step("GRID_MODEL_START", job, total, pair=pair, model=model_name, events=len(keys), train=len(trf))
                            t = time.perf_counter()
                            model.fit(trf[FEATURES], ytr)
                            raw_pred = np.asarray(model.predict(va[FEATURES])).reshape(-1).astype(int)
                            pred = np.where(va["event_key"].isin(keys), raw_pred, 0)
                            result = score(va["trade_label_1h"].to_numpy(int), pred, va["domestic_return_pips_1h"].to_numpy(float))
                            rows.append({
                                "pair": pair,
                                "model": model_name,
                                "min_events": min_events,
                                "min_profit_factor": min_pf,
                                "min_positive_year_ratio": min_pos_year_ratio,
                                "min_avg_pips": min_avg_pips,
                                "selected_event_units": len(keys),
                                "train_rows": len(trf),
                                "seconds": time.perf_counter() - t,
                                **result,
                            })
                            progress_step("GRID_MODEL_DONE", pair=pair, model=model_name, net_pips=f"{result['net_pips']:.1f}", pf=f"{result['profit_factor']:.4f}", trades=result["trades"])
    return pd.DataFrame(rows), selected_cache


def choose_and_retrain(d, grid, selected_cache):
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
        selected_rows.append(best)
        key = (pair, int(best["min_events"]), float(best["min_profit_factor"]), float(best["min_positive_year_ratio"]), float(best["min_avg_pips"]))
        selected_events = selected_cache[key]
        event_keys = set(selected_events["pair"].astype(str) + "|" + selected_events["CountryCode"].astype(str) + "|" + selected_events["event_name_norm"].astype(str))
        p = d[d["pair"].eq(pair)].copy()
        tr = p[p["split"].eq("train") & p["event_key"].isin(event_keys) & p["trade_label_1h"].ne(0)].copy()
        model = model_defs(seed=777)[best["model"]]
        model.fit(tr[FEATURES], tr["trade_label_1h"].astype(int))
        bundles[pair] = {
            "model": model,
            "model_name": best["model"],
            "event_keys": event_keys,
            "selected_events": selected_events,
            "best": best,
        }
        joblib.dump(
            {"model": model, "features": FEATURES, "event_keys": sorted(event_keys), "best": best},
            MODELS / f"{best['model']}_{pair}_v3_sparse_event_selection.joblib",
        )
        progress_step("PAIR_SELECTED", pair=pair, model=best["model"], events=len(event_keys), validation_net_pips=f"{best['net_pips']:.1f}")
    return pd.DataFrame(selected_rows), bundles


def evaluate_oos(oos, bundles):
    rows = []
    preds = []
    for pair, bundle in bundles.items():
        part = oos[oos["pair"].eq(pair)].copy()
        if part.empty:
            continue
        raw_pred = np.asarray(bundle["model"].predict(part[FEATURES])).reshape(-1).astype(int)
        pred = np.where(part["event_key"].isin(bundle["event_keys"]), raw_pred, 0)
        p = part[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "year", "event_key"]].copy()
        p["model"] = bundle["model_name"]
        p["variant"] = "v3_sparse_event_selection"
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
                "variant": "v3_sparse_event_selection",
                "trades": int(len(x)),
                "net_pips": float(x.sum()) if len(x) else 0.0,
                "profit_factor": safe_pf(x) if len(x) else 0.0,
                "winrate": float((x > 0).mean()) if len(x) else 0.0,
                "selected_event_units": len(bundle["event_keys"]),
            })
        progress_step("OOS_DONE", pair=pair, model=bundle["model_name"])
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


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

    total = oos_results.groupby("period").agg(trades=("trades", "sum"), net_pips=("net_pips", "sum")).reset_index()
    lines = [
        "# Laporan Paper 1 V3 - Sparse Event Selection",
        "",
        "Eksperimen ini memilih event relevan per `pair + country + event_name` dari training 2010-2020 saja.",
        "Event yang tidak lolos selection dipaksa menjadi NO_TRADE saat validation dan OOS.",
        "",
        "## Ringkasan OOS",
        "",
        md_table(total),
        "",
        "## Model Terpilih Berdasarkan Validation",
        "",
        md_table(selected[["pair", "model", "min_events", "min_profit_factor", "min_positive_year_ratio", "min_avg_pips", "selected_event_units", "trades", "net_pips", "profit_factor"]]),
        "",
        "## Hasil OOS Per Pair",
        "",
        md_table(oos_results.sort_values(["period", "net_pips"], ascending=[True, False])),
        "",
        "## Keputusan",
        "",
        "Bandingkan hasil ini dengan baseline aktif di `paper1_pair_best_model_selection`. Jika OOS development 2024-2025 tidak lebih baik, V3 tidak mengganti baseline.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d = load_train_validation()
    stats = event_stats(d[d["split"].eq("train")])
    stats.to_csv(OUT / "v3_sparse_event_stats_train_2010_2020.csv", index=False)
    grid, selected_cache = validation_grid(d, stats)
    grid.to_csv(OUT / "v3_validation_grid_results.csv", index=False)
    selected, bundles = choose_and_retrain(d, grid, selected_cache)
    selected.to_csv(OUT / "v3_selected_models.csv", index=False)
    selected_events = []
    for pair, bundle in bundles.items():
        x = bundle["selected_events"].copy()
        x["selected_model"] = bundle["model_name"]
        selected_events.append(x)
    if selected_events:
        pd.concat(selected_events, ignore_index=True).to_csv(OUT / "v3_selected_events_per_pair.csv", index=False)
    oos_results, oos_predictions = evaluate_oos(build_oos(), bundles)
    oos_results.to_csv(OUT / "v3_oos_results.csv", index=False)
    oos_predictions.to_csv(OUT / "v3_oos_predictions.csv", index=False)
    metadata = {
        "variant": "v3_sparse_event_selection",
        "event_selection_fit": "train_2010_2020_only",
        "validation": "2021_2023",
        "oos_development": "2024_2025",
        "final_oos": "2026_audit_only",
        "features": FEATURES,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(selected, oos_results)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
