from common import *

import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PHASE2_DIR = ARTIFACTS / "phase2_two_stage"
OUT_DIR = ARTIFACTS / "phase3_two_stage_models"
JOBLIB_DIR = OUT_DIR / "joblib"
REPORT_DIR = OUT_DIR / "reports"
DATASET = PHASE2_DIR / "two_stage_modeling_dataset.csv"
METADATA = PHASE2_DIR / "two_stage_metadata.json"

TARGET = "trade_target"
POSITIVE_CLASS = 1
THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_features():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    return meta["features"]


def build_model(name):
    if name == "elastic_net_trade_filter":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                penalty="elasticnet",
                solver="saga",
                C=0.5,
                l1_ratio=0.5,
                random_state=42,
                n_jobs=-1,
            ),
        )
    if name == "random_forest_trade_filter":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
    if name == "catboost_trade_filter":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=100,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {name}")


def positive_probability(model, x):
    probs = model.predict_proba(x)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    pos_idx = int(np.where(classes == POSITIVE_CLASS)[0][0])
    return np.asarray(probs)[:, pos_idx]


def metric_dict(model_name, split_name, pair, y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).reshape(-1)
    pred = (prob >= threshold).astype(int)
    if len(y_true) == 0:
        return {
            "model": model_name,
            "stage": "stage1_trade_filter",
            "split": split_name,
            "pair": pair,
            "n": 0,
            "threshold": threshold,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "trade_precision": np.nan,
            "trade_recall": np.nan,
            "trade_f1": np.nan,
            "pred_trade_count": 0,
            "true_trade_count": 0,
            "pred_trade_ratio": np.nan,
            "true_trade_ratio": np.nan,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0
    )
    return {
        "model": model_name,
        "stage": "stage1_trade_filter",
        "split": split_name,
        "pair": pair,
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "trade_precision": float(precision[0]),
        "trade_recall": float(recall[0]),
        "trade_f1": float(f1[0]),
        "pred_trade_count": int(pred.sum()),
        "true_trade_count": int(y_true.sum()),
        "pred_trade_ratio": float(pred.mean()),
        "true_trade_ratio": float(y_true.mean()),
    }


def threshold_score(row):
    # Stage 1 should not be too timid: a good filter keeps recall useful while avoiding trade-everything.
    trade_ratio_gap = abs(row["pred_trade_ratio"] - row["true_trade_ratio"])
    collapse_penalty = 0.20 if row["pred_trade_ratio"] < 0.05 or row["pred_trade_ratio"] > 0.95 else 0.0
    return (
        row["macro_f1"]
        + 0.50 * row["balanced_accuracy"]
        + 0.25 * row["trade_f1"]
        - 0.25 * trade_ratio_gap
        - collapse_penalty
    )


def df_to_markdown(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.6f}"
            else:
                value = "" if pd.isna(value) else str(value)
            values.append(value.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def choose_thresholds(model_name, validation_df, probs):
    rows = []
    for pair, idx in validation_df.groupby("pair").groups.items():
        y_pair = validation_df.loc[idx, TARGET].astype(int).values
        p_pair = probs[validation_df.index.get_indexer(idx)]
        best = None
        for threshold in THRESHOLD_GRID:
            row = metric_dict(model_name, "validation", pair, y_pair, p_pair, threshold)
            row["score"] = threshold_score(row)
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
        if best is None:
            continue
    return pd.DataFrame(rows)


def summarize_model(model_name, model, df, features, threshold_by_pair):
    rows = []
    for split_name in ["train", "validation"]:
        split_df = df[df["split"].eq(split_name)].copy()
        if split_df.empty:
            continue
        probs = positive_probability(model, split_df[features])
        split_df["_prob_trade"] = probs
        for pair, part in split_df.groupby("pair", sort=True):
            threshold = float(threshold_by_pair.get(pair, 0.50))
            rows.append(metric_dict(model_name, split_name, pair, part[TARGET], part["_prob_trade"], threshold))
        rows.append(metric_dict(model_name, split_name, "ALL", split_df[TARGET], split_df["_prob_trade"], 0.50))
    return rows


def main():
    print("[PROGRESS] START 40_phase3_stage1_train_trade_filter.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLIB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    usecols = ["event_time", "Name", "CountryCode", "pair", "event_year", "split", TARGET] + features
    print(f"[DATA] reading {DATASET}", flush=True)
    df = pd.read_csv(DATASET, usecols=usecols, low_memory=False)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df[df["split"].isin(["train", "validation"])].copy()

    print(f"[DATA] rows={len(df)} features={len(features)} pairs={sorted(df['pair'].dropna().unique().tolist())}", flush=True)
    print(f"[DATA] target_counts={df[TARGET].value_counts().sort_index().to_dict()}", flush=True)
    for split_name in ["train", "validation"]:
        part = df[df["split"].eq(split_name)]
        print(f"[DATA] {split_name}_rows={len(part)} target_counts={part[TARGET].value_counts().sort_index().to_dict()}", flush=True)

    train_df = df[df["split"].eq("train")].copy()
    validation_df = df[df["split"].eq("validation")].copy()

    models = ["elastic_net_trade_filter", "random_forest_trade_filter", "catboost_trade_filter"]
    all_threshold_grid_rows = []
    selected_threshold_rows = []
    all_metric_rows = []
    status_rows = []

    for i, model_name in enumerate(models, start=1):
        started = time.perf_counter()
        print(f"[MODEL] START stage1 {i}/{len(models)} {model_name}", flush=True)
        try:
            model = build_model(model_name)
            model.fit(train_df[features], train_df[TARGET])
            elapsed = round(time.perf_counter() - started, 2)
            print(f"[MODEL] FIT_DONE {model_name} seconds={elapsed}", flush=True)

            val_probs = positive_probability(model, validation_df[features])
            threshold_grid = choose_thresholds(model_name, validation_df, val_probs)
            all_threshold_grid_rows.append(threshold_grid)
            selected = threshold_grid.sort_values(["pair", "score"], ascending=[True, False]).groupby("pair", as_index=False).head(1)
            selected["selected_on"] = "validation_2021_2023"
            selected_threshold_rows.append(selected)
            threshold_by_pair = dict(zip(selected["pair"], selected["threshold"]))

            all_metric_rows.extend(summarize_model(model_name, model, df, features, threshold_by_pair))

            model_path = JOBLIB_DIR / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            status_rows.append({
                "model": model_name,
                "stage": "stage1_trade_filter",
                "status": "trained",
                "seconds": elapsed,
                "artifact": str(model_path),
                "note": "",
            })
            print(f"[MODEL] SAVED {model_path}", flush=True)
        except Exception as exc:
            status_rows.append({
                "model": model_name,
                "stage": "stage1_trade_filter",
                "status": "failed",
                "seconds": round(time.perf_counter() - started, 2),
                "artifact": "",
                "note": str(exc),
            })
            print(f"[MODEL] FAILED {model_name} error={exc}", flush=True)

    threshold_grid_df = pd.concat(all_threshold_grid_rows, ignore_index=True) if all_threshold_grid_rows else pd.DataFrame()
    selected_threshold_df = pd.concat(selected_threshold_rows, ignore_index=True) if selected_threshold_rows else pd.DataFrame()
    metric_df = pd.DataFrame(all_metric_rows)
    status_df = pd.DataFrame(status_rows)

    threshold_grid_df.to_csv(OUT_DIR / "stage1_trade_filter_threshold_grid.csv", index=False)
    selected_threshold_df.to_csv(OUT_DIR / "stage1_trade_filter_thresholds.csv", index=False)
    metric_df.to_csv(OUT_DIR / "stage1_trade_filter_metrics.csv", index=False)
    status_df.to_csv(OUT_DIR / "stage1_training_status.csv", index=False)

    metadata = {
        "stage": "stage1_trade_filter",
        "target": TARGET,
        "target_definition": "1=TRADE, 0=NO_TRADE",
        "features": features,
        "feature_count": len(features),
        "train_period": "2010-2020",
        "validation_period": "2021-2023",
        "final_test_policy": "2024+ is not used in Stage 1 training or threshold selection.",
        "threshold_file": str(OUT_DIR / "stage1_trade_filter_thresholds.csv"),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT_DIR / "stage1_trade_filter_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# Phase 3 Stage 1 - Trade Filter",
        "",
        "Stage 1 melatih filter `TRADE` vs `NO_TRADE` memakai dataset Fase 2.",
        "",
        f"- Rows used: {len(df):,}",
        f"- Features: {len(features)}",
        "- Training: 2010-2020",
        "- Validation / threshold selection: 2021-2023",
        "- Final OOS 2024+ tidak dipakai.",
        "",
        "## Status",
        "",
        df_to_markdown(status_df),
        "",
        "## Validation Metrics - ALL Threshold 0.50",
        "",
        df_to_markdown(metric_df[(metric_df["split"].eq("validation")) & (metric_df["pair"].eq("ALL"))]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'stage1_trade_filter_thresholds.csv'}`",
        f"- `{OUT_DIR / 'stage1_trade_filter_metrics.csv'}`",
        f"- `{OUT_DIR / 'stage1_training_status.csv'}`",
        f"- `{OUT_DIR / 'stage1_trade_filter_metadata.json'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE3_STAGE1_TRADE_FILTER.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OUTPUT] {OUT_DIR / 'stage1_trade_filter_thresholds.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'stage1_trade_filter_metrics.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'stage1_training_status.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE3_STAGE1_TRADE_FILTER.md'}", flush=True)
    print("[PROGRESS] DONE 40_phase3_stage1_train_trade_filter.py", flush=True)


if __name__ == "__main__":
    main()
