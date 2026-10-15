from common import *

import json
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PHASE2_DIR = ARTIFACTS / "phase2_two_stage"
OUT_DIR = ARTIFACTS / "phase3_two_stage_models"
JOBLIB_DIR = OUT_DIR / "joblib"
DATASET = PHASE2_DIR / "two_stage_modeling_dataset.csv"
METADATA = PHASE2_DIR / "two_stage_metadata.json"

TARGET = "direction_target"
CONFIDENCE_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_features():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    return meta["features"]


def build_model(name):
    if name == "elastic_net_direction":
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
    if name == "random_forest_direction":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
    if name == "catboost_direction":
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


def class_probabilities(model, x):
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = probs[:, sell_idx]
    buy_prob = probs[:, buy_idx]
    pred = np.where(buy_prob >= sell_prob, 1, -1)
    confidence = np.maximum(buy_prob, sell_prob)
    return sell_prob, buy_prob, pred, confidence


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


def metric_dict(model_name, split_name, pair, y_true, pred, confidence=None, confidence_threshold=0.0):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    if confidence is not None and confidence_threshold > 0:
        mask = np.asarray(confidence) >= confidence_threshold
        y_eval = y_true[mask]
        pred_eval = pred[mask]
        selected_count = int(mask.sum())
    else:
        y_eval = y_true
        pred_eval = pred
        selected_count = int(len(y_true))

    if len(y_eval) == 0:
        return {
            "model": model_name,
            "stage": "stage2_direction",
            "split": split_name,
            "pair": pair,
            "n": int(len(y_true)),
            "selected_count": 0,
            "confidence_threshold": float(confidence_threshold),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "buy_precision": np.nan,
            "buy_recall": np.nan,
            "sell_precision": np.nan,
            "sell_recall": np.nan,
            "pred_buy_count": 0,
            "pred_sell_count": 0,
            "true_buy_count": int((y_true == 1).sum()),
            "true_sell_count": int((y_true == -1).sum()),
            "selected_ratio": 0.0,
        }

    labels = [-1, 1]
    precision, recall, _, _ = precision_recall_fscore_support(y_eval, pred_eval, labels=labels, zero_division=0)
    return {
        "model": model_name,
        "stage": "stage2_direction",
        "split": split_name,
        "pair": pair,
        "n": int(len(y_true)),
        "selected_count": selected_count,
        "confidence_threshold": float(confidence_threshold),
        "accuracy": accuracy_score(y_eval, pred_eval),
        "balanced_accuracy": balanced_accuracy_score(y_eval, pred_eval),
        "macro_f1": f1_score(y_eval, pred_eval, average="macro", zero_division=0),
        "buy_precision": float(precision[1]),
        "buy_recall": float(recall[1]),
        "sell_precision": float(precision[0]),
        "sell_recall": float(recall[0]),
        "pred_buy_count": int((pred_eval == 1).sum()),
        "pred_sell_count": int((pred_eval == -1).sum()),
        "true_buy_count": int((y_true == 1).sum()),
        "true_sell_count": int((y_true == -1).sum()),
        "selected_ratio": float(selected_count / max(1, len(y_true))),
    }


def confidence_score(row):
    # Direction model should be accurate without shrinking too aggressively.
    selection_penalty = 0.20 if row["selected_ratio"] < 0.25 else 0.0
    collapse_penalty = 0.15 if row["pred_buy_count"] == 0 or row["pred_sell_count"] == 0 else 0.0
    return row["macro_f1"] + 0.50 * row["balanced_accuracy"] + 0.10 * row["selected_ratio"] - selection_penalty - collapse_penalty


def choose_confidence_thresholds(model_name, validation_df, pred, confidence):
    rows = []
    for pair, idx in validation_df.groupby("pair").groups.items():
        y_pair = validation_df.loc[idx, TARGET].astype(int).values
        loc = validation_df.index.get_indexer(idx)
        pred_pair = pred[loc]
        conf_pair = confidence[loc]
        for threshold in CONFIDENCE_GRID:
            row = metric_dict(model_name, "validation", pair, y_pair, pred_pair, conf_pair, threshold)
            row["score"] = confidence_score(row)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_model(model_name, model, df, features, threshold_by_pair):
    rows = []
    for split_name in ["train", "validation"]:
        split_df = df[df["split"].eq(split_name)].copy()
        if split_df.empty:
            continue
        _, _, pred, confidence = class_probabilities(model, split_df[features])
        split_df["_pred_direction"] = pred
        split_df["_confidence"] = confidence
        for pair, part in split_df.groupby("pair", sort=True):
            threshold = float(threshold_by_pair.get(pair, 0.50))
            rows.append(metric_dict(model_name, split_name, pair, part[TARGET], part["_pred_direction"], part["_confidence"], threshold))
        rows.append(metric_dict(model_name, split_name, "ALL", split_df[TARGET], split_df["_pred_direction"], split_df["_confidence"], 0.50))
    return rows


def main():
    print("[PROGRESS] START 41_phase3_stage2_train_direction.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLIB_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    usecols = ["event_time", "Name", "CountryCode", "pair", "event_year", "split", "trade_target", TARGET] + features
    print(f"[DATA] reading {DATASET}", flush=True)
    df = pd.read_csv(DATASET, usecols=usecols, low_memory=False)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df[df["split"].isin(["train", "validation"])].copy()

    print(f"[DATA] rows={len(df)} features={len(features)} pairs={sorted(df['pair'].dropna().unique().tolist())}", flush=True)
    print(f"[DATA] direction_counts={df[TARGET].value_counts().sort_index().to_dict()}", flush=True)
    for split_name in ["train", "validation"]:
        part = df[df["split"].eq(split_name)]
        print(f"[DATA] {split_name}_rows={len(part)} direction_counts={part[TARGET].value_counts().sort_index().to_dict()}", flush=True)

    train_df = df[df["split"].eq("train")].copy()
    validation_df = df[df["split"].eq("validation")].copy()

    models = ["elastic_net_direction", "random_forest_direction", "catboost_direction"]
    all_threshold_grid_rows = []
    selected_threshold_rows = []
    all_metric_rows = []
    status_rows = []

    for i, model_name in enumerate(models, start=1):
        started = time.perf_counter()
        print(f"[MODEL] START stage2 {i}/{len(models)} {model_name}", flush=True)
        try:
            model = build_model(model_name)
            model.fit(train_df[features], train_df[TARGET])
            elapsed = round(time.perf_counter() - started, 2)
            print(f"[MODEL] FIT_DONE {model_name} seconds={elapsed}", flush=True)

            _, _, val_pred, val_confidence = class_probabilities(model, validation_df[features])
            threshold_grid = choose_confidence_thresholds(model_name, validation_df, val_pred, val_confidence)
            all_threshold_grid_rows.append(threshold_grid)
            selected = threshold_grid.sort_values(["pair", "score"], ascending=[True, False]).groupby("pair", as_index=False).head(1)
            selected["selected_on"] = "validation_2021_2023"
            selected_threshold_rows.append(selected)
            threshold_by_pair = dict(zip(selected["pair"], selected["confidence_threshold"]))

            all_metric_rows.extend(summarize_model(model_name, model, df, features, threshold_by_pair))

            model_path = JOBLIB_DIR / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            status_rows.append({
                "model": model_name,
                "stage": "stage2_direction",
                "status": "trained",
                "seconds": elapsed,
                "artifact": str(model_path),
                "note": "",
            })
            print(f"[MODEL] SAVED {model_path}", flush=True)
        except Exception as exc:
            status_rows.append({
                "model": model_name,
                "stage": "stage2_direction",
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

    threshold_grid_df.to_csv(OUT_DIR / "stage2_direction_confidence_grid.csv", index=False)
    selected_threshold_df.to_csv(OUT_DIR / "stage2_direction_thresholds.csv", index=False)
    metric_df.to_csv(OUT_DIR / "stage2_direction_metrics.csv", index=False)
    status_df.to_csv(OUT_DIR / "stage2_training_status.csv", index=False)

    metadata = {
        "stage": "stage2_direction",
        "target": TARGET,
        "target_definition": "1=BUY, -1=SELL; trained only on Stage-1 historical TRADE rows",
        "features": features,
        "feature_count": len(features),
        "train_period": "2010-2020",
        "validation_period": "2021-2023",
        "final_test_policy": "2024+ is not used in Stage 2 training or confidence selection.",
        "threshold_file": str(OUT_DIR / "stage2_direction_thresholds.csv"),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT_DIR / "stage2_direction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# Phase 3 Stage 2 - Direction Model",
        "",
        "Stage 2 melatih arah `SELL` vs `BUY` hanya pada event yang secara historis masuk kelas `TRADE`.",
        "",
        f"- Rows used: {len(df):,}",
        f"- Features: {len(features)}",
        "- Training: 2010-2020",
        "- Validation / confidence selection: 2021-2023",
        "- Final OOS 2024+ tidak dipakai.",
        "",
        "## Status",
        "",
        df_to_markdown(status_df),
        "",
        "## Validation Metrics - ALL Confidence 0.50",
        "",
        df_to_markdown(metric_df[(metric_df["split"].eq("validation")) & (metric_df["pair"].eq("ALL"))]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'stage2_direction_thresholds.csv'}`",
        f"- `{OUT_DIR / 'stage2_direction_metrics.csv'}`",
        f"- `{OUT_DIR / 'stage2_training_status.csv'}`",
        f"- `{OUT_DIR / 'stage2_direction_metadata.json'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE3_STAGE2_DIRECTION.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OUTPUT] {OUT_DIR / 'stage2_direction_thresholds.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'stage2_direction_metrics.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'stage2_training_status.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE3_STAGE2_DIRECTION.md'}", flush=True)
    print("[PROGRESS] DONE 41_phase3_stage2_train_direction.py", flush=True)


if __name__ == "__main__":
    main()
