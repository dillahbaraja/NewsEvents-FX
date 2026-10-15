from common import *

import json
import os
import threading
import time
import warnings

import joblib
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PHASE2_DIR = ARTIFACTS / "phase2_two_stage"
OUT_DIR = ARTIFACTS / "phase8_svm_ebm_two_stage_models"
JOBLIB_DIR = OUT_DIR / "joblib"
DATASET = PHASE2_DIR / "two_stage_modeling_dataset.csv"
METADATA = PHASE2_DIR / "two_stage_metadata.json"

STAGE1_TARGET = "trade_target"
STAGE2_TARGET = "direction_target"
STAGE1_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
STAGE2_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MODELS = [m.strip().lower() for m in os.environ.get("PHASE8_MODELS", "svm_rbf_exact,ebm").split(",") if m.strip()]
EVAL_CHUNK_SIZE = int(os.environ.get("PHASE8_EVAL_CHUNK_SIZE", "10000"))

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def load_features():
    return json.loads(METADATA.read_text(encoding="utf-8"))["features"]


def build_model(kind):
    if kind == "svm_rbf_exact":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                cache_size=4000,
                random_state=42,
            ),
        )
    if kind in {"svm_rbf", "svm_rbf_approx"}:
        # Approximation is kept as a compute-friendly robustness variant, not as the primary SVM-RBF paper model.
        return make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=0.05, n_components=700, random_state=42),
            LogisticRegression(max_iter=4000, class_weight="balanced", solver="saga", n_jobs=-1, random_state=42),
        )
    if kind == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(
            interactions=0,
            outer_bags=4,
            max_bins=256,
            learning_rate=0.015,
            max_rounds=8000,
            early_stopping_rounds=100,
            validation_size=0.15,
            n_jobs=1,
            random_state=42,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def model_names(kind, stage):
    if stage == 1:
        return f"{kind}_trade_filter", f"{kind}_trade_filter.joblib"
    return f"{kind}_direction", f"{kind}_direction.joblib"


def fit_with_heartbeat(model, x_train, y_train, model_name, stage, interval=60):
    stop_event = threading.Event()
    started = time.perf_counter()
    rows = len(x_train)
    features = x_train.shape[1] if hasattr(x_train, "shape") and len(x_train.shape) > 1 else "?"
    class_counts = pd.Series(y_train).value_counts().sort_index().to_dict()

    def heartbeat():
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - started
            print(
                f"[HEARTBEAT] {model_name} stage={stage} fitting elapsed={elapsed:.1f}s "
                f"rows={rows} features={features} class_counts={class_counts}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        model.fit(x_train, y_train)
    finally:
        stop_event.set()
        thread.join(timeout=1)
    elapsed = time.perf_counter() - started
    print(f"[MODEL] FIT_DONE {model_name} stage={stage} seconds={elapsed:.2f}", flush=True)
    return model


def positive_probability(model, x, positive_class=1):
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        if score.ndim > 1:
            classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
            idx = int(np.where(classes == positive_class)[0][0])
            score = score[:, idx]
        score = np.clip(score, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-score))
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    idx = int(np.where(classes == positive_class)[0][0])
    return probs[:, idx]


def direction_outputs(model, x):
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        if score.ndim > 1:
            classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
            buy_idx = int(np.where(classes == 1)[0][0])
            sell_idx = int(np.where(classes == -1)[0][0])
            buy_score = score[:, buy_idx]
            sell_score = score[:, sell_idx]
            pred = np.where(buy_score >= sell_score, 1, -1)
            raw_confidence = np.maximum(buy_score, sell_score)
            confidence = 1.0 / (1.0 + np.exp(-np.clip(raw_confidence, -30.0, 30.0)))
            sell_prob = 1.0 / (1.0 + np.exp(-np.clip(sell_score, -30.0, 30.0)))
            buy_prob = 1.0 / (1.0 + np.exp(-np.clip(buy_score, -30.0, 30.0)))
            return sell_prob, buy_prob, pred, confidence
        prob_buy = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
        prob_sell = 1.0 - prob_buy
        pred = np.where(prob_buy >= prob_sell, 1, -1)
        confidence = np.maximum(prob_sell, prob_buy)
        return prob_sell, prob_buy, pred, confidence
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = probs[:, sell_idx]
    buy_prob = probs[:, buy_idx]
    pred = np.where(buy_prob >= sell_prob, 1, -1)
    confidence = np.maximum(sell_prob, buy_prob)
    return sell_prob, buy_prob, pred, confidence


def positive_probability_chunked(model, x, model_name, stage, positive_class=1):
    rows = []
    total = len(x)
    started = time.perf_counter()
    for start in range(0, total, EVAL_CHUNK_SIZE):
        end = min(start + EVAL_CHUNK_SIZE, total)
        rows.append(positive_probability(model, x.iloc[start:end], positive_class))
        print(
            f"[EVAL] {model_name} stage={stage} positive_score rows={end}/{total} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return np.concatenate(rows) if rows else np.array([])


def direction_outputs_chunked(model, x, model_name, stage):
    sell_parts, buy_parts, pred_parts, conf_parts = [], [], [], []
    total = len(x)
    started = time.perf_counter()
    for start in range(0, total, EVAL_CHUNK_SIZE):
        end = min(start + EVAL_CHUNK_SIZE, total)
        sell_prob, buy_prob, pred, confidence = direction_outputs(model, x.iloc[start:end])
        sell_parts.append(sell_prob)
        buy_parts.append(buy_prob)
        pred_parts.append(pred)
        conf_parts.append(confidence)
        print(
            f"[EVAL] {model_name} stage={stage} direction_score rows={end}/{total} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not sell_parts:
        empty = np.array([])
        return empty, empty, empty, empty
    return (
        np.concatenate(sell_parts),
        np.concatenate(buy_parts),
        np.concatenate(pred_parts),
        np.concatenate(conf_parts),
    )


def stage1_metrics(model_name, split_name, pair, y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[1], average=None, zero_division=0)
    return {
        "model": model_name, "stage": "stage1_trade_filter", "split": split_name, "pair": pair,
        "n": int(len(y_true)), "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "trade_precision": float(precision[0]), "trade_recall": float(recall[0]), "trade_f1": float(f1[0]),
        "pred_trade_count": int(pred.sum()), "true_trade_count": int(y_true.sum()),
        "pred_trade_ratio": float(pred.mean()), "true_trade_ratio": float(y_true.mean()),
    }


def stage1_score(row):
    gap = abs(row["pred_trade_ratio"] - row["true_trade_ratio"])
    collapse = 0.20 if row["pred_trade_ratio"] < 0.05 or row["pred_trade_ratio"] > 0.95 else 0.0
    return row["macro_f1"] + 0.50 * row["balanced_accuracy"] + 0.25 * row["trade_f1"] - 0.25 * gap - collapse


def stage2_metrics(model_name, split_name, pair, y_true, pred, confidence, threshold):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    mask = np.asarray(confidence) >= threshold
    y_eval = y_true[mask]
    pred_eval = pred[mask]
    if len(y_eval) == 0:
        return {
            "model": model_name, "stage": "stage2_direction", "split": split_name, "pair": pair,
            "n": int(len(y_true)), "selected_count": 0, "confidence_threshold": float(threshold),
            "accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan,
            "buy_precision": np.nan, "buy_recall": np.nan, "sell_precision": np.nan, "sell_recall": np.nan,
            "pred_buy_count": 0, "pred_sell_count": 0,
            "true_buy_count": int((y_true == 1).sum()), "true_sell_count": int((y_true == -1).sum()),
            "selected_ratio": 0.0,
        }
    precision, recall, _, _ = precision_recall_fscore_support(y_eval, pred_eval, labels=[-1, 1], zero_division=0)
    return {
        "model": model_name, "stage": "stage2_direction", "split": split_name, "pair": pair,
        "n": int(len(y_true)), "selected_count": int(mask.sum()), "confidence_threshold": float(threshold),
        "accuracy": accuracy_score(y_eval, pred_eval),
        "balanced_accuracy": balanced_accuracy_score(y_eval, pred_eval),
        "macro_f1": f1_score(y_eval, pred_eval, average="macro", zero_division=0),
        "buy_precision": float(precision[1]), "buy_recall": float(recall[1]),
        "sell_precision": float(precision[0]), "sell_recall": float(recall[0]),
        "pred_buy_count": int((pred_eval == 1).sum()), "pred_sell_count": int((pred_eval == -1).sum()),
        "true_buy_count": int((y_true == 1).sum()), "true_sell_count": int((y_true == -1).sum()),
        "selected_ratio": float(mask.mean()),
    }


def stage2_score(row):
    selection_penalty = 0.20 if row["selected_ratio"] < 0.25 else 0.0
    collapse_penalty = 0.15 if row["pred_buy_count"] == 0 or row["pred_sell_count"] == 0 else 0.0
    return row["macro_f1"] + 0.50 * row["balanced_accuracy"] + 0.10 * row["selected_ratio"] - selection_penalty - collapse_penalty


def select_stage1_thresholds(model_name, validation_df, probs):
    rows = []
    for pair, idx in validation_df.groupby("pair").groups.items():
        loc = validation_df.index.get_indexer(idx)
        for threshold in STAGE1_GRID:
            row = stage1_metrics(model_name, "validation", pair, validation_df.loc[idx, STAGE1_TARGET], probs[loc], threshold)
            row["score"] = stage1_score(row)
            rows.append(row)
    grid = pd.DataFrame(rows)
    return grid, grid.sort_values(["pair", "score"], ascending=[True, False]).groupby("pair", as_index=False).head(1)


def select_stage2_thresholds(model_name, validation_df, pred, confidence):
    rows = []
    for pair, idx in validation_df.groupby("pair").groups.items():
        loc = validation_df.index.get_indexer(idx)
        for threshold in STAGE2_GRID:
            row = stage2_metrics(model_name, "validation", pair, validation_df.loc[idx, STAGE2_TARGET], pred[loc], confidence[loc], threshold)
            row["score"] = stage2_score(row)
            rows.append(row)
    grid = pd.DataFrame(rows)
    return grid, grid.sort_values(["pair", "score"], ascending=[True, False]).groupby("pair", as_index=False).head(1)


def df_to_markdown(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.6f}"
            else:
                value = "" if pd.isna(value) else str(value)
            vals.append(value.replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    print("[PROGRESS] START 46_phase8_train_svm_ebm_two_stage.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLIB_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()

    usecols = ["event_time", "Name", "CountryCode", "pair", "event_year", "split", STAGE1_TARGET, STAGE2_TARGET] + features
    print(f"[DATA] reading={DATASET}", flush=True)
    df = pd.read_csv(DATASET, usecols=usecols, low_memory=False)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df[df["split"].isin(["train", "validation"])].copy()
    print(f"[DATA] rows={len(df)} models={MODELS}", flush=True)

    stage1_df = df.dropna(subset=[STAGE1_TARGET]).copy()
    stage1_df[STAGE1_TARGET] = stage1_df[STAGE1_TARGET].astype(int)
    stage2_df = df.dropna(subset=[STAGE2_TARGET]).copy()
    stage2_df[STAGE2_TARGET] = stage2_df[STAGE2_TARGET].astype(int)

    s1_train = stage1_df[stage1_df["split"].eq("train")]
    s1_val = stage1_df[stage1_df["split"].eq("validation")]
    s2_train = stage2_df[stage2_df["split"].eq("train")]
    s2_val = stage2_df[stage2_df["split"].eq("validation")]
    print(f"[DATA] stage1 train={len(s1_train)} validation={len(s1_val)}", flush=True)
    print(f"[DATA] stage2 train={len(s2_train)} validation={len(s2_val)}", flush=True)

    status_rows, s1_grid_rows, s1_threshold_rows, s2_grid_rows, s2_threshold_rows = [], [], [], [], []

    for kind in MODELS:
        for stage, train_df, val_df, target in [
            (1, s1_train, s1_val, STAGE1_TARGET),
            (2, s2_train, s2_val, STAGE2_TARGET),
        ]:
            model_name, filename = model_names(kind, stage)
            started = time.perf_counter()
            print(f"[MODEL] START {model_name} rows={len(train_df)}", flush=True)
            try:
                artifact_path = JOBLIB_DIR / filename
                if artifact_path.exists():
                    print(f"[MODEL] RESUME_LOAD {artifact_path}", flush=True)
                    model = joblib.load(artifact_path)
                    elapsed = round(time.perf_counter() - started, 2)
                    status_rows.append({"model": model_name, "stage": stage, "status": "loaded_existing", "seconds": elapsed, "artifact": str(artifact_path), "note": "Existing artifact reused; no retraining."})
                else:
                    model = build_model(kind)
                    model = fit_with_heartbeat(model, train_df[features], train_df[target], model_name, stage)
                    elapsed = round(time.perf_counter() - started, 2)
                    joblib.dump(model, artifact_path)
                    print(f"[MODEL] SAVED {artifact_path} seconds={elapsed}", flush=True)
                    status_rows.append({"model": model_name, "stage": stage, "status": "trained", "seconds": elapsed, "artifact": str(artifact_path), "note": ""})

                if stage == 1:
                    probs = positive_probability_chunked(model, val_df[features], model_name, stage, 1)
                    grid, selected = select_stage1_thresholds(model_name, val_df, probs)
                    selected["selected_on"] = "validation_2021_2023"
                    s1_grid_rows.append(grid)
                    s1_threshold_rows.append(selected)
                else:
                    _, _, pred, conf = direction_outputs_chunked(model, val_df[features], model_name, stage)
                    grid, selected = select_stage2_thresholds(model_name, val_df, pred, conf)
                    selected["selected_on"] = "validation_2021_2023"
                    s2_grid_rows.append(grid)
                    s2_threshold_rows.append(selected)
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 2)
                status_rows.append({"model": model_name, "stage": stage, "status": "failed", "seconds": elapsed, "artifact": "", "note": str(exc)})
                print(f"[MODEL] FAILED {model_name} error={exc}", flush=True)

    s1_grid = pd.concat(s1_grid_rows, ignore_index=True) if s1_grid_rows else pd.DataFrame()
    s1_thresholds = pd.concat(s1_threshold_rows, ignore_index=True) if s1_threshold_rows else pd.DataFrame()
    s2_grid = pd.concat(s2_grid_rows, ignore_index=True) if s2_grid_rows else pd.DataFrame()
    s2_thresholds = pd.concat(s2_threshold_rows, ignore_index=True) if s2_threshold_rows else pd.DataFrame()
    status = pd.DataFrame(status_rows)

    s1_grid.to_csv(OUT_DIR / "stage1_trade_filter_threshold_grid.csv", index=False)
    s1_thresholds.to_csv(OUT_DIR / "stage1_trade_filter_thresholds.csv", index=False)
    s2_grid.to_csv(OUT_DIR / "stage2_direction_confidence_grid.csv", index=False)
    s2_thresholds.to_csv(OUT_DIR / "stage2_direction_thresholds.csv", index=False)
    status.to_csv(OUT_DIR / "phase8_training_status.csv", index=False)

    metadata = {
        "models": MODELS,
        "stage1_target": STAGE1_TARGET,
        "stage2_target": STAGE2_TARGET,
        "features": features,
        "train_period": "2010-2020",
        "validation_period": "2021-2023",
        "oos_policy": "2024+ is not used for fitting or threshold selection.",
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT_DIR / "phase8_svm_ebm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "# Phase 8 - SVM RBF and EBM Two-Stage Training",
        "",
        "Training tambahan untuk `svm_rbf_two_stage` dan `ebm_two_stage`.",
        "",
        "## Status",
        "",
        df_to_markdown(status),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'stage1_trade_filter_thresholds.csv'}`",
        f"- `{OUT_DIR / 'stage2_direction_thresholds.csv'}`",
        f"- `{OUT_DIR / 'phase8_training_status.csv'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE8_SVM_EBM_TRAINING.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'phase8_training_status.csv'}", flush=True)
    print("[PROGRESS] DONE 46_phase8_train_svm_ebm_two_stage.py", flush=True)


if __name__ == "__main__":
    main()
