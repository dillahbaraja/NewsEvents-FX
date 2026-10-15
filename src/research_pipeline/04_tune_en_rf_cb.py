from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import json
import joblib
import shutil
import time

FEATURES = model_feature_names()
TARGET = "trade_label_1h"


def cleanup_previous_outputs():
    backup_dir = ARTIFACTS / "backup_before_cleanup" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    targets = [
        ARTIFACTS / "joblib" / "elastic_net.joblib",
        ARTIFACTS / "joblib" / "random_forest.joblib",
        ARTIFACTS / "joblib" / "catboost.joblib",
        ARTIFACTS / "en_rf_cb_best_params.json",
        ROOT / "model_comparison_results.csv",
        ROOT / "hyperparameter_tuning_results.csv",
        ROOT / "repeated_validation_results.csv",
        ROOT / "model_training_status.csv",
    ]
    for path in targets:
        if path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
            path.unlink()


def prediction_counts(pred):
    s = pd.Series(np.asarray(pred).reshape(-1))
    return int((s == 1).sum()), int((s == -1).sum()), int((s == 0).sum())


def collapse_penalty(pred):
    buy_count, sell_count, no_trade_count = prediction_counts(pred)
    return 1.0 if collapse_reason(buy_count, sell_count, no_trade_count) else 0.0


def metric_row(model_name, split_name, y_true, pred, status="trained", note="", params=None, seconds=np.nan):
    y_true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y_true) == 0:
        return {
            "model": model_name,
            "target": TARGET,
            "split": split_name,
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "buy_count": 0,
            "sell_count": 0,
            "no_trade_count": 0,
            "collapse_flag": False,
            "collapse_reason": "no_rows",
            "status": status,
            "note": note or "No rows available for this split.",
            "params": json.dumps(params or {}),
            "seconds": seconds,
        }
    buy_count, sell_count, no_trade_count = prediction_counts(pred)
    reason = collapse_reason(buy_count, sell_count, no_trade_count)
    return {
        "model": model_name,
        "target": TARGET,
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "no_trade_count": no_trade_count,
        "collapse_flag": bool(reason),
        "collapse_reason": reason,
        "status": status,
        "note": note,
        "params": json.dumps(params or {}),
        "seconds": seconds,
    }


def build_model(name, params):
    if name == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                penalty="elasticnet",
                solver="saga",
                C=params["C"],
                l1_ratio=params["l1_ratio"],
                random_state=42,
                verbose=1,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
            verbose=1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="MultiClass",
            eval_metric="TotalF1",
            verbose=100,
            random_seed=42,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {name}")


def grids():
    return {
        "elastic_net": list(ParameterGrid({
            "C": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        })),
        "random_forest": list(ParameterGrid({
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 6, 10, 16],
            "min_samples_leaf": [5, 10, 25],
        })),
        "catboost": list(ParameterGrid({
            "iterations": [300, 500, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.10],
            "l2_leaf_reg": [3, 5, 10],
        })),
    }


def main():
    print("[PROGRESS] START 04_tune_en_rf_cb.py", flush=True)
    cleanup_previous_outputs()
    print("[CLEANUP] previous EN_RF_CB tuning artifacts removed", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", TARGET] + FEATURES)
    d = d.dropna(subset=FEATURES + [TARGET])
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = d[FEATURES]
    y = d[TARGET].astype(int)
    split = research_split(d.event_year)
    train_mask = split == "train"
    val_mask = split == "validation"
    print(f"[DATA] rows={len(d)} features={len(FEATURES)}", flush=True)
    print(f"[DATA] train_rows={int(train_mask.sum())}", flush=True)
    print(f"[DATA] validation_rows={int(val_mask.sum())}", flush=True)
    print(f"[DATA] train_label_counts={y.loc[train_mask].value_counts().sort_index().to_dict()}", flush=True)

    tuning_rows = []
    metric_rows = []
    repeated_rows = []
    status_rows = []
    best_params = {}
    (ARTIFACTS / "joblib").mkdir(parents=True, exist_ok=True)

    for model_name, grid in grids().items():
        print(f"[TUNING] START {model_name} trials={len(grid)}", flush=True)
        best = None
        for idx, params in enumerate(grid, start=1):
            started = time.perf_counter()
            print(f"[TUNING] {model_name} trial={idx}/{len(grid)} params={params}", flush=True)
            try:
                model = build_model(model_name, params)
                model.fit(X.loc[train_mask], y.loc[train_mask])
                pred_val = model.predict(X.loc[val_mask])
                row = metric_row(model_name, "validation", y.loc[val_mask], pred_val, params=params, seconds=round(time.perf_counter() - started, 2))
                tuning_rows.append(row)
                rank_key = (
                    row["balanced_accuracy"],
                    row["macro_f1"],
                    -collapse_penalty(pred_val),
                    -abs(row["buy_count"] - row["sell_count"]) / max(1, row["n"]),
                )
                if best is None or rank_key > best["rank_key"]:
                    best = {"rank_key": rank_key, "params": params, "model": model}
                print(
                    f"[TUNING] DONE {model_name} trial={idx}/{len(grid)} bal_acc={row['balanced_accuracy']:.6f} macro_f1={row['macro_f1']:.6f} collapse={row['collapse_flag']}",
                    flush=True,
                )
            except Exception as exc:
                tuning_rows.append({
                    "model": model_name,
                    "target": TARGET,
                    "split": "validation",
                    "n": int(val_mask.sum()),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "buy_count": 0,
                    "sell_count": 0,
                    "no_trade_count": 0,
                    "collapse_flag": False,
                    "collapse_reason": "",
                    "status": "failed",
                    "note": str(exc),
                    "params": json.dumps(params),
                    "seconds": round(time.perf_counter() - started, 2),
                })
                print(f"[TUNING] FAILED {model_name} trial={idx}/{len(grid)} error={exc}", flush=True)

        if best is None:
            status_rows.append({"model": model_name, "status": "failed", "note": "No successful tuning trial."})
            continue

        params = best["params"]
        print(f"[FINAL] REFIT {model_name} params={params}", flush=True)
        final_model = build_model(model_name, params)
        final_model.fit(X.loc[train_mask], y.loc[train_mask])
        joblib.dump(final_model, ARTIFACTS / "joblib" / f"{model_name}.joblib")
        best_params[model_name] = {"params": params, "target": TARGET}
        status_rows.append({"model": model_name, "status": "trained", "note": ""})

        for split_name in ["train", "validation", "final_test"]:
            mask = split == split_name
            pred = final_model.predict(X.loc[mask]) if mask.any() else np.array([])
            metric_rows.append(metric_row(model_name, split_name, y.loc[mask], pred, params=params))

        for year in sorted(d.loc[val_mask, "event_year"].dropna().astype(int).unique()):
            mask = val_mask & (d["event_year"].astype(int) == year)
            pred = final_model.predict(X.loc[mask])
            row = metric_row(model_name, f"validation_{year}", y.loc[mask], pred, params=params)
            row["event_year"] = year
            repeated_rows.append(row)

    pd.DataFrame(tuning_rows).to_csv(ROOT / "hyperparameter_tuning_results.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(ROOT / "model_comparison_results.csv", index=False)
    pd.DataFrame(repeated_rows).to_csv(ROOT / "repeated_validation_results.csv", index=False)
    pd.DataFrame(status_rows).to_csv(ROOT / "model_training_status.csv", index=False)
    (ARTIFACTS / "en_rf_cb_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print("[OUTPUT] hyperparameter_tuning_results.csv", flush=True)
    print("[OUTPUT] repeated_validation_results.csv", flush=True)
    print("[OUTPUT] model_comparison_results.csv", flush=True)
    print("[OUTPUT] model_training_status.csv", flush=True)
    print("[OUTPUT] en_rf_cb_best_params.json", flush=True)


if __name__ == "__main__":
    main()
