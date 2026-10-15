from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import time

FEATURES = model_feature_names()


def add_metric(rows, name, split_name, y_true, proba, status="trained", note=""):
    if len(y_true) == 0:
        rows.append({
            "model": name,
            "target": "domestic_currency_direction_1h",
            "split": split_name,
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "status": status,
            "note": note or "No rows available for this split.",
        })
        return
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "model": name,
        "target": "domestic_currency_direction_1h",
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "status": status,
        "note": note,
    })


def optional_models():
    models = {}
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
        )
    except Exception as exc:
        models["catboost"] = exc
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        models["ebm"] = ExplainableBoostingClassifier(random_state=42)
    except Exception as exc:
        models["ebm"] = exc
    try:
        from sklearn.svm import SVC
        models["svm_rbf"] = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, max_iter=1000, random_state=42))
    except Exception as exc:
        models["svm_rbf"] = exc
    return models


def main():
    print("[PROGRESS] START 04_train_models.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", "label_1h"] + FEATURES)
    d = d.dropna(subset=FEATURES + ["label_1h"])
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = d.label_1h.astype(int)
    s = research_split(d.event_year)
    print(f"[DATA] rows={len(d)} features={len(FEATURES)}", flush=True)
    for split_name in ["train", "validation", "final_test"]:
        print(f"[DATA] {split_name}_rows={int((s == split_name).sum())}", flush=True)
    rows = []
    models = {
        "elastic_net": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=42,
        )),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
    }
    models.update(optional_models())
    (ARTIFACTS / "joblib").mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        started = time.perf_counter()
        print(f"[MODEL] START {name}", flush=True)
        if isinstance(model, Exception):
            rows.append({
                "model": name,
                "target": "domestic_currency_direction_1h",
                "split": "not_run",
                "n": 0,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "auc": np.nan,
                "status": "dependency_missing",
                "note": str(model),
            })
            continue
        model.fit(X[s == "train"], y[s == "train"])
        print(f"[MODEL] FIT DONE {name} seconds={time.perf_counter() - started:.1f}", flush=True)
        joblib.dump(model, ARTIFACTS / "joblib" / (name + ".joblib"))
        for sp in ["train", "validation", "final_test"]:
            print(f"[MODEL] {name} EVALUATE {sp}", flush=True)
            q = s == sp
            if not q.any():
                add_metric(rows, name, sp, y[q], np.array([]))
                continue
            if name == "svm_rbf":
                score = model.decision_function(X[q])
                pr = 1.0 / (1.0 + np.exp(-score))
            else:
                pr = model.predict_proba(X[q])[:, 1]
            add_metric(rows, name, sp, y[q], pr)
    pd.DataFrame(rows).to_csv(ROOT / "model_comparison_results.csv", index=False)


if __name__ == "__main__":
    main()







