from common import *
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import shutil
import threading
import time
import warnings

FEATURES = model_feature_names()
TARGET = "trade_label_1h"
HEARTBEAT_SECONDS = max(5, int(os.environ.get("FX_HEARTBEAT_SECONDS", "60")))


def fit_with_heartbeat(model, X_train, y_train, model_name):
    done = threading.Event()

    def heartbeat():
        started = time.perf_counter()
        while not done.wait(HEARTBEAT_SECONDS):
            print(
                f"[HEARTBEAT] {model_name} fitting elapsed={time.perf_counter() - started:.1f}s "
                f"rows={len(X_train)} features={X_train.shape[1]}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    convergence_warning = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
            convergence_warning = any(isinstance(w.message, ConvergenceWarning) for w in caught)
    finally:
        done.set()
        thread.join(timeout=1)
    return convergence_warning


def cleanup_previous_outputs():
    backup_dir = ARTIFACTS / "backup_before_cleanup" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    targets = [
        ARTIFACTS / "joblib" / "ebm.joblib",
        ARTIFACTS / "joblib" / "svm_rbf.joblib",
        ROOT / "model_comparison_results_ebm_svm.csv",
    ]
    for path in targets:
        if path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
            path.unlink()


def metric_row(name, split_name, y_true, pred, status="trained", note=""):
    if len(y_true) == 0:
        return {
            "model": name,
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
        }
    buy_count = int((pred == 1).sum())
    sell_count = int((pred == -1).sum())
    no_trade_count = int((pred == 0).sum())
    reason = collapse_reason(buy_count, sell_count, no_trade_count)
    return {
        "model": name,
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
    }


def model_registry():
    models = {}
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        models["ebm"] = ExplainableBoostingClassifier(
            random_state=42,
            n_jobs=1,
            interactions=0,
            outer_bags=4,
            max_rounds=5000,
        )
    except Exception as exc:
        models["ebm"] = exc
    models["svm_rbf"] = make_pipeline(
        StandardScaler(),
        RBFSampler(
            gamma=0.05,
            n_components=500,
            random_state=42,
        ),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=-1,
            random_state=42,
        ),
    )
    return models


def main():
    print("[PROGRESS] START 04_train_ebm_svm.py", flush=True)
    cleanup_previous_outputs()
    print("[CLEANUP] previous EBM_SVM artifacts removed", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", TARGET] + FEATURES)
    d = d.dropna(subset=FEATURES + [TARGET])
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = d[TARGET].astype(int)
    s = research_split(d.event_year)
    print(f"[DATA] rows={len(d)} features={len(FEATURES)}", flush=True)
    for split_name in ["train", "validation", "final_test"]:
        print(f"[DATA] {split_name}_rows={int((s == split_name).sum())}", flush=True)

    rows = []
    status_rows = []
    (ARTIFACTS / "joblib").mkdir(parents=True, exist_ok=True)
    for name, model in model_registry().items():
        started = time.perf_counter()
        print(f"[MODEL] START {name}", flush=True)
        if isinstance(model, Exception):
            rows.append(metric_row(name, "not_run", np.array([]), np.array([]), status="dependency_missing", note=str(model)))
            status_rows.append({"model": name, "status": "dependency_missing", "note": str(model)})
            continue
        convergence_warning = fit_with_heartbeat(model, X[s == "train"], y[s == "train"], name)
        status = "not_converged" if convergence_warning else "trained"
        print(f"[MODEL] FIT DONE {name} status={status} seconds={time.perf_counter() - started:.1f}", flush=True)
        joblib.dump(model, ARTIFACTS / "joblib" / (name + ".joblib"))
        status_rows.append({"model": name, "status": status, "note": "SVM solver reached max_iter." if convergence_warning else ""})
        for sp in ["train", "validation", "final_test"]:
            print(f"[MODEL] {name} EVALUATE {sp}", flush=True)
            q = s == sp
            if not q.any():
                rows.append(metric_row(name, sp, y[q], np.array([]), status=status))
                continue
            pred = model.predict(X[q])
            rows.append(metric_row(name, sp, y[q], pred, status=status))

    out_path = ROOT / "model_comparison_results_ebm_svm.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    status_path = ROOT / "model_training_status.csv"
    existing = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(status_rows)], ignore_index=True).drop_duplicates("model", keep="last").to_csv(status_path, index=False)
    print(f"[OUTPUT] {out_path}", flush=True)


if __name__ == "__main__":
    main()
