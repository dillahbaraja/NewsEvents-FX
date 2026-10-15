from common import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import joblib
import threading
import time

FEATURES = model_feature_names()
TARGET = "trade_label_1h"


def metric_row(split_name, y_true, pred):
    pred = np.asarray(pred).reshape(-1)
    buy_count = int((pred == 1).sum())
    sell_count = int((pred == -1).sum())
    no_trade_count = int((pred == 0).sum())
    reason = collapse_reason(buy_count, sell_count, no_trade_count)
    return {
        "model": "ebm",
        "target": TARGET,
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred) if len(y_true) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(y_true) else np.nan,
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0) if len(y_true) else np.nan,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "no_trade_count": no_trade_count,
        "collapse_flag": bool(reason),
        "collapse_reason": reason,
        "status": "trained",
        "note": "",
    }


def main():
    print("[PROGRESS] START 04d_retrain_ebm_only.py", flush=True)
    from interpret.glassbox import ExplainableBoostingClassifier

    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", TARGET] + FEATURES)
    d = d.dropna(subset=FEATURES + [TARGET])
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = d[FEATURES]
    y = d[TARGET].astype(int)
    split = research_split(d.event_year)
    train_rows = int((split == "train").sum())
    validation_rows = int((split == "validation").sum())
    print(f"[DATA] rows={len(d)} train_rows={train_rows} validation_rows={validation_rows} features={len(FEATURES)}", flush=True)
    print(f"[DATA] train_label_counts={y.loc[split == 'train'].value_counts().sort_index().to_dict()}", flush=True)

    model = ExplainableBoostingClassifier(
        random_state=42,
        n_jobs=1,
        interactions=0,
        outer_bags=4,
        max_rounds=5000,
    )
    done = threading.Event()

    def heartbeat():
        started = time.perf_counter()
        while not done.wait(60):
            elapsed = time.perf_counter() - started
            print(f"[HEARTBEAT] ebm_only fitting elapsed={elapsed:.1f}s train_rows={train_rows} features={len(FEATURES)}", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        print("[MODEL] START ebm_only fit", flush=True)
        model.fit(X.loc[split == "train"], y.loc[split == "train"])
        print("[MODEL] DONE ebm_only fit", flush=True)
    finally:
        done.set()
        thread.join(timeout=1)
    (ARTIFACTS / "joblib").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "joblib" / "ebm.joblib")

    rows = []
    for split_name in ["train", "validation", "final_test"]:
        mask = split == split_name
        pred = model.predict(X.loc[mask]) if mask.any() else np.array([])
        rows.append(metric_row(split_name, y.loc[mask], pred))
    out_path = ROOT / "model_comparison_results_ebm_only.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    status_path = ROOT / "model_training_status.csv"
    existing = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    status = pd.DataFrame([{"model": "ebm", "status": "trained", "note": ""}])
    pd.concat([existing, status], ignore_index=True).drop_duplicates("model", keep="last").to_csv(status_path, index=False)
    print(f"[OUTPUT] {ARTIFACTS / 'joblib' / 'ebm.joblib'}", flush=True)
    print(f"[OUTPUT] {out_path}", flush=True)


if __name__ == "__main__":
    main()
