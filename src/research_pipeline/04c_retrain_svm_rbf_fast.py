from common import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES = model_feature_names()
TARGET = "trade_label_1h"


def prediction_counts(pred):
    s = pd.Series(np.asarray(pred).reshape(-1))
    return int((s == 1).sum()), int((s == -1).sum()), int((s == 0).sum())


def metric_row(split_name, y_true, pred):
    buy_count, sell_count, no_trade_count = prediction_counts(pred)
    return {
        "model": "svm_rbf",
        "target": TARGET,
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred) if len(y_true) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(y_true) else np.nan,
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0) if len(y_true) else np.nan,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "no_trade_count": no_trade_count,
        "collapse_flag": bool(collapse_reason(buy_count, sell_count, no_trade_count)),
        "collapse_reason": collapse_reason(buy_count, sell_count, no_trade_count),
        "status": "trained",
        "note": "RBFSampler approximation for fast SVM-style RBF probability inference.",
    }


def main():
    print("[PROGRESS] START 04c_retrain_svm_rbf_fast.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", TARGET] + FEATURES)
    d = d.dropna(subset=FEATURES + [TARGET])
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = d[FEATURES]
    y = d[TARGET].astype(int)
    split = research_split(d.event_year)

    model = make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.05, n_components=500, random_state=42),
        LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, random_state=42),
    )
    model.fit(X.loc[split == "train"], y.loc[split == "train"])
    (ARTIFACTS / "joblib").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "joblib" / "svm_rbf.joblib")

    rows = []
    for split_name in ["train", "validation", "final_test"]:
        mask = split == split_name
        pred = model.predict(X.loc[mask]) if mask.any() else np.array([])
        rows.append(metric_row(split_name, y.loc[mask], pred))
    out_path = ROOT / "model_comparison_results_svm_rbf_fast.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    status_path = ROOT / "model_training_status.csv"
    existing = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    status = pd.DataFrame([{"model": "svm_rbf", "status": "trained", "note": "RBFSampler approximation for fast SVM-style RBF probability inference."}])
    pd.concat([existing, status], ignore_index=True).drop_duplicates("model", keep="last").to_csv(status_path, index=False)
    print(f"[OUTPUT] {ARTIFACTS / 'joblib' / 'svm_rbf.joblib'}", flush=True)
    print(f"[OUTPUT] {out_path}", flush=True)


if __name__ == "__main__":
    main()
