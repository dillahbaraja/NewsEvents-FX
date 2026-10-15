from common import *
import argparse
import json
from pathlib import Path

import joblib
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description="Parity check between CatBoost joblib and CatBoost ONNX.")
    p.add_argument("--sample", type=int, default=1000)
    p.add_argument("--split", default="validation", choices=["train", "validation", "final_test", "all"])
    return p.parse_args()


def probability_from_onnx_output(outputs):
    probabilities = outputs[1]
    values = []
    for item in probabilities:
        if isinstance(item, dict):
            values.append(float(item.get(1, item.get(1.0, 0.0))))
        else:
            values.append(float(item[1]))
    return np.asarray(values, dtype=float)


def main():
    args = parse_args()
    print("[PROGRESS] START 11_onnx_parity_check.py", flush=True)
    metadata_path = ARTIFACTS / "model_metadata.json"
    model_path = ARTIFACTS / "joblib" / "catboost.joblib"
    onnx_path = ARTIFACTS / "catboost.onnx"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["features"]
    usecols = ["event_year", "label_1h"] + features
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=usecols)
    d = d.dropna(subset=features + ["label_1h"])
    if args.split != "all":
        d = d[research_split(d["event_year"]).astype(str) == args.split].copy()
    if len(d) > args.sample:
        d = d.sample(args.sample, random_state=42)
    X = d[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    model = joblib.load(model_path)
    py_prob = model.predict_proba(X)[:, 1]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.to_numpy(dtype=np.float32)})
    onnx_prob = probability_from_onnx_output(outputs)

    diff = np.abs(py_prob - onnx_prob)
    rows = pd.DataFrame({
        "python_probability": py_prob,
        "onnx_probability": onnx_prob,
        "abs_diff": diff,
    })
    out_csv = ARTIFACTS / "onnx_parity_results.csv"
    rows.to_csv(out_csv, index=False)
    summary = {
        "sample": int(len(rows)),
        "split": args.split,
        "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        "mean_abs_diff": float(diff.mean()) if len(diff) else np.nan,
        "p95_abs_diff": float(np.quantile(diff, 0.95)) if len(diff) else np.nan,
        "status": "ok" if len(diff) and float(diff.max()) < 1e-5 else "check",
    }
    out_json = ARTIFACTS / "onnx_parity_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OUTPUT] {out_csv}", flush=True)
    print(f"[OUTPUT] {out_json}", flush=True)
    print(f"[PARITY] {summary}", flush=True)


if __name__ == "__main__":
    main()
