from common import *
from pathlib import Path
import json
import shutil

import joblib
import onnx
from onnx import helper

SRC_DIR = ARTIFACTS / "catboost_v2_pair_specific_loop"
BEST_CSV = SRC_DIR / "catboost_v2_pair_best_results.csv"
ONNX_DIR = ARTIFACTS / "catboost_v2_pair_onnx"
SIGNAL_TABLE = ARTIFACTS / "catboost_v2_pair_signal_table.csv"
MODEL_NAME = "catboost_v2_pair"


def make_onnx_mt5_tensor_output(path):
    model = onnx.load(str(path))

    zip_nodes = [node for node in model.graph.node if node.op_type == "ZipMap"]
    if not zip_nodes:
        return

    probability_tensor_name = zip_nodes[0].input[0]
    class_count = None
    for value_info in list(model.graph.value_info) + list(model.graph.output):
        if value_info.name != probability_tensor_name:
            continue
        shape = value_info.type.tensor_type.shape.dim
        if len(shape) >= 2 and shape[1].HasField("dim_value"):
            class_count = int(shape[1].dim_value)
            break

    if class_count is None:
        class_count = 3 if "AUDUSD" in str(path).upper() else 2

    kept_nodes = [node for node in model.graph.node if node.op_type != "ZipMap"]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    label_output = model.graph.output[0]
    del model.graph.output[:]
    model.graph.output.extend([
        label_output,
        helper.make_tensor_value_info(probability_tensor_name, onnx.TensorProto.FLOAT, [None, class_count]),
    ])

    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def mt5_time(value):
    return pd.to_datetime(value).strftime("%Y.%m.%d %H:%M")


def domestic_signal_from_trade_signal(signal, orientation):
    if pd.isna(signal) or str(signal).upper() == "NO_TRADE":
        return 0
    trade_signal = 1 if str(signal).upper() == "BUY" else -1
    return trade_signal if str(orientation).lower() == "base" else -trade_signal


def export_onnx(pair, row):
    best_dir = SRC_DIR / pair / "best_model"
    joblibs = sorted(best_dir.glob("*.joblib"))
    if not joblibs:
        raise FileNotFoundError(f"No joblib found in {best_dir}")
    model = joblib.load(joblibs[0])
    out_path = ONNX_DIR / f"catboost_v2_pair_{pair}.onnx"
    cbm_path = ONNX_DIR / f"catboost_v2_pair_{pair}.cbm"
    model.save_model(str(cbm_path))
    model.save_model(
        str(out_path),
        format="onnx",
        export_parameters={
            "onnx_graph_name": f"catboost_v2_pair_{pair}",
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": f"Macro News FX CatBoost V2 pair-specific {pair}",
        },
    )
    make_onnx_mt5_tensor_output(out_path)
    return out_path, cbm_path, joblibs[0]


def export_pair_signals(pair):
    pred_path = SRC_DIR / pair / "best_model" / pair / "backtrader_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    d = pd.read_csv(pred_path)
    d = d[d["signal"].isin(["BUY", "SELL"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL"])
    mapping = available_currency_pair_mapping()[["pair", "CountryCode", "orientation"]].drop_duplicates()
    d = d.merge(mapping, on=["pair", "CountryCode"], how="left")
    d["EVENT_TIME"] = d["event_time"].map(mt5_time)
    d["SYMBOL"] = d["pair"].astype(str)
    d["COUNTRY"] = d["CountryCode"].astype(str)
    d["EVENT_NAME"] = d["Name"].astype(str)
    d["PROBABILITY"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(1.0)
    d["SIGNAL"] = [
        domestic_signal_from_trade_signal(sig, ori)
        for sig, ori in zip(d["signal"], d["orientation"])
    ]
    d["MODEL"] = MODEL_NAME
    out = d[["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL"]].copy()
    return out


def main():
    print("[PROGRESS] START 30_export_catboost_v2_pair_mt5_artifacts.py", flush=True)
    if not BEST_CSV.exists():
        raise FileNotFoundError(f"Missing best results: {BEST_CSV}")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(BEST_CSV)
    rows = []
    status_rows = []
    for _, row in best.iterrows():
        pair = str(row["pair"]).upper()
        print(f"[EXPORT] START pair={pair} model={row['model']}", flush=True)
        try:
            onnx_path, cbm_path, joblib_path = export_onnx(pair, row)
            sig = export_pair_signals(pair)
            rows.append(sig)
            status_rows.append({
                "pair": pair,
                "status": "exported",
                "joblib": str(joblib_path),
                "onnx": str(onnx_path),
                "cbm": str(cbm_path),
                "signals": int(len(sig)),
                "win_rate": row.get("win_rate", ""),
                "trades": row.get("trades", ""),
                "net_pips": row.get("net_pips", ""),
                "model": row.get("model", ""),
            })
            print(f"[EXPORT] DONE pair={pair} signals={len(sig)} onnx={onnx_path}", flush=True)
        except Exception as exc:
            status_rows.append({"pair": pair, "status": "failed", "error": str(exc)})
            print(f"[EXPORT] FAILED pair={pair} error={exc}", flush=True)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL"]
    )
    out = out.drop_duplicates(subset=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "MODEL"], keep="last")
    out.to_csv(SIGNAL_TABLE, index=False)
    status = pd.DataFrame(status_rows)
    status.to_csv(ARTIFACTS / "catboost_v2_pair_mt5_export_status.csv", index=False)
    metadata = {
        "model_name": MODEL_NAME,
        "onnx_dir": str(ONNX_DIR),
        "signal_table": str(SIGNAL_TABLE),
        "pairs": status_rows,
        "note": "Pair-specific CatBoost V2 exports for MT5. ONNX file is selected by active symbol; CSV rows use MODEL=catboost_v2_pair.",
    }
    (ARTIFACTS / "catboost_v2_pair_mt5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OUTPUT] {SIGNAL_TABLE} rows={len(out)}", flush=True)
    print(f"[OUTPUT] {ARTIFACTS / 'catboost_v2_pair_mt5_export_status.csv'}", flush=True)
    print(f"[OUTPUT] {ARTIFACTS / 'catboost_v2_pair_mt5_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
