from common import *
from pathlib import Path
import json

import onnxruntime as ort
from catboost import CatBoostClassifier

SCRIPT = Path(__file__).name
OUT_DIR = ARTIFACTS / "catboost_v2_onnx_csv_parity_audit"
SRC_DIR = ARTIFACTS / "catboost_v2_pair_specific_loop"
ONNX_DIR = ARTIFACTS / "catboost_v2_pair_onnx"
CSV_SIGNALS = ARTIFACTS / "catboost_v2_pair_signal_table.csv"


def catboost_pair_threshold(pair):
    pair = str(pair).upper()
    if pair in {"GBPUSD", "AUDUSD"}:
        return 0.65
    if pair in {"NZDUSD", "USDCHF", "GBPJPY"}:
        return 0.75
    return 0.70


def catboost_pair_min_impact(pair):
    pair = str(pair).upper()
    return 1 if pair in {"EURUSD", "GBPUSD", "EURJPY"} else 2


def build_features(d):
    out = d.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out["category"] = out["Name"].map(event_category)
    out["surprise_fc"] = pd.to_numeric(out["Actual"], errors="coerce") - pd.to_numeric(out["Forecast"], errors="coerce")
    out["surprise_prev"] = pd.to_numeric(out["Actual"], errors="coerce") - pd.to_numeric(out["Previous"], errors="coerce")
    out["surprise_pct_fc"] = out["surprise_fc"] / (pd.to_numeric(out["Forecast"], errors="coerce").abs() + 1e-9)
    out["release_hour"] = out["event_time"].dt.hour.fillna(0).astype(int)
    mapping = available_currency_pair_mapping()[["pair", "CountryCode", "orientation"]].drop_duplicates()
    out = out.merge(mapping, on=["pair", "CountryCode"], how="left", suffixes=("", "_map"))
    out["orientation"] = out["orientation"].fillna("other")
    out = add_context_features(out)
    features = model_feature_names()
    for col in features:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, features


def onnx_probability(session, X):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.astype("float32")})
    label = np.asarray(outputs[0]).reshape(-1)
    proba = outputs[1]
    if isinstance(proba, list):
        rows = []
        for item in proba:
            if isinstance(item, dict):
                keys = sorted(item.keys())
                rows.append([float(item[k]) for k in keys])
            else:
                rows.append(list(item))
        proba = np.asarray(rows, dtype=float)
    else:
        proba = np.asarray(proba, dtype=float)
    return label, proba


def native_probability(model, X):
    proba = np.asarray(model.predict_proba(X), dtype=float)
    pred = np.asarray(model.predict(X)).reshape(-1)
    return pred, proba


def final_signal_from_binary(prob_sell, prob_buy, threshold):
    if prob_buy >= threshold and prob_buy >= prob_sell:
        return 1
    if prob_sell >= threshold and prob_sell > prob_buy:
        return -1
    return 0


def final_signal_from_multiclass(proba, threshold):
    # Expected class order in CatBoost multiclass export: SELL, NO_TRADE, BUY.
    if proba.shape[1] < 3:
        return np.array([final_signal_from_binary(row[0], row[1], threshold) for row in proba], dtype=int)
    cls = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    signal = np.zeros(len(proba), dtype=int)
    signal[(cls == 0) & (conf >= threshold)] = -1
    signal[(cls == 2) & (conf >= threshold)] = 1
    return signal


def probability_columns(proba, is_multiclass):
    if is_multiclass:
        if proba.shape[1] >= 3:
            return proba[:, 0], proba[:, 1], proba[:, 2]
        return proba[:, 0], np.zeros(len(proba)), proba[:, 1]
    return proba[:, 0], np.zeros(len(proba)), proba[:, 1]


def audit_pair(pair, csv_signals):
    pair = pair.upper()
    pred_path = SRC_DIR / pair / "best_model" / pair / "backtrader_predictions.csv"
    cbm_path = ONNX_DIR / f"catboost_v2_pair_{pair}.cbm"
    onnx_path = ONNX_DIR / f"catboost_v2_pair_{pair}.onnx"
    if not pred_path.exists() or not cbm_path.exists() or not onnx_path.exists():
        return None, pd.DataFrame()

    raw = pd.read_csv(pred_path, low_memory=False)
    raw = raw[pd.to_datetime(raw["event_time"], errors="coerce").notna()].copy()
    d, features = build_features(raw)
    X = d[features].to_numpy(dtype=np.float32)

    model = CatBoostClassifier()
    model.load_model(str(cbm_path))
    native_pred, native_proba = native_probability(model, d[features])

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_pred, onnx_proba = onnx_probability(session, X)

    is_multiclass = pair == "AUDUSD" or native_proba.shape[1] >= 3 or onnx_proba.shape[1] >= 3
    threshold = catboost_pair_threshold(pair)
    ns, nn, nb = probability_columns(native_proba, is_multiclass)
    os, on, ob = probability_columns(onnx_proba, is_multiclass)
    if is_multiclass:
        native_signal = final_signal_from_multiclass(native_proba, threshold)
        onnx_signal = final_signal_from_multiclass(onnx_proba, threshold)
    else:
        native_signal = np.array([final_signal_from_binary(s, b, threshold) for s, b in zip(ns, nb)], dtype=int)
        onnx_signal = np.array([final_signal_from_binary(s, b, threshold) for s, b in zip(os, ob)], dtype=int)

    # Compare to exported CSV signal table. Missing rows mean CSV says NO_TRADE.
    compare = d[["event_time", "pair", "CountryCode", "Name"]].copy()
    compare["EVENT_TIME"] = pd.to_datetime(compare["event_time"]).dt.strftime("%Y.%m.%d %H:%M")
    compare["key"] = compare["EVENT_TIME"] + "|" + compare["pair"].astype(str).str.upper() + "|" + compare["CountryCode"].astype(str).str.upper() + "|" + compare["Name"].astype(str)
    sig = csv_signals[csv_signals["SYMBOL"].astype(str).str.upper().eq(pair)].copy()
    sig["key"] = sig["EVENT_TIME"].astype(str) + "|" + sig["SYMBOL"].astype(str).str.upper() + "|" + sig["COUNTRY"].astype(str).str.upper() + "|" + sig["EVENT_NAME"].astype(str)
    sig_map = sig.drop_duplicates("key").set_index("key")["SIGNAL"].astype(int).to_dict()
    csv_signal = compare["key"].map(sig_map).fillna(0).astype(int).to_numpy()

    max_proba_diff = float(np.nanmax(np.abs(native_proba[:, :onnx_proba.shape[1]] - onnx_proba[:, :native_proba.shape[1]]))) if len(d) else np.nan
    mean_buy_diff = float(np.nanmean(np.abs(nb - ob))) if len(d) else np.nan
    native_onnx_mismatch = native_signal != onnx_signal
    onnx_csv_mismatch = onnx_signal != csv_signal
    native_csv_mismatch = native_signal != csv_signal

    details = pd.DataFrame({
        "pair": pair,
        "event_time": compare["EVENT_TIME"],
        "country": d["CountryCode"].astype(str),
        "event_name": d["Name"].astype(str),
        "native_signal": native_signal,
        "onnx_signal": onnx_signal,
        "csv_signal": csv_signal,
        "native_prob_sell": ns,
        "native_prob_no_trade": nn,
        "native_prob_buy": nb,
        "onnx_prob_sell": os,
        "onnx_prob_no_trade": on,
        "onnx_prob_buy": ob,
        "native_onnx_mismatch": native_onnx_mismatch,
        "onnx_csv_mismatch": onnx_csv_mismatch,
        "native_csv_mismatch": native_csv_mismatch,
    })
    summary = {
        "pair": pair,
        "rows": int(len(details)),
        "threshold": threshold,
        "is_multiclass": bool(is_multiclass),
        "native_trade_count": int(np.count_nonzero(native_signal)),
        "onnx_trade_count": int(np.count_nonzero(onnx_signal)),
        "csv_trade_count": int(np.count_nonzero(csv_signal)),
        "native_onnx_signal_mismatch_rate": float(native_onnx_mismatch.mean()) if len(details) else np.nan,
        "onnx_csv_signal_mismatch_rate": float(onnx_csv_mismatch.mean()) if len(details) else np.nan,
        "native_csv_signal_mismatch_rate": float(native_csv_mismatch.mean()) if len(details) else np.nan,
        "max_probability_abs_diff": max_proba_diff,
        "mean_buy_probability_abs_diff": mean_buy_diff,
    }
    return summary, details


def main():
    started = progress_start(SCRIPT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_SIGNALS.exists():
        raise FileNotFoundError(CSV_SIGNALS)
    csv_signals = pd.read_csv(CSV_SIGNALS, low_memory=False)
    pairs = sorted([p.name.replace("catboost_v2_pair_", "").replace(".onnx", "") for p in ONNX_DIR.glob("catboost_v2_pair_*.onnx")])
    summaries = []
    details = []
    for i, pair in enumerate(pairs, 1):
        progress_step("PAIR_AUDIT_START", i, len(pairs), pair=pair)
        summary, detail = audit_pair(pair, csv_signals)
        if summary is not None:
            summaries.append(summary)
            details.append(detail)
            progress_step(
                "PAIR_AUDIT_DONE",
                i,
                len(pairs),
                pair=pair,
                native_onnx_mismatch=f"{summary['native_onnx_signal_mismatch_rate']:.4f}",
                onnx_csv_mismatch=f"{summary['onnx_csv_signal_mismatch_rate']:.4f}",
            )
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_path = OUT_DIR / "catboost_v2_onnx_csv_parity_summary.csv"
    detail_path = OUT_DIR / "catboost_v2_onnx_csv_parity_details.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    if not summary_df.empty:
        weighted_native_onnx = float(np.average(summary_df["native_onnx_signal_mismatch_rate"], weights=summary_df["rows"]))
        weighted_onnx_csv = float(np.average(summary_df["onnx_csv_signal_mismatch_rate"], weights=summary_df["rows"]))
        recommendation = "use_single_route"
        if weighted_native_onnx > 0.01 or weighted_onnx_csv > 0.01:
            recommendation = "keep_two_routes_until_mismatch_is_resolved"
        report = {
            "weighted_native_onnx_signal_mismatch_rate": weighted_native_onnx,
            "weighted_onnx_csv_signal_mismatch_rate": weighted_onnx_csv,
            "recommendation": recommendation,
            "rule": "If mismatch > 1%, keep ONNX and CSV separate; otherwise one route is acceptable.",
        }
    else:
        report = {"recommendation": "no_data"}
    (OUT_DIR / "catboost_v2_onnx_csv_parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OUTPUT] {summary_path}", flush=True)
    print(f"[OUTPUT] {detail_path}", flush=True)
    print(f"[PARITY] {report}", flush=True)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
