from common import *
import joblib

FEATURES = model_feature_names()
TARGET = "trade_label_1h"
STRICT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
DIRECTIONAL_THRESHOLDS = [0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]
MIN_TRADES = 20


def load_models():
    model_dir = ARTIFACTS / "joblib"
    return {p.stem: joblib.load(p) for p in sorted(model_dir.glob("*.joblib"))}


def svm_decision_probability_frame(model, X):
    if not hasattr(model, "named_steps") or "svc" not in model.named_steps:
        return None
    svc = model.named_steps["svc"]
    if not hasattr(svc, "classes_"):
        return None
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    proba = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    for idx, cls in enumerate(svc.classes_):
        if int(cls) == -1:
            out["prob_sell"] = proba[:, idx]
        elif int(cls) == 0:
            out["prob_no_trade"] = proba[:, idx]
        elif int(cls) == 1:
            out["prob_buy"] = proba[:, idx]
    return out


def class_probability_frame(model, X):
    svm_probs = svm_decision_probability_frame(model, X)
    if svm_probs is not None:
        return svm_probs
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
        out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
        for idx, cls in enumerate(classes):
            if int(cls) == -1:
                out["prob_sell"] = proba[:, idx]
            elif int(cls) == 0:
                out["prob_no_trade"] = proba[:, idx]
            elif int(cls) == 1:
                out["prob_buy"] = proba[:, idx]
        return out
    pred = pd.Series(np.asarray(model.predict(X)).reshape(-1), index=X.index)
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    out.loc[pred.eq(-1), "prob_sell"] = 1.0
    out.loc[pred.eq(0), "prob_no_trade"] = 1.0
    out.loc[pred.eq(1), "prob_buy"] = 1.0
    return out


def signals_from_thresholds(probs, threshold, mode="strict"):
    signal = np.zeros(len(probs), dtype=int)
    buy = probs["prob_buy"].to_numpy()
    sell = probs["prob_sell"].to_numpy()
    no_trade = probs["prob_no_trade"].to_numpy()
    if mode == "directional":
        buy_ok = (buy >= threshold) & (buy >= sell)
        sell_ok = (sell >= threshold) & (sell > buy)
    else:
        buy_ok = (buy >= threshold) & (buy >= sell) & (buy >= no_trade)
        sell_ok = (sell >= threshold) & (sell >= buy) & (sell >= no_trade)
    signal[buy_ok] = 1
    signal[sell_ok] = -1
    return signal


def evaluate_threshold(g, signal):
    buy_count = int((signal == 1).sum())
    sell_count = int((signal == -1).sum())
    no_trade_count = int((signal == 0).sum())
    trade_count = buy_count + sell_count
    pnl = g["domestic_return_pips_1h"].to_numpy() * signal
    trade_pnl = pnl[signal != 0]
    reason = collapse_reason(buy_count, sell_count, no_trade_count, min_trades=MIN_TRADES)
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "no_trade_count": no_trade_count,
        "trade_count": trade_count,
        "buy_ratio": buy_count / len(signal) if len(signal) else 0.0,
        "sell_ratio": sell_count / len(signal) if len(signal) else 0.0,
        "no_trade_ratio": no_trade_count / len(signal) if len(signal) else 0.0,
        "net_pips": float(np.nansum(trade_pnl)) if trade_count else 0.0,
        "avg_pips_per_trade": float(np.nanmean(trade_pnl)) if trade_count else 0.0,
        "profit_factor": float(profit_factor(trade_pnl)) if trade_count else 0.0,
        "win_rate": float((trade_pnl > 0).mean()) if trade_count else 0.0,
        "collapse_flag": bool(reason),
        "collapse_reason": reason,
    }


def main():
    print("[PROGRESS] START 04b_optimize_model_pair_thresholds.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv")
    d["split"] = research_split(d["event_year"])
    d = d[d["split"].eq("validation")].dropna(subset=FEATURES + [TARGET, "domestic_return_pips_1h"]).copy()
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    models = load_models()
    print(f"[DATA] validation_rows={len(d)} models={list(models)}", flush=True)

    threshold_rows = []
    bias_rows = []
    for model_name, model in models.items():
        print(f"[THRESHOLD] START model={model_name}", flush=True)
        probs = class_probability_frame(model, d[FEATURES])
        work = pd.concat([d[["pair", TARGET, "domestic_return_pips_1h"]].reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
        for pair, g in work.groupby("pair", sort=False):
            best = None
            for mode, thresholds in [("strict", STRICT_THRESHOLDS), ("directional", DIRECTIONAL_THRESHOLDS)]:
              for threshold in thresholds:
                signal = signals_from_thresholds(g[["prob_sell", "prob_no_trade", "prob_buy"]], threshold, mode=mode)
                metrics = evaluate_threshold(g, signal)
                row = {
                    "model": model_name,
                    "pair": pair,
                    "threshold": threshold,
                    "signal_mode": mode,
                    "selection_split": "validation_2021_2023",
                    **metrics,
                }
                rank = (
                    metrics["net_pips"] > 0,
                    0 if metrics["collapse_flag"] else 1,
                    metrics["profit_factor"],
                    metrics["net_pips"],
                    metrics["trade_count"],
                    -abs(metrics["buy_count"] - metrics["sell_count"]) / max(1, metrics["trade_count"]),
                )
                if best is None or rank > best["rank"]:
                    best = {"rank": rank, "row": row, "signal": signal}
            threshold_rows.append(best["row"])
            bias = {k: best["row"][k] for k in [
                "model", "pair", "buy_count", "sell_count", "no_trade_count",
                "buy_ratio", "sell_ratio", "no_trade_ratio", "collapse_flag", "collapse_reason",
            ]}
            bias_rows.append(bias)
            print(f"[THRESHOLD] DONE model={model_name} pair={pair} threshold={best['row']['threshold']} trades={best['row']['trade_count']} collapse={best['row']['collapse_flag']}", flush=True)

    pd.DataFrame(threshold_rows).to_csv(ROOT / "model_pair_thresholds.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(ROOT / "model_pair_bias_report.csv", index=False)
    print("[OUTPUT] model_pair_thresholds.csv", flush=True)
    print("[OUTPUT] model_pair_bias_report.csv", flush=True)


if __name__ == "__main__":
    main()
