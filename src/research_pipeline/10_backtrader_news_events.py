from common import *
import argparse
import json
import math
import shutil
from pathlib import Path

import backtrader as bt
import joblib


DATATEST = DATA / "datatest"
DEFAULT_NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"
BACKTRADER_ROOT = DATA / "backtrader"


class NewsEventsBacktraderMethod(bt.Strategy):
    params = (("method_note", "Signals are prepared event-by-event; Backtrader is the testing framework marker."),)


def parse_args():
    p = argparse.ArgumentParser(description="Backtrader-style testing for Macro News FX models.")
    p.add_argument("--pairs", default="all", help="Comma-separated pairs or 'all'.")
    p.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    p.add_argument("--news-test", default=str(DEFAULT_NEWS_TEST), help="NEWS_EVENTS_TEST.csv path.")
    p.add_argument("--start", default="2024-01-01", help="Test start date.")
    p.add_argument("--end", default="2026-12-31", help="Test end date.")
    p.add_argument("--holding-bars", type=int, default=1, help="Exit after this many H1 bars.")
    p.add_argument("--initial-cash", type=float, default=10000.0)
    p.add_argument("--trade-size", type=float, default=1.0)
    p.add_argument("--tolerance-hours", type=int, default=2)
    return p.parse_args()


def load_metadata():
    path = ARTIFACTS / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds():
    path = ROOT / "model_pair_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model pair thresholds: {path}")
    return pd.read_csv(path)


def available_model_files():
    model_dir = ARTIFACTS / "joblib"
    return {p.stem: p for p in model_dir.glob("*.joblib")}


def selected_values(raw, available):
    if raw.strip().lower() == "all":
        return list(available)
    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    return [x for x in wanted if x in set(available)]


def prepare_pair_events(pair, news_path, start, end, holding_bars, tolerance_hours):
    mapping = available_currency_pair_mapping()
    pair_map = mapping[mapping["pair"].eq(pair)].copy()
    if pair_map.empty:
        return pd.DataFrame()

    price_path = DATATEST / pair_file(pair)
    if not price_path.exists():
        return pd.DataFrame()

    px = read_prices(price_path).set_index("time")
    px["close_fwd"] = px["Close"].shift(-holding_bars)
    px["exit_time"] = px.index.to_series().shift(-holding_bars)
    px = px.reset_index()

    news = read_news(news_path)
    news = news[(news["event_time"] >= pd.Timestamp(start)) & (news["event_time"] <= pd.Timestamp(end))]
    news = news[news["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
    if news.empty:
        return pd.DataFrame()

    news["pair"] = pair
    news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
    news = news[news["orientation"].isin(["base", "quote"])].copy()
    news["category"] = news["Name"].map(event_category)
    news["release_hour"] = news["event_time"].dt.hour
    news["surprise_fc"] = news["Actual"] - news["Forecast"]
    news["surprise_prev"] = news["Actual"] - news["Previous"]
    news["surprise_pct_fc"] = news["surprise_fc"] / (news["Forecast"].abs() + 1e-9)
    news = add_context_features(news)

    joined = pd.merge_asof(
        news.sort_values("event_time"),
        px.sort_values("time"),
        left_on="event_time",
        right_on="time",
        direction="forward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )
    joined = joined.dropna(subset=["Close", "close_fwd", "exit_time"]).copy()
    if joined.empty:
        return joined

    raw_return = np.log(joined["close_fwd"] / joined["Close"])
    sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
    joined["domestic_return_1h"] = raw_return * sign
    joined["pip_size"] = joined["pair"].map(pip_size)
    joined["domestic_return_pips_1h"] = ((joined["close_fwd"] - joined["Close"]) / joined["pip_size"]) * sign
    joined["domestic_return_pct_1h"] = np.expm1(joined["domestic_return_1h"])
    return joined


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


def apply_thresholds(events, probs, threshold_row):
    threshold = float(threshold_row["threshold"])
    signal_mode = str(threshold_row.get("signal_mode", "strict") or "strict")
    collapse = bool(threshold_row.get("collapse_flag", False))
    final_signal = []
    reason = []
    raw_pred = []
    confidence = []
    for _, row in probs.iterrows():
        values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
        if signal_mode == "directional":
            raw = "BUY" if row.prob_buy >= row.prob_sell else "SELL"
            conf = float(max(row.prob_buy, row.prob_sell))
        else:
            raw = max(values, key=values.get)
            conf = float(values[raw])
        raw_pred.append(raw)
        confidence.append(conf)
        if collapse and signal_mode != "directional":
            final_signal.append("NO_TRADE")
            reason.append("COLLAPSE_FILTERED")
        elif raw == "NO_TRADE":
            final_signal.append("NO_TRADE")
            reason.append("NO_TRADE")
        elif conf < threshold:
            final_signal.append("NO_TRADE")
            reason.append("LOW_CONFIDENCE")
        else:
            final_signal.append(raw)
            reason.append(raw)
    out = events.copy()
    out[["prob_sell", "prob_no_trade", "prob_buy"]] = probs[["prob_sell", "prob_no_trade", "prob_buy"]].to_numpy()
    out["raw_predicted_class"] = raw_pred
    out["confidence"] = confidence
    out["threshold"] = threshold
    out["signal_mode"] = signal_mode
    out["signal"] = final_signal
    out["signal_reason"] = reason
    return out


def max_drawdown(equity):
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return worst


def run_pair_model(pair, model_name, model_path, metadata, thresholds, args):
    out_dir = BACKTRADER_ROOT / pair / model_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = prepare_pair_events(pair, Path(args.news_test), args.start, args.end, args.holding_bars, args.tolerance_hours)
    if events.empty:
        pd.DataFrame().to_csv(out_dir / "backtrader_predictions.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "backtrader_orders.csv", index=False)
        summary = {"pair": pair, "model": model_name, "period_start": args.start, "period_end": args.end, "trades": 0, "status": "no_events_or_no_price_coverage"}
        pd.DataFrame([summary]).to_csv(out_dir / "backtrader_summary.csv", index=False)
        pd.DataFrame([summary]).to_csv(out_dir / "backtrader_normalized_summary.csv", index=False)
        (out_dir / "backtrader_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    threshold_match = thresholds[(thresholds["model"].eq(model_name)) & (thresholds["pair"].eq(pair))]
    if threshold_match.empty:
        raise ValueError(f"Missing threshold row for model={model_name} pair={pair}")
    threshold_row = threshold_match.iloc[0]

    feature_names = metadata["features"]
    for col in feature_names:
        if col not in events.columns:
            events[col] = 0
    X = events[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = joblib.load(model_path)
    probs = class_probability_frame(model, X)
    events = apply_thresholds(events.reset_index(drop=True), probs.reset_index(drop=True), threshold_row)
    events["model"] = model_name

    pred_cols = [
        "event_time", "pair", "CountryCode", "currency", "Name", "Actual", "Forecast", "Previous",
        "Importance", "Impact", "model", "prob_sell", "prob_no_trade", "prob_buy", "raw_predicted_class",
        "confidence", "threshold", "signal_mode", "signal", "signal_reason", "Close", "domestic_return_1h",
        "domestic_return_pips_1h", "domestic_return_pct_1h",
    ]
    events[pred_cols].rename(columns={"Close": "close"}).to_csv(out_dir / "backtrader_predictions.csv", index=False)

    orders = []
    cash = args.initial_cash
    equity = [cash]
    for _, row in events[events["signal"].isin(["BUY", "SELL"])].iterrows():
        domestic_direction = 1.0 if row["signal"] == "BUY" else -1.0
        raw_direction = domestic_direction if row["orientation"] == "base" else -domestic_direction
        pnl = (row["close_fwd"] - row["Close"]) * raw_direction * args.trade_size
        pnl_pips = row["domestic_return_pips_1h"] * domestic_direction
        cash += pnl
        equity.append(cash)
        orders.append({
            "entry_time": row["time"],
            "exit_time": row["exit_time"],
            "pair": pair,
            "model": model_name,
            "event_name": row["Name"],
            "signal": row["signal"],
            "entry_price": row["Close"],
            "exit_price": row["close_fwd"],
            "size": args.trade_size,
            "pnl": pnl,
            "pnl_pips": pnl_pips,
            "return_pct": pnl / args.initial_cash if args.initial_cash else np.nan,
            "reason": f"event_holding_{args.holding_bars}h",
        })
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(out_dir / "backtrader_orders.csv", index=False)

    trade_count = int(len(orders_df))
    buy_count = int(events["signal"].eq("BUY").sum())
    sell_count = int(events["signal"].eq("SELL").sum())
    no_trade_count = int(events["signal"].eq("NO_TRADE").sum())
    wins = int((orders_df["pnl_pips"] > 0).sum()) if not orders_df.empty else 0
    gross_profit = float(orders_df.loc[orders_df["pnl_pips"] > 0, "pnl_pips"].sum()) if not orders_df.empty else 0.0
    gross_loss = float(-orders_df.loc[orders_df["pnl_pips"] < 0, "pnl_pips"].sum()) if not orders_df.empty else 0.0
    net_pips = float(orders_df["pnl_pips"].sum()) if not orders_df.empty else 0.0
    normalized_return = net_pips / max(1, trade_count)
    summary = {
        "pair": pair,
        "model": model_name,
        "period_start": args.start,
        "period_end": args.end,
        "trades": trade_count,
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "no_trade_signals": no_trade_count,
        "win_rate": wins / trade_count if trade_count else 0.0,
        "net_profit": float(orders_df["pnl"].sum()) if not orders_df.empty else 0.0,
        "net_pips": net_pips,
        "avg_pips_per_trade": normalized_return,
        "net_return_pct": (cash - args.initial_cash) / args.initial_cash if args.initial_cash else np.nan,
        "normalized_return": normalized_return,
        "max_drawdown": max_drawdown(equity),
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
        "threshold": float(threshold_row["threshold"]),
        "signal_mode": str(threshold_row.get("signal_mode", "strict") or "strict"),
        "events": int(len(events)),
        "status": "ok",
    }
    pd.DataFrame([summary]).to_csv(out_dir / "backtrader_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "backtrader_normalized_summary.csv", index=False)
    (out_dir / "backtrader_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    print("[PROGRESS] START 10_backtrader_news_events.py", flush=True)
    if not Path(args.news_test).exists():
        raise FileNotFoundError(f"Missing news test file: {args.news_test}")
    metadata = load_metadata()
    thresholds = load_thresholds()
    models = available_model_files()
    if not models:
        raise FileNotFoundError(f"No joblib models found in {ARTIFACTS / 'joblib'}")
    pairs = selected_values(args.pairs, available_pairs())
    model_names = selected_values(args.models, models.keys())
    print(f"[DATA] pairs={pairs}", flush=True)
    print(f"[DATA] models={model_names}", flush=True)
    print(f"[DATA] features={len(metadata['features'])}", flush=True)
    BACKTRADER_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for pair in pairs:
        for model_name in model_names:
            print(f"[BACKTRADER] START pair={pair} model={model_name}", flush=True)
            summary = run_pair_model(pair, model_name, models[model_name], metadata, thresholds, args)
            summaries.append(summary)
            print(f"[BACKTRADER] DONE pair={pair} model={model_name} trades={summary.get('trades', 0)} status={summary.get('status')}", flush=True)
    all_summary = pd.DataFrame(summaries)
    summary_path = BACKTRADER_ROOT / "backtrader_all_summary.csv"
    normalized_summary_path = BACKTRADER_ROOT / "backtrader_normalized_summary.csv"
    if summary_path.exists() and args.models.strip().lower() != "all":
        previous = pd.read_csv(summary_path)
        if {"pair", "model"}.issubset(previous.columns) and {"pair", "model"}.issubset(all_summary.columns):
            keys = set(zip(all_summary["pair"].astype(str), all_summary["model"].astype(str)))
            previous = previous[
                ~previous.apply(lambda row: (str(row["pair"]), str(row["model"])) in keys, axis=1)
            ]
            all_summary = pd.concat([previous, all_summary], ignore_index=True, sort=False)
    all_summary.to_csv(BACKTRADER_ROOT / "backtrader_all_summary.csv", index=False)
    all_summary.to_csv(normalized_summary_path, index=False)
    print(f"[OUTPUT] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
