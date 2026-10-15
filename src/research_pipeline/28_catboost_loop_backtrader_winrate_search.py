from common import *
from pathlib import Path
import json
import math
import shutil
import time

import joblib
from catboost import CatBoostClassifier

from importlib.machinery import SourceFileLoader

btmod = SourceFileLoader("btmod", str(ROOT / "10_backtrader_news_events.py")).load_module()

TARGET_WINRATE = float(os.environ.get("CATBOOST_LOOP_TARGET_WINRATE", "0.60"))
MIN_TRADES = int(os.environ.get("CATBOOST_LOOP_MIN_TRADES", "300"))
TRAIN_END = int(os.environ.get("CATBOOST_LOOP_TRAIN_END", "2023"))
TEST_START = os.environ.get("CATBOOST_LOOP_TEST_START", "2024-01-01")
TEST_END = os.environ.get("CATBOOST_LOOP_TEST_END", "2026-08-31")
PAIRS = [p.strip().upper() for p in os.environ.get("CATBOOST_LOOP_PAIRS", ",".join(available_pairs())).split(",") if p.strip()]
OUT_DIR = ARTIFACTS / "catboost_loop_winrate_search"
FEATURES = model_feature_names()


def attempts():
    return [
        {
            "name": "cb01_global_multiclass_balanced_thr045",
            "scope": "global",
            "target_mode": "multiclass",
            "iterations": 350,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "threshold": 0.45,
            "importance_min": 0,
            "impact_min": 0,
            "class_weights": [1.3, 0.55, 1.3],
            "note": "Global CatBoost multiclass, balanced toward BUY/SELL, moderate threshold.",
        },
        {
            "name": "cb02_global_multiclass_conservative_thr055",
            "scope": "global",
            "target_mode": "multiclass",
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 8,
            "threshold": 0.55,
            "importance_min": 1,
            "impact_min": 0,
            "class_weights": [1.5, 0.45, 1.5],
            "note": "Global multiclass with stronger confidence and low-importance filter.",
        },
        {
            "name": "cb03_global_directional_binary_thr060",
            "scope": "global",
            "target_mode": "binary_directional",
            "iterations": 450,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6,
            "threshold": 0.60,
            "importance_min": 1,
            "impact_min": 0,
            "note": "Binary BUY/SELL model trained only on significant historical moves.",
        },
        {
            "name": "cb04_global_directional_binary_high_impact_thr065",
            "scope": "global",
            "target_mode": "binary_directional",
            "iterations": 550,
            "depth": 6,
            "learning_rate": 0.035,
            "l2_leaf_reg": 10,
            "threshold": 0.65,
            "importance_min": 2,
            "impact_min": 1,
            "note": "Binary model restricted to stronger news events and stronger confidence.",
        },
        {
            "name": "cb05_pair_specific_multiclass_thr055",
            "scope": "pair",
            "target_mode": "multiclass",
            "iterations": 350,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6,
            "threshold": 0.55,
            "importance_min": 1,
            "impact_min": 0,
            "class_weights": [1.4, 0.5, 1.4],
            "note": "Separate CatBoost multiclass model per pair.",
        },
        {
            "name": "cb06_pair_specific_directional_binary_thr060",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 450,
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 8,
            "threshold": 0.60,
            "importance_min": 1,
            "impact_min": 0,
            "note": "Separate binary BUY/SELL model per pair.",
        },
        {
            "name": "cb07_pair_specific_directional_binary_high_impact_thr065",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 550,
            "depth": 5,
            "learning_rate": 0.035,
            "l2_leaf_reg": 10,
            "threshold": 0.65,
            "importance_min": 2,
            "impact_min": 1,
            "note": "Pair-specific binary model restricted to important and impactful news.",
        },
        {
            "name": "cb08_pair_specific_directional_binary_ultra_selective_thr070",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 650,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12,
            "threshold": 0.70,
            "importance_min": 2,
            "impact_min": 2,
            "note": "Ultra-selective pair-specific binary model for winrate search.",
        },
    ]


def load_training_frame():
    cols = [
        "event_year", "pair", "CountryCode", "Name", "Importance", "Impact",
        "trade_label_1h", "domestic_return_pips_1h",
    ] + FEATURES
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=lambda c: c in set(cols), low_memory=False)
    d = d[pd.to_numeric(d["event_year"], errors="coerce").le(TRAIN_END)].copy()
    d = d.dropna(subset=FEATURES + ["trade_label_1h"])
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    d["trade_label_1h"] = d["trade_label_1h"].astype(int)
    d["direction_label"] = np.where(pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce") >= 0, 1, -1)
    return d


def make_model(cfg, class_values):
    params = {
        "iterations": cfg["iterations"],
        "depth": cfg["depth"],
        "learning_rate": cfg["learning_rate"],
        "l2_leaf_reg": cfg["l2_leaf_reg"],
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "random_seed": 42,
        "verbose": 100,
        "allow_writing_files": False,
    }
    if cfg.get("class_weights") is not None and len(class_values) == 3:
        params["class_weights"] = cfg["class_weights"]
    return CatBoostClassifier(**params)


def train_one_model(train_df, cfg, pair=None):
    data = train_df.copy()
    if pair:
        data = data[data["pair"].eq(pair)].copy()
    data = data[
        pd.to_numeric(data["Importance"], errors="coerce").fillna(0).ge(cfg["importance_min"]) &
        pd.to_numeric(data["Impact"], errors="coerce").fillna(0).ge(cfg["impact_min"])
    ].copy()

    if cfg["target_mode"] == "binary_directional":
        data = data[data["trade_label_1h"].ne(0)].copy()
        y = data["direction_label"].astype(int)
    else:
        y = data["trade_label_1h"].astype(int)

    if data.empty or y.nunique() < 2:
        return None, {"rows": len(data), "classes": sorted(y.unique().tolist()) if len(y) else []}

    model = make_model(cfg, sorted(y.unique().tolist()))
    model.fit(data[FEATURES], y)
    return model, {"rows": len(data), "classes": sorted(y.unique().tolist())}


def probability_to_signal(model, X, cfg):
    classes = list(model.classes_)
    proba = np.asarray(model.predict_proba(X))
    out = pd.DataFrame(0.0, index=X.index, columns=["prob_sell", "prob_no_trade", "prob_buy"])
    for idx, cls in enumerate(classes):
        cls = int(cls)
        if cls == -1:
            out["prob_sell"] = proba[:, idx]
        elif cls == 0:
            out["prob_no_trade"] = proba[:, idx]
        elif cls == 1:
            out["prob_buy"] = proba[:, idx]

    signals = []
    reasons = []
    confidence = []
    raw = []
    for _, row in out.iterrows():
        if cfg["target_mode"] == "binary_directional":
            pred = "BUY" if row.prob_buy >= row.prob_sell else "SELL"
            conf = float(max(row.prob_buy, row.prob_sell))
        else:
            values = {"SELL": row.prob_sell, "NO_TRADE": row.prob_no_trade, "BUY": row.prob_buy}
            pred = max(values, key=values.get)
            conf = float(values[pred])
        raw.append(pred)
        confidence.append(conf)
        if pred == "NO_TRADE":
            signals.append("NO_TRADE")
            reasons.append("NO_TRADE")
        elif conf < cfg["threshold"]:
            signals.append("NO_TRADE")
            reasons.append("LOW_CONFIDENCE")
        else:
            signals.append(pred)
            reasons.append(pred)
    return out, raw, confidence, signals, reasons


def evaluate_pair(pair, cfg, models, attempt_dir):
    events = btmod.prepare_pair_events(pair, btmod.DEFAULT_NEWS_TEST, TEST_START, TEST_END, 1, 2)
    out_dir = attempt_dir / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        return {"pair": pair, "trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "status": "no_events"}

    events = events[
        pd.to_numeric(events["Importance"], errors="coerce").fillna(0).ge(cfg["importance_min"]) &
        pd.to_numeric(events["Impact"], errors="coerce").fillna(0).ge(cfg["impact_min"])
    ].copy()
    if events.empty:
        return {"pair": pair, "trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "status": "filtered_empty"}

    for col in FEATURES:
        if col not in events.columns:
            events[col] = 0
    X = events[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = models.get(pair) if cfg["scope"] == "pair" else models.get("global")
    if model is None:
        return {"pair": pair, "trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "status": "no_model"}

    probs, raw, confidence, signals, reasons = probability_to_signal(model, X, cfg)
    events = events.reset_index(drop=True)
    events[["prob_sell", "prob_no_trade", "prob_buy"]] = probs.reset_index(drop=True)
    events["raw_predicted_class"] = raw
    events["confidence"] = confidence
    events["threshold"] = cfg["threshold"]
    events["signal_mode"] = cfg["target_mode"]
    events["signal"] = signals
    events["signal_reason"] = reasons
    events["model"] = cfg["name"]

    pred_cols = [
        "event_time", "pair", "CountryCode", "currency", "Name", "Actual", "Forecast", "Previous",
        "Importance", "Impact", "model", "prob_sell", "prob_no_trade", "prob_buy", "raw_predicted_class",
        "confidence", "threshold", "signal_mode", "signal", "signal_reason", "Close", "domestic_return_1h",
        "domestic_return_pips_1h", "domestic_return_pct_1h",
    ]
    events[pred_cols].rename(columns={"Close": "close"}).to_csv(out_dir / "backtrader_predictions.csv", index=False)

    orders = []
    for _, row in events[events["signal"].isin(["BUY", "SELL"])].iterrows():
        domestic_direction = 1.0 if row["signal"] == "BUY" else -1.0
        raw_direction = domestic_direction if row["orientation"] == "base" else -domestic_direction
        pnl = (row["close_fwd"] - row["Close"]) * raw_direction
        pnl_pips = row["domestic_return_pips_1h"] * domestic_direction
        orders.append({
            "entry_time": row["time"],
            "exit_time": row["exit_time"],
            "pair": pair,
            "model": cfg["name"],
            "event_name": row["Name"],
            "country": row["CountryCode"],
            "signal": row["signal"],
            "entry_price": row["Close"],
            "exit_price": row["close_fwd"],
            "pnl": pnl,
            "pnl_pips": pnl_pips,
            "confidence": row["confidence"],
            "reason": "catboost_loop_event_holding_1h",
        })
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(out_dir / "backtrader_orders.csv", index=False)

    trades = len(orders_df)
    wins = int((orders_df["pnl_pips"] > 0).sum()) if trades else 0
    gross_profit = float(orders_df.loc[orders_df["pnl_pips"] > 0, "pnl_pips"].sum()) if trades else 0.0
    gross_loss = float(-orders_df.loc[orders_df["pnl_pips"] < 0, "pnl_pips"].sum()) if trades else 0.0
    net_pips = float(orders_df["pnl_pips"].sum()) if trades else 0.0
    return {
        "pair": pair,
        "trades": trades,
        "wins": wins,
        "losses": int((orders_df["pnl_pips"] < 0).sum()) if trades else 0,
        "win_rate": wins / trades if trades else 0.0,
        "net_pips": net_pips,
        "avg_pips_per_trade": net_pips / trades if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
        "events": len(events),
        "buy_signals": int(events["signal"].eq("BUY").sum()),
        "sell_signals": int(events["signal"].eq("SELL").sum()),
        "no_trade_signals": int(events["signal"].eq("NO_TRADE").sum()),
        "status": "ok",
    }


def aggregate_summary(rows):
    d = pd.DataFrame(rows)
    trades = int(d["trades"].sum()) if not d.empty else 0
    wins = int(d.get("wins", pd.Series(dtype=float)).sum()) if not d.empty and "wins" in d else 0
    net_pips = float(d["net_pips"].sum()) if not d.empty else 0.0
    gp = 0.0
    gl = 0.0
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades if trades else 0.0,
        "net_pips": net_pips,
        "avg_pips_per_trade": net_pips / trades if trades else 0.0,
        "positive_pairs": int((d["net_pips"] > 0).sum()) if not d.empty else 0,
        "pairs": int(d["pair"].nunique()) if not d.empty else 0,
    }


def main():
    print("[PROGRESS] START 28_catboost_loop_backtrader_winrate_search.py", flush=True)
    print(f"[CONFIG] target_winrate={TARGET_WINRATE} min_trades={MIN_TRADES} train_end={TRAIN_END} test={TEST_START}..{TEST_END}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_training_frame()
    print(f"[DATA] train_rows={len(train_df)} pairs={PAIRS} features={len(FEATURES)}", flush=True)

    loop_rows = []
    best = None
    best_model_dir = OUT_DIR / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)

    for idx, cfg in enumerate(attempts(), start=1):
        started = time.perf_counter()
        attempt_dir = OUT_DIR / f"{idx:02d}_{cfg['name']}"
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[LOOP] START {idx}/{len(attempts())} {cfg['name']} | {cfg['note']}", flush=True)

        models = {}
        train_status = []
        if cfg["scope"] == "global":
            model, status = train_one_model(train_df, cfg)
            train_status.append({"scope": "global", **status})
            if model is not None:
                models["global"] = model
                joblib.dump(model, attempt_dir / f"{cfg['name']}.joblib")
        else:
            for pair in PAIRS:
                print(f"[TRAIN] pair={pair}", flush=True)
                model, status = train_one_model(train_df, cfg, pair=pair)
                train_status.append({"scope": pair, **status})
                if model is not None:
                    models[pair] = model
                    joblib.dump(model, attempt_dir / f"{cfg['name']}_{pair}.joblib")
        pd.DataFrame(train_status).to_csv(attempt_dir / "training_status.csv", index=False)

        pair_rows = []
        for pair in PAIRS:
            print(f"[BACKTRADER] attempt={cfg['name']} pair={pair}", flush=True)
            row = evaluate_pair(pair, cfg, models, attempt_dir)
            pair_rows.append(row)
            print(f"[BACKTRADER] pair={pair} trades={row['trades']} win_rate={row['win_rate']:.4f} net_pips={row['net_pips']:.1f} status={row['status']}", flush=True)

        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(attempt_dir / "pair_summary.csv", index=False)
        agg = aggregate_summary(pair_rows)
        elapsed = round(time.perf_counter() - started, 2)
        loop_row = {
            "attempt": idx,
            "model": cfg["name"],
            "scope": cfg["scope"],
            "target_mode": cfg["target_mode"],
            "threshold": cfg["threshold"],
            "importance_min": cfg["importance_min"],
            "impact_min": cfg["impact_min"],
            "trades": agg["trades"],
            "wins": agg["wins"],
            "win_rate": agg["win_rate"],
            "net_pips": agg["net_pips"],
            "avg_pips_per_trade": agg["avg_pips_per_trade"],
            "positive_pairs": agg["positive_pairs"],
            "pairs": agg["pairs"],
            "seconds": elapsed,
            "target_reached": bool(agg["win_rate"] >= TARGET_WINRATE and agg["trades"] >= MIN_TRADES),
            "note": cfg["note"],
        }
        loop_rows.append(loop_row)
        pd.DataFrame(loop_rows).to_csv(OUT_DIR / "catboost_loop_results.csv", index=False)
        print(f"[LOOP] DONE {cfg['name']} trades={agg['trades']} win_rate={agg['win_rate']:.4f} net_pips={agg['net_pips']:.1f} positive_pairs={agg['positive_pairs']} seconds={elapsed}", flush=True)

        rank_key = (loop_row["target_reached"], loop_row["win_rate"], loop_row["net_pips"], loop_row["trades"])
        if best is None or rank_key > best["rank_key"]:
            best = {"rank_key": rank_key, "row": loop_row, "dir": attempt_dir}
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            shutil.copytree(attempt_dir, best_model_dir)

        if loop_row["target_reached"]:
            print(f"[TARGET] REACHED by {cfg['name']}", flush=True)
            break

    results = pd.DataFrame(loop_rows)
    results.to_csv(OUT_DIR / "catboost_loop_results.csv", index=False)
    best_row = best["row"] if best else {}
    report = [
        "# CatBoost Loop Backtrader Winrate Search",
        "",
        f"- Target winrate: `{TARGET_WINRATE:.2%}`",
        f"- Minimum trades: `{MIN_TRADES}`",
        f"- Training period used: `2010-{TRAIN_END}`",
        f"- Backtrader test period: `{TEST_START}` to `{TEST_END}`",
        f"- Pairs: `{', '.join(PAIRS)}`",
        "",
        "## Important Caveat",
        "",
        "This is a development search because the loop checks test performance repeatedly. The selected model must be validated again on a fresh holdout or with MT5 Strategy Tester before it is used as final paper evidence.",
        "",
        "## Best Attempt",
        "",
        json.dumps(best_row, indent=2),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'catboost_loop_results.csv'}`",
        f"- `{best_model_dir}`",
    ]
    (OUT_DIR / "LAPORAN_CATBOOST_LOOP_BACKTRADER_WINRATE_SEARCH.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'catboost_loop_results.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_CATBOOST_LOOP_BACKTRADER_WINRATE_SEARCH.md'}", flush=True)
    if best_row:
        print(f"[BEST] {best_row}", flush=True)


if __name__ == "__main__":
    main()
