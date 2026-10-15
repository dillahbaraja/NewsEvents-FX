from common import *

import importlib.util
import math
from itertools import product

import joblib
from sklearn.ensemble import RandomForestClassifier


ROOT50 = ROOT / "50_catboost_block_inverse_pipeline.py"
spec50 = importlib.util.spec_from_file_location("cbi_v1", ROOT50)
cbi = importlib.util.module_from_spec(spec50)
spec50.loader.exec_module(cbi)


OUT_DIR = ARTIFACTS / "random_forest_v2_base_pair_specific"
JOBLIB_DIR = OUT_DIR / "joblib"
MODEL_NAME = "random_forest_v2_base_pair_specific"

TRAIN_END = 2020
VALIDATION_START = 2021
VALIDATION_END = 2023

STAGE1_TARGET = "trade_target"
STAGE2_TARGET = "direction_target"

RF_GRID = [
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "class_weight": class_weight,
    }
    for n_estimators, max_depth, min_samples_leaf, max_features, class_weight in product(
        [500, 800],
        [8, 12, 18, None],
        [5, 10, 20],
        ["sqrt", "log2", 0.5],
        ["balanced_subsample"],
    )
]

STAGE1_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
STAGE2_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def fit_rf(params, random_state=42):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )


def positive_probability(model, x):
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_).astype(int)
    idx = int(np.where(classes == 1)[0][0])
    return probs[:, idx]


def direction_outputs(model, x):
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_).astype(int)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = probs[:, sell_idx]
    buy_prob = probs[:, buy_idx]
    pred = np.where(buy_prob >= sell_prob, 1, -1)
    conf = np.maximum(sell_prob, buy_prob)
    return sell_prob, buy_prob, pred, conf


def make_orders(df, stage1_model, stage2_model, features, trade_threshold, direction_threshold):
    out = df.copy()
    out["prob_trade"] = positive_probability(stage1_model, out[features])
    sell_prob, buy_prob, pred, conf = direction_outputs(stage2_model, out[features])
    out["prob_sell"] = sell_prob
    out["prob_buy"] = buy_prob
    out["pred_direction"] = pred.astype(int)
    out["direction_confidence"] = conf
    out["final_signal"] = np.where(
        (out["prob_trade"] >= trade_threshold) & (out["direction_confidence"] >= direction_threshold),
        out["pred_direction"],
        0,
    )
    out["pnl_pips"] = out["final_signal"] * out["domestic_return_pips_1h"]
    return out


def summarize_orders(pair, orders, split, params=None, trade_threshold=None, direction_threshold=None):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    pnl = pd.to_numeric(active["pnl_pips"], errors="coerce").fillna(0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    n = int(len(active))
    row = {
        "pair": pair,
        "split": split,
        "trade_count": n,
        "buy_count": int((active["final_signal"] == 1).sum()),
        "sell_count": int((active["final_signal"] == -1).sum()),
        "net_pips": float(pnl.sum()),
        "avg_pips": float(pnl.mean()) if n else 0.0,
        "winrate": float((pnl > 0).mean()) if n else 0.0,
        "profit_factor": gp / gl if gl else (math.inf if gp > 0 else 0.0),
        "trade_threshold": trade_threshold,
        "direction_threshold": direction_threshold,
    }
    if params:
        row.update(params)
    return row


def score_validation(row):
    if row["trade_count"] < 100:
        return -1e12
    if row["profit_factor"] < 1.0:
        return row["net_pips"] - 5000
    balance_penalty = abs(row["buy_count"] - row["sell_count"]) / max(row["trade_count"], 1) * 300
    return row["net_pips"] + min(row["profit_factor"], 3.0) * 1000 + row["avg_pips"] * 200 - balance_penalty


def signal_table(orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    return pd.DataFrame({
        "EVENT_TIME": pd.to_datetime(active["event_time"]).dt.strftime("%Y.%m.%d %H:%M"),
        "SYMBOL": active["pair"].astype(str),
        "COUNTRY": active["CountryCode"].astype(str),
        "EVENT_NAME": active["Name"].astype(str),
        "PROBABILITY": active[["prob_trade", "direction_confidence"]].min(axis=1),
        "SIGNAL": active["final_signal"].astype(int),
        "MODEL": MODEL_NAME,
    })


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in view.columns) + " |")
    return "\n".join(lines)


def run_pair(pair, df, oos_df, features):
    pair_started = time.perf_counter()
    print(f"[PAIR] START {pair}", flush=True)
    train = df[(df["pair"].eq(pair)) & (df["event_year"].le(TRAIN_END))].copy()
    validation = df[(df["pair"].eq(pair)) & (df["event_year"].between(VALIDATION_START, VALIDATION_END))].copy()
    oos = oos_df[oos_df["pair"].eq(pair)].copy()
    s1_train = train.dropna(subset=[STAGE1_TARGET])
    s2_train = train.dropna(subset=[STAGE2_TARGET])
    s1_val = validation.dropna(subset=[STAGE1_TARGET])
    if len(s1_train) < 500 or len(s2_train) < 200 or len(s1_val) < 100:
        print(f"[PAIR] SKIP {pair} insufficient_rows", flush=True)
        return [], [], []

    rows = []
    artifacts = []
    best = None
    best_orders = None
    total = len(RF_GRID)
    for i, params in enumerate(RF_GRID, start=1):
        trial_started = time.perf_counter()
        print(f"[TRAIN] START {pair} trial={i}/{total} params={params}", flush=True)
        stage1 = fit_rf(params, random_state=42 + i)
        stage2 = fit_rf(params, random_state=142 + i)
        print(f"[TRAIN] FIT_STAGE1 {pair} trial={i}/{total} rows={len(s1_train)}", flush=True)
        stage1.fit(s1_train[features], s1_train[STAGE1_TARGET].astype(int))
        print(f"[TRAIN] FIT_STAGE1_DONE {pair} trial={i}/{total} elapsed={time.perf_counter() - trial_started:.1f}s", flush=True)
        print(f"[TRAIN] FIT_STAGE2 {pair} trial={i}/{total} rows={len(s2_train)}", flush=True)
        stage2.fit(s2_train[features], s2_train[STAGE2_TARGET].astype(int))
        print(f"[TRAIN] FIT_STAGE2_DONE {pair} trial={i}/{total} elapsed={time.perf_counter() - trial_started:.1f}s", flush=True)
        print(f"[TUNE] START {pair} trial={i}/{total} thresholds={len(STAGE1_THRESHOLDS) * len(STAGE2_THRESHOLDS)}", flush=True)
        for t1 in STAGE1_THRESHOLDS:
            for t2 in STAGE2_THRESHOLDS:
                val_orders = make_orders(s1_val, stage1, stage2, features, t1, t2)
                row = summarize_orders(pair, val_orders, "validation", params, t1, t2)
                row["score"] = score_validation(row)
                rows.append(row)
                if best is None or row["score"] > best["row"]["score"]:
                    best = {"row": row, "stage1": stage1, "stage2": stage2, "params": params, "t1": t1, "t2": t2}
                    best_orders = val_orders
        print(
            f"[TRAIN] DONE {pair} trial={i}/{total} best_score={best['row']['score']:.2f} best_val_net={best['row']['net_pips']:.1f} elapsed={time.perf_counter() - trial_started:.1f}s",
            flush=True,
        )
        if i % 5 == 0 or i == total:
            print(f"[PAIR] HEARTBEAT {pair} trials={i}/{total} elapsed={time.perf_counter() - pair_started:.1f}s", flush=True)

    if best is None:
        return rows, [], []

    pair_dir = JOBLIB_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["stage1"], pair_dir / "random_forest_v2_trade_filter.joblib")
    joblib.dump(best["stage2"], pair_dir / "random_forest_v2_direction.joblib")
    print(f"[OUTPUT] {pair} saved_models={pair_dir}", flush=True)
    best_orders.to_csv(OUT_DIR / f"{pair}_validation_orders.csv", index=False)
    print(f"[OOS] START {pair} rows={len(oos)}", flush=True)
    oos_orders = make_orders(oos, best["stage1"], best["stage2"], features, best["t1"], best["t2"])
    oos_orders.to_csv(OUT_DIR / f"{pair}_oos_orders.csv", index=False)
    oos_summary = summarize_orders(pair, oos_orders, "oos_2024_2026", best["params"], best["t1"], best["t2"])
    oos_summary["score"] = np.nan
    print(f"[PAIR] DONE {pair} validation_net={best['row']['net_pips']:.1f} oos_net={oos_summary['net_pips']:.1f} oos_pf={oos_summary['profit_factor']:.4f} elapsed={time.perf_counter() - pair_started:.1f}s", flush=True)
    return rows, [best["row"], oos_summary], [oos_orders]


def aggregate(summary):
    oos = summary[summary["split"].eq("oos_2024_2026")].copy()
    if oos.empty:
        return pd.DataFrame()
    trades = int(oos["trade_count"].sum())
    net = float(oos["net_pips"].sum())
    return pd.DataFrame([{
        "pair": "ALL",
        "split": "oos_2024_2026",
        "trade_count": trades,
        "buy_count": int(oos["buy_count"].sum()),
        "sell_count": int(oos["sell_count"].sum()),
        "net_pips": net,
        "avg_pips": net / trades if trades else 0.0,
        "winrate": np.nan,
        "profit_factor": np.nan,
        "positive_pairs": int((oos["net_pips"] > 0).sum()),
    }])


def main():
    started = progress_start("55_random_forest_v2_base_pair_specific.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLIB_DIR.mkdir(parents=True, exist_ok=True)
    progress_step("LOAD_FEATURES")
    features = cbi.load_features()
    progress_step("LOAD_TRAINING_DATA")
    df = cbi.read_modeling_dataset(features)
    progress_step("LOAD_OOS_DATA")
    oos_df = cbi.prepare_oos_events(features)
    pairs = sorted(oos_df["pair"].dropna().unique().tolist())
    progress_step("PAIR_LIST", pairs=",".join(pairs), pair_count=len(pairs), rf_grid=len(RF_GRID))

    grid_rows = []
    summary_rows = []
    all_oos_orders = []
    for idx, pair in enumerate(pairs, start=1):
        progress_step("PAIR_BEGIN", idx, len(pairs), pair=pair)
        rows, summary, orders = run_pair(pair, df, oos_df, features)
        grid_rows.extend(rows)
        summary_rows.extend(summary)
        all_oos_orders.extend(orders)
        pd.DataFrame(grid_rows).to_csv(OUT_DIR / "random_forest_v2_grid_results.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "random_forest_v2_summary_by_pair.csv", index=False)
        progress_step("PAIR_SAVED_INTERMEDIATE", idx, len(pairs), pair=pair, grid_rows=len(grid_rows), summary_rows=len(summary_rows))

    summary = pd.DataFrame(summary_rows)
    agg = aggregate(summary)
    final_summary = pd.concat([summary, agg], ignore_index=True) if not agg.empty else summary
    final_summary.to_csv(OUT_DIR / "random_forest_v2_summary.csv", index=False)
    if all_oos_orders:
        oos_orders = pd.concat(all_oos_orders, ignore_index=True)
        oos_orders.to_csv(OUT_DIR / "random_forest_v2_oos_orders.csv", index=False)
        st = signal_table(oos_orders)
        st.to_csv(OUT_DIR / "random_forest_v2_mt5_signal_table.csv", index=False)
        st.to_csv(ARTIFACTS / "random_forest_v2_mt5_signal_table.csv", index=False)

    report = "\n".join([
        "# Random Forest V2 Base Pair-Specific",
        "",
        "Pipeline ini memperbaiki base Random Forest sebelum block/inverse.",
        "",
        "Metode:",
        "",
        "- Satu RF trade-filter dan satu RF direction-model per pair.",
        "- Hyperparameter dan threshold dipilih dari validation `2021-2023`.",
        "- OOS `2024-08.2026` hanya untuk evaluasi final.",
        "",
        "## Summary",
        "",
        md_table(final_summary.sort_values(["split", "net_pips"], ascending=[True, False]), ".4f"),
    ]) + "\n"
    (OUT_DIR / "LAPORAN_RANDOM_FOREST_V2_BASE_PAIR_SPECIFIC.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_RANDOM_FOREST_V2_BASE_PAIR_SPECIFIC.md'}", flush=True)
    progress_done("55_random_forest_v2_base_pair_specific.py", started)


if __name__ == "__main__":
    main()
