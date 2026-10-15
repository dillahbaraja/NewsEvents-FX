from common import *

import importlib.util
import math
from pathlib import Path

import joblib


ROOT50 = ROOT / "50_catboost_block_inverse_pipeline.py"
spec50 = importlib.util.spec_from_file_location("cbi_v1", ROOT50)
cbi = importlib.util.module_from_spec(spec50)
spec50.loader.exec_module(cbi)

ROOT52 = ROOT / "52_catboost_block_inverse_v2_pair_gate.py"
spec52 = importlib.util.spec_from_file_location("cbi_v2", ROOT52)
cbi2 = importlib.util.module_from_spec(spec52)
spec52.loader.exec_module(cbi2)


PHASE3_JOBLIB = ARTIFACTS / "phase3_two_stage_models" / "joblib"
OUT_DIR = ARTIFACTS / "model_block_inverse_v2_all_models"
MODEL_KINDS = ["elastic_net", "random_forest"]
CAL_START = 2019
CAL_END = 2023


def load_two_stage_model(kind):
    return (
        joblib.load(PHASE3_JOBLIB / f"{kind}_trade_filter.joblib"),
        joblib.load(PHASE3_JOBLIB / f"{kind}_direction.joblib"),
    )


def positive_probability(model, x):
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_).astype(int)
    idx = int(np.where(classes == 1)[0][0])
    return probs[:, idx]


def direction_outputs(model, x):
    probs = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_).astype(int)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = probs[:, sell_idx]
    buy_prob = probs[:, buy_idx]
    pred = np.where(buy_prob >= sell_prob, 1, -1)
    conf = np.maximum(sell_prob, buy_prob)
    return sell_prob, buy_prob, pred, conf


def add_predictions(df, trade_model, direction_model, features):
    out = df.copy()
    out["prob_trade"] = positive_probability(trade_model, out[features])
    sell_prob, buy_prob, pred, conf = direction_outputs(direction_model, out[features])
    out["prob_sell"] = sell_prob
    out["prob_buy"] = buy_prob
    out["pred_direction"] = pred.astype(int)
    out["direction_confidence"] = conf
    return out


def base_orders(pred):
    out = pred.copy()
    out["base_signal"] = np.where(
        (out["prob_trade"] >= 0.50) & (out["direction_confidence"] >= 0.50),
        out["pred_direction"],
        0,
    )
    out["action"] = "BASE_MODEL"
    out["final_signal"] = out["base_signal"]
    out["pnl_pips"] = out["final_signal"] * out["domestic_return_pips_1h"]
    out["status"] = np.where(out["final_signal"].isin([1, -1]), "TRADE_BASE", "NO_BASE_SIGNAL")
    return out


def summary_all(model, version, split, orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    pnl = pd.to_numeric(active["pnl_pips"], errors="coerce").fillna(0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    n = int(len(active))
    return {
        "model": model,
        "version": version,
        "split": split,
        "trade_count": n,
        "buy_count": int((active["final_signal"] == 1).sum()),
        "sell_count": int((active["final_signal"] == -1).sum()),
        "net_pips": float(pnl.sum()),
        "avg_pips": float(pnl.mean()) if n else 0.0,
        "winrate": float((pnl > 0).mean()) if n else 0.0,
        "profit_factor": gp / gl if gl else (math.inf if gp > 0 else 0.0),
        "positive_pairs": int(active.groupby("pair")["pnl_pips"].sum().gt(0).sum()) if n else 0,
        "active_pairs": int(active["pair"].nunique()) if n else 0,
    }


def summary_by_pair(model, version, split, orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    rows = []
    for pair, g in active.groupby("pair"):
        pnl = pd.to_numeric(g["pnl_pips"], errors="coerce").fillna(0)
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        rows.append({
            "model": model,
            "version": version,
            "split": split,
            "pair": pair,
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == 1).sum()),
            "sell_count": int((g["final_signal"] == -1).sum()),
            "net_pips": float(pnl.sum()),
            "avg_pips": float(pnl.mean()) if len(g) else 0.0,
            "winrate": float((pnl > 0).mean()) if len(g) else 0.0,
            "profit_factor": gp / gl if gl else (math.inf if gp > 0 else 0.0),
        })
    return pd.DataFrame(rows)


def write_signal_table(model, orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    out = pd.DataFrame({
        "EVENT_TIME": pd.to_datetime(active["event_time"]).dt.strftime("%Y.%m.%d %H:%M"),
        "SYMBOL": active["pair"].astype(str),
        "COUNTRY": active["CountryCode"].astype(str),
        "EVENT_NAME": active["Name"].astype(str),
        "PROBABILITY": active[["prob_trade", "direction_confidence"]].min(axis=1),
        "SIGNAL": active["final_signal"].astype(int),
        "MODEL": f"{model}_block_inverse_v2",
    })
    out.to_csv(OUT_DIR / f"{model}_block_inverse_v2_mt5_signal_table.csv", index=False)
    return out


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


def run_model(kind, df, oos_df, features):
    print(f"[MODEL] START {kind}", flush=True)
    trade_model, direction_model = load_two_stage_model(kind)
    calibration = df[df["event_year"].between(CAL_START, CAL_END)].dropna(subset=[cbi.STAGE1_TARGET]).copy()
    cal_pred = add_predictions(calibration, trade_model, direction_model, features)
    oos_pred = add_predictions(oos_df, trade_model, direction_model, features)

    cal_base = base_orders(cal_pred)
    oos_base = base_orders(oos_pred)
    stats = cbi.build_event_stats(cal_base)
    stats.to_csv(OUT_DIR / f"{kind}_event_stats.csv", index=False)

    best = None
    grid_rows = []
    for i, cfg in enumerate(cbi2.STRICT_GRID, start=1):
        event_map = cbi2.make_strict_event_map(stats, cfg)
        cal_bi = cbi.apply_event_map(cal_pred, event_map, cfg)
        base_pair = cbi2.summarize_pair(cal_base, kind, "BASE_MODEL")
        bi_pair = cbi2.summarize_pair(cal_bi, kind, "BLOCK_INVERSE")
        pair_sel = cbi2.select_pairs(base_pair, bi_pair)
        cal_final = cbi2.apply_pair_selection(cal_base, cal_bi, pair_sel)
        row = cbi2.summary_all(cal_final, "calibration")
        row.update(cfg)
        row["model"] = kind
        row["score"] = row["net_pips"] + min(row["profit_factor"], 3.0) * 1000 + row["positive_pairs"] * 400
        grid_rows.append(row)
        if best is None or row["score"] > best["row"]["score"]:
            best = {"row": row, "event_map": event_map, "pair_selection": pair_sel}
        if i % 100 == 0:
            print(f"[GRID] {kind} {i}/{len(cbi2.STRICT_GRID)} best_net={best['row']['net_pips']:.1f} best_pf={best['row']['profit_factor']:.4f}", flush=True)

    event_map = best["event_map"]
    pair_sel = best["pair_selection"]
    cal_bi = cbi.apply_event_map(cal_pred, event_map, best["row"])
    oos_bi = cbi.apply_event_map(oos_pred, event_map, best["row"])
    cal_final = cbi2.apply_pair_selection(cal_base, cal_bi, pair_sel)
    oos_final = cbi2.apply_pair_selection(oos_base, oos_bi, pair_sel)

    event_map.to_csv(OUT_DIR / f"{kind}_block_inverse_v2_event_map.csv", index=False)
    pair_sel.to_csv(OUT_DIR / f"{kind}_block_inverse_v2_pair_selection.csv", index=False)
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(OUT_DIR / f"{kind}_block_inverse_v2_grid_results.csv", index=False)
    cal_base.to_csv(OUT_DIR / f"{kind}_base_calibration_orders.csv", index=False)
    oos_base.to_csv(OUT_DIR / f"{kind}_base_oos_orders.csv", index=False)
    cal_final.to_csv(OUT_DIR / f"{kind}_block_inverse_v2_calibration_orders.csv", index=False)
    oos_final.to_csv(OUT_DIR / f"{kind}_block_inverse_v2_oos_orders.csv", index=False)
    write_signal_table(kind, oos_final)

    summaries = [
        summary_all(kind, "base_no_block_inverse", "calibration", cal_base),
        summary_all(kind, "block_inverse_v2", "calibration", cal_final),
        summary_all(kind, "base_no_block_inverse", "oos_2024_2026", oos_base),
        summary_all(kind, "block_inverse_v2", "oos_2024_2026", oos_final),
    ]
    pairs = pd.concat([
        summary_by_pair(kind, "base_no_block_inverse", "oos_2024_2026", oos_base),
        summary_by_pair(kind, "block_inverse_v2", "oos_2024_2026", oos_final),
    ], ignore_index=True)
    print(f"[MODEL] DONE {kind}", flush=True)
    return pd.DataFrame(summaries), pairs, pair_sel


def main():
    print("[PROGRESS] START 54_model_block_inverse_v2_all_models.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = cbi.load_features()
    df = cbi.read_modeling_dataset(features)
    oos_df = cbi.prepare_oos_events(features)

    all_summary = []
    all_pairs = []
    all_selection = []
    for kind in MODEL_KINDS:
        summary, pairs, selection = run_model(kind, df, oos_df, features)
        all_summary.append(summary)
        all_pairs.append(pairs)
        selection.insert(0, "model", kind)
        all_selection.append(selection)

    summary = pd.concat(all_summary, ignore_index=True)
    pair_summary = pd.concat(all_pairs, ignore_index=True)
    selection = pd.concat(all_selection, ignore_index=True)
    summary.to_csv(OUT_DIR / "model_block_inverse_v2_summary.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "model_block_inverse_v2_pair_summary.csv", index=False)
    selection.to_csv(OUT_DIR / "model_block_inverse_v2_pair_selection.csv", index=False)

    oos = summary[summary["split"].eq("oos_2024_2026")].sort_values("net_pips", ascending=False)
    report = "\n".join([
        "# Model Block/Inverse V2 All Models",
        "",
        "Laporan ini membandingkan dua cara untuk setiap model:",
        "",
        "- `base_no_block_inverse`: model two-stage tanpa block/inverse.",
        "- `block_inverse_v2`: model two-stage dengan strict event block/inverse dan pair gate.",
        "",
        "Unit block/inverse: `pair + country + event_name`, sehingga berita negara yang sama tidak digeneralisasi antar pair.",
        "",
        "## OOS Summary",
        "",
        md_table(oos, ".4f"),
        "",
        "## OOS Pair Matrix",
        "",
        md_table(pair_summary.sort_values(["model", "version", "net_pips"], ascending=[True, True, False]), ".4f"),
        "",
        "## Pair Selection",
        "",
        md_table(selection, ".4f"),
    ]) + "\n"
    (OUT_DIR / "LAPORAN_MODEL_BLOCK_INVERSE_V2_ALL_MODELS.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_MODEL_BLOCK_INVERSE_V2_ALL_MODELS.md'}", flush=True)
    print("[PROGRESS] DONE 54_model_block_inverse_v2_all_models.py", flush=True)


if __name__ == "__main__":
    main()
