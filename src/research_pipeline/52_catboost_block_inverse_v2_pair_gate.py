from common import *

import importlib.util
import math
import shutil
from pathlib import Path

import joblib


ROOT50 = ROOT / "50_catboost_block_inverse_pipeline.py"
spec = importlib.util.spec_from_file_location("cbi_v1", ROOT50)
cbi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbi)

SRC_DIR = ARTIFACTS / "catboost_block_inverse"
SRC_VARIANT = SRC_DIR / "v3_cal_2019_2023"
OUT_DIR = ARTIFACTS / "catboost_block_inverse_v2"
MODEL_NAME = "catboost_block_inverse_v2"

TRAIN_END = 2018
CAL_START = 2019
CAL_END = 2023

STRICT_GRID = []
for min_trade_count in [20, 30, 50, 75]:
    for min_pf in [1.05, 1.10, 1.20, 1.30]:
        for min_avg in [0.10, 0.25, 0.50]:
            for min_pos_year_ratio in [0.50, 0.60]:
                for max_dominant_year_share in [0.50, 0.60, 0.75]:
                    STRICT_GRID.append({
                        "min_trade_count": min_trade_count,
                        "min_profit_factor": min_pf,
                        "min_avg_pips": min_avg,
                        "min_positive_year_ratio": min_pos_year_ratio,
                        "max_dominant_year_share": max_dominant_year_share,
                        "unknown_policy": "BLOCK",
                        "min_inverse_trade_count": max(30, min_trade_count),
                        "min_inverse_profit_factor": max(1.20, min_pf),
                        "min_inverse_avg_pips": max(0.25, min_avg),
                        "min_inverse_positive_year_ratio": max(0.60, min_pos_year_ratio),
                        "inverse_advantage_ratio": 1.25,
                    })


def strict_action_for_row(row, cfg):
    original_ok = (
        row["trade_count"] >= cfg["min_trade_count"]
        and row["original_net_pips"] > 0
        and row["original_profit_factor"] >= cfg["min_profit_factor"]
        and row["original_avg_pips"] >= cfg["min_avg_pips"]
        and row["original_positive_year_ratio"] >= cfg["min_positive_year_ratio"]
        and row["original_dominant_year_share"] <= cfg["max_dominant_year_share"]
    )
    inverse_ok = (
        row["trade_count"] >= cfg["min_inverse_trade_count"]
        and row["inverse_net_pips"] > 0
        and row["inverse_profit_factor"] >= cfg["min_inverse_profit_factor"]
        and row["inverse_avg_pips"] >= cfg["min_inverse_avg_pips"]
        and row["inverse_positive_year_ratio"] >= cfg["min_inverse_positive_year_ratio"]
        and row["inverse_dominant_year_share"] <= cfg["max_dominant_year_share"]
        and row["inverse_net_pips"] >= max(row["original_net_pips"], 0) * cfg["inverse_advantage_ratio"]
    )
    if inverse_ok and (not original_ok or row["inverse_net_pips"] > row["original_net_pips"]):
        return "INVERSE", "strict_inverse_pass"
    if original_ok:
        return "ALLOW", "strict_original_pass"
    return "BLOCK", "rare_or_unreliable"


def make_strict_event_map(stats, cfg):
    out = stats.copy()
    if "country" in out.columns and "CountryCode" not in out.columns:
        out["CountryCode"] = out["country"]
    actions = out.apply(lambda r: strict_action_for_row(r, cfg), axis=1)
    out["action"] = [a[0] for a in actions]
    out["reason"] = [a[1] for a in actions]
    for k, v in cfg.items():
        out[k] = v
    return out


def base_orders(pred):
    out = pred.copy()
    out["base_signal"] = np.where(
        (out["prob_trade"] >= 0.50) & (out["direction_confidence"] >= 0.50),
        out["pred_direction"],
        0,
    )
    out["action"] = "BASE_CATBOOST"
    out["final_signal"] = out["base_signal"]
    out["pnl_pips"] = out["final_signal"] * out["domestic_return_pips_1h"]
    out["status"] = np.where(out["final_signal"].isin([1, -1]), "TRADE_BASE", "NO_BASE_SIGNAL")
    return out


def summarize_pair(orders, variant, method):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    rows = []
    for pair, g in active.groupby("pair"):
        pnl = pd.to_numeric(g["pnl_pips"], errors="coerce").fillna(0)
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        rows.append({
            "variant": variant,
            "method": method,
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


def score_pair(row):
    if row["trade_count"] < 100:
        return -1e9
    return row["net_pips"] + min(row["profit_factor"], 3.0) * 500 + row["avg_pips"] * 100


def select_pairs(base_summary, bi_summary):
    pairs = sorted(set(base_summary["pair"]).union(set(bi_summary["pair"])))
    rows = []
    for pair in pairs:
        b = base_summary[base_summary["pair"].eq(pair)]
        bi = bi_summary[bi_summary["pair"].eq(pair)]
        b = b.iloc[0].to_dict() if not b.empty else {"net_pips": 0, "profit_factor": 0, "trade_count": 0, "avg_pips": 0}
        bi = bi.iloc[0].to_dict() if not bi.empty else {"net_pips": 0, "profit_factor": 0, "trade_count": 0, "avg_pips": 0}
        use_bi = (
            bi["trade_count"] >= 100
            and bi["net_pips"] > b["net_pips"]
            and bi["profit_factor"] >= max(1.05, b["profit_factor"])
            and score_pair(bi) > score_pair(b)
        )
        use_base = (
            not use_bi
            and b["trade_count"] >= 100
            and b["net_pips"] > 0
            and b["profit_factor"] >= 1.00
        )
        if use_bi:
            selected = "BLOCK_INVERSE"
        elif use_base:
            selected = "BASE_CATBOOST"
        else:
            selected = "BLOCK_PAIR"
        rows.append({
            "pair": pair,
            "selected_method": selected,
            "base_trade_count": int(b["trade_count"]),
            "base_net_pips": float(b["net_pips"]),
            "base_profit_factor": float(b["profit_factor"]),
            "block_inverse_trade_count": int(bi["trade_count"]),
            "block_inverse_net_pips": float(bi["net_pips"]),
            "block_inverse_profit_factor": float(bi["profit_factor"]),
        })
    return pd.DataFrame(rows)


def apply_pair_selection(base, bi, pair_selection):
    rows = []
    for _, sel in pair_selection.iterrows():
        pair = sel["pair"]
        method = sel["selected_method"]
        if method == "BASE_CATBOOST":
            part = base[base["pair"].eq(pair)].copy()
        elif method == "BLOCK_INVERSE":
            part = bi[bi["pair"].eq(pair)].copy()
        else:
            continue
        part["selected_method"] = method
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summary_all(orders, split):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    pnl = pd.to_numeric(active["pnl_pips"], errors="coerce").fillna(0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    return {
        "split": split,
        "trade_count": int(len(active)),
        "buy_count": int((active["final_signal"] == 1).sum()),
        "sell_count": int((active["final_signal"] == -1).sum()),
        "net_pips": float(pnl.sum()),
        "avg_pips": float(pnl.mean()) if len(active) else 0,
        "winrate": float((pnl > 0).mean()) if len(active) else 0,
        "profit_factor": gp / gl if gl else (math.inf if gp > 0 else 0),
        "positive_pairs": int(active.groupby("pair")["pnl_pips"].sum().gt(0).sum()) if len(active) else 0,
        "active_pairs": int(active["pair"].nunique()) if len(active) else 0,
    }


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


def write_signal_table(orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    out = pd.DataFrame({
        "EVENT_TIME": pd.to_datetime(active["event_time"]).dt.strftime("%Y.%m.%d %H:%M"),
        "SYMBOL": active["pair"].astype(str),
        "COUNTRY": active["CountryCode"].astype(str),
        "EVENT_NAME": active["Name"].astype(str),
        "PROBABILITY": active[["prob_trade", "direction_confidence"]].min(axis=1),
        "SIGNAL": active["final_signal"].astype(int),
        "MODEL": MODEL_NAME,
    })
    out.to_csv(OUT_DIR / "catboost_block_inverse_v2_mt5_signal_table.csv", index=False)
    out.to_csv(ARTIFACTS / "catboost_block_inverse_v2_mt5_signal_table.csv", index=False)
    return out


def main():
    print("[PROGRESS] START 52_catboost_block_inverse_v2_pair_gate.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = cbi.load_features()
    df = cbi.read_modeling_dataset(features)
    oos_df = cbi.prepare_oos_events(features)
    stage1 = joblib.load(SRC_VARIANT / "catboost_block_inverse_trade_filter.joblib")
    stage2 = joblib.load(SRC_VARIANT / "catboost_block_inverse_direction.joblib")
    calibration = df[df["event_year"].between(CAL_START, CAL_END)].dropna(subset=[cbi.STAGE1_TARGET]).copy()

    print("[PREDICT] calibration", flush=True)
    cal_pred = cbi.add_predictions(calibration, stage1, stage2, features)
    print("[PREDICT] oos", flush=True)
    oos_pred = cbi.add_predictions(oos_df, stage1, stage2, features)
    cal_base = base_orders(cal_pred)
    oos_base = base_orders(oos_pred)
    stats = cbi.build_event_stats(cal_base)
    stats.to_csv(OUT_DIR / "catboost_block_inverse_v2_event_stats.csv", index=False)

    best = None
    grid_rows = []
    for i, cfg in enumerate(STRICT_GRID, start=1):
        event_map = make_strict_event_map(stats, cfg)
        cal_bi = cbi.apply_event_map(cal_pred, event_map, cfg)
        base_pair = summarize_pair(cal_base, "v2", "BASE_CATBOOST")
        bi_pair = summarize_pair(cal_bi, "v2", "BLOCK_INVERSE")
        pair_sel = select_pairs(base_pair, bi_pair)
        cal_final = apply_pair_selection(cal_base, cal_bi, pair_sel)
        row = summary_all(cal_final, "calibration")
        row.update(cfg)
        row["score"] = row["net_pips"] + min(row["profit_factor"], 3.0) * 1000 + row["positive_pairs"] * 400
        row["allow_count"] = int(event_map["action"].eq("ALLOW").sum())
        row["inverse_count"] = int(event_map["action"].eq("INVERSE").sum())
        row["block_count"] = int(event_map["action"].eq("BLOCK").sum())
        grid_rows.append(row)
        if best is None or row["score"] > best["row"]["score"]:
            best = {"row": row, "event_map": event_map, "pair_selection": pair_sel}
        if i % 100 == 0:
            print(f"[GRID] {i}/{len(STRICT_GRID)} best_net={best['row']['net_pips']:.1f} best_pf={best['row']['profit_factor']:.4f}", flush=True)

    grid = pd.DataFrame(grid_rows).sort_values("score", ascending=False)
    grid.to_csv(OUT_DIR / "catboost_block_inverse_v2_grid_results.csv", index=False)
    best_map = best["event_map"]
    pair_sel = best["pair_selection"]
    best_map.to_csv(OUT_DIR / "catboost_block_inverse_v2_event_map.csv", index=False)
    pair_sel.to_csv(OUT_DIR / "catboost_block_inverse_v2_pair_selection.csv", index=False)

    cal_bi = cbi.apply_event_map(cal_pred, best_map, best["row"])
    oos_bi = cbi.apply_event_map(oos_pred, best_map, best["row"])
    cal_final = apply_pair_selection(cal_base, cal_bi, pair_sel)
    oos_final = apply_pair_selection(oos_base, oos_bi, pair_sel)
    cal_final.to_csv(OUT_DIR / "catboost_block_inverse_v2_calibration_orders.csv", index=False)
    oos_final.to_csv(OUT_DIR / "catboost_block_inverse_v2_oos_orders.csv", index=False)
    signal_table = write_signal_table(oos_final)

    cal_summary = summary_all(cal_final, "calibration")
    oos_summary = summary_all(oos_final, "oos_2024_2026")
    pd.DataFrame([cal_summary, oos_summary]).to_csv(OUT_DIR / "catboost_block_inverse_v2_summary.csv", index=False)
    oos_pair = summarize_pair(oos_final, "v2", "PAIR_SELECTED").sort_values("net_pips", ascending=False)
    oos_pair.to_csv(OUT_DIR / "catboost_block_inverse_v2_oos_summary_by_pair.csv", index=False)

    report = "\n".join([
        "# CatBoost Block/Inverse V2 Pair Gate",
        "",
        "V2 applies stricter inverse rules, rare event blocking, and pair-specific method selection.",
        "",
        "## Summary",
        "",
        md_table(pd.DataFrame([cal_summary, oos_summary]), ".4f"),
        "",
        "## Pair Selection",
        "",
        md_table(pair_sel, ".4f"),
        "",
        "## OOS Per Pair",
        "",
        md_table(oos_pair, ".4f"),
        "",
        "## MT5",
        "",
        f"- Signal table: `{OUT_DIR / 'catboost_block_inverse_v2_mt5_signal_table.csv'}`",
        f"- EA model name: `{MODEL_NAME}`",
    ]) + "\n"
    (OUT_DIR / "LAPORAN_CATBOOST_BLOCK_INVERSE_V2.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_CATBOOST_BLOCK_INVERSE_V2.md'}", flush=True)
    print("[PROGRESS] DONE 52_catboost_block_inverse_v2_pair_gate.py", flush=True)


if __name__ == "__main__":
    main()
