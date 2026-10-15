from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
STATS_IN = MANUAL_DIR / "manual_block_inverse_event_stats_2010_2022_literal.csv"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2022_reset.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_2010_2022_winrate65.csv"
SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_winrate65_summary_2010_2022.csv"
ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_winrate65_orders_2010_2022.csv"
GRID_OUT = MANUAL_DIR / "manual_block_inverse_winrate65_grid_results.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_WINRATE65_2010_2022.md"

TARGET_WIN_RATE = 0.65
MIN_TOTAL_TRADES = 100


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    gp = float(values[values > 0].sum())
    gl = float(-values[values < 0].sum())
    if gl == 0:
        return np.inf if gp > 0 else 0.0
    return gp / gl


def markdown_table(df):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def evaluate(action_by_stat, stat_idx, pnl, inverse_pnl, pair_codes):
    valid = stat_idx >= 0
    action = np.zeros(len(stat_idx), dtype=np.int8)
    action[valid] = action_by_stat[stat_idx[valid]]
    active = action != 0
    if not active.any():
        return None
    vals = np.where(action[active] == -1, inverse_pnl[active], pnl[active])
    trades = int(vals.size)
    if trades < MIN_TOTAL_TRADES:
        return None
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals < 0].sum())
    pf = gp / gl if gl else (np.inf if gp > 0 else 0.0)
    by_pair = pd.DataFrame({"pair_code": pair_codes[active], "pnl": vals}).groupby("pair_code")["pnl"].sum()
    return {
        "trades": trades,
        "win_rate": float((vals > 0).mean()),
        "net_pips": float(vals.sum()),
        "profit_factor": pf,
        "positive_pairs": int((by_pair > 0).sum()),
        "active_pairs": int(len(by_pair)),
    }


def score(row):
    if row["win_rate"] < TARGET_WIN_RATE:
        return -1e18
    return row["net_pips"] + row["trades"] * 0.05 + row["positive_pairs"] * 100.0 + row["profit_factor"] * 50.0


def summarize_orders(orders):
    rows = []
    for pair, g in orders.groupby("pair", sort=False):
        rows.append({
            "pair": pair,
            "trades": int(len(g)),
            "win_rate": float((g["pnl_pips_final"] > 0).mean()) if len(g) else 0.0,
            "net_pips": float(g["pnl_pips_final"].sum()) if len(g) else 0.0,
            "profit_factor": profit_factor(g["pnl_pips_final"]) if len(g) else 0.0,
            "allow_trades": int(g["action"].eq("ALLOW").sum()),
            "inverse_trades": int(g["action"].eq("INVERSE").sum()),
        })
    return pd.DataFrame(rows).sort_values("net_pips", ascending=False)


def main():
    print("[PROGRESS] START 19b_optimize_manual_block_inverse_winrate65_fast.py", flush=True)
    stats = pd.read_csv(STATS_IN)
    trades = pd.read_csv(TRADES_IN, low_memory=False)
    numeric_cols = [
        "trade_count", "active_years", "dominant_year_share", "net_pips",
        "profit_factor", "hit_rate", "inverse_net_pips", "inverse_profit_factor",
        "inverse_hit_rate", "inverse_active_years", "inverse_dominant_year_share",
        "positive_years", "inverse_positive_years",
    ]
    for col in numeric_cols:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
    for col in ["pnl_pips", "inverse_pnl_pips"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0)

    stats = stats.reset_index(drop=True)
    key_to_idx = {(r.pair, r.country, r.event_name_norm): i for i, r in enumerate(stats.itertuples(index=False))}
    stat_idx = np.array(
        [key_to_idx.get((r.pair, r.CountryCode, r.event_name_norm), -1) for r in trades.itertuples(index=False)],
        dtype=np.int32,
    )
    pair_cat = trades["pair"].astype("category")
    pair_codes = pair_cat.cat.codes.to_numpy(dtype=np.int16)
    pnl = trades["pnl_pips"].to_numpy(dtype=float)
    inverse_pnl = trades["inverse_pnl_pips"].to_numpy(dtype=float)

    print(f"[DATA] stats={len(stats)} trades={len(trades)}", flush=True)

    s = stats
    pos_year_ratio = s["positive_years"].to_numpy(dtype=float) / np.maximum(1.0, s["active_years"].to_numpy(dtype=float))
    inv_pos_year_ratio = s["inverse_positive_years"].to_numpy(dtype=float) / np.maximum(1.0, s["inverse_active_years"].to_numpy(dtype=float))
    grid_rows = []
    best = None
    total = 0
    for min_events in [3, 5, 10, 20, 30, 50]:
        for min_active_years in [1, 2, 3, 4, 6]:
            for max_dom in [0.75, 0.90, 1.00]:
                base = (
                    (s["trade_count"].to_numpy(dtype=float) >= min_events)
                    & (s["active_years"].to_numpy(dtype=float) >= min_active_years)
                    & (s["dominant_year_share"].to_numpy(dtype=float) <= max_dom)
                )
                for min_hit in [0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
                    for min_pf in [1.20, 1.50, 2.00, 3.00]:
                        for min_posyr in [0.00, 0.40, 0.50, 0.60]:
                            total += 1
                            if total % 500 == 0:
                                print(f"[GRID] {total}/8640", flush=True)
                            allow = (
                                base
                                & (s["net_pips"].to_numpy(dtype=float) > 0)
                                & (s["profit_factor"].to_numpy(dtype=float) >= min_pf)
                                & (s["hit_rate"].to_numpy(dtype=float) >= min_hit)
                                & (pos_year_ratio >= min_posyr)
                            )
                            inverse = (
                                base
                                & (s["net_pips"].to_numpy(dtype=float) < 0)
                                & (s["inverse_net_pips"].to_numpy(dtype=float) > 0)
                                & (s["inverse_profit_factor"].to_numpy(dtype=float) >= min_pf)
                                & (s["inverse_hit_rate"].to_numpy(dtype=float) >= min_hit)
                                & (s["inverse_active_years"].to_numpy(dtype=float) >= min_active_years)
                                & (s["inverse_dominant_year_share"].to_numpy(dtype=float) <= max_dom)
                                & (inv_pos_year_ratio >= min_posyr)
                            )
                            actions = np.zeros(len(stats), dtype=np.int8)
                            actions[allow] = 1
                            actions[inverse] = -1
                            metrics = evaluate(actions, stat_idx, pnl, inverse_pnl, pair_codes)
                            if metrics is None:
                                continue
                            row = {
                                "min_events": min_events,
                                "min_active_years": min_active_years,
                                "max_dominant_year_share": max_dom,
                                "min_hit_rate": min_hit,
                                "min_profit_factor": min_pf,
                                "min_positive_year_ratio": min_posyr,
                                "allow_events": int(allow.sum()),
                                "inverse_events": int(inverse.sum()),
                                "block_events": int((actions == 0).sum()),
                                **metrics,
                            }
                            row["score"] = score(row)
                            grid_rows.append(row)
                            if best is None or row["score"] > best["row"]["score"]:
                                best = {"row": row, "actions": actions.copy()}

    grid = pd.DataFrame(grid_rows).sort_values(["score", "win_rate", "net_pips"], ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    if best is None or best["row"]["win_rate"] < TARGET_WIN_RATE:
        print("[RESULT] no_candidate_reached_target", flush=True)
        print(grid.head(20).to_string(index=False), flush=True)
        return

    action_names = np.full(len(stats), "BLOCK", dtype=object)
    reason_names = np.full(len(stats), "winrate65_blocked_by_threshold", dtype=object)
    action_names[best["actions"] == 1] = "ALLOW"
    reason_names[best["actions"] == 1] = "winrate65_allow_high_hit_rate"
    action_names[best["actions"] == -1] = "INVERSE"
    reason_names[best["actions"] == -1] = "winrate65_inverse_high_hit_rate"

    model = stats.copy()
    model["action"] = action_names
    model["reason"] = reason_names
    model["selection_period"] = "optimized_in_sample_2010_2022_winrate65"
    for key in [
        "min_events", "min_active_years", "max_dominant_year_share",
        "min_hit_rate", "min_profit_factor", "min_positive_year_ratio",
    ]:
        model[key] = best["row"][key]
    model.to_csv(MODEL_OUT, index=False)

    action_by_trade = np.zeros(len(trades), dtype=np.int8)
    valid = stat_idx >= 0
    action_by_trade[valid] = best["actions"][stat_idx[valid]]
    active = action_by_trade != 0
    orders = trades.loc[active].copy()
    orders["action"] = np.where(action_by_trade[active] == -1, "INVERSE", "ALLOW")
    orders["signal"] = np.where(
        orders["action"].eq("INVERSE"),
        np.where(orders["initial_signal"].eq("BUY"), "SELL", "BUY"),
        orders["initial_signal"],
    )
    orders["pnl_pips_final"] = np.where(orders["action"].eq("INVERSE"), orders["inverse_pnl_pips"], orders["pnl_pips"])
    orders.to_csv(ORDERS_OUT, index=False)
    summary = summarize_orders(orders)
    summary.to_csv(SUMMARY_OUT, index=False)

    total_metrics = {
        "trades": int(len(orders)),
        "win_rate": float((orders["pnl_pips_final"] > 0).mean()) if len(orders) else 0.0,
        "net_pips": float(orders["pnl_pips_final"].sum()) if len(orders) else 0.0,
        "profit_factor": profit_factor(orders["pnl_pips_final"]) if len(orders) else 0.0,
        "positive_pairs": int((summary["net_pips"] > 0).sum()) if not summary.empty else 0,
        "active_pairs": int(len(summary)),
    }
    report = [
        "# Laporan Manual Block/Inverse Optimized Winrate 2010-2022",
        "",
        "Model ini dioptimasi secara in-sample pada 2010-2022 untuk mencapai winrate total di atas 65%.",
        "Ini menjawab eksperimen optimasi manual, tetapi bukan bukti out-of-sample.",
        "",
        "## Konfigurasi Terpilih",
        markdown_table(pd.DataFrame([{k: best["row"][k] for k in [
            "min_events", "min_active_years", "max_dominant_year_share",
            "min_hit_rate", "min_profit_factor", "min_positive_year_ratio",
        ]}])),
        "",
        "## Ringkasan Total 2010-2022",
        f"- Trades: {total_metrics['trades']}",
        f"- Winrate total: {total_metrics['win_rate']:.2%}",
        f"- Net pips: {total_metrics['net_pips']:.2f}",
        f"- Profit factor: {total_metrics['profit_factor']:.4f}",
        f"- Positive pairs: {total_metrics['positive_pairs']}/{total_metrics['active_pairs']}",
        f"- Event actions: {model['action'].value_counts().to_dict()}",
        "",
        "## Ringkasan Per Pair",
        markdown_table(summary),
        "",
        "## Output",
        f"- `{MODEL_OUT}`",
        f"- `{ORDERS_OUT}`",
        f"- `{SUMMARY_OUT}`",
        f"- `{GRID_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {ORDERS_OUT}", flush=True)
    print(f"[OUTPUT] {SUMMARY_OUT}", flush=True)
    print(f"[OUTPUT] {GRID_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)
    print(
        f"[RESULT] trades={total_metrics['trades']} win_rate={total_metrics['win_rate']:.4f} "
        f"net_pips={total_metrics['net_pips']:.2f} pf={total_metrics['profit_factor']:.4f} "
        f"positive_pairs={total_metrics['positive_pairs']}/{total_metrics['active_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
