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
    s = pd.Series(values).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    if gl == 0:
        return np.inf if gp > 0 else 0.0
    return gp / gl


def score_candidate(row):
    if row["win_rate"] < TARGET_WIN_RATE:
        return -1e18
    if row["trades"] < MIN_TOTAL_TRADES:
        return -1e18
    # Prefer a model that clears the winrate target and still keeps enough trades and profit.
    return (
        row["net_pips"] * 1.0
        + row["trades"] * 0.05
        + row["positive_pairs"] * 100.0
        + row["profit_factor"] * 50.0
    )


def build_candidate_model(stats, cfg):
    s = stats
    pos_year_ratio = s["positive_years"] / np.maximum(1, s["active_years"])
    inv_pos_year_ratio = s["inverse_positive_years"] / np.maximum(1, s["inverse_active_years"])
    base = (
        (s["trade_count"] >= cfg["min_events"])
        & (s["active_years"] >= cfg["min_active_years"])
        & (s["dominant_year_share"] <= cfg["max_dominant_year_share"])
    )
    allow = (
        base
        & (s["net_pips"] > 0)
        & (s["profit_factor"] >= cfg["min_profit_factor"])
        & (s["hit_rate"] >= cfg["min_hit_rate"])
        & (pos_year_ratio >= cfg["min_positive_year_ratio"])
    )
    inverse = (
        base
        & (s["net_pips"] < 0)
        & (s["inverse_net_pips"] > 0)
        & (s["inverse_profit_factor"] >= cfg["min_profit_factor"])
        & (s["inverse_hit_rate"] >= cfg["min_hit_rate"])
        & (s["inverse_active_years"] >= cfg["min_active_years"])
        & (s["inverse_dominant_year_share"] <= cfg["max_dominant_year_share"])
        & (inv_pos_year_ratio >= cfg["min_positive_year_ratio"])
    )
    actions = np.full(len(s), "BLOCK", dtype=object)
    reasons = np.full(len(s), "winrate65_blocked_by_threshold", dtype=object)
    actions[allow.to_numpy()] = "ALLOW"
    reasons[allow.to_numpy()] = "winrate65_allow_high_hit_rate"
    actions[inverse.to_numpy()] = "INVERSE"
    reasons[inverse.to_numpy()] = "winrate65_inverse_high_hit_rate"
    return actions, reasons


def evaluate_actions(trades, stats, stat_index_by_key, actions):
    action_by_trade = np.full(len(trades), "BLOCK", dtype=object)
    valid = trades["stat_idx"].to_numpy() >= 0
    idx = trades.loc[valid, "stat_idx"].to_numpy(dtype=np.int32)
    action_by_trade[valid] = actions[idx]
    active = (action_by_trade == "ALLOW") | (action_by_trade == "INVERSE")
    if not active.any():
        return {
            "trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "profit_factor": 0.0,
            "positive_pairs": 0,
            "active_pairs": 0,
        }
    pnl = np.where(
        action_by_trade[active] == "INVERSE",
        trades.loc[active, "inverse_pnl_pips"].to_numpy(dtype=float),
        trades.loc[active, "pnl_pips"].to_numpy(dtype=float),
    )
    pairs = trades.loc[active, "pair"].to_numpy()
    by_pair = pd.DataFrame({"pair": pairs, "pnl": pnl}).groupby("pair")["pnl"].sum()
    return {
        "trades": int(active.sum()),
        "win_rate": float((pnl > 0).mean()),
        "net_pips": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "positive_pairs": int((by_pair > 0).sum()),
        "active_pairs": int(len(by_pair)),
    }


def materialize_orders(trades, actions):
    action_by_trade = np.full(len(trades), "BLOCK", dtype=object)
    valid = trades["stat_idx"].to_numpy() >= 0
    idx = trades.loc[valid, "stat_idx"].to_numpy(dtype=np.int32)
    action_by_trade[valid] = actions[idx]
    active = (action_by_trade == "ALLOW") | (action_by_trade == "INVERSE")
    orders = trades.loc[active].copy()
    orders["action"] = action_by_trade[active]
    orders["signal"] = np.where(
        orders["action"].eq("INVERSE"),
        np.where(orders["initial_signal"].eq("BUY"), "SELL", "BUY"),
        orders["initial_signal"],
    )
    orders["pnl_pips_final"] = np.where(
        orders["action"].eq("INVERSE"),
        orders["inverse_pnl_pips"],
        orders["pnl_pips"],
    )
    return orders


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


def main():
    print("[PROGRESS] START 19_optimize_manual_block_inverse_winrate_2010_2022.py", flush=True)
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
    key_to_idx = {
        (row.pair, row.country, row.event_name_norm): i
        for i, row in enumerate(stats.itertuples(index=False))
    }
    trades["stat_idx"] = [
        key_to_idx.get((row.pair, row.CountryCode, row.event_name_norm), -1)
        for row in trades.itertuples(index=False)
    ]
    print(f"[DATA] stats={len(stats)} trades={len(trades)}", flush=True)

    configs = []
    for min_events in [3, 5, 10, 20, 30, 50]:
        for min_active_years in [1, 2, 3, 4, 6]:
            for max_dom in [0.75, 0.90, 1.00]:
                for min_hit in [0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
                    for min_pf in [1.20, 1.50, 2.00, 3.00]:
                        for min_posyr in [0.00, 0.40, 0.50, 0.60]:
                            configs.append({
                                "min_events": min_events,
                                "min_active_years": min_active_years,
                                "max_dominant_year_share": max_dom,
                                "min_hit_rate": min_hit,
                                "min_profit_factor": min_pf,
                                "min_positive_year_ratio": min_posyr,
                            })

    grid_rows = []
    best = None
    for i, cfg in enumerate(configs, start=1):
        if i % 500 == 0:
            print(f"[GRID] {i}/{len(configs)}", flush=True)
        actions, reasons = build_candidate_model(stats, cfg)
        metrics = evaluate_actions(trades, stats, key_to_idx, actions)
        row = {**cfg, **metrics}
        row["allow_events"] = int((actions == "ALLOW").sum())
        row["inverse_events"] = int((actions == "INVERSE").sum())
        row["block_events"] = int((actions == "BLOCK").sum())
        row["score"] = score_candidate(row)
        grid_rows.append(row)
        if best is None or row["score"] > best["row"]["score"]:
            best = {"row": row, "actions": actions, "reasons": reasons, "cfg": cfg}

    grid = pd.DataFrame(grid_rows).sort_values(["score", "win_rate", "net_pips"], ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    if best is None or best["row"]["win_rate"] < TARGET_WIN_RATE:
        print("[RESULT] no_candidate_reached_target", flush=True)
        print(grid.head(20).to_string(index=False), flush=True)
        return

    model = stats.copy()
    model["action"] = best["actions"]
    model["reason"] = best["reasons"]
    model["selection_period"] = "optimized_in_sample_2010_2022_winrate65"
    for key, value in best["cfg"].items():
        model[key] = value
    model.to_csv(MODEL_OUT, index=False)

    orders = materialize_orders(trades, best["actions"])
    orders.to_csv(ORDERS_OUT, index=False)
    summary = summarize_orders(orders)
    summary.to_csv(SUMMARY_OUT, index=False)

    total = {
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
        "Model ini sengaja dioptimasi in-sample pada periode 2010-2022 untuk memenuhi target winrate total di atas 65%.",
        "Karena target dipilih dan dievaluasi pada periode yang sama, hasil ini harus dibedakan dari out-of-sample test.",
        "",
        "## Konfigurasi Terpilih",
        markdown_table(pd.DataFrame([best["cfg"]])),
        "",
        "## Ringkasan Total 2010-2022",
        f"- Trades: {total['trades']}",
        f"- Winrate total: {total['win_rate']:.2%}",
        f"- Net pips: {total['net_pips']:.2f}",
        f"- Profit factor: {total['profit_factor']:.4f}",
        f"- Positive pairs: {total['positive_pairs']}/{total['active_pairs']}",
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
        f"[RESULT] trades={total['trades']} win_rate={total['win_rate']:.4f} "
        f"net_pips={total['net_pips']:.2f} pf={total['profit_factor']:.4f} "
        f"positive_pairs={total['positive_pairs']}/{total['active_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
