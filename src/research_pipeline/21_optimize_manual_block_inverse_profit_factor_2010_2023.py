from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"
STATS_IN = MANUAL_DIR / "manual_block_inverse_event_stats.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_2010_2023_profit_factor.csv"
ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_profit_factor_orders_2010_2023.csv"
SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_profit_factor_summary_2010_2023.csv"
GRID_OUT = MANUAL_DIR / "manual_block_inverse_profit_factor_grid_results.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_PROFIT_FACTOR_2010_2023.md"

MIN_TOTAL_TRADES = 500
MIN_ACTIVE_PAIRS = 5


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


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


def summarize_orders(orders):
    rows = []
    for pair, g in orders.groupby("pair", sort=False):
        pnl = g["pnl_pips_final"].to_numpy(dtype=float)
        rows.append({
            "pair": pair,
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()) if len(g) else 0.0,
            "net_pips": float(pnl.sum()) if len(g) else 0.0,
            "avg_pips_per_trade": float(pnl.mean()) if len(g) else 0.0,
            "gross_profit": float(pnl[pnl > 0].sum()) if len(g) else 0.0,
            "gross_loss": float(-pnl[pnl < 0].sum()) if len(g) else 0.0,
            "profit_factor": profit_factor(pnl) if len(g) else 0.0,
            "max_drawdown_pips": max_drawdown(pnl),
            "allow_trades": int(g["action"].eq("ALLOW").sum()),
            "inverse_trades": int(g["action"].eq("INVERSE").sum()),
        })
    return pd.DataFrame(rows).sort_values("net_pips", ascending=False)


def total_metrics(orders):
    if orders.empty:
        return {
            "trades": 0, "win_rate": 0.0, "net_pips": 0.0, "avg_pips_per_trade": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            "max_drawdown_pips": 0.0, "positive_pairs": 0, "active_pairs": 0,
        }
    pnl = orders["pnl_pips_final"].to_numpy(dtype=float)
    by_pair = orders.groupby("pair")["pnl_pips_final"].sum()
    return {
        "trades": int(len(orders)),
        "win_rate": float((pnl > 0).mean()),
        "net_pips": float(pnl.sum()),
        "avg_pips_per_trade": float(pnl.mean()),
        "gross_profit": float(pnl[pnl > 0].sum()),
        "gross_loss": float(-pnl[pnl < 0].sum()),
        "profit_factor": profit_factor(pnl),
        "max_drawdown_pips": max_drawdown(pnl),
        "positive_pairs": int((by_pair > 0).sum()),
        "active_pairs": int(len(by_pair)),
    }


def score(row):
    if row["trades"] < MIN_TOTAL_TRADES or row["active_pairs"] < MIN_ACTIVE_PAIRS:
        return -1e18
    drawdown_penalty = min(row["max_drawdown_pips"], row["gross_profit"])
    collapse_penalty = 1000.0 * max(0, MIN_ACTIVE_PAIRS - row["positive_pairs"])
    return (
        row["net_pips"]
        + 1000.0 * row["profit_factor"]
        + 200.0 * row["positive_pairs"]
        + 10.0 * row["avg_pips_per_trade"]
        - 0.25 * drawdown_penalty
        - collapse_penalty
    )


def main():
    print("[PROGRESS] START 21_optimize_manual_block_inverse_profit_factor_2010_2023.py", flush=True)
    if not TRADES_IN.exists():
        raise FileNotFoundError(f"Missing trades input: {TRADES_IN}")
    if not STATS_IN.exists():
        raise FileNotFoundError(f"Missing stats input: {STATS_IN}")

    stats = pd.read_csv(STATS_IN)
    trades = pd.read_csv(TRADES_IN, low_memory=False)
    numeric_stats = [
        "trade_count", "active_years", "positive_years", "dominant_year_share",
        "net_pips", "profit_factor", "hit_rate",
        "inverse_net_pips", "inverse_profit_factor", "inverse_hit_rate",
        "inverse_active_years", "inverse_positive_years", "inverse_dominant_year_share",
    ]
    for col in numeric_stats:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
    for col in ["pnl_pips", "inverse_pnl_pips"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0)
    trades["event_time"] = pd.to_datetime(trades["event_time"], errors="coerce")
    trades = trades.sort_values(["event_time", "pair"]).reset_index(drop=True)

    print(f"[DATA] stats={len(stats)} trades={len(trades)}", flush=True)

    stats = stats.reset_index(drop=True)
    key_to_idx = {(r.pair, r.country, r.event_name_norm): i for i, r in enumerate(stats.itertuples(index=False))}
    stat_idx = np.array(
        [key_to_idx.get((r.pair, r.CountryCode, r.event_name_norm), -1) for r in trades.itertuples(index=False)],
        dtype=np.int32,
    )
    s = stats
    pos_year_ratio = s["positive_years"].to_numpy(dtype=float) / np.maximum(1.0, s["active_years"].to_numpy(dtype=float))
    inv_pos_year_ratio = s["inverse_positive_years"].to_numpy(dtype=float) / np.maximum(1.0, s["inverse_active_years"].to_numpy(dtype=float))

    min_trade_count_grid = [10, 20, 30, 50]
    min_pf_grid = [1.10, 1.20, 1.30, 1.50, 2.00]
    min_avg_grid = [0.10, 0.25, 0.50, 1.00]
    min_posyr_grid = [0.40, 0.50, 0.60]
    max_dom_grid = [0.50, 0.60, 0.75]
    expected_total = len(min_trade_count_grid) * len(min_pf_grid) * len(min_avg_grid) * len(min_posyr_grid) * len(max_dom_grid)

    grid_rows = []
    best = None
    total = 0
    for min_trade_count in min_trade_count_grid:
        for min_pf in min_pf_grid:
            for min_avg in min_avg_grid:
                for min_posyr in min_posyr_grid:
                    for max_dom in max_dom_grid:
                        total += 1
                        if total % 50 == 0:
                            print(f"[GRID] {total}/{expected_total}", flush=True)
                        allow_avg = s["net_pips"].to_numpy(dtype=float) / np.maximum(1.0, s["trade_count"].to_numpy(dtype=float))
                        inverse_avg = s["inverse_net_pips"].to_numpy(dtype=float) / np.maximum(1.0, s["trade_count"].to_numpy(dtype=float))
                        allow = (
                            (s["trade_count"].to_numpy(dtype=float) >= min_trade_count)
                            & (s["net_pips"].to_numpy(dtype=float) > 0)
                            & (s["profit_factor"].to_numpy(dtype=float) >= min_pf)
                            & (allow_avg >= min_avg)
                            & (pos_year_ratio >= min_posyr)
                            & (s["dominant_year_share"].to_numpy(dtype=float) <= max_dom)
                        )
                        inverse = (
                            (s["trade_count"].to_numpy(dtype=float) >= min_trade_count)
                            & (s["inverse_net_pips"].to_numpy(dtype=float) > 0)
                            & (s["inverse_profit_factor"].to_numpy(dtype=float) >= min_pf)
                            & (inverse_avg >= min_avg)
                            & (inv_pos_year_ratio >= min_posyr)
                            & (s["inverse_dominant_year_share"].to_numpy(dtype=float) <= max_dom)
                            & (s["inverse_net_pips"].to_numpy(dtype=float) > s["net_pips"].to_numpy(dtype=float))
                        )

                        both = allow & inverse
                        if both.any():
                            prefer_inverse = s["inverse_net_pips"].to_numpy(dtype=float) > s["net_pips"].to_numpy(dtype=float)
                            allow = allow & ~(both & prefer_inverse)
                            inverse = inverse & ~(both & ~prefer_inverse)

                        actions = np.zeros(len(stats), dtype=np.int8)
                        actions[allow] = 1
                        actions[inverse] = -1

                        valid = stat_idx >= 0
                        action_by_trade = np.zeros(len(trades), dtype=np.int8)
                        action_by_trade[valid] = actions[stat_idx[valid]]
                        active = action_by_trade != 0
                        if not active.any():
                            continue
                        pnl = np.where(
                            action_by_trade[active] == -1,
                            trades.loc[active, "inverse_pnl_pips"].to_numpy(dtype=float),
                            trades.loc[active, "pnl_pips"].to_numpy(dtype=float),
                        )
                        pairs = trades.loc[active, "pair"].to_numpy()
                        by_pair = pd.DataFrame({"pair": pairs, "pnl": pnl}).groupby("pair")["pnl"].sum()
                        row = {
                            "min_trade_count": min_trade_count,
                            "min_profit_factor": min_pf,
                            "min_avg_pips_per_trade": min_avg,
                            "min_positive_year_ratio": min_posyr,
                            "max_dominant_year_share": max_dom,
                            "allow_events": int(allow.sum()),
                            "inverse_events": int(inverse.sum()),
                            "block_events": int((actions == 0).sum()),
                            "trades": int(active.sum()),
                            "win_rate": float((pnl > 0).mean()),
                            "net_pips": float(pnl.sum()),
                            "avg_pips_per_trade": float(pnl.mean()),
                            "gross_profit": float(pnl[pnl > 0].sum()),
                            "gross_loss": float(-pnl[pnl < 0].sum()),
                            "profit_factor": profit_factor(pnl),
                            "max_drawdown_pips": max_drawdown(pnl),
                            "positive_pairs": int((by_pair > 0).sum()),
                            "active_pairs": int(len(by_pair)),
                        }
                        row["score"] = score(row)
                        grid_rows.append(row)
                        if best is None or row["score"] > best["row"]["score"]:
                            best = {"row": row, "actions": actions.copy()}

    grid = pd.DataFrame(grid_rows).sort_values(["score", "net_pips", "profit_factor"], ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    if best is None:
        raise RuntimeError("No candidate actions were generated.")

    action_names = np.full(len(stats), "BLOCK", dtype=object)
    reason_names = np.full(len(stats), "profit_factor_blocked_by_threshold", dtype=object)
    action_names[best["actions"] == 1] = "ALLOW"
    reason_names[best["actions"] == 1] = "profit_factor_allow_profitable_stable"
    action_names[best["actions"] == -1] = "INVERSE"
    reason_names[best["actions"] == -1] = "profit_factor_inverse_profitable_stable"

    model = stats.copy()
    model["action"] = action_names
    model["reason"] = reason_names
    model["selection_period"] = "optimized_in_sample_2010_2023_profit_factor"
    for key in [
        "min_trade_count", "min_profit_factor", "min_avg_pips_per_trade",
        "min_positive_year_ratio", "max_dominant_year_share",
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
    metrics = total_metrics(orders)
    report = [
        "# Laporan Manual Block/Inverse Berbasis Profit Factor 2010-2023",
        "",
        "Model ini dioptimasi secara in-sample pada periode 2010-2023 dengan objective profit/loss ratio, bukan winrate.",
        "",
        "## Konfigurasi Terpilih",
        markdown_table(pd.DataFrame([{k: best["row"][k] for k in [
            "min_trade_count", "min_profit_factor", "min_avg_pips_per_trade",
            "min_positive_year_ratio", "max_dominant_year_share",
        ]}])),
        "",
        "## Ringkasan Total",
        f"- Trades: {metrics['trades']}",
        f"- Winrate: {metrics['win_rate']:.2%}",
        f"- Net pips: {metrics['net_pips']:.2f}",
        f"- Avg pips per trade: {metrics['avg_pips_per_trade']:.4f}",
        f"- Gross profit: {metrics['gross_profit']:.2f}",
        f"- Gross loss: {metrics['gross_loss']:.2f}",
        f"- Profit factor: {metrics['profit_factor']:.4f}",
        f"- Max drawdown pips: {metrics['max_drawdown_pips']:.2f}",
        f"- Positive pairs: {metrics['positive_pairs']}/{metrics['active_pairs']}",
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
        f"[RESULT] trades={metrics['trades']} net_pips={metrics['net_pips']:.2f} "
        f"pf={metrics['profit_factor']:.4f} avg={metrics['avg_pips_per_trade']:.4f} "
        f"positive_pairs={metrics['positive_pairs']}/{metrics['active_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
