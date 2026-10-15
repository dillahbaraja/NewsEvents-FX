from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_country_event_2010_2023_profit_loss.csv"
EXPANDED_MODEL_OUT = MANUAL_DIR / "model_block_inverse_country_event_2010_2023_profit_loss_expanded_for_tester.csv"
STATS_OUT = MANUAL_DIR / "manual_block_inverse_country_event_stats_2010_2023.csv"
ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_country_event_orders_2010_2023.csv"
SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_country_event_summary_2010_2023.csv"
GRID_OUT = MANUAL_DIR / "manual_block_inverse_country_event_grid_results.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_COUNTRY_EVENT_PROFIT_LOSS_2010_2023.md"

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


def yearly_stats(group, col):
    yearly = group.groupby("event_year", as_index=False).agg(
        yearly_pips=(col, "sum"),
        yearly_trades=(col, "size"),
    )
    active_years = int((yearly["yearly_trades"] > 0).sum())
    positive_years = int((yearly["yearly_pips"] > 0).sum())
    abs_total = float(yearly["yearly_pips"].abs().sum())
    dominant_share = float(yearly["yearly_pips"].abs().max() / abs_total) if abs_total > 0 else 1.0
    return active_years, positive_years, dominant_share


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


def build_country_event_stats(trades):
    rows = []
    for (country, event_norm), g in trades.groupby(["CountryCode", "event_name_norm"], sort=False):
        original = g["pnl_pips"].to_numpy(dtype=float)
        inverse = g["inverse_pnl_pips"].to_numpy(dtype=float)
        active_years, positive_years, dominant_share = yearly_stats(g, "pnl_pips")
        inv_active_years, inv_positive_years, inv_dominant_share = yearly_stats(g, "inverse_pnl_pips")
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        rows.append({
            "country": country,
            "event_name": event_name,
            "event_name_norm": event_norm,
            "trade_count": int(len(g)),
            "active_years": active_years,
            "positive_years_original": positive_years,
            "positive_year_ratio_original": positive_years / max(1, active_years),
            "dominant_year_share_original": dominant_share,
            "gross_profit_original": float(original[original > 0].sum()),
            "gross_loss_original": float(-original[original < 0].sum()),
            "net_pips_original": float(original.sum()),
            "profit_factor_original": profit_factor(original),
            "avg_pips_original": float(original.mean()) if len(original) else 0.0,
            "inverse_active_years": inv_active_years,
            "positive_years_inverse": inv_positive_years,
            "positive_year_ratio_inverse": inv_positive_years / max(1, inv_active_years),
            "dominant_year_share_inverse": inv_dominant_share,
            "gross_profit_inverse": float(inverse[inverse > 0].sum()),
            "gross_loss_inverse": float(-inverse[inverse < 0].sum()),
            "net_pips_inverse": float(inverse.sum()),
            "profit_factor_inverse": profit_factor(inverse),
            "avg_pips_inverse": float(inverse.mean()) if len(inverse) else 0.0,
            "pairs_covered": int(g["pair"].nunique()),
        })
    return pd.DataFrame(rows)


def main():
    print("[PROGRESS] START 22_optimize_manual_block_inverse_country_event_profit_loss_2010_2023.py", flush=True)
    if not TRADES_IN.exists():
        raise FileNotFoundError(f"Missing trades input: {TRADES_IN}")

    trades = pd.read_csv(TRADES_IN, low_memory=False)
    for col in ["pnl_pips", "inverse_pnl_pips"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0)
    trades["event_year"] = pd.to_numeric(trades["event_year"], errors="coerce")
    trades["event_time"] = pd.to_datetime(trades["event_time"], errors="coerce")
    trades = trades.dropna(subset=["event_year", "CountryCode", "event_name_norm", "pair"]).copy()
    trades = trades.sort_values(["event_time", "pair"]).reset_index(drop=True)
    print(f"[DATA] train_trades={len(trades)}", flush=True)

    stats = build_country_event_stats(trades)
    stats.to_csv(STATS_OUT, index=False)
    print(f"[OUTPUT] {STATS_OUT}", flush=True)
    print(f"[DATA] country_event_units={len(stats)}", flush=True)

    key_to_idx = {(r.country, r.event_name_norm): i for i, r in enumerate(stats.itertuples(index=False))}
    unit_idx = np.array(
        [key_to_idx.get((r.CountryCode, r.event_name_norm), -1) for r in trades.itertuples(index=False)],
        dtype=np.int32,
    )

    min_trade_count_grid = [20, 50, 100, 200]
    min_pf_grid = [1.05, 1.10, 1.20, 1.30, 1.50]
    min_avg_grid = [0.05, 0.10, 0.25, 0.50]
    min_posyr_grid = [0.40, 0.50, 0.60]
    max_dom_grid = [0.50, 0.60, 0.75]
    expected_total = len(min_trade_count_grid) * len(min_pf_grid) * len(min_avg_grid) * len(min_posyr_grid) * len(max_dom_grid)

    grid_rows = []
    best = None
    s = stats
    total = 0
    for min_trade_count in min_trade_count_grid:
        for min_pf in min_pf_grid:
            for min_avg in min_avg_grid:
                for min_posyr in min_posyr_grid:
                    for max_dom in max_dom_grid:
                        total += 1
                        if total % 50 == 0:
                            print(f"[GRID] {total}/{expected_total}", flush=True)
                        allow = (
                            (s["trade_count"].to_numpy(dtype=float) >= min_trade_count)
                            & (s["net_pips_original"].to_numpy(dtype=float) > 0)
                            & (s["profit_factor_original"].to_numpy(dtype=float) >= min_pf)
                            & (s["avg_pips_original"].to_numpy(dtype=float) >= min_avg)
                            & (s["positive_year_ratio_original"].to_numpy(dtype=float) >= min_posyr)
                            & (s["dominant_year_share_original"].to_numpy(dtype=float) <= max_dom)
                        )
                        inverse = (
                            (s["trade_count"].to_numpy(dtype=float) >= min_trade_count)
                            & (s["net_pips_inverse"].to_numpy(dtype=float) > 0)
                            & (s["profit_factor_inverse"].to_numpy(dtype=float) >= min_pf)
                            & (s["avg_pips_inverse"].to_numpy(dtype=float) >= min_avg)
                            & (s["positive_year_ratio_inverse"].to_numpy(dtype=float) >= min_posyr)
                            & (s["dominant_year_share_inverse"].to_numpy(dtype=float) <= max_dom)
                            & (s["net_pips_inverse"].to_numpy(dtype=float) > s["net_pips_original"].to_numpy(dtype=float))
                            & (s["profit_factor_inverse"].to_numpy(dtype=float) > s["profit_factor_original"].to_numpy(dtype=float))
                        )
                        both = allow & inverse
                        if both.any():
                            prefer_inverse = s["net_pips_inverse"].to_numpy(dtype=float) > s["net_pips_original"].to_numpy(dtype=float)
                            allow = allow & ~(both & prefer_inverse)
                            inverse = inverse & ~(both & ~prefer_inverse)

                        actions = np.zeros(len(stats), dtype=np.int8)
                        actions[allow] = 1
                        actions[inverse] = -1

                        valid = unit_idx >= 0
                        action_by_trade = np.zeros(len(trades), dtype=np.int8)
                        action_by_trade[valid] = actions[unit_idx[valid]]
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
    reason_names = np.full(len(stats), "country_event_blocked_by_profit_loss_threshold", dtype=object)
    action_names[best["actions"] == 1] = "ALLOW"
    reason_names[best["actions"] == 1] = "country_event_allow_profit_loss_stable"
    action_names[best["actions"] == -1] = "INVERSE"
    reason_names[best["actions"] == -1] = "country_event_inverse_profit_loss_better"

    model = stats.copy()
    model["action"] = action_names
    model["reason"] = reason_names
    model["selection_period"] = "optimized_country_event_2010_2023_profit_loss"
    for key in [
        "min_trade_count", "min_profit_factor", "min_avg_pips_per_trade",
        "min_positive_year_ratio", "max_dominant_year_share",
    ]:
        model[key] = best["row"][key]
    model.to_csv(MODEL_OUT, index=False)

    pair_country = trades[["pair", "CountryCode", "event_name", "event_name_norm"]].drop_duplicates()
    expanded = pair_country.merge(
        model[["country", "event_name_norm", "action", "reason", "selection_period"]],
        left_on=["CountryCode", "event_name_norm"],
        right_on=["country", "event_name_norm"],
        how="left",
    )
    expanded["action"] = expanded["action"].fillna("BLOCK")
    expanded["reason"] = expanded["reason"].fillna("country_event_missing_blocked")
    expanded["country"] = expanded["CountryCode"]
    expanded = expanded[["pair", "country", "event_name", "event_name_norm", "action", "reason", "selection_period"]]
    expanded.to_csv(EXPANDED_MODEL_OUT, index=False)

    action_by_trade = np.zeros(len(trades), dtype=np.int8)
    valid = unit_idx >= 0
    action_by_trade[valid] = best["actions"][unit_idx[valid]]
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

    metrics = {
        "trades": int(len(orders)),
        "win_rate": float((orders["pnl_pips_final"] > 0).mean()) if len(orders) else 0.0,
        "net_pips": float(orders["pnl_pips_final"].sum()) if len(orders) else 0.0,
        "avg_pips_per_trade": float(orders["pnl_pips_final"].mean()) if len(orders) else 0.0,
        "gross_profit": float(orders.loc[orders["pnl_pips_final"] > 0, "pnl_pips_final"].sum()) if len(orders) else 0.0,
        "gross_loss": float(-orders.loc[orders["pnl_pips_final"] < 0, "pnl_pips_final"].sum()) if len(orders) else 0.0,
        "profit_factor": profit_factor(orders["pnl_pips_final"]) if len(orders) else 0.0,
        "max_drawdown_pips": max_drawdown(orders["pnl_pips_final"]) if len(orders) else 0.0,
        "positive_pairs": int((summary["net_pips"] > 0).sum()) if not summary.empty else 0,
        "active_pairs": int(len(summary)),
    }
    report = [
        "# Laporan Manual Block/Inverse Country + Event Profit-Loss 2010-2023",
        "",
        "Model ini memakai satu action konsisten untuk setiap CountryCode + event_name_norm.",
        "Objective utama adalah profit total dan profit/loss ratio, bukan winrate.",
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
        f"- `{EXPANDED_MODEL_OUT}`",
        f"- `{STATS_OUT}`",
        f"- `{ORDERS_OUT}`",
        f"- `{SUMMARY_OUT}`",
        f"- `{GRID_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {EXPANDED_MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {STATS_OUT}", flush=True)
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
