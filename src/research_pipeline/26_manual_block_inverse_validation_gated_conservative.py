from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_validation_gated_conservative_2010_2023.csv"
VALIDATION_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_validation_2018_2023.csv"
EVENT_VALIDATION_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_event_validation_2018_2023.csv"
EVENT_SCORES_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_event_scores.csv"
FINAL_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_orders_2010_2023.csv"
TEST_SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_test_summary_2024_2026.csv"
TEST_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_validation_gated_conservative_orders_2024_2026.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_VALIDATION_GATED_CONSERVATIVE.md"

VALIDATION_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

MIN_TRADE_COUNT = 25
MIN_ACTIVE_YEARS = 4
MAX_DOMINANT_YEAR_SHARE = 0.55

ALLOW_MIN_POSITIVE_YEAR_RATIO = 0.60
ALLOW_MIN_PROFIT_FACTOR = 1.08
ALLOW_MIN_AVG_PIPS = 0.15

INVERSE_MIN_POSITIVE_YEAR_RATIO = 0.65
INVERSE_MIN_PROFIT_FACTOR = 1.18
INVERSE_MIN_AVG_PIPS = 0.25
INVERSE_MIN_SCORE_RATIO = 1.20

MIN_VALIDATION_ACTIVE_YEARS = 3
MIN_VALIDATION_POSITIVE_YEAR_RATIO = 0.50
MIN_VALIDATION_NET_PIPS = 20.0
MIN_VALIDATION_AVG_PIPS = 0.05
MIN_VALIDATION_ACTION_SHARE = 0.60
TOP_EVENTS_PER_PAIR_COUNTRY = 35


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


def direction_metrics(group, col):
    values = group[col].to_numpy(dtype=float)
    yearly = group.groupby("event_year", as_index=False).agg(
        yearly_pips=(col, "sum"),
        yearly_trades=(col, "size"),
    )
    trade_count = int(values.size)
    active_years = int((yearly["yearly_trades"] > 0).sum())
    positive_years = int((yearly["yearly_pips"] > 0).sum())
    positive_year_ratio = positive_years / max(1, active_years)
    abs_total = float(yearly["yearly_pips"].abs().sum())
    dominant_year_share = float(yearly["yearly_pips"].abs().max() / abs_total) if abs_total > 0 else 1.0
    net_pips = float(values.sum())
    avg_pips = net_pips / max(1, trade_count)
    pf = profit_factor(values)
    stability_score = positive_year_ratio * (1.0 - dominant_year_share)
    score = net_pips * np.log1p(trade_count) * max(0.0, stability_score)
    return {
        "net_pips": net_pips,
        "gross_profit": float(values[values > 0].sum()),
        "gross_loss": float(-values[values < 0].sum()),
        "profit_factor": pf,
        "avg_pips": avg_pips,
        "trade_count": trade_count,
        "active_years": active_years,
        "positive_years": positive_years,
        "positive_year_ratio": positive_year_ratio,
        "dominant_year_share": dominant_year_share,
        "selection_score": score,
    }


def direction_ok(metrics, action):
    if not (
        metrics["trade_count"] >= MIN_TRADE_COUNT
        and metrics["active_years"] >= MIN_ACTIVE_YEARS
        and metrics["dominant_year_share"] <= MAX_DOMINANT_YEAR_SHARE
        and metrics["net_pips"] > 0
    ):
        return False
    if action == "ALLOW":
        return (
            metrics["positive_year_ratio"] >= ALLOW_MIN_POSITIVE_YEAR_RATIO
            and metrics["profit_factor"] > ALLOW_MIN_PROFIT_FACTOR
            and metrics["avg_pips"] >= ALLOW_MIN_AVG_PIPS
        )
    return (
        metrics["positive_year_ratio"] >= INVERSE_MIN_POSITIVE_YEAR_RATIO
        and metrics["profit_factor"] > INVERSE_MIN_PROFIT_FACTOR
        and metrics["avg_pips"] >= INVERSE_MIN_AVG_PIPS
    )


def choose_action(original, inverse):
    original_ok = direction_ok(original, "ALLOW")
    inverse_ok = direction_ok(inverse, "INVERSE")
    inverse_margin_ok = inverse["selection_score"] > max(0.0, original["selection_score"]) * INVERSE_MIN_SCORE_RATIO
    if original_ok and inverse_ok and inverse_margin_ok:
        return "INVERSE", "conservative_inverse_better_with_margin"
    if original_ok:
        return "ALLOW", "conservative_allow_stable"
    if inverse_ok and inverse_margin_ok:
        return "INVERSE", "conservative_inverse_only_with_margin"
    return "BLOCK", "conservative_unstable_or_weak"


def build_model(train, selection_period):
    rows = []
    for (pair, country, event_norm), g in train.groupby(["pair", "CountryCode", "event_name_norm"], sort=False):
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        original = direction_metrics(g, "pnl_pips")
        inverse = direction_metrics(g, "inverse_pnl_pips")
        action, reason = choose_action(original, inverse)
        selected = inverse if action == "INVERSE" else original
        rows.append({
            "pair": pair,
            "country": country,
            "event_name": event_name,
            "event_name_norm": event_norm,
            "action": action,
            "reason": reason,
            "selection_period": selection_period,
            "trade_count": original["trade_count"],
            "active_years": selected["active_years"] if action != "BLOCK" else max(original["active_years"], inverse["active_years"]),
            "net_pips": selected["net_pips"] if action != "BLOCK" else 0.0,
            "profit_factor": selected["profit_factor"] if action != "BLOCK" else 0.0,
            "avg_pips": selected["avg_pips"] if action != "BLOCK" else 0.0,
            "positive_year_ratio": selected["positive_year_ratio"] if action != "BLOCK" else 0.0,
            "dominant_year_share": selected["dominant_year_share"] if action != "BLOCK" else 1.0,
            "hit_rate": np.nan,
            "original_net_pips": original["net_pips"],
            "original_profit_factor": original["profit_factor"],
            "original_avg_pips": original["avg_pips"],
            "original_active_years": original["active_years"],
            "original_positive_year_ratio": original["positive_year_ratio"],
            "original_dominant_year_share": original["dominant_year_share"],
            "original_selection_score": original["selection_score"],
            "inverse_net_pips": inverse["net_pips"],
            "inverse_profit_factor": inverse["profit_factor"],
            "inverse_avg_pips": inverse["avg_pips"],
            "inverse_active_years": inverse["active_years"],
            "inverse_positive_year_ratio": inverse["positive_year_ratio"],
            "inverse_dominant_year_share": inverse["dominant_year_share"],
            "inverse_selection_score": inverse["selection_score"],
            "inverse_hit_rate": np.nan,
        })
    return pd.DataFrame(rows)


def apply_model(data, model):
    d = data.merge(
        model[["pair", "country", "event_name_norm", "action", "reason"]].rename(
            columns={"action": "model_action", "reason": "model_reason"}
        ),
        left_on=["pair", "CountryCode", "event_name_norm"],
        right_on=["pair", "country", "event_name_norm"],
        how="left",
    )
    d["action"] = d["model_action"].fillna("BLOCK_UNKNOWN")
    d["reason"] = d["model_reason"].fillna("unknown_event_blocked")
    d["final_signal"] = "NO_TRADE"
    allow = d["action"].eq("ALLOW")
    inverse = d["action"].eq("INVERSE")
    d.loc[allow, "final_signal"] = d.loc[allow, "initial_signal"]
    d.loc[inverse, "final_signal"] = np.where(d.loc[inverse, "initial_signal"].eq("BUY"), "SELL", "BUY")
    d["final_pnl_pips"] = np.where(
        inverse,
        d["inverse_pnl_pips"],
        np.where(allow, d["pnl_pips"], 0.0),
    )
    return d, d[d["final_signal"].isin(["BUY", "SELL"])].copy()


def summarize_orders(orders, label, year=None):
    if orders.empty:
        return pd.DataFrame([{
            "sample": label,
            "year": year,
            "pair": "ALL",
            "trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
            "allow_trades": 0,
            "inverse_trades": 0,
        }])
    rows = []
    for pair, g in orders.groupby("pair", sort=False):
        pnl = g["final_pnl_pips"].to_numpy(dtype=float)
        rows.append({
            "sample": label,
            "year": year,
            "pair": pair,
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()),
            "net_pips": float(pnl.sum()),
            "profit_factor": profit_factor(pnl),
            "max_drawdown_pips": max_drawdown(pnl),
            "allow_trades": int(g["action"].eq("ALLOW").sum()),
            "inverse_trades": int(g["action"].eq("INVERSE").sum()),
        })
    return pd.DataFrame(rows)


def summarize_event_validation(orders, year):
    if orders.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["pair", "CountryCode", "event_name_norm", "action"]
    for (pair, country, event_norm, action), g in orders.groupby(group_cols, sort=False):
        pnl = g["final_pnl_pips"].to_numpy(dtype=float)
        rows.append({
            "validation_year": year,
            "pair": pair,
            "country": country,
            "event_name_norm": event_norm,
            "action": action,
            "validation_trades": int(len(g)),
            "validation_net_pips": float(pnl.sum()),
            "validation_avg_pips": float(pnl.mean()) if len(pnl) else 0.0,
            "validation_profit_factor": profit_factor(pnl),
            "validation_win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        })
    return pd.DataFrame(rows)


def build_validation_gate(event_validation):
    if event_validation.empty:
        return pd.DataFrame(columns=[
            "pair", "country", "event_name_norm", "validation_action", "validation_active_years",
            "validation_positive_years", "validation_positive_year_ratio", "validation_net_pips",
            "validation_avg_pips", "validation_profit_factor", "validation_action_share",
            "validation_rank_score",
        ])
    agg = event_validation.groupby(["pair", "country", "event_name_norm", "action"], as_index=False).agg(
        validation_active_years=("validation_year", "nunique"),
        validation_positive_years=("validation_net_pips", lambda s: int((s > 0).sum())),
        validation_trades=("validation_trades", "sum"),
        validation_net_pips=("validation_net_pips", "sum"),
    )
    agg["validation_positive_year_ratio"] = agg["validation_positive_years"] / agg["validation_active_years"].clip(lower=1)
    agg["validation_avg_pips"] = agg["validation_net_pips"] / agg["validation_trades"].clip(lower=1)

    pf_rows = []
    for key, g in event_validation.groupby(["pair", "country", "event_name_norm", "action"], sort=False):
        expanded = []
        for _, row in g.iterrows():
            n = int(row["validation_trades"])
            if n <= 0:
                continue
            expanded.extend([float(row["validation_avg_pips"])] * n)
        pf_rows.append((*key, profit_factor(expanded)))
    pf = pd.DataFrame(pf_rows, columns=["pair", "country", "event_name_norm", "action", "validation_profit_factor"])
    agg = agg.merge(pf, on=["pair", "country", "event_name_norm", "action"], how="left")

    totals = event_validation.groupby(["pair", "country", "event_name_norm"], as_index=False).agg(
        total_validation_trades=("validation_trades", "sum")
    )
    agg = agg.merge(totals, on=["pair", "country", "event_name_norm"], how="left")
    agg["validation_action_share"] = agg["validation_trades"] / agg["total_validation_trades"].clip(lower=1)
    agg["validation_rank_score"] = (
        agg["validation_net_pips"]
        * agg["validation_positive_year_ratio"]
        * agg["validation_action_share"]
        * np.log1p(agg["validation_trades"])
    )
    gate = agg[
        (agg["validation_active_years"] >= MIN_VALIDATION_ACTIVE_YEARS)
        & (agg["validation_positive_year_ratio"] >= MIN_VALIDATION_POSITIVE_YEAR_RATIO)
        & (agg["validation_net_pips"] >= MIN_VALIDATION_NET_PIPS)
        & (agg["validation_avg_pips"] >= MIN_VALIDATION_AVG_PIPS)
        & (agg["validation_action_share"] >= MIN_VALIDATION_ACTION_SHARE)
    ].copy()
    gate = gate.sort_values("validation_rank_score", ascending=False)
    gate["rank_in_pair_country"] = gate.groupby(["pair", "country"])["validation_rank_score"].rank(method="first", ascending=False)
    gate = gate[gate["rank_in_pair_country"] <= TOP_EVENTS_PER_PAIR_COUNTRY].copy()
    return gate.rename(columns={"action": "validation_action"})


def apply_validation_gate(final_model, gate):
    gated = final_model.merge(
        gate,
        on=["pair", "country", "event_name_norm"],
        how="left",
    )
    keep = (
        gated["action"].isin(["ALLOW", "INVERSE"])
        & gated["validation_action"].eq(gated["action"])
    )
    gated.loc[~keep, "reason"] = np.where(
        gated.loc[~keep, "action"].isin(["ALLOW", "INVERSE"]),
        "blocked_by_event_validation_gate",
        gated.loc[~keep, "reason"],
    )
    gated.loc[~keep, "action"] = "BLOCK"
    return gated


def aggregate_summary(summary, sample_label):
    active = summary[summary["trades"].gt(0)].copy()
    if active.empty:
        return {
            "sample": sample_label,
            "pairs": 0,
            "trades": 0,
            "net_pips": 0.0,
            "positive_pairs": 0,
            "weighted_win_rate": 0.0,
        }
    return {
        "sample": sample_label,
        "pairs": int(len(active)),
        "trades": int(active["trades"].sum()),
        "net_pips": float(active["net_pips"].sum()),
        "positive_pairs": int((active["net_pips"] > 0).sum()),
        "weighted_win_rate": float((active["trades"] * active["win_rate"]).sum() / active["trades"].sum()),
    }


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
    print("[PROGRESS] START 26_manual_block_inverse_validation_gated_conservative.py", flush=True)
    if not TRADES_IN.exists():
        raise FileNotFoundError(f"Missing trades input: {TRADES_IN}")

    data = pd.read_csv(TRADES_IN, low_memory=False)
    data["event_year"] = pd.to_numeric(data["event_year"], errors="coerce")
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    for col in ["pnl_pips", "inverse_pnl_pips"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data = data.dropna(subset=["event_year", "event_time", "pair", "CountryCode", "event_name_norm"]).copy()
    data = data.sort_values(["event_time", "pair"]).reset_index(drop=True)
    print(f"[DATA] train_trades={len(data)}", flush=True)

    validation_rows = []
    event_validation_rows = []
    score_rows = []
    for year in VALIDATION_YEARS:
        train = data[(data["event_year"] >= 2010) & (data["event_year"] < year)].copy()
        valid = data[data["event_year"].eq(year)].copy()
        print(f"[WF] year={year} train={len(train)} valid={len(valid)}", flush=True)
        model = build_model(train, f"validation_gated_train_2010_{year-1}_validate_{year}")
        model["validation_year"] = year
        score_rows.append(model)
        _, orders = apply_model(valid, model)
        validation_rows.append(summarize_orders(orders, "validation_gated_validation", year))
        event_validation_rows.append(summarize_event_validation(orders, year))

    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    validation.to_csv(VALIDATION_OUT, index=False)
    event_validation = pd.concat(event_validation_rows, ignore_index=True) if event_validation_rows else pd.DataFrame()
    event_validation.to_csv(EVENT_VALIDATION_OUT, index=False)
    event_scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    event_scores.to_csv(EVENT_SCORES_OUT, index=False)

    gate = build_validation_gate(event_validation)
    final_train = data[(data["event_year"] >= 2010) & (data["event_year"] <= 2023)].copy()
    final_model_raw = build_model(final_train, "validation_gated_conservative_final_train_2010_2023")
    final_model = apply_validation_gate(final_model_raw, gate)
    final_model.to_csv(MODEL_OUT, index=False)

    _, final_orders = apply_model(final_train, final_model)
    final_orders.to_csv(FINAL_ORDERS_OUT, index=False)

    validation_agg = pd.DataFrame([aggregate_summary(validation, "validation_gated_validation_2018_2023")])
    validation_pair = validation.groupby("pair", as_index=False).agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        positive_years=("net_pips", lambda s: int((s > 0).sum())),
        allow_trades=("allow_trades", "sum"),
        inverse_trades=("inverse_trades", "sum"),
    ).sort_values("net_pips", ascending=False)
    final_action_count = final_model.groupby("action", as_index=False).size().rename(columns={"size": "count"})
    gate_action_count = gate.groupby("validation_action", as_index=False).size().rename(columns={"size": "count"}) if not gate.empty else pd.DataFrame()

    report = [
        "# Laporan Manual Block/Inverse Validation-Gated Conservative",
        "",
        "Model ini memakai train stability yang lebih ketat, inverse margin, dan event-level validation gate dari 2018-2023. Data 2024-2026 tidak dipakai memilih action.",
        "",
        "## Train Filter",
        f"- trade_count >= {MIN_TRADE_COUNT}",
        f"- active_years >= {MIN_ACTIVE_YEARS}",
        f"- dominant_year_share <= {MAX_DOMINANT_YEAR_SHARE}",
        f"- ALLOW: positive_year_ratio >= {ALLOW_MIN_POSITIVE_YEAR_RATIO}, profit_factor > {ALLOW_MIN_PROFIT_FACTOR}, avg_pips >= {ALLOW_MIN_AVG_PIPS}",
        f"- INVERSE: positive_year_ratio >= {INVERSE_MIN_POSITIVE_YEAR_RATIO}, profit_factor > {INVERSE_MIN_PROFIT_FACTOR}, avg_pips >= {INVERSE_MIN_AVG_PIPS}, score margin >= {INVERSE_MIN_SCORE_RATIO}x",
        "",
        "## Validation Gate",
        f"- active_validation_years >= {MIN_VALIDATION_ACTIVE_YEARS}",
        f"- validation_positive_year_ratio >= {MIN_VALIDATION_POSITIVE_YEAR_RATIO}",
        f"- validation_net_pips >= {MIN_VALIDATION_NET_PIPS}",
        f"- validation_avg_pips >= {MIN_VALIDATION_AVG_PIPS}",
        f"- validation_action_share >= {MIN_VALIDATION_ACTION_SHARE}",
        f"- top events per pair-country <= {TOP_EVENTS_PER_PAIR_COUNTRY}",
        "",
        "## Validasi Walk-Forward 2018-2023",
        markdown_table(validation_agg),
        "",
        "## Pair Validation Summary",
        markdown_table(validation_pair),
        "",
        "## Gate Action Count",
        markdown_table(gate_action_count),
        "",
        "## Final Action Count",
        markdown_table(final_action_count),
        "",
        "## Output",
        f"- `{MODEL_OUT}`",
        f"- `{VALIDATION_OUT}`",
        f"- `{EVENT_VALIDATION_OUT}`",
        f"- `{EVENT_SCORES_OUT}`",
        f"- `{FINAL_ORDERS_OUT}`",
        f"- `{TEST_SUMMARY_OUT}`",
        f"- `{TEST_ORDERS_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_OUT}", flush=True)
    print(f"[OUTPUT] {EVENT_VALIDATION_OUT}", flush=True)
    print(f"[OUTPUT] {EVENT_SCORES_OUT}", flush=True)
    print(f"[OUTPUT] {FINAL_ORDERS_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)
    print(f"[DATA] gate_rows={len(gate)} final_actions={final_model['action'].value_counts().to_dict()}", flush=True)
    print(
        f"[RESULT] validation_trades={int(validation_agg['trades'].iloc[0])} "
        f"validation_net_pips={float(validation_agg['net_pips'].iloc[0]):.2f} "
        f"positive_pairs={int(validation_agg['positive_pairs'].iloc[0])}/{int(validation_agg['pairs'].iloc[0])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
