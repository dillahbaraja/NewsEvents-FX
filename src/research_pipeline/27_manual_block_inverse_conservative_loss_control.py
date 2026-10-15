from common import *
from importlib.machinery import SourceFileLoader

base26 = SourceFileLoader(
    "base26",
    str(ROOT / "26_manual_block_inverse_validation_gated_conservative.py"),
).load_module()

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_conservative_loss_control_2010_2023.csv"
VALIDATION_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_validation_2018_2023.csv"
EVENT_VALIDATION_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_event_validation_2018_2023.csv"
EVENT_GATE_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_event_gate.csv"
EVENT_SCORES_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_event_scores.csv"
FINAL_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_orders_2010_2023.csv"
TEST_SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_test_summary_2024_2026.csv"
TEST_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_conservative_loss_control_orders_2024_2026.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_CONSERVATIVE_LOSS_CONTROL.md"

VALIDATION_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

MIN_VALIDATION_ACTIVE_YEARS = 3
MIN_VALIDATION_POSITIVE_YEAR_RATIO = 0.50
MIN_INVERSE_VALIDATION_ACTIVE_YEARS = 4
MIN_INVERSE_VALIDATION_POSITIVE_YEAR_RATIO = 0.67
MIN_VALIDATION_NET_PIPS = 20.0
MIN_VALIDATION_AVG_PIPS = 0.05
MIN_VALIDATION_ACTION_SHARE = 0.60
MIN_VALIDATION_NET_DD_RATIO = 0.65
TOP_EVENTS_PER_PAIR_COUNTRY = 35
TOP_EVENTS_PER_PAIR_COUNTRY_CATEGORY = 8


def event_category(event_name_norm):
    text = str(event_name_norm).lower()
    checks = [
        ("rate", ["rate decision", "interest rate", "cash rate", "fomc", "ecb", "boj", "boe", "rba", "rbnz", "boc", "snb"]),
        ("inflation", ["cpi", "inflation", "ppi", "price index", "prices"]),
        ("labor", ["unemployment", "jobless", "claims", "employment", "non-farm", "payroll", "earnings", "claimant"]),
        ("pmi", ["pmi", "manufacturing", "services", "ism", "business activity"]),
        ("retail", ["retail sales", "consumer spending"]),
        ("growth", ["gdp", "growth", "industrial production", "capacity utilization"]),
        ("housing", ["housing", "home sales", "building permits", "construction"]),
        ("trade", ["trade balance", "current account", "exports", "imports"]),
        ("sentiment", ["confidence", "sentiment", "expectations", "survey", "ifo", "zew", "sentix"]),
        ("commodity", ["oil", "crude", "inventories", "commodity", "gas"]),
    ]
    for category, needles in checks:
        if any(needle in text for needle in needles):
            return category
    return "other"


def drawdown_from_years(frame):
    yearly = frame.sort_values("validation_year")["validation_net_pips"].to_numpy(dtype=float)
    if yearly.size == 0:
        return 0.0
    equity = np.cumsum(yearly)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def build_loss_control_gate(event_validation):
    if event_validation.empty:
        return pd.DataFrame()
    agg = event_validation.groupby(["pair", "country", "event_name_norm", "action"], as_index=False).agg(
        validation_active_years=("validation_year", "nunique"),
        validation_positive_years=("validation_net_pips", lambda s: int((s > 0).sum())),
        validation_trades=("validation_trades", "sum"),
        validation_net_pips=("validation_net_pips", "sum"),
    )
    agg["validation_positive_year_ratio"] = agg["validation_positive_years"] / agg["validation_active_years"].clip(lower=1)
    agg["validation_avg_pips"] = agg["validation_net_pips"] / agg["validation_trades"].clip(lower=1)

    dd_rows = []
    for key, g in event_validation.groupby(["pair", "country", "event_name_norm", "action"], sort=False):
        dd_rows.append((*key, drawdown_from_years(g)))
    dd = pd.DataFrame(dd_rows, columns=["pair", "country", "event_name_norm", "action", "validation_max_drawdown_pips"])
    agg = agg.merge(dd, on=["pair", "country", "event_name_norm", "action"], how="left")
    agg["validation_net_dd_ratio"] = agg["validation_net_pips"] / agg["validation_max_drawdown_pips"].clip(lower=1.0)

    totals = event_validation.groupby(["pair", "country", "event_name_norm"], as_index=False).agg(
        total_validation_trades=("validation_trades", "sum")
    )
    agg = agg.merge(totals, on=["pair", "country", "event_name_norm"], how="left")
    agg["validation_action_share"] = agg["validation_trades"] / agg["total_validation_trades"].clip(lower=1)
    agg["event_category"] = agg["event_name_norm"].map(event_category)
    agg["validation_rank_score"] = (
        agg["validation_net_pips"]
        * agg["validation_positive_year_ratio"]
        * agg["validation_action_share"]
        * np.log1p(agg["validation_trades"])
        * np.minimum(2.0, agg["validation_net_dd_ratio"].clip(lower=0.0))
    )

    base_ok = (
        (agg["validation_net_pips"] >= MIN_VALIDATION_NET_PIPS)
        & (agg["validation_avg_pips"] >= MIN_VALIDATION_AVG_PIPS)
        & (agg["validation_action_share"] >= MIN_VALIDATION_ACTION_SHARE)
        & (agg["validation_net_dd_ratio"] >= MIN_VALIDATION_NET_DD_RATIO)
    )
    allow_ok = (
        agg["action"].eq("ALLOW")
        & (agg["validation_active_years"] >= MIN_VALIDATION_ACTIVE_YEARS)
        & (agg["validation_positive_year_ratio"] >= MIN_VALIDATION_POSITIVE_YEAR_RATIO)
    )
    inverse_ok = (
        agg["action"].eq("INVERSE")
        & (agg["validation_active_years"] >= MIN_INVERSE_VALIDATION_ACTIVE_YEARS)
        & (agg["validation_positive_year_ratio"] >= MIN_INVERSE_VALIDATION_POSITIVE_YEAR_RATIO)
    )
    gate = agg[base_ok & (allow_ok | inverse_ok)].copy()
    gate = gate.sort_values("validation_rank_score", ascending=False)
    gate["rank_in_pair_country"] = gate.groupby(["pair", "country"])["validation_rank_score"].rank(method="first", ascending=False)
    gate["rank_in_pair_country_category"] = gate.groupby(
        ["pair", "country", "event_category"]
    )["validation_rank_score"].rank(method="first", ascending=False)
    gate = gate[
        (gate["rank_in_pair_country"] <= TOP_EVENTS_PER_PAIR_COUNTRY)
        & (gate["rank_in_pair_country_category"] <= TOP_EVENTS_PER_PAIR_COUNTRY_CATEGORY)
    ].copy()
    return gate.rename(columns={"action": "validation_action"})


def aggregate_summary(summary, sample_label):
    return base26.aggregate_summary(summary, sample_label)


def markdown_table(df):
    return base26.markdown_table(df)


def main():
    print("[PROGRESS] START 27_manual_block_inverse_conservative_loss_control.py", flush=True)
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
        model = base26.build_model(train, f"loss_control_train_2010_{year-1}_validate_{year}")
        model["validation_year"] = year
        score_rows.append(model)
        _, orders = base26.apply_model(valid, model)
        validation_rows.append(base26.summarize_orders(orders, "loss_control_validation", year))
        event_validation_rows.append(base26.summarize_event_validation(orders, year))

    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    validation.to_csv(VALIDATION_OUT, index=False)
    event_validation = pd.concat(event_validation_rows, ignore_index=True) if event_validation_rows else pd.DataFrame()
    event_validation.to_csv(EVENT_VALIDATION_OUT, index=False)
    event_scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    event_scores.to_csv(EVENT_SCORES_OUT, index=False)

    gate = build_loss_control_gate(event_validation)
    gate.to_csv(EVENT_GATE_OUT, index=False)

    final_train = data[(data["event_year"] >= 2010) & (data["event_year"] <= 2023)].copy()
    final_model_raw = base26.build_model(final_train, "conservative_loss_control_final_train_2010_2023")
    final_model = base26.apply_validation_gate(final_model_raw, gate)
    final_model.to_csv(MODEL_OUT, index=False)

    _, final_orders = base26.apply_model(final_train, final_model)
    final_orders.to_csv(FINAL_ORDERS_OUT, index=False)

    validation_agg = pd.DataFrame([aggregate_summary(validation, "loss_control_validation_2018_2023")])
    validation_pair = validation.groupby("pair", as_index=False).agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        positive_years=("net_pips", lambda s: int((s > 0).sum())),
        allow_trades=("allow_trades", "sum"),
        inverse_trades=("inverse_trades", "sum"),
    ).sort_values("net_pips", ascending=False)
    final_action_count = final_model.groupby("action", as_index=False).size().rename(columns={"size": "count"})
    gate_action_count = gate.groupby("validation_action", as_index=False).size().rename(columns={"size": "count"}) if not gate.empty else pd.DataFrame()
    gate_category_count = gate.groupby("event_category", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False) if not gate.empty else pd.DataFrame()

    report = [
        "# Laporan Manual Block/Inverse Conservative Loss Control",
        "",
        "Model ini melanjutkan validation-gated conservative dengan drawdown gate, inverse validation yang lebih ketat, dan cap event-category. Data 2024-2026 tidak dipakai memilih action.",
        "",
        "## Loss-Control Gate",
        f"- ALLOW: active validation years >= {MIN_VALIDATION_ACTIVE_YEARS}, positive year ratio >= {MIN_VALIDATION_POSITIVE_YEAR_RATIO}",
        f"- INVERSE: active validation years >= {MIN_INVERSE_VALIDATION_ACTIVE_YEARS}, positive year ratio >= {MIN_INVERSE_VALIDATION_POSITIVE_YEAR_RATIO}",
        f"- validation_net_pips >= {MIN_VALIDATION_NET_PIPS}",
        f"- validation_avg_pips >= {MIN_VALIDATION_AVG_PIPS}",
        f"- validation_action_share >= {MIN_VALIDATION_ACTION_SHARE}",
        f"- validation_net_dd_ratio >= {MIN_VALIDATION_NET_DD_RATIO}",
        f"- top events per pair-country <= {TOP_EVENTS_PER_PAIR_COUNTRY}",
        f"- top events per pair-country-category <= {TOP_EVENTS_PER_PAIR_COUNTRY_CATEGORY}",
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
        "## Gate Category Count",
        markdown_table(gate_category_count),
        "",
        "## Final Action Count",
        markdown_table(final_action_count),
        "",
        "## Output",
        f"- `{MODEL_OUT}`",
        f"- `{VALIDATION_OUT}`",
        f"- `{EVENT_VALIDATION_OUT}`",
        f"- `{EVENT_GATE_OUT}`",
        f"- `{EVENT_SCORES_OUT}`",
        f"- `{FINAL_ORDERS_OUT}`",
        f"- `{TEST_SUMMARY_OUT}`",
        f"- `{TEST_ORDERS_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_OUT}", flush=True)
    print(f"[OUTPUT] {EVENT_VALIDATION_OUT}", flush=True)
    print(f"[OUTPUT] {EVENT_GATE_OUT}", flush=True)
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
