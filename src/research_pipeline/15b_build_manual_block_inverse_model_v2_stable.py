from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
STATS_IN = MANUAL_DIR / "manual_block_inverse_event_stats.csv"
MODEL_OUT = MANUAL_DIR / "model_block_inverse_v2_stable.csv"

MIN_EVENTS = 30
MIN_ACTIVE_YEARS = 6
MAX_DOMINANT_YEAR_SHARE = 0.60


def classify(row):
    if row["trade_count"] < MIN_EVENTS:
        return "BLOCK", "v2_trade_count_lt_30"
    if row["active_years"] < MIN_ACTIVE_YEARS:
        return "BLOCK", "v2_active_years_lt_6"
    if row["dominant_year_share"] > MAX_DOMINANT_YEAR_SHARE:
        return "BLOCK", "v2_dominant_year_share_gt_0.60"

    inverse_positive_year_ratio = row["inverse_positive_years"] / max(1, row["inverse_active_years"])
    inverse_ok = (
        row["net_pips"] < 0
        and row["inverse_net_pips"] > 0
        and row["inverse_profit_factor"] >= 1.25
        and row["inverse_hit_rate"] >= 0.55
        and row["inverse_active_years"] >= MIN_ACTIVE_YEARS
        and row["inverse_dominant_year_share"] <= MAX_DOMINANT_YEAR_SHARE
        and inverse_positive_year_ratio >= 0.55
    )
    if inverse_ok:
        return "INVERSE", "v2_inverse_pf_hit_yearly_stable"

    positive_year_ratio = row["positive_years"] / max(1, row["active_years"])
    allow_ok = (
        row["net_pips"] > 0
        and row["profit_factor"] >= 1.15
        and row["hit_rate"] >= 0.53
        and positive_year_ratio >= 0.50
    )
    if allow_ok:
        return "ALLOW", "v2_allow_pf_hit_yearly_stable"

    reasons = []
    if row["profit_factor"] < 1.00:
        reasons.append("v2_profit_factor_lt_1.00")
    if row["net_pips"] <= 0:
        reasons.append("v2_net_pips_le_0")
    if row["hit_rate"] < 0.52:
        reasons.append("v2_hit_rate_lt_0.52")
    if positive_year_ratio < 0.50:
        reasons.append("v2_positive_year_ratio_lt_0.50")
    return "BLOCK", ";".join(reasons) if reasons else "v2_not_meeting_allow_or_inverse"


def main():
    print("[PROGRESS] START 15b_build_manual_block_inverse_model_v2_stable.py", flush=True)
    if not STATS_IN.exists():
        raise FileNotFoundError(f"Missing v1 stats: {STATS_IN}")
    d = pd.read_csv(STATS_IN)
    required = [
        "trade_count", "active_years", "dominant_year_share", "net_pips", "profit_factor", "hit_rate",
        "inverse_net_pips", "inverse_profit_factor", "inverse_hit_rate", "inverse_active_years",
        "inverse_positive_years", "inverse_dominant_year_share", "positive_years",
    ]
    for col in required:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    actions = d.apply(lambda row: classify(row), axis=1)
    d["action"] = [x[0] for x in actions]
    d["reason"] = [x[1] for x in actions]
    d["selection_period"] = "calibration_2010_2023_v2_stable"
    d.to_csv(MODEL_OUT, index=False)
    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[DATA] actions={d['action'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
