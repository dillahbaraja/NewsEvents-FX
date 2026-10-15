"""Conservative event gate using split validation, cost-aware P/L and bootstrap support."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
ROOT = ARTIFACTS / "event_character_pair_specific"
REPORTS = ROOT / "reports"
OUT = ROOT / "conservative_event_gate"
KEY = ["pair", "CountryCode", "event_name_norm"]
MIN_TRADES = 30
MIN_PF = 1.05
MIN_POSITIVE_SUBPERIODS = 2
BOOTSTRAPS = 200
ROUND_TRIP_COST_PIPS = 1.0


def pf(x):
    return float(profit_factor(np.asarray(x, dtype=float))) if len(x) else 0.0


def metrics(x):
    x = np.asarray(x, dtype=float)
    net = float(x.sum()) if len(x) else 0.0
    return {"trades": len(x), "net": net, "pf": pf(x), "avg": float(x.mean()) if len(x) else 0.0, "winrate": float((x > 0).mean()) if len(x) else 0.0}


def bootstrap_lower(x, seed):
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_TRADES:
        return -np.inf
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.10) * len(x))


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(REPORTS / "pair_specific_selected_models_stability.csv", low_memory=False)
    selected = selected[selected["deployable"].astype(str).str.lower().eq("true")]
    v = pd.read_csv(REPORTS / "pair_specific_validation_predictions.csv", low_memory=False)
    o = pd.read_csv(REPORTS / "pair_specific_oos_predictions.csv", low_memory=False)
    pairs = set(selected["pair"])
    v = v[v["pair"].isin(pairs)].copy(); o = o[o["pair"].isin(pairs)].copy()
    v["year"] = pd.to_datetime(v["event_time"], errors="coerce").dt.year
    o["year"] = pd.to_datetime(o["event_time"], errors="coerce").dt.year
    v["strategy_pips"] = pd.to_numeric(v["strategy_pips"], errors="coerce").fillna(0.0)
    o["strategy_pips"] = pd.to_numeric(o["strategy_pips"], errors="coerce").fillna(0.0)
    rows = []
    for (pair, country, event), g in v.groupby(KEY, dropna=False, sort=False):
        active = g[g["signal"].astype(float).ne(0)].copy()
        if len(active) < MIN_TRADES:
            continue
        original = active["strategy_pips"].to_numpy(float) - ROUND_TRIP_COST_PIPS
        inverse = -active["strategy_pips"].to_numpy(float) - ROUND_TRIP_COST_PIPS
        sub = []
        for label, h in [("v1_2021_2022", active[active["year"].between(2021, 2022)]), ("v2_2023", active[active["year"].eq(2023)])]:
            ix = h.index
            loc = active.index.get_indexer(ix)
            sub.append((label, metrics(original[loc]), metrics(inverse[loc])))
        orig_ok = all(m["trades"] >= 5 and m["net"] > 0 and m["pf"] >= MIN_PF for _, m, _ in sub)
        inv_ok = all(m["trades"] >= 5 and m["net"] > 0 and m["pf"] >= MIN_PF for _, _, m in sub)
        orig_boot = bootstrap_lower(original, abs(hash((pair, country, event))) % (2**32))
        inv_boot = bootstrap_lower(inverse, abs(hash((event, country, pair))) % (2**32))
        action = "BLOCK"; chosen = "BLOCK"
        if orig_ok and orig_boot > 0 and (not inv_ok or original.sum() >= inverse.sum()):
            action = chosen = "ALLOW"
        elif inv_ok and inv_boot > 0:
            action = chosen = "INVERSE"
        rows.append({"pair": pair, "CountryCode": country, "event_name_norm": event, "action": action, "validation_trades": len(active), "original_net_pips": float(original.sum()), "inverse_net_pips": float(inverse.sum()), "original_pf": pf(original), "inverse_pf": pf(inverse), "original_bootstrap_lower_net": orig_boot, "inverse_bootstrap_lower_net": inv_boot, "v1_original_net": sub[0][1]["net"], "v2_original_net": sub[1][1]["net"], "v1_inverse_net": sub[0][2]["net"], "v2_inverse_net": sub[1][2]["net"], "selection_period": "validation_2021_2023_split_2021_2022_2023"})
    rules = pd.DataFrame(rows)
    rules.to_csv(OUT / "conservative_event_gate_rules.csv", index=False)
    o = o.merge(rules[KEY + ["action"]], on=KEY, how="inner")
    o["final_signal"] = np.where(o["action"].eq("BLOCK"), 0, np.where(o["action"].eq("INVERSE"), -o["signal"].astype(float), o["signal"].astype(float)))
    o["final_pips"] = o["domestic_return_pips"].astype(float) * o["final_signal"] - np.where(o["final_signal"].ne(0), ROUND_TRIP_COST_PIPS, 0.0)
    summaries = []
    for _, s in selected.iterrows():
        g = o[(o["pair"] == s["pair"]) & (o["model"] == s["model"]) & o["final_signal"].ne(0)]
        x = g["final_pips"].to_numpy(float)
        summaries.append({"pair": s["pair"], "model": s["model"], "trades": len(x), "net_pips": float(x.sum()) if len(x) else 0.0, "profit_factor": pf(x), "winrate": float((x > 0).mean()) if len(x) else 0.0})
        progress_step("PAIR_CONSERVATIVE_GATE", pair=s["pair"], trades=len(x), net_pips=f"{x.sum() if len(x) else 0.0:.1f}", pf=f"{pf(x):.4f}")
    pd.DataFrame(summaries).to_csv(OUT / "conservative_gate_oos_summary.csv", index=False)
    o.to_csv(OUT / "conservative_gate_oos_predictions.csv", index=False)
    (OUT / "README.md").write_text("Conservative event gate. Rules selected from validation 2021-2023 split into 2021-2022 and 2023 because the stored pair-specific prediction artifact begins in 2021. OOS 2024-2026 is evaluation only. Costs: 1 pip round trip.\n", encoding="utf-8")
    progress_step("CONSERVATIVE_GATE_DONE", rules=len(rules), allowed=int(rules.action.ne("BLOCK").sum()), oos_rows=len(o))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
