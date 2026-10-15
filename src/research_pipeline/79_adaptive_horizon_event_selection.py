"""Select event-specific horizons on validation and evaluate them on untouched OOS."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
ROOT = ARTIFACTS / "event_character_pair_specific"
REPORTS = ROOT / "reports"
OUT = ROOT / "adaptive_horizon_experiment"
KEY = ["pair", "CountryCode", "event_name_norm"]


def pf(values):
    return float(profit_factor(values)) if len(values) else 0.0


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(REPORTS / "pair_specific_selected_models_stability.csv", low_memory=False)
    selected = selected[selected["deployable"].astype(str).str.lower().eq("true")]
    v = pd.read_csv(REPORTS / "pair_specific_validation_predictions.csv", low_memory=False)
    o = pd.read_csv(REPORTS / "pair_specific_oos_predictions.csv", low_memory=False)
    v = v[v["pair"].isin(selected["pair"])].copy()
    o = o[o["pair"].isin(selected["pair"])].copy()
    v["strategy_pips"] = pd.to_numeric(v["strategy_pips"], errors="coerce").fillna(0.0)
    o["strategy_pips"] = pd.to_numeric(o["strategy_pips"], errors="coerce").fillna(0.0)
    rows = []
    for (pair, country, event), g in v.groupby(KEY, dropna=False, sort=False):
        candidates = []
        for horizon, h in g.groupby("horizon_hours"):
            active = h[h["signal"].astype(float).ne(0)]
            if len(active) < 10:
                continue
            net = float(active["strategy_pips"].sum())
            pfv = pf(active["strategy_pips"].to_numpy())
            score = net + 250.0 * min(pfv, 3.0) + 5.0 * np.log1p(len(active))
            candidates.append((score, horizon, len(active), net, pfv))
        if candidates:
            score, horizon, n, net, pfv = max(candidates, key=lambda x: x[0])
            rows.append({"pair": pair, "CountryCode": country, "event_name_norm": event, "selected_horizon_hours": horizon, "validation_trades": n, "validation_net_pips": net, "validation_profit_factor": pfv, "selection_score": score, "selection_period": "2021-2023"})
    choices = pd.DataFrame(rows)
    choices.to_csv(OUT / "event_specific_horizon_selection.csv", index=False)
    o = o.merge(choices[KEY + ["selected_horizon_hours"]], on=KEY, how="inner")
    o = o[pd.to_numeric(o["horizon_hours"], errors="coerce") == pd.to_numeric(o["selected_horizon_hours"], errors="coerce")].copy()
    summaries = []
    for _, s in selected.iterrows():
        g = o[(o["pair"] == s["pair"]) & (o["model"] == s["model"])]
        active = g[g["signal"].astype(float).ne(0)]
        p = active["strategy_pips"].to_numpy(dtype=float)
        summaries.append({"pair": s["pair"], "model": s["model"], "trades": len(p), "net_pips": float(p.sum()) if len(p) else 0.0, "profit_factor": pf(p), "winrate": float((p > 0).mean()) if len(p) else 0.0, "events_selected": int(g[["CountryCode", "event_name_norm"]].drop_duplicates().shape[0]) if not g.empty else 0})
        progress_step("PAIR_ADAPTIVE_HORIZON", pair=s["pair"], trades=len(p), net_pips=f"{p.sum() if len(p) else 0.0:.1f}", pf=f"{pf(p):.4f}")
    pd.DataFrame(summaries).to_csv(OUT / "adaptive_horizon_oos_summary.csv", index=False)
    o.to_csv(OUT / "adaptive_horizon_oos_predictions.csv", index=False)
    progress_step("ADAPTIVE_HORIZON_DONE", selected_events=len(choices), oos_rows=len(o), output=str(OUT))
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
