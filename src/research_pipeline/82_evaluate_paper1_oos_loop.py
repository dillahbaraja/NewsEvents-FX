"""Evaluate Paper 1 baseline on OOS development (2024-25) and locked final OOS (2026)."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from common import ARTIFACTS, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
PIPE = Path(__file__).resolve().parent
OUT = ARTIFACTS / "paper1_baseline"
MODELS = OUT / "models"
FEATURES = ["surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour"] + [f"category_{x}" for x in ["inflation","labor","growth","rates","sentiment","trade","housing","commodities","other"]] + ["orientation_base", "orientation_quote"]


def main():
    started = progress_start(SCRIPT)
    sys.path.insert(0, str(PIPE))
    from importlib import import_module
    stage65 = import_module("65_test_event_character_aware_oos")
    d = stage65.attach_profiles(stage65.build_oos_dataset())
    d = d[pd.to_numeric(d["horizon_hours"], errors="coerce").eq(1)].copy()
    progress_step("OOS_DATA_READY", rows=len(d), pairs=d["pair"].nunique())
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["year"] = pd.to_datetime(d["event_time"], errors="coerce").dt.year
    rows=[]; preds=[]
    for pair in sorted(d["pair"].unique()):
        part=d[d["pair"].eq(pair)].copy()
        for name in ["elastic_net","random_forest","catboost"]:
            path=MODELS/f"{name}_{pair}.joblib"
            if not path.exists(): continue
            model=joblib.load(path); pred=np.asarray(model.predict(part[FEATURES])).reshape(-1).astype(int)
            part2=part[["event_time","Name","CountryCode","pair","domestic_return_pips","year"]].copy(); part2["model"]=name; part2["signal"]=pred; part2["strategy_pips"]=part2["domestic_return_pips"].astype(float)*pred; preds.append(part2)
            for period, mask in [("oos_dev_2024_2025",part2.year.between(2024,2025)),("final_oos_2026",part2.year.eq(2026))]:
                active=part2[mask & part2.signal.ne(0)]; x=active.strategy_pips.to_numpy(float)
                rows.append({"period":period,"pair":pair,"model":name,"trades":len(x),"net_pips":float(x.sum()) if len(x) else 0.0,"profit_factor":float(profit_factor(x)) if len(x) else 0.0,"winrate":float((x>0).mean()) if len(x) else 0.0})
            progress_step("MODEL_OOS_DONE",pair=pair,model=name)
    pd.DataFrame(rows).to_csv(OUT/"paper1_oos_loop_results.csv",index=False)
    if preds: pd.concat(preds,ignore_index=True).to_csv(OUT/"paper1_oos_loop_predictions.csv",index=False)
    progress_done(SCRIPT,started)

if __name__=="__main__": main()
