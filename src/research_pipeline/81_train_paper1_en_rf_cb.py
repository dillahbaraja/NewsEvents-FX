"""Paper 1 standard baseline: EN, RF and CatBoost per FX pair."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from common import DATA, progress_done, progress_start, progress_step, profit_factor

SCRIPT = Path(__file__).name
OUT = DATA / "artifacts" / "paper1_baseline"
MODELS = OUT / "models"
SOURCE = Path(__file__).resolve().parent / "modeling_dataset_multi_currency.csv"
FEATURES = ["surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour"] + [f"category_{x}" for x in ["inflation","labor","growth","rates","sentiment","trade","housing","commodities","other"]] + ["orientation_base", "orientation_quote"]


def score(y, pred, pnl):
    active = pnl[pred != 0]
    return {"trades": int(len(active)), "net_pips": float(active.sum()) if len(active) else 0.0, "profit_factor": float(profit_factor(active)) if len(active) else 0.0, "winrate": float((active > 0).mean()) if len(active) else 0.0, "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else 0.0, "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def main():
    started = progress_start(SCRIPT)
    MODELS.mkdir(parents=True, exist_ok=True)
    use = ["pair", "event_year", "split", "trade_label_1h", "domestic_return_pips_1h"] + FEATURES
    d = pd.read_csv(SOURCE, usecols=use, low_memory=False)
    d = d[d["pair"].notna()].copy()
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["trade_label_1h"] = pd.to_numeric(d["trade_label_1h"], errors="coerce").fillna(0).astype(int)
    d["domestic_return_pips_1h"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce").fillna(0.0)
    rows=[]
    for pair in sorted(d["pair"].unique()):
        part=d[d["pair"].eq(pair)].copy(); tr=part[part.split.eq("train")]; va=part[part.split.eq("validation")]; te=part[part.split.eq("final_test")]
        progress_step("PAIR_START", pair=pair, train=len(tr), validation=len(va), test=len(te))
        models={"elastic_net":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,solver="saga",penalty="elasticnet",l1_ratio=0.5,class_weight="balanced",multi_class="multinomial",random_state=42)),"random_forest":RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=10,class_weight="balanced_subsample",n_jobs=-1,random_state=42)}
        if CatBoostClassifier is not None: models["catboost"]=CatBoostClassifier(iterations=500,depth=6,learning_rate=0.05,l2_leaf_reg=8,loss_function="MultiClass",auto_class_weights="Balanced",verbose=False,allow_writing_files=False,random_seed=42)
        X=tr[FEATURES]; y=tr.trade_label_1h; 
        for name,model in models.items():
            t=time.perf_counter(); progress_step("MODEL_START",pair=pair,model=name)
            model.fit(X,y); joblib.dump(model,MODELS/f"{name}_{pair}.joblib")
            for split,q in [("train",tr),("validation",va),("final_test",te)]:
                if q.empty: continue
                pred=np.asarray(model.predict(q[FEATURES])).reshape(-1).astype(int); s=score(q.trade_label_1h.to_numpy(),pred,q.domestic_return_pips_1h.to_numpy()*pred); s.update({"pair":pair,"model":name,"split":split,"seconds":time.perf_counter()-t}); rows.append(s)
            progress_step("MODEL_DONE",pair=pair,model=name,seconds=f"{time.perf_counter()-t:.1f}")
    pd.DataFrame(rows).to_csv(OUT/"paper1_baseline_results.csv",index=False)
    (OUT/"metadata.json").write_text('{"models":["elastic_net","random_forest","catboost"],"features":"news_only_standard","train":"2010-2020","validation":"2021-2023","oos":"2024-2026","target":"trade_label_1h"}',encoding="utf-8")
    progress_done(SCRIPT,started)

if __name__=="__main__": main()
