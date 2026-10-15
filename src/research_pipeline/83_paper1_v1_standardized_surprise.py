"""Paper 1 V1: train on standardized macro surprise and event interactions."""
from pathlib import Path
import sys, time, joblib
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import DATA, ARTIFACTS, progress_start, progress_step, progress_done, profit_factor

SCRIPT=Path(__file__).name; OUT=DATA/'artifacts'/'paper1_v1_standardized_surprise'; MODELS=OUT/'models'; SOURCE=Path(__file__).resolve().parent/'modeling_dataset_multi_currency.csv'
BASE=["surprise_fc","surprise_prev","surprise_pct_fc","Importance","Impact","release_hour"]+[f"category_{x}" for x in ["inflation","labor","growth","rates","sentiment","trade","housing","commodities","other"]]+["orientation_base","orientation_quote"]
NEW=["surprise_fc_z","surprise_prev_z","surprise_fc_abs","surprise_prev_abs","surprise_fc_x_importance","surprise_prev_x_importance","surprise_fc_x_impact","surprise_prev_x_impact"]
FEATURES=BASE+NEW

def add_features(d, stats=None, fit=False):
    d=d.copy(); d["CountryCode"]=d.get("CountryCode", "US").astype(str)
    for c in ["surprise_fc","surprise_prev","Importance","Impact"]: d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0.0)
    if fit:
        stats=d[d["split"].eq("train")].groupby(["pair","CountryCode"])[["surprise_fc","surprise_prev"]].agg(["mean","std"]).reset_index()
        stats.columns=["pair","CountryCode","fc_mean","fc_std","prev_mean","prev_std"]
    d=d.merge(stats,on=["pair","CountryCode"],how="left")
    for c in ["fc_mean","prev_mean"]: d[c]=d[c].fillna(0.0)
    for c in ["fc_std","prev_std"]: d[c]=d[c].replace(0,np.nan).fillna(1.0)
    d["surprise_fc_z"]=(d.surprise_fc-d.fc_mean)/d.fc_std; d["surprise_prev_z"]=(d.surprise_prev-d.prev_mean)/d.prev_std
    d["surprise_fc_abs"]=d.surprise_fc_z.abs(); d["surprise_prev_abs"]=d.surprise_prev_z.abs()
    d["surprise_fc_x_importance"]=d.surprise_fc_z*d.Importance; d["surprise_prev_x_importance"]=d.surprise_prev_z*d.Importance
    d["surprise_fc_x_impact"]=d.surprise_fc_z*d.Impact; d["surprise_prev_x_impact"]=d.surprise_prev_z*d.Impact
    return d,stats

def main():
    started=progress_start(SCRIPT); MODELS.mkdir(parents=True,exist_ok=True)
    use=["pair","CountryCode","split","trade_label_1h","domestic_return_pips_1h"]+BASE
    d=pd.read_csv(SOURCE,usecols=use,low_memory=False); d=d[d.pair.notna()].copy(); d,stats=add_features(d,fit=True)
    stats.to_csv(OUT/'standardization_statistics_train.csv',index=False)
    for c in FEATURES: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0)
    d.trade_label_1h=pd.to_numeric(d.trade_label_1h,errors='coerce').fillna(0).astype(int); d.domestic_return_pips_1h=pd.to_numeric(d.domestic_return_pips_1h,errors='coerce').fillna(0.0)
    rows=[]
    for pair in sorted(d.pair.unique()):
        p=d[d.pair.eq(pair)]; tr=p[p.split.eq('train')]; va=p[p.split.eq('validation')];
        models={'elastic_net':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1800,solver='saga',penalty='elasticnet',l1_ratio=.35,class_weight='balanced',multi_class='multinomial',random_state=43)),'random_forest':RandomForestClassifier(n_estimators=350,max_depth=12,min_samples_leaf=12,class_weight='balanced_subsample',n_jobs=-1,random_state=43)}
        try:
            from catboost import CatBoostClassifier
            models['catboost']=CatBoostClassifier(iterations=600,depth=6,learning_rate=.04,l2_leaf_reg=10,loss_function='MultiClass',auto_class_weights='Balanced',verbose=False,allow_writing_files=False,random_seed=43)
        except Exception: pass
        for name,m in models.items():
            progress_step('MODEL_START',pair=pair,model=name); t=time.perf_counter(); m.fit(tr[FEATURES],tr.trade_label_1h); joblib.dump({'model':m,'features':FEATURES,'stats':stats},MODELS/f'{name}_{pair}.joblib')
            for split,q in [('validation',va)]:
                pred=np.asarray(m.predict(q[FEATURES])).reshape(-1).astype(int); x=q.domestic_return_pips_1h.to_numpy()*pred; active=x[pred!=0]
                rows.append({'pair':pair,'model':name,'split':split,'trades':len(active),'net_pips':float(active.sum()) if len(active) else 0.0,'profit_factor':float(profit_factor(active)) if len(active) else 0.0,'winrate':float((active>0).mean()) if len(active) else 0.0})
            progress_step('MODEL_DONE',pair=pair,model=name,seconds=f'{time.perf_counter()-t:.1f}')
    pd.DataFrame(rows).to_csv(OUT/'validation_results.csv',index=False); (OUT/'metadata.json').write_text('{"variant":"standardized_surprise_interactions","stats_fit":"train_2010_2020","validation":"2021_2023","oos_dev":"2024_2025","final_oos":"2026"}',encoding='utf-8'); progress_done(SCRIPT,started)
if __name__=='__main__': main()
