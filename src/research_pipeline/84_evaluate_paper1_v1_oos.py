from pathlib import Path
import sys, joblib
import numpy as np, pandas as pd
from common import ARTIFACTS, progress_start, progress_step, progress_done, profit_factor

SCRIPT=Path(__file__).name; PIPE=Path(__file__).resolve().parent; OUT=ARTIFACTS/'paper1_v1_standardized_surprise'; MODELS=OUT/'models'
BASE=["surprise_fc","surprise_prev","surprise_pct_fc","Importance","Impact","release_hour"]+[f"category_{x}" for x in ["inflation","labor","growth","rates","sentiment","trade","housing","commodities","other"]]+["orientation_base","orientation_quote"]
NEW=["surprise_fc_z","surprise_prev_z","surprise_fc_abs","surprise_prev_abs","surprise_fc_x_importance","surprise_prev_x_importance","surprise_fc_x_impact","surprise_prev_x_impact"]; FEATURES=BASE+NEW
def add(d,stats):
 d=d.copy(); d['CountryCode']=d['CountryCode'].astype(str)
 for c in ['surprise_fc','surprise_prev','Importance','Impact']: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0.0)
 d=d.merge(stats,on=['pair','CountryCode'],how='left')
 for c in ['fc_mean','prev_mean']: d[c]=d[c].fillna(0.0)
 for c in ['fc_std','prev_std']: d[c]=d[c].replace(0,np.nan).fillna(1.0)
 d['surprise_fc_z']=(d.surprise_fc-d.fc_mean)/d.fc_std; d['surprise_prev_z']=(d.surprise_prev-d.prev_mean)/d.prev_std
 d['surprise_fc_abs']=d.surprise_fc_z.abs(); d['surprise_prev_abs']=d.surprise_prev_z.abs(); d['surprise_fc_x_importance']=d.surprise_fc_z*d.Importance; d['surprise_prev_x_importance']=d.surprise_prev_z*d.Importance; d['surprise_fc_x_impact']=d.surprise_fc_z*d.Impact; d['surprise_prev_x_impact']=d.surprise_prev_z*d.Impact
 for c in FEATURES: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0)
 return d
def main():
 st=progress_start(SCRIPT); sys.path.insert(0,str(PIPE)); from importlib import import_module; s=import_module('65_test_event_character_aware_oos'); d=s.attach_profiles(s.build_oos_dataset()); d=d[pd.to_numeric(d.horizon_hours,errors='coerce').eq(1)].copy(); d['year']=pd.to_datetime(d.event_time,errors='coerce').dt.year; stats=pd.read_csv(OUT/'standardization_statistics_train.csv'); d=add(d,stats)
 rows=[]
 for pair in sorted(d.pair.unique()):
  p=d[d.pair.eq(pair)]
  for name in ['elastic_net','random_forest','catboost']:
   path=MODELS/f'{name}_{pair}.joblib';
   if not path.exists(): continue
   bundle=joblib.load(path); m=bundle['model']; pred=np.asarray(m.predict(p[FEATURES])).reshape(-1).astype(int); p=p.copy(); p['signal']=pred; p['pnl']=p.domestic_return_pips.astype(float)*pred
   for period,mask in [('oos_dev_2024_2025',p.year.between(2024,2025)),('final_oos_2026',p.year.eq(2026))]:
    x=p.loc[mask & p.signal.ne(0),'pnl'].to_numpy(float); rows.append({'period':period,'pair':pair,'model':name,'trades':len(x),'net_pips':float(x.sum()) if len(x) else 0.0,'profit_factor':float(profit_factor(x)) if len(x) else 0.0,'winrate':float((x>0).mean()) if len(x) else 0.0})
   progress_step('MODEL_OOS_DONE',pair=pair,model=name)
 pd.DataFrame(rows).to_csv(OUT/'v1_oos_results.csv',index=False); progress_done(SCRIPT,st)
if __name__=='__main__': main()
