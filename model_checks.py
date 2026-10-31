import re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open("model_run.py").read().split("rows=[]")[0]   # reuse data prep and models
exec(src)
from sklearn.metrics import roc_auc_score
gb = models["Gradient boosting"].fit(X(tr), tr["mismatch_any"])
p = gb.predict_proba(X(te))[:,1]; yt = te["mismatch_any"].values
rng = np.random.default_rng(42); aucs=[]
for _ in range(1000):
    i = rng.integers(0,len(yt),len(yt)); 
    if yt[i].min()==yt[i].max(): continue
    aucs.append(roc_auc_score(yt[i],p[i]))
print("GB test AUC", round(roc_auc_score(yt,p),3), "95% interval", np.percentile(aucs,[2.5,97.5]).round(3))
# paired: GB vs LR
lr = models["Logistic regression"].fit(X(tr), tr["mismatch_any"]); pl = lr.predict_proba(X(te))[:,1]
d=[]
for _ in range(1000):
    i = rng.integers(0,len(yt),len(yt))
    d.append(roc_auc_score(yt[i],p[i])-roc_auc_score(yt[i],pl[i]))
print("GB minus LR AUC diff 95%", np.percentile(d,[2.5,97.5]).round(3))
for col in ["skin_type","skin_tone_group"]:
    for g,s in te.groupby(col):
        if s["mismatch_any"].nunique()<2 or len(s)<100: print(col,g,len(s),"skipped"); continue
        print(col,g,len(s),"mismatch",round(s.mismatch_any.mean(),3),"AUC",round(roc_auc_score(s.mismatch_any,gb.predict_proba(X(s))[:,1]),3))