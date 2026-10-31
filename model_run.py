import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

r = pd.read_csv("out/reviews_clean.csv.gz", low_memory=False)
r["log_price_per_100ml"] = np.log1p(r["price_per_100ml"])
r["dry_x_alcohol"] = ((r["skin_type"]=="dry")&(r["ing_drying_alcohol"]==1)).astype(int)
r["dry_x_fragrance"] = ((r["skin_type"]=="dry")&(r["ing_fragrance"]==1)).astype(int)
r["oily_x_oil"] = ((r["skin_type"]=="oily")&(r["ing_oil_butter"]==1)).astype(int)
irr=["ing_fragrance","ing_fragrance_allergen","ing_drying_alcohol","ing_retinoid"]
r["irritant_load"]=r[irr].sum(axis=1,min_count=1)
r["year"]=pd.to_datetime(r["submission_time"]).dt.year
r["claimed_fit"]=r["claimed_fit"]
ing=[c for c in r.columns if c.startswith("ing_") and c!="ing_suspect_parse"]
cat=["skin_type","skin_tone_group","price_tier","secondary_category"]
num=ing+["log_price_per_100ml","n_ingredients","is_kit","claim_vegan","claim_clean","claim_fragrance_free","claimed_fit",
         "dry_x_alcohol","dry_x_fragrance","oily_x_oil","irritant_load","new","limited_edition","online_only","sephora_exclusive"]
y=r["mismatch_any"]
tr=r[r["split"]=="train"]; te=r[r["split"]=="test"]
print("train",len(tr),"test",len(te),"prev train",round(tr.mismatch_any.mean(),3),"prev test",round(te.mismatch_any.mean(),3))

def prep(scale):
    n = [("imp",SimpleImputer(strategy="median",add_indicator=True))]
    if scale: n.append(("sc",StandardScaler()))
    return ColumnTransformer([("n",Pipeline(n),num),("c",OneHotEncoder(handle_unknown="ignore",min_frequency=50),cat)])
models={
 "Logistic regression":Pipeline([("p",prep(True)),("m",LogisticRegression(C=0.1,max_iter=500,class_weight="balanced"))]),
 "Decision tree":Pipeline([("p",prep(False)),("m",DecisionTreeClassifier(max_depth=4,min_samples_leaf=200,class_weight="balanced",random_state=42))]),
 "Random forest":Pipeline([("p",prep(False)),("m",RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=100,n_jobs=-1,class_weight="balanced",random_state=42))]),
 "Gradient boosting":Pipeline([("p",prep(False)),("m",HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=150,l2_regularization=5,random_state=42))]),
}
X=lambda d:d[num+cat]
rows=[]
gkf=GroupKFold(n_splits=5)
for name,m in models.items():
    cv=cross_val_score(m,X(tr),tr["mismatch_any"],groups=tr["product_id"],cv=gkf,scoring="roc_auc",n_jobs=1)
    m.fit(X(tr),tr["mismatch_any"])
    p_tr=m.predict_proba(X(tr))[:,1]; p=m.predict_proba(X(te))[:,1]
    thr=np.quantile(p,0.80); pred=(p>=thr).astype(int)
    rows.append(dict(model=name,cv_auc=cv.mean(),cv_sd=cv.std(),train_auc=roc_auc_score(tr.mismatch_any,p_tr),
        test_auc=roc_auc_score(te.mismatch_any,p),test_ap=average_precision_score(te.mismatch_any,p),
        prec_top20=precision_score(te.mismatch_any,pred),rec_top20=recall_score(te.mismatch_any,pred),f1_top20=f1_score(te.mismatch_any,pred)))
res=pd.DataFrame(rows).set_index("model").round(3)
print(res.to_string())
print("baseline AP (prevalence) test",round(te.mismatch_any.mean(),3))
# baseline: skin type only
b=Pipeline([("p",ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore"),["skin_type"])])),("m",LogisticRegression())])
b.fit(tr[["skin_type"]],tr.mismatch_any); pb=b.predict_proba(te[["skin_type"]])[:,1]
print("skin-type-only AUC",round(roc_auc_score(te.mismatch_any,pb),3))
# year feature check: add year
# importance of LR coefficients
lr=models["Logistic regression"]; lr.fit(X(tr),tr.mismatch_any)
names=lr.named_steps["p"].get_feature_names_out()
co=pd.Series(lr.named_steps["m"].coef_[0],index=names).sort_values()
print(co.head(8).round(3).to_dict()); print(co.tail(8).round(3).to_dict())
# GB permutation importance on test
from sklearn.inspection import permutation_importance
gb=models["Gradient boosting"].fit(X(tr),tr.mismatch_any)
pi=permutation_importance(gb,X(te),te.mismatch_any,scoring="roc_auc",n_repeats=5,random_state=42,n_jobs=1)
imp=pd.Series(pi.importances_mean,index=num+cat).sort_values(ascending=False)
print(imp.head(10).round(4).to_dict())
# random split AUC (to show leakage by products/time)
from sklearn.model_selection import train_test_split
a,bb=train_test_split(r,test_size=0.2,random_state=42,stratify=r.mismatch_any)
g=models["Gradient boosting"].fit(X(a),a.mismatch_any)
print("random split GB AUC",round(roc_auc_score(bb.mismatch_any,g.predict_proba(X(bb))[:,1]),3))
res.to_csv("model_results.csv")