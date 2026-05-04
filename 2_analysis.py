"""
STEP 2 — ML Analysis: Churn Prediction + A/B Test Stats
Outputs: data/churn_scored.csv, data/model_metrics.csv,
         data/feature_importance.csv, data/ab_stats.csv, data/summary.json
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from scipy import stats

# ── LOAD ───────────────────────────────────────────────────────────
df = pd.read_csv('data/telecom_churn.csv')
ab = pd.read_csv('data/ab_test_results.csv')
print(f"Loaded: churn={len(df):,}  ab={len(ab):,}")

# ── FEATURE ENGINEERING ───────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['ChargePerMonth']  = df['TotalCharges'] / (df['Tenure'] + 1)
df['FiberNoSupport']  = ((df['InternetService']=='Fiber optic') & (df['TechSupport']=='No')).astype(int)
df['HighValue']       = (df['MonthlyCharges'] > 80).astype(int)
df['Churn_bin']       = (df['Churn']=='Yes').astype(int)

cat_cols = ['Gender','Partner','Dependents','PhoneService','InternetService','OnlineSecurity',
            'OnlineBackup','StreamingTV','TechSupport','Contract','PaperlessBilling','PaymentMethod']
le = LabelEncoder()
for c in cat_cols:
    df[c+'_enc'] = le.fit_transform(df[c].astype(str))

FEATURES = ['Tenure','MonthlyCharges','TotalCharges','ChargePerMonth','FiberNoSupport','HighValue',
            'SeniorCitizen','Contract_enc','InternetService_enc','TechSupport_enc',
            'OnlineSecurity_enc','PaymentMethod_enc','Partner_enc']

X = df[FEATURES]; y = df['Churn_bin']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

# ── TRAIN 4 MODELS ────────────────────────────────────────────────
MODELS = {
    'Logistic Regression': (LogisticRegression(max_iter=1000,random_state=42), True),
    'Random Forest':        (RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1), False),
    'Gradient Boosting':    (GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,random_state=42), False),
    'XGBoost':              (XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
                                           eval_metric='logloss',random_state=42,verbosity=0), False),
}
rows, best_name, best_auc, best_model = [], '', 0, None
for name,(model,scaled) in MODELS.items():
    Xtr,Xte = (X_tr_s,X_te_s) if scaled else (X_tr,X_te)
    model.fit(Xtr,y_tr); yp=model.predict(Xte); ypr=model.predict_proba(Xte)[:,1]
    auc=roc_auc_score(y_te,ypr)
    rows.append({'Model':name,'Accuracy':round(accuracy_score(y_te,yp),4),
                 'Precision':round(precision_score(y_te,yp),4),'Recall':round(recall_score(y_te,yp),4),
                 'F1':round(f1_score(y_te,yp),4),'AUC_ROC':round(auc,4)})
    if auc > best_auc: best_auc,best_name,best_model,best_scaled = auc,name,model,scaled
    print(f"  {name:28s} AUC={auc:.4f}  Acc={accuracy_score(y_te,yp):.4f}")

pd.DataFrame(rows).to_csv('data/model_metrics.csv',index=False)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────
xgb = [m for n,(m,_) in MODELS.items() if 'XGBoost' in n][0]
fi = pd.DataFrame({'Feature':FEATURES,'Importance':xgb.feature_importances_}).sort_values('Importance',ascending=False)
fi.to_csv('data/feature_importance.csv',index=False)

# ── RISK SCORING ──────────────────────────────────────────────────
best_X = sc.transform(X) if best_scaled else X
df['ChurnProbability'] = best_model.predict_proba(best_X)[:,1]
df['RiskLevel'] = pd.cut(df['ChurnProbability'],[0,0.35,0.65,1.0],labels=['Low','Medium','High'])
df.to_csv('data/churn_scored.csv',index=False)
print(f"\nRisk: High={df['RiskLevel'].eq('High').sum():,}  Medium={df['RiskLevel'].eq('Medium').sum():,}  Low={df['RiskLevel'].eq('Low').sum():,}")

# ── A/B HYPOTHESIS TEST ───────────────────────────────────────────
ctrl=ab[ab.Variant=='Control_A']; var=ab[ab.Variant=='Variant_B']
nA,cA=len(ctrl),ctrl.Converted.sum(); nB,cB=len(var),var.Converted.sum()
pA,pB=cA/nA,cB/nB; lift=(pB-pA)/pA
pp=(cA+cB)/(nA+nB); se=np.sqrt(pp*(1-pp)*(1/nA+1/nB))
z=(pB-pA)/se; pval=1-stats.norm.cdf(z)
ci_lo,ci_hi=(pB-pA)-1.96*se,(pB-pA)+1.96*se

seg_rows=[]
for seg,mask in [('Mobile',ab.Device=='Mobile'),('Desktop',ab.Device=='Desktop'),
                 ('New Visitors',ab.UserType=='New'),('Returning',ab.UserType=='Returning'),
                 ('High-Risk Churners',ab.HighRiskChurner==1)]:
    sc2=ab[mask&(ab.Variant=='Control_A')]; sv=ab[mask&(ab.Variant=='Variant_B')]
    if len(sc2)<50: continue
    pc,pv=sc2.Converted.mean(),sv.Converted.mean()
    pp2=(sc2.Converted.sum()+sv.Converted.sum())/(len(sc2)+len(sv))
    se2=np.sqrt(pp2*(1-pp2)*(1/len(sc2)+1/len(sv)))
    zs=(pv-pc)/se2 if se2>0 else 0; pvs=1-stats.norm.cdf(zs)
    seg_rows.append({'Segment':seg,'Control_CVR':round(pc,4),'Variant_CVR':round(pv,4),
                     'Lift':round((pv-pc)/pc,4),'p_value':round(pvs,4),'Significant':pvs<0.05})
pd.DataFrame(seg_rows).to_csv('data/ab_stats.csv',index=False)

summary={
    'churn_rate':round(df.Churn_bin.mean(),4),'total':50000,'churned':int(df.Churn_bin.sum()),
    'retained':int((df.Churn_bin==0).sum()),
    'risk_high':int(df.RiskLevel.eq('High').sum()),'risk_med':int(df.RiskLevel.eq('Medium').sum()),
    'risk_low':int(df.RiskLevel.eq('Low').sum()),
    'best_model':best_name,'best_auc':round(best_auc,4),
    'acc':round(rows[[r['Model'] for r in rows].index(best_name)]['Accuracy'],4) if best_name in [r['Model'] for r in rows] else 0,
    'prec':round(rows[[r['Model'] for r in rows].index(best_name)]['Precision'],4) if best_name in [r['Model'] for r in rows] else 0,
    'rec':round(rows[[r['Model'] for r in rows].index(best_name)]['Recall'],4) if best_name in [r['Model'] for r in rows] else 0,
    'pA':round(pA,4),'pB':round(pB,4),'lift':round(lift,4),
    'z':round(z,3),'pval':round(pval,6),'ci_lo':round(ci_lo,4),'ci_hi':round(ci_hi,4),
    'mtm_churn':round(df[df.Contract=='Month-to-Month'].Churn_bin.mean(),4),
    'twyr_churn':round(df[df.Contract=='Two Year'].Churn_bin.mean(),4),
}
with open('data/summary.json','w') as f: json.dump(summary,f,indent=2)
print(f"\n[OK] A/B: Control={pA:.2%}  Variant={pB:.2%}  Lift={lift:.1%}  p={pval:.4f}")
print(f"[OK] summary.json saved"); print("=== ANALYSIS COMPLETE ===")
