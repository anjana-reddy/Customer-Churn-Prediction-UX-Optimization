"""
STEP 1 — Generate Datasets
Outputs: data/telecom_churn.csv, data/ab_test_results.csv
"""
import numpy as np, pandas as pd
np.random.seed(42); N = 50000

# ── CHURN DATASET ─────────────────────────────────────────────────
contract = np.random.choice(['Month-to-Month','One Year','Two Year'], N, p=[0.456,0.304,0.240])
tenure   = np.where(contract=='Month-to-Month', np.random.exponential(15,N).clip(1,72).astype(int),
           np.where(contract=='One Year',       np.random.normal(28,10,N).clip(12,48).astype(int),
                                                np.random.normal(48,12,N).clip(24,72).astype(int)))
internet = np.random.choice(['Fiber optic','DSL','No'], N, p=[0.44,0.34,0.22])
monthly  = (np.where(internet=='Fiber optic',75,np.where(internet=='DSL',55,30)) + np.random.normal(0,10,N)).clip(18,119)
total    = (monthly * tenure + np.random.normal(0,50,N)).clip(18)
tech_sup = np.where(contract=='Two Year',
           np.random.choice(['Yes','No'],N,p=[0.65,0.35]),
           np.random.choice(['Yes','No'],N,p=[0.30,0.70]))
security = np.random.choice(['Yes','No','No internet service'],N,p=[0.28,0.50,0.22])
payment  = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'],N,p=[0.34,0.23,0.22,0.21])
phone    = np.random.choice(['Yes','No'],N,p=[0.90,0.10])
gender   = np.random.choice(['Male','Female'],N)
senior   = np.random.choice([0,1],N,p=[0.84,0.16])
partner  = np.random.choice(['Yes','No'],N,p=[0.48,0.52])
depends  = np.random.choice(['Yes','No'],N,p=[0.30,0.70])
paperless= np.random.choice(['Yes','No'],N,p=[0.59,0.41])
backup   = np.random.choice(['Yes','No','No internet service'],N,p=[0.34,0.44,0.22])
streaming= np.random.choice(['Yes','No','No internet service'],N,p=[0.38,0.40,0.22])

score = (np.where(contract=='Month-to-Month',-0.30,np.where(contract=='One Year',-1.65,-3.20))
       + np.where(internet=='Fiber optic',0.18,np.where(internet=='DSL',0.05,-0.10))
       + np.where(tech_sup=='No',0.12,-0.05)
       + np.where(security=='No',0.08,0.0)
       + np.where(payment=='Electronic check',0.10,0.0)
       + (-0.009*tenure) + (0.0014*monthly) + np.random.normal(0,0.9,N))
churn_prob = 1/(1+np.exp(-score))
churn = (np.random.random(N) < churn_prob).astype(int)

df = pd.DataFrame({
    'CustomerID':['CUST-'+str(i).zfill(5) for i in range(1,N+1)],
    'Gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':depends,
    'Tenure':tenure,'PhoneService':phone,'InternetService':internet,
    'OnlineSecurity':security,'OnlineBackup':backup,'StreamingTV':streaming,
    'TechSupport':tech_sup,'Contract':contract,'PaperlessBilling':paperless,
    'PaymentMethod':payment,'MonthlyCharges':monthly.round(2),
    'TotalCharges':total.round(2),'Churn':np.where(churn==1,'Yes','No')
})
df.to_csv('data/telecom_churn.csv', index=False)
print(f"[OK] telecom_churn.csv — {N:,} rows | Churn: {churn.mean():.1%}")

# ── A/B TEST DATASET ──────────────────────────────────────────────
M = 29000
variant  = np.random.choice(['Control_A','Variant_B'],M)
days     = np.random.randint(1,21,M)
device   = np.random.choice(['Mobile','Desktop','Tablet'],M,p=[0.427,0.400,0.173])
usertype = np.random.choice(['New','Returning'],M,p=[0.626,0.374])
pages    = (np.random.poisson(4,M)+1).clip(1,20)
duration = np.random.exponential(180,M).clip(10,1200).round(0).astype(int)
cvr      = np.where(variant=='Control_A',0.113,0.134)
cvr     += np.where((device=='Mobile')&(variant=='Variant_B'),0.014,0.0)
cvr     += np.where((usertype=='New')&(variant=='Variant_B'),0.005,0.0)
converted= (np.random.random(M) < cvr).astype(int)
revenue  = np.where(converted==1, np.random.normal(89,22,M).clip(20,250).round(2), 0.0)
highrisk = np.random.choice([0,1],M,p=[0.936,0.064])

ab = pd.DataFrame({
    'VisitorID':['VIS-'+str(i).zfill(5) for i in range(1,M+1)],
    'Variant':variant,'Day':days,'UserType':usertype,'Device':device,
    'PagesViewed':pages,'SessionDuration_sec':duration,
    'Converted':converted,'Revenue':revenue,'HighRiskChurner':highrisk
})
ab.to_csv('data/ab_test_results.csv', index=False)
c=ab[ab.Variant=='Control_A']; v=ab[ab.Variant=='Variant_B']
print(f"[OK] ab_test_results.csv — {M:,} rows | Control: {c.Converted.mean():.2%} | Variant: {v.Converted.mean():.2%} | Lift: {(v.Converted.mean()-c.Converted.mean())/c.Converted.mean():.1%}")
