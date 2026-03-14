"""
06_eicu_validation.py
eICU外部验证：FIB-4≥2.67 → AKI
输出：
    output/tables/table7_eicu_validation.csv
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
EICU    = f"{PROJECT}/data/raw/cohort_eicu_raw.csv"
OUT     = f"{PROJECT}/output/tables/table7_eicu_validation.csv"

df = pd.read_csv(EICU)
df['gender_num'] = (df['gender'].str.lower() == 'male').astype(int)
df['age_num']    = pd.to_numeric(df['age'], errors='coerce')
df['fib4_high']  = (df['fib4'] >= 2.67).astype('Int64')
df['aki']        = df['aki_flag']

print(f"eICU总队列: {len(df):,}")
print(f"MASLD: {df['masld_lab'].sum():,}")
print(f"FIB-4可计算: {df['fib4'].notna().sum():,}")

# ============================================================
# 仅MASLD组
# ============================================================
df_m = df[df['masld_lab'] == 1].copy()
print(f"\nMASLD组: {len(df_m):,}")
print(f"FIB-4高风险(≥2.67): {df_m['fib4_high'].sum():,}")
print(f"AKI: {df_m['aki'].sum():,} ({df_m['aki'].mean()*100:.1f}%)")

# FIB-4分层AKI率
print("\nFIB-4分层AKI率:")
for label, mask in [
    ('FIB-4 <2.67',  df_m['fib4_high']==0),
    ('FIB-4 >=2.67', df_m['fib4_high']==1),
]:
    sub = df_m[mask]
    n   = sub['aki'].notna().sum()
    r   = sub['aki'].mean() * 100
    print(f"  {label}: N={n}  AKI={r:.1f}%")

# ============================================================
# 多因素回归（与MIMIC-IV相同协变量，取可用子集）
# ============================================================
COVARS = ['age_num', 'gender_num', 'dm', 'hypertension', 'ckd']
covs   = [c for c in COVARS if c in df_m.columns]
print(f"\n可用协变量: {covs}")

rows = []

# 单因素
d = df_m.dropna(subset=['aki','fib4_high'])
m_uni = smf.logit('aki ~ fib4_high', data=d).fit(disp=0)
coef  = m_uni.params['fib4_high']
ci    = m_uni.conf_int().loc['fib4_high']
p     = m_uni.pvalues['fib4_high']
p_s   = "<0.001" if p < 0.001 else f"{p:.3f}"
print(f"\n单因素: OR={np.exp(coef):.2f}({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) P={p_s} N={len(d)}")
rows.append({'Model':'Univariate','OR':round(np.exp(coef),2),
             'CI_low':round(np.exp(ci[0]),2),'CI_high':round(np.exp(ci[1]),2),
             'P':round(p,3),'N':len(d)})

# 多因素
d = df_m.dropna(subset=['aki','fib4_high']+covs)
# 检查各协变量方差
print("\n协变量方差检查:")
for c in covs:
    print(f"  {c}: unique={d[c].nunique()}  mean={d[c].mean():.3f}")
# 移除无方差的列
covs = [c for c in covs if d[c].nunique() > 1]
print(f"有效协变量: {covs}")
formula = 'aki ~ fib4_high + ' + ' + '.join(covs)
formula = 'aki ~ fib4_high + ' + ' + '.join(covs)
m_mul = smf.logit(formula, data=d).fit(disp=0)
coef  = m_mul.params['fib4_high']
ci    = m_mul.conf_int().loc['fib4_high']
p     = m_mul.pvalues['fib4_high']
p_s   = "<0.001" if p < 0.001 else f"{p:.3f}"
print(f"多因素: OR={np.exp(coef):.2f}({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) P={p_s} N={len(d)}")
rows.append({'Model':'Multivariable','OR':round(np.exp(coef),2),
             'CI_low':round(np.exp(ci[0]),2),'CI_high':round(np.exp(ci[1]),2),
             'P':round(p,3),'N':len(d)})

# ============================================================
# 与MIMIC-IV结果对比
# ============================================================
print("\n" + "="*50)
print("  MIMIC-IV vs eICU 结果对比")
print("="*50)
print(f"  MIMIC-IV: FIB-4≥2.67 OR=1.63(1.36-1.95) P<0.001")
print(f"  eICU:     FIB-4≥2.67 OR={rows[-1]['OR']}({rows[-1]['CI_low']}-{rows[-1]['CI_high']}) P={rows[-1]['P']}")

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\n已保存: {OUT}")
print("外部验证完成！")