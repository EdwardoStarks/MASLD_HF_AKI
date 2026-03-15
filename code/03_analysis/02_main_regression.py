"""
02_main_regression_v2.py
主效应回归分析（修订版）：
  主要结局：AKI、院内死亡、30天再入院
  暴露：MASLD（主定义）+ FIB-4分层
  次要分析：DE阴性结果保留报告
输出：
    output/tables/table2_outcomes_regression.csv
    output/tables/table2_fib4_aki.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT  = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW      = f"{PROJECT}/data/raw/cohort_mimic_raw.csv"
PSM_ALL  = f"{PROJECT}/data/final/cohort_psm_all.csv"   # 全队列PSM（结局分析）
OUT1     = f"{PROJECT}/output/tables/table2_outcomes_regression.csv"
OUT2     = f"{PROJECT}/output/tables/table2_fib4_aki.csv"

df_raw = pd.read_csv(RAW)
df_raw['aki']            = df_raw['aki_flag']
df_raw['hospital_death'] = df_raw['hospital_expire_flag']
df_raw['readmission_30'] = df_raw['readmit_30d']
df_raw['gender_num'] = (df_raw['gender'] == 'M').astype(int)
df_raw['log_fib4']   = np.log1p(df_raw['fib4'])
df_raw['log_nlr']    = np.log1p(df_raw['nlr'])
df_raw['log_alt']    = np.log1p(df_raw['alt_final'])



# FIB-4分层
df_raw['fib4_group'] = pd.cut(
    df_raw['fib4'],
    bins=[0, 1.30, 2.67, np.inf],
    labels=[0, 1, 2]   # 0=低, 1=中, 2=高
).astype('Int64')

# FIB-4哑变量（参照：非MASLD）
df_raw['masld_fib4_low']  = ((df_raw['masld_main']==1) & (df_raw['fib4_group'].fillna(-1)==0)).astype(int)
df_raw['masld_fib4_mid']  = ((df_raw['masld_main']==1) & (df_raw['fib4_group'].fillna(-1)==1)).astype(int)
df_raw['masld_fib4_high'] = ((df_raw['masld_main']==1) & (df_raw['fib4_group'].fillna(-1)==2)).astype(int)

# 尝试读取全队列PSM文件（若不存在则用原始队列+备注）
try:
    df_psm = pd.read_csv(PSM_ALL)
    df_psm['aki']            = df_psm['aki_flag']
    df_psm['hospital_death'] = df_psm['hospital_expire_flag']
    df_psm['readmission_30'] = df_psm['readmit_30d']
    df_psm['gender_num'] = (df_psm['gender'] == 'M').astype(int)
    df_psm['log_fib4']   = np.log1p(df_psm['fib4'])
    df_psm['log_nlr']    = np.log1p(df_psm['nlr'])
    df_psm['fib4_group'] = pd.cut(
        df_psm['fib4'], bins=[0,1.30,2.67,np.inf], labels=[0,1,2]
    ).astype('Int64')
    df_psm['masld_fib4_low']  = ((df_psm['masld_main']==1) & (df_psm['fib4_group'].fillna(-1)==0)).astype(int)
    df_psm['masld_fib4_mid']  = ((df_psm['masld_main']==1) & (df_psm['fib4_group'].fillna(-1)==1)).astype(int)
    df_psm['masld_fib4_high'] = ((df_psm['masld_main']==1) & (df_psm['fib4_group'].fillna(-1)==2)).astype(int)
    psm_available = True
    print(f"全队列PSM数据集: {len(df_psm):,} 例")
except:
    psm_available = False
    print("⚠️  cohort_psm_all.csv未找到，结局分析使用原始队列多因素调整")

print(f"原始队列: {len(df_raw):,} 例  MASLD: {df_raw['masld_main'].sum():.0f}")

# ============================================================
# 协变量
# ============================================================
COVARS_BASE = [
    'age', 'gender_num',
    'dm_t2', 'hypertension', 'dyslipidemia', 'ckd',
    'afib', 'copd_asthma', 'cad',
    'hfref', 'hfpef',
    'egfr', 'sodium_final', 'hemoglobin_final',
]

COVARS_FULL = COVARS_BASE + ['albumin_final', 'log_fib4', 'log_nlr', 'bmi']

BASE_COVARS = COVARS_BASE  # 保持后续代码兼容

# ── 插入验证代码 ────────────────────────────────────────
print("=== 协变量缺失情况（Model A，n=11,517）===")
check_cols = COVARS_BASE + ['aki']
print(df_raw[check_cols].isnull().sum().to_string())
print(f"\n任一协变量缺失的总人数: {df_raw[check_cols].isnull().any(axis=1).sum()}")

print("\n=== FIB-4原始成分缺失情况（MASLD组，n=1,114）===")
masld = df_raw[df_raw['masld_main'] == 1]
fib4_components = ['age', 'ast_final', 'alt_final', 'platelet_final']
for col in fib4_components:
    if col in masld.columns:
        print(f"  {col}: {masld[col].isnull().sum()} 缺失")
print(f"  fib4直接缺失: {masld['fib4'].isnull().sum()}")
print(f"  fib4_group缺失(即NaN): {masld['fib4_group'].isna().sum()}")
# ── 验证代码结束 ────────────────────────────────────────

def avail(df, cols):
    return [c for c in cols if c in df.columns]

def clean(df, cols):
    return df.dropna(subset=cols).copy()

def fmt_logistic(model, var):
    if var not in model.params: return 'N/A'
    coef = model.params[var]
    ci   = model.conf_int().loc[var]
    p    = model.pvalues[var]
    or_  = np.exp(coef)
    p_s  = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"OR={or_:.2f} ({np.exp(ci[0]):.2f}~{np.exp(ci[1]):.2f}), P={p_s}"

def fmt_linear(model, var):
    if var not in model.params: return 'N/A'
    coef = model.params[var]
    ci   = model.conf_int().loc[var]
    p    = model.pvalues[var]
    p_s  = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"β={coef:.3f} ({ci[0]:.3f}~{ci[1]:.3f}), P={p_s}"

rows = []

# ============================================================
# PART A：MASLD对三大结局的影响
# ============================================================
OUTCOMES = [
    ('aki',            'AKI',                'logistic'),
    ('hospital_death', 'In-hospital death',  'logistic'),
    ('readmission_30', '30-day readmission', 'logistic'),
]

for outcome, label, mtype in OUTCOMES:
    if outcome not in df_raw.columns:
        print(f"⚠️  列不存在: {outcome}")
        continue

    print(f"\n{'='*55}")
    print(f"  {label}（{outcome}）")
    print(f"{'='*55}")

    # 描述性
    m_grp = df_raw.groupby('masld_main')[outcome].agg(['sum','count','mean'])
    for g, row_d in m_grp.iterrows():
        tag = 'MASLD' if g==1 else '非MASLD'
        print(f"  {tag}: {int(row_d['sum'])}/{int(row_d['count'])} ({row_d['mean']*100:.1f}%)")

    # 单因素
    df_m = clean(df_raw, [outcome, 'masld_main'])
    df_m = df_m[df_m[outcome].notna()]
    m_uni = smf.logit(f'{outcome} ~ masld_main', data=df_m).fit(disp=0)
    r = fmt_logistic(m_uni, 'masld_main')
    print(f"  单因素:          {r}  N={len(df_m)}")
    rows.append({'Outcome': label, 'Model': 'Univariable',          'Result': r, 'N': len(df_m)})

    # Model A 基础
    covs_a = avail(df_raw, COVARS_BASE)
    df_m = clean(df_raw, [outcome,'masld_main'] + covs_a)
    df_m = df_m[df_m[outcome].notna()]
    m_a = smf.logit(f'{outcome} ~ masld_main + ' + ' + '.join(covs_a), data=df_m).fit(disp=0)
    r = fmt_logistic(m_a, 'masld_main')
    print(f"  Model A(基础):   {r}  N={len(df_m)}")
    rows.append({'Outcome': label, 'Model': 'Model A (base)',        'Result': r, 'N': len(df_m)})

    # Model B 完整
    covs_b = avail(df_raw, COVARS_FULL)
    df_m = clean(df_raw, [outcome,'masld_main'] + covs_b)
    df_m = df_m[df_m[outcome].notna()]
    m_b = smf.logit(f'{outcome} ~ masld_main + ' + ' + '.join(covs_b), data=df_m).fit(disp=0)
    r = fmt_logistic(m_b, 'masld_main')
    print(f"  Model B(完整):   {r}  N={len(df_m)}")
    rows.append({'Outcome': label, 'Model': 'Model B (full)',        'Result': r, 'N': len(df_m)})

    # 多因素（PSM后）
    if psm_available and outcome in df_psm.columns:
        covs_p = avail(df_psm, BASE_COVARS)
        df_m = clean(df_psm, [outcome,'masld_main'] + covs_p)
        df_m = df_m[df_m[outcome].notna()]
        formula = f'{outcome} ~ masld_main + ' + ' + '.join(covs_p)
        m_psm = smf.logit(formula, data=df_m).fit(disp=0)
        r = fmt_logistic(m_psm, 'masld_main')
        rows.append({'Outcome': label, 'Model': 'Multivariable (PSM)',   'Result': r, 'N': len(df_m)})

# ============================================================
# PART B：FIB-4分层对AKI的剂量-反应
# ============================================================
print(f"\n{'='*55}")
print("  FIB-4分层 × MASLD → AKI（剂量-反应）")
print(f"{'='*55}")

# 描述
print("\n  各组AKI发生率：")
grp_labels = {-1:'Non-MASLD', 0:'MASLD-FIB4 low risk', 1:'MASLD-FIB4 intermediate risk', 2:'MASLD-FIB4 high risk'}
df_raw['fib4_grp_4cat'] = -1  # 默认非MASLD
df_raw.loc[df_raw['masld_fib4_low']==1,  'fib4_grp_4cat'] = 0
df_raw.loc[df_raw['masld_fib4_mid']==1,  'fib4_grp_4cat'] = 1
df_raw.loc[df_raw['masld_fib4_high']==1, 'fib4_grp_4cat'] = 2

# fib4_rows = []
# for g in [-1, 0, 1, 2]:
#     if g == -1:
#         sub = df_m[df_m['fib4_grp_4cat'] == -1]  # 用dropna后的df_m，N=10436
#     else:
#         sub = df_raw[df_raw['fib4_grp_4cat'] == g]  # MASLD三组用df_raw
#     aki_r = sub['aki'].mean() * 100 if 'aki' in sub else np.nan
#     n = len(sub)
#     lbl = grp_labels[g]
#     print(f"  {lbl}: N={n}  AKI率={aki_r:.1f}%")
#     fib4_rows.append({'Group': lbl, 'N': n, 'AKI rate (%)': round(aki_r, 1)})

# 多因素回归（FIB-4三哑变量，参照非MASLD）
print("\n  多因素Logistic（参照：非MASLD）：")
covs = avail(df_raw, BASE_COVARS)
# 去掉log_fib4避免多重共线性（FIB-4已通过哑变量进入）
covs_no_fib4 = [c for c in covs if c != 'log_fib4']
formula_fib4 = ('aki ~ masld_fib4_low + masld_fib4_mid + masld_fib4_high + '
                + ' + '.join(covs_no_fib4))
df_m = clean(df_raw, ['aki','masld_fib4_low','masld_fib4_mid','masld_fib4_high']
             + covs_no_fib4)
df_m = df_m[df_m['aki'].notna()]
df_m['fib4_grp_4cat'] = df_raw.loc[df_m.index, 'fib4_grp_4cat']  # 同步分组标签

fib4_rows = []
for g in [-1, 0, 1, 2]:
    if g == -1:
        sub = df_m[df_m['fib4_grp_4cat'] == -1]  # 用dropna后的df_m，N=10436
    else:
        sub = df_raw[df_raw['fib4_grp_4cat'] == g]  # MASLD三组用df_raw
    aki_r = sub['aki'].mean() * 100 if 'aki' in sub else np.nan
    n = len(sub)
    lbl = grp_labels[g]
    print(f"  {lbl}: N={n}  AKI率={aki_r:.1f}%")
    fib4_rows.append({'Group': lbl, 'N': n, 'AKI rate (%)': round(aki_r, 1)})

# ↓ 插入这两行验证代码
print(f"  Model A 总样本: {len(df_m)}")
print(f"  Non-MASLD N (dropna后): {(df_m['fib4_grp_4cat']==-1).sum()}")

m_fib4 = smf.logit(formula_fib4, data=df_m).fit(disp=0)

for var, label in [
    ('masld_fib4_low',  'MASLD FIB4<1.30'),
    ('masld_fib4_mid',  'MASLD FIB4 1.30-2.67'),
    ('masld_fib4_high', 'MASLD FIB4≥2.67'),
]:
    r = fmt_logistic(m_fib4, var)
    print(f"  {label:<25}: {r}")
    fib4_rows.append({'Group': label, 'Multivariable OR': r})

# 趋势检验（MASLD组内FIB-4连续值对AKI）
print("\n  趋势检验（MASLD组内FIB-4连续值→AKI）：")
df_trend = df_raw[df_raw['masld_main']==1].copy()
df_trend = df_trend[df_trend['aki'].notna() & df_trend['fib4'].notna()]
trend_covs = [c for c in covs_no_fib4 if c in df_trend.columns]
df_trend_m = clean(df_trend, ['aki','log_fib4'] + trend_covs)
m_trend = smf.logit('aki ~ log_fib4 + ' + ' + '.join(trend_covs),
                    data=df_trend_m).fit(disp=0)
r = fmt_logistic(m_trend, 'log_fib4')
print(f"  FIB-4(log)→AKI（MASLD组内）: {r}  N={len(df_trend_m)}")
rows.append({'Outcome': 'AKI', 'Model': 'Trend test (log-FIB4 continuous, MASLD only)', 'Result': r, 'N': len(df_trend_m)})

# ============================================================
# PART C：复合终点（死亡 OR AKI OR 再入院）
# ============================================================
print(f"\n{'='*55}")
print("  复合终点（死亡/AKI/30天再入院）")
print(f"{'='*55}")

outcome_cols = ['hospital_death','aki','readmission_30']
avail_outcomes = [c for c in outcome_cols if c in df_raw.columns]
if len(avail_outcomes) >= 2:
    df_raw['composite'] = df_raw[avail_outcomes].max(axis=1)
    m_grp = df_raw.groupby('masld_main')['composite'].agg(['sum','count','mean'])
    for g, row_d in m_grp.iterrows():
        tag = 'MASLD' if g==1 else '非MASLD'
        print(f"  {tag}: {int(row_d['sum'])}/{int(row_d['count'])} ({row_d['mean']*100:.1f}%)")

    covs = avail(df_raw, BASE_COVARS)
    df_m = clean(df_raw, ['composite','masld_main'] + covs)
    df_m = df_m[df_m['composite'].notna()]
    m_comp = smf.logit('composite ~ masld_main + ' + ' + '.join(covs),
                       data=df_m).fit(disp=0)
    r = fmt_logistic(m_comp, 'masld_main')
    print(f"  多因素: {r}  N={len(df_m)}")
    rows.append({'Outcome': 'Composite endpoint', 'Model': 'Multivariable (unmatched)', 'Result': r, 'N': len(df_m)})

# ============================================================
# PART D：敏感性分析（不同MASLD定义）
# ============================================================
print(f"\n{'='*55}")
print("  敏感性分析：不同MASLD定义对AKI的影响")
print(f"{'='*55}")

for masld_col, label in [
    ('masld_icd_this', 'Current ICD only'),
    ('masld_icd_ever', 'Historical ICD'),
    ('masld_lab',      'Laboratory-based'),
    ('masld_main',     'Three-tier union'),
]:
    if masld_col not in df_raw.columns: continue
    covs = avail(df_raw, BASE_COVARS)
    df_m = clean(df_raw, ['aki', masld_col] + covs)
    df_m = df_m[df_m['aki'].notna()]
    try:
        m = smf.logit(f'aki ~ {masld_col} + ' + ' + '.join(covs),
                      data=df_m).fit(disp=0)
        r = fmt_logistic(m, masld_col)
        print(f"  {label:<15}: {r}  N={len(df_m)}")
        rows.append({'Outcome': 'AKI', 'Model': f'Sensitivity-{label}', 'Result': r, 'N': len(df_m)})
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# ============================================================
# 保存
# ============================================================
pd.DataFrame(rows).to_csv(OUT1, index=False, encoding='utf-8-sig')
pd.DataFrame(fib4_rows).to_csv(OUT2, index=False, encoding='utf-8-sig')
print(f"\n  已保存:\n  {OUT1}\n  {OUT2}")
print("  下一步: 运行 03_rcs_analysis.py（FIB-4→AKI RCS曲线）")
