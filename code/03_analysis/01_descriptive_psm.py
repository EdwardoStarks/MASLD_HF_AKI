"""
01_descriptive_psm.py
基线描述统计 + 倾向评分匹配（PSM 1:2）
输出：
    output/tables/table1_baseline.csv
    data/final/cohort_psm.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW     = f"{PROJECT}/data/raw/cohort_mimic_raw.csv"
FINAL   = f"{PROJECT}/data/final/cohort_psm.csv"
TABLE1  = f"{PROJECT}/output/tables/table1_baseline.csv"

# ============================================================
# 加载数据
# ============================================================
df = pd.read_csv(RAW)
print(f"总样本: {len(df):,}  MASLD: {df['masld_main'].sum():.0f}")

# DE分析子集（有DE值）
df_de = df[df['de_value'].notna()].copy()
print(f"DE分析子集: {len(df_de):,}  MASLD: {df_de['masld_main'].sum():.0f}")

# ============================================================
# PSM变量定义
# ============================================================
PSM_VARS = [
    'age', 'gender_num',
    'dm_t2', 'hypertension', 'dyslipidemia', 'ckd',
    'afib', 'copd_asthma', 'cad',
    'hfref', 'hfpef', 'obesity_icd',
    'egfr', 'sodium_final', 'hemoglobin_final',
]

BASELINE_VARS = {
    'continuous': [
        ('age',              'Age (years)'),
        ('bmi',              'BMI (kg/m²)'),
        ('egfr',             'eGFR (mL/min/1.73m²)'),
        ('albumin_final',    'Albumin (g/dL)'),
        ('creatinine_final', 'Creatinine (mg/dL)'),
        ('sodium_final',     'Sodium (mEq/L)'),
        ('potassium_final',  'Potassium (mEq/L)'),
        ('hemoglobin_final', 'Haemoglobin (g/dL)'),
        ('alt_final',        'ALT (U/L)'),
        ('ast_final',        'AST (U/L)'),
        ('bilirubin_final',  'Total bilirubin (mg/dL)'),
        ('inr_final',        'INR'),
        ('fib4',             'FIB-4'),
        ('nlr',              'NLR'),
        ('sii',              'SII'),
        ('furo_eq_72h',      'Furosemide equivalent dose 72h (mg)'),
        ('de_value',         'Diuretic efficiency (mL/40mg)'),
    ],
    'categorical': [
        ('gender_num',           'Male, %'),
        ('hfref',                'HFrEF, %'),
        ('hfpef',                'HFpEF, %'),
        ('dm_t2',                'Type 2 diabetes, %'),
        ('hypertension',         'Hypertension, %'),
        ('dyslipidemia',         'Dyslipidaemia, %'),
        ('ckd',                  'CKD, %'),
        ('afib',                 'Atrial fibrillation, %'),
        ('copd_asthma',          'COPD/asthma, %'),
        ('cad',                  'Coronary artery disease, %'),
        ('obesity_icd',          'Obesity (ICD-coded), %'),
        ('dr_1400',              'Diuretic resistance (DE <1400), %'),
        ('hospital_expire_flag', 'In-hospital mortality, %'),
        ('readmit_30d',          '30-day readmission, %'),
        ('aki_flag',             'AKI, %'),
    ]
}

# ============================================================
# 数据预处理
# ============================================================
def preprocess(df):
    df = df.copy()
    df['gender_num'] = (df['gender'] == 'M').astype(int)
    # 对数变换右偏变量
    for col in ['alt_final','ast_final','fib4','nlr','sii',
                'bilirubin_final','furo_eq_72h','de_value']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df

df    = preprocess(df)
df_de = preprocess(df_de)

# ============================================================
# Table 1：基线描述统计
# ============================================================
def describe_var(df, col, is_cat, group_col='masld_main'):
    g0 = df[df[group_col]==0][col].dropna()
    g1 = df[df[group_col]==1][col].dropna()
    all_ = df[col].dropna()

    if is_cat:
        r0   = f"{g0.sum():.0f} ({g0.mean()*100:.1f}%)"
        r1   = f"{g1.sum():.0f} ({g1.mean()*100:.1f}%)"
        rall = f"{all_.sum():.0f} ({all_.mean()*100:.1f}%)"
        _, p = stats.chi2_contingency(
            pd.crosstab(df[group_col], df[col].fillna(0))
        )[:2]
    else:
        # 正态性检验决定用均值还是中位数
        _, p_norm = stats.shapiro(g1.sample(min(50,len(g1)), random_state=42))
        if p_norm > 0.05:
            r0   = f"{g0.mean():.1f} ± {g0.std():.1f}"
            r1   = f"{g1.mean():.1f} ± {g1.std():.1f}"
            rall = f"{all_.mean():.1f} ± {all_.std():.1f}"
            _, p = stats.ttest_ind(g0, g1)
        else:
            r0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}, {g0.quantile(.75):.1f}]"
            r1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}, {g1.quantile(.75):.1f}]"
            rall = f"{all_.median():.1f} [{all_.quantile(.25):.1f}, {all_.quantile(.75):.1f}]"
            _, p = stats.mannwhitneyu(g0, g1)

    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return r0, r1, rall, p_str

def build_table1(df, title="全队列基线"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    n0 = (df['masld_main']==0).sum()
    n1 = (df['masld_main']==1).sum()
    print(f"  非MASLD n={n0:,}  MASLD n={n1:,}")

    rows = []
    for col, label in BASELINE_VARS['continuous']:
        if col not in df.columns: continue
        r0,r1,rall,p = describe_var(df, col, False)
        # 改后
        rows.append({'Variable': label, 'Overall': rall, 'Non-MASLD': r0, 'MASLD': r1, 'P value': p})

    for col, label in BASELINE_VARS['categorical']:
        if col not in df.columns: continue
        r0,r1,rall,p = describe_var(df, col, True)
        rows.append({'Variable': label, 'Overall': rall, 'Non-MASLD': r0, 'MASLD': r1, 'P value': p})

    tb = pd.DataFrame(rows)
    print(tb.to_string(index=False))
    return tb

tb_full = build_table1(df, "Table 1A：全队列基线（n=11,517）")
tb_de   = build_table1(df_de, "Table 1B：DE分析子集基线（n=2,605）")

# ============================================================
# PSM：倾向评分匹配 1:2
# ============================================================
def run_psm(df, label="全队列"):
    print(f"\n{'='*50}")
    print(f"  PSM：{label}")

    df_psm = df.copy()

    # 填充PSM变量缺失值（均值填充，仅用于PS计算）
    psm_data = df_psm[PSM_VARS + ['masld_main','hadm_id']].copy()
    for col in PSM_VARS:
        if col in psm_data.columns:
            psm_data[col] = psm_data[col].fillna(psm_data[col].median())

    X = psm_data[PSM_VARS].values
    y = psm_data['masld_main'].values

    # 标准化 + Logistic回归计算PS
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    lr     = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_sc, y)
    ps     = lr.predict_proba(X_sc)[:, 1]
    df_psm['ps'] = ps

    # C统计量
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, ps)
    print(f"  PS模型C统计量: {auc:.3f}")

    # 1:2最近邻匹配（caliper=0.2*SD of logit PS）
    logit_ps = np.log(ps / (1 - ps + 1e-8))
    caliper  = 0.2 * np.std(logit_ps)
    print(f"  Caliper: {caliper:.4f}")

    treated  = df_psm[df_psm['masld_main']==1].copy()
    control  = df_psm[df_psm['masld_main']==0].copy()
    treated['logit_ps']  = logit_ps[df_psm['masld_main']==1]
    control['logit_ps']  = logit_ps[df_psm['masld_main']==0]
    control['matched']   = False

    matched_ctrl_idx = []
    used_ctrl = set()

    for _, t_row in treated.iterrows():
        t_lps = t_row['logit_ps']
        ctrl_avail = control[~control.index.isin(used_ctrl)].copy()
        ctrl_avail['dist'] = abs(ctrl_avail['logit_ps'] - t_lps)
        candidates = ctrl_avail[ctrl_avail['dist'] <= caliper].nsmallest(2, 'dist')
        for idx in candidates.index:
            matched_ctrl_idx.append(idx)
            used_ctrl.add(idx)

    matched_ctrl = control.loc[matched_ctrl_idx]
    matched_df   = pd.concat([treated, matched_ctrl], ignore_index=True)

    n_t  = len(treated)
    n_c  = len(matched_ctrl)
    print(f"  匹配后：MASLD={n_t}  对照={n_c}  比例=1:{n_c/n_t:.2f}")

    # 匹配质量：标准化均差（SMD）
    print(f"\n  匹配质量（SMD，目标<0.1）：")
    for col in ['age','bmi','egfr','dm_t2','hypertension','ckd','afib','hfref']:
        if col not in matched_df.columns: continue
        g0 = matched_df[matched_df['masld_main']==0][col].dropna()
        g1 = matched_df[matched_df['masld_main']==1][col].dropna()
        pooled_sd = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
        smd = abs(g1.mean() - g0.mean()) / (pooled_sd + 1e-8)
        flag = "✅" if smd < 0.1 else "⚠️"
        print(f"    {flag} {col:<20} SMD={smd:.3f}")

    return matched_df

# PSM on DE子集（主分析）
df_psm_de = run_psm(df_de, "DE分析子集（主分析）")

# PSM on 全队列（用于结局分析）
df_psm_all = run_psm(df, "全队列（结局分析）")

# ============================================================
# 保存
# ============================================================
# 主分析数据集：DE子集PSM后
df_psm_de.to_csv(FINAL, index=False, encoding='utf-8-sig')
print(f"\n  主分析数据集已保存: {FINAL}")
print(f"  行数: {len(df_psm_de):,}  列数: {len(df_psm_de.columns)}")

# 全队列PSM（用于结局分析）
all_psm_path = f"{PROJECT}/data/final/cohort_psm_all.csv"
df_psm_all.to_csv(all_psm_path, index=False, encoding='utf-8-sig')

# Table 1保存
tb_full.to_csv(TABLE1.replace('.csv','_full.csv'), index=False, encoding='utf-8-sig')
tb_de.to_csv(TABLE1.replace('.csv','_de_subset.csv'), index=False, encoding='utf-8-sig')

# PSM后的Table 1
tb_psm = build_table1(df_psm_de, "Table 1C：PSM匹配后基线（DE子集）")
tb_psm.to_csv(TABLE1.replace('.csv','_psm.csv'), index=False, encoding='utf-8-sig')

print("\n  所有Table 1文件已保存至 output/tables/")
print("  下一步：运行 02_main_regression.py")