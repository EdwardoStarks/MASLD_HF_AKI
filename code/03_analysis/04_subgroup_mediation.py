"""
04_subgroup_mediation.py
亚组分析 + 中介分析
  亚组：FIB-4≥2.67 × AKI，按8个临床变量分层
  中介：FIB-4→AKI路径中albumin/NLR的中介效应
输出：
    output/tables/table4_subgroup.csv
    output/figures/fig_subgroup_forest.png
    output/tables/table5_mediation.csv
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

PROJECT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW     = f"{PROJECT}/data/raw/cohort_mimic_raw.csv"
OUT_SG  = f"{PROJECT}/output/tables/table4_subgroup.csv"
OUT_FIG = f"{PROJECT}/output/figures/fig_subgroup_forest.png"
OUT_MED = f"{PROJECT}/output/tables/table5_mediation.csv"

df = pd.read_csv(RAW)
df['aki']        = df['aki_flag']
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['log_fib4']   = np.log1p(df['fib4'])
df['log_nlr']    = np.log1p(df['nlr'])

COVARS_BASE = [
    'age', 'gender_num', 'dm_t2', 'hypertension', 'dyslipidemia', 'ckd',
    'afib', 'copd_asthma', 'cad', 'hfref', 'hfpef',
    'egfr', 'sodium_final', 'hemoglobin_final',
]

# FIB-4高风险标志（暴露变量）
df['fib4_high'] = (df['fib4'] >= 2.67).astype('Int64')

def avail(df, cols):
    return [c for c in cols if c in df.columns]

def logistic_or(df_sub, exposure, outcome, covs):
    """返回(OR, CI_low, CI_high, P, N)"""
    cols = [outcome, exposure] + covs
    d = df_sub.dropna(subset=cols)
    d = d[d[outcome].notna()]
    if len(d) < 30 or d[outcome].sum() < 5:
        return None
    try:
        formula = f"{outcome} ~ {exposure} + " + " + ".join(covs)
        m = smf.logit(formula, data=d).fit(disp=0)
        if exposure not in m.params:
            return None
        coef = m.params[exposure]
        ci   = m.conf_int().loc[exposure]
        p    = m.pvalues[exposure]
        return (np.exp(coef), np.exp(ci[0]), np.exp(ci[1]), p, len(d))
    except:
        return None

# ============================================================
# PART A：亚组分析（FIB-4高风险→AKI）
# ============================================================
print("=" * 55)
print("  亚组分析：FIB-4≥2.67 → AKI")
print("=" * 55)

# 仅MASLD组
df_masld = df[df['masld_main'] == 1].copy()

SUBGROUPS = [
    ('age',          'Age',       lambda d: d['age'] >= 65,          '>=65y',      '<65y'),
    ('gender_num',   'Gender',    lambda d: d['gender_num'] == 1,    'Male',       'Female'),
    ('hfref',        'HF Type',   lambda d: d['hfref'] == 1,         'HFrEF',      'HFpEF/Other'),
    ('ckd',          'CKD',       lambda d: d['ckd'] == 1,           'CKD',        'No CKD'),
    ('dm_t2',        'Diabetes',  lambda d: d['dm_t2'] == 1,         'DM',         'No DM'),
    ('egfr',         'eGFR',      lambda d: d['egfr'] < 60,          '<60',        '>=60'),
    ('albumin_final','Albumin',   lambda d: d['albumin_final'] < 3.5,'<3.5g/dL',   '>=3.5g/dL'),
    ('hypertension', 'HTN',       lambda d: d['hypertension'] == 1,  'HTN',        'No HTN'),
]

covs_sg = avail(df_masld, [c for c in COVARS_BASE if c not in ['age','gender_num']])

sg_rows = []
for col, label, condition, pos_label, neg_label in SUBGROUPS:
    if col not in df_masld.columns:
        continue
    for flag, sublabel in [(True, pos_label), (False, neg_label)]:
        try:
            mask = condition(df_masld) if flag else ~condition(df_masld)
            sub  = df_masld[mask].copy()
            # 亚组协变量去掉分层变量本身
            covs_sub = [c for c in covs_sg if c != col]
            res = logistic_or(sub, 'fib4_high', 'aki', covs_sub)
            if res:
                or_, lo, hi, p, n = res
                p_s = "<0.001" if p < 0.001 else f"{p:.3f}"
                print(f"  {label}-{sublabel:<12}: OR={or_:.2f}({lo:.2f}-{hi:.2f}) P={p_s} N={n}")
                sg_rows.append({
                    'subgroup': label, 'level': sublabel,
                    'OR': or_, 'CI_low': lo, 'CI_high': hi, 'P': p, 'N': n
                })
        except Exception as e:
            print(f"  {label}-{sublabel}: ERROR {e}")

# 交互检验P值
print("\n  交互检验（分层变量×FIB-4高风险）：")
int_rows = []
for col, label, _, _, _ in SUBGROUPS:
    if col not in df_masld.columns:
        continue
    d = df_masld.dropna(subset=['aki','fib4_high', col] + covs_sg)
    d = d[d['aki'].notna() & d['fib4_high'].notna()]
    if len(d) < 50:
        continue
    try:
        covs_no = [c for c in covs_sg if c != col]
        formula = (f"aki ~ fib4_high * {col} + " + " + ".join(covs_no))
        m = smf.logit(formula, data=d).fit(disp=0)
        int_term = f"fib4_high:{col}"
        if int_term not in m.pvalues:
            int_term = f"{col}:fib4_high"
        if int_term in m.pvalues:
            p_int = m.pvalues[int_term]
            p_s = "<0.001" if p_int < 0.001 else f"{p_int:.3f}"
            print(f"  {label:<10} 交互P={p_s}")
            for row in sg_rows:
                if row['subgroup'] == label:
                    row['P_interaction'] = round(p_int, 3)
    except Exception as e:
        print(f"  {label}: 交互检验ERROR {e}")

pd.DataFrame(sg_rows).to_csv(OUT_SG, index=False, encoding='utf-8-sig')

# ============================================================
# PART B：森林图
# ============================================================
df_sg = pd.DataFrame(sg_rows)

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = list(range(len(df_sg)))[::-1]

colors = ['#2166ac' if p < 0.05 else 'gray' for p in df_sg['P']]

for i, (_, row) in enumerate(df_sg.iterrows()):
    y = y_pos[i]
    ax.plot([row['CI_low'], row['CI_high']], [y, y], color=colors[i], lw=1.5)
    ax.plot(row['OR'], y, 'o', color=colors[i],
        markersize=max(4, min(10, row['N']/50)))
    if row['N'] < 150:
        ax.text(row['CI_high'] + 0.05, y, f'* small n={row["N"]}',
            fontsize=9, color='orange', va='center')

ax.axvline(x=1.0, color='black', linestyle='--', lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['subgroup']}-{r['level']} (n={r['N']})"
                    for _, r in df_sg.iterrows()], fontsize=8)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=10)
ax.set_title('Subgroup Analysis: FIB-4≥2.67 → AKI Risk\n(MASLD patients, adjusted)', fontsize=10)
ax.set_xlim(0.3, 5.0)

sig_patch  = mpatches.Patch(color='#2166ac', label='P<0.05')
ns_patch   = mpatches.Patch(color='gray',    label='P≥0.05')
ax.legend(handles=[sig_patch, ns_patch], fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n森林图已保存: {OUT_FIG}")

# ============================================================
# PART C：中介分析（Baron-Kenny + Sobel）
# ============================================================
print("\n" + "=" * 55)
print("  中介分析：FIB-4高风险 → [白蛋白/NLR] → AKI")
print("=" * 55)

def mediation_bk(df_in, exposure, mediator, outcome, covs):
    """Baron-Kenny三步法 + Sobel检验"""
    d = df_in.dropna(subset=[exposure, mediator, outcome] + covs)
    d = d[d[outcome].notna()]
    if len(d) < 50:
        return None
    try:
        # Step1: X→Y (total effect)
        m1 = smf.logit(f"{outcome} ~ {exposure} + " + "+".join(covs), data=d).fit(disp=0)
        # Step2: X→M
        m2 = smf.ols(f"{mediator} ~ {exposure} + " + "+".join(covs), data=d).fit()
        # Step3: X+M→Y
        m3 = smf.logit(f"{outcome} ~ {exposure} + {mediator} + " + "+".join(covs), data=d).fit(disp=0)

        a  = m2.params[exposure];   se_a = m2.bse[exposure]
        b  = m3.params[mediator];   se_b = m3.bse[mediator]
        c  = m1.params[exposure]    # total
        c2 = m3.params[exposure]    # direct

        # Sobel
        sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
        sobel_z  = (a * b) / sobel_se
        sobel_p  = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(sobel_z)))

        prop = abs(a * b) / (abs(a * b) + abs(c2)) * 100

        print(f"  {mediator}:")
        print(f"    总效应 c={c:.3f}  直接效应 c'={c2:.3f}")
        print(f"    a={a:.3f}  b={b:.3f}  中介效应a*b={a*b:.3f}")
        print(f"    Sobel Z={sobel_z:.2f}  P={sobel_p:.3f}")
        print(f"    中介比例={prop:.1f}%  N={len(d)}")

        return {'mediator': mediator, 'a': a, 'b': b, 'ab': a*b,
                'c_total': c, 'c_direct': c2, 'sobel_z': sobel_z,
                'sobel_p': sobel_p, 'prop_mediated': prop, 'N': len(d)}
    except Exception as e:
        print(f"  {mediator}: ERROR {e}")
        return None

covs_med = avail(df_masld, [c for c in COVARS_BASE
                             if c not in ['albumin_final','log_nlr']])
med_rows = []
for med, med_label in [('albumin_final', '白蛋白'), ('log_nlr', 'NLR(log)')]:
    if med not in df_masld.columns:
        continue
    res = mediation_bk(df_masld, 'fib4_high', med, 'aki', covs_med)
    if res:
        med_rows.append(res)

pd.DataFrame(med_rows).to_csv(OUT_MED, index=False, encoding='utf-8-sig')
print(f"\n中介分析已保存: {OUT_MED}")
print("下一步: 运行 05_outcomes_finegray.py")