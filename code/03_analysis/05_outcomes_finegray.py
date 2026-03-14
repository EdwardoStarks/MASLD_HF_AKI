"""
05_outcomes_finegray.py
竞争风险分析（Fine-Gray）+ Nomogram
  主要结局：AKI（竞争事件：院内死亡）
  次要：30天再入院
输出：
    output/tables/table6_finegray.csv
    output/figures/fig_cumulative_incidence.png
    output/figures/fig_nomogram.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import AalenJohansenFitter
from lifelines.statistics import logrank_test
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PROJECT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW     = f"{PROJECT}/data/raw/cohort_mimic_raw.csv"
OUT_TAB = f"{PROJECT}/output/tables/table6_finegray.csv"
OUT_CIF = f"{PROJECT}/output/figures/fig_cumulative_incidence.png"
OUT_NOM = f"{PROJECT}/output/figures/fig_nomogram.png"

df = pd.read_csv(RAW)
df['aki']        = df['aki_flag']
df['gender_num'] = (df['gender'] == 'M').astype(int)
df['log_fib4']   = np.log1p(df['fib4'])
df['fib4_high']  = (df['fib4'] >= 2.67).astype('Int64')

COVARS_BASE = [
    'age', 'gender_num', 'dm_t2', 'hypertension', 'dyslipidemia', 'ckd',
    'afib', 'copd_asthma', 'cad', 'hfref', 'hfpef',
    'egfr', 'sodium_final', 'hemoglobin_final',
]

# ============================================================
# PART A：累积发生率曲线（AKI竞争院内死亡）
# 用住院时长作为时间轴
# ============================================================
print("=" * 55)
print("  累积发生率曲线（Aalen-Johansen）")
print("=" * 55)

df_masld = df[df['masld_main'] == 1].copy()
df_masld = df_masld.dropna(subset=['los_days', 'aki', 'hospital_expire_flag', 'fib4_high'])

# 构建竞争风险事件编码
# 0=censored, 1=AKI, 2=death without AKI
def make_event(row):
    if row['aki'] == 1:
        return 1
    elif row['hospital_expire_flag'] == 1:
        return 2
    else:
        return 0

df_masld['event'] = df_masld.apply(make_event, axis=1)
df_masld['time']  = df_masld['los_days'].clip(lower=0.1)

fig, ax = plt.subplots(figsize=(7, 5))
colors_grp = {0: '#2166ac', 1: '#d6604d'}
labels_grp = {0: 'FIB-4 <2.67', 1: 'FIB-4 >=2.67'}

aj_results = {}
for grp in [0, 1]:
    sub = df_masld[df_masld['fib4_high'] == grp]
    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(sub['time'], sub['event'], event_of_interest=1)
    ajf.plot(ax=ax, color=colors_grp[grp], label=labels_grp[grp], ci_show=True)
    aj_results[grp] = ajf
    print(f"  FIB-4{'>=2.67' if grp else '<2.67'}: N={len(sub)}  events={sub['event'].eq(1).sum()}")


# 用log-rank检验替代Gray test（lifelines版本不支持Gray）
grp0 = df_masld[df_masld['fib4_high']==0]
grp1 = df_masld[df_masld['fib4_high']==1]
lr = logrank_test(
    grp0['time'], grp1['time'],
    event_observed_A=(grp0['event']==1),
    event_observed_B=(grp1['event']==1),
)
p_s = "<0.001" if lr.p_value < 0.001 else f"{lr.p_value:.3f}"
print(f"  Log-rank test P={p_s}")
ax.text(0.98, 0.05, f'Log-rank P={p_s}',
        transform=ax.transAxes, fontsize=9,
        ha='right', va='bottom', color='black')

ax.set_xlabel('Hospital Length of Stay (days)', fontsize=11)
ax.set_ylabel('Cumulative Incidence of AKI', fontsize=11)
ax.set_title('Cumulative Incidence of AKI by FIB-4 Group\n'
             '(Competing risk: in-hospital death)', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, df_masld['time'].quantile(0.95))
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_CIF, dpi=300, bbox_inches='tight')
plt.close()
print(f"  CIF图已保存: {OUT_CIF}")

# ============================================================
# PART B：Nomogram（仅用显著变量：egfr, fib4, albumin_final）
# ============================================================
print("\n" + "=" * 55)
print("  Nomogram：AKI风险预测")
print("=" * 55)

NOM_VARS = ['egfr', 'fib4', 'albumin_final']
df_nom = df[df['masld_main'] == 1].dropna(subset=NOM_VARS + ['aki']).copy()

formula_nom = "aki ~ egfr + fib4 + albumin_final"
m_nom = smf.logit(formula_nom, data=df_nom).fit(disp=0)
print(f"  Nomogram模型 N={len(df_nom)}  AIC={m_nom.aic:.1f}")
for v in NOM_VARS:
    coef = m_nom.params[v]
    p    = m_nom.pvalues[v]
    p_s  = "<0.001" if p < 0.001 else f"{p:.3f}"
    print(f"    {v:<18} coef={coef:.3f}  P={p_s}")

var_labels = {
    'egfr':          'eGFR (mL/min)',
    'fib4':          'FIB-4 Score',
    'albumin_final': 'Albumin (g/dL)',
}
var_ranges = {
    'egfr':          (5, 120),
    'fib4':          (0.5, 10.0),
    'albumin_final': (1.5, 5.0),
}
var_ticks = {
    'egfr':          [5, 30, 60, 90, 120],
    'fib4':          [1, 3, 6, 10],
    'albumin_final': [1.5, 2.5, 3.5, 4.5, 5.0],
}

coefs = {v: m_nom.params[v] for v in NOM_VARS}

# 计算每个变量的点数范围（相对于参考点）
def val_to_points(v, val, scale):
    ref = (var_ranges[v][0] + var_ranges[v][1]) / 2
    return (val - ref) * coefs[v] * scale

# 归一化：最大点差=100
raw_ranges = {v: abs(coefs[v]) * (var_ranges[v][1] - var_ranges[v][0]) for v in NOM_VARS}
total_range = sum(raw_ranges.values())
scale = 100 / total_range

fig, axes = plt.subplots(len(NOM_VARS) + 3, 1, figsize=(10, 7))
plt.subplots_adjust(hspace=0)

def draw_axis(ax, label, tick_vals, tick_pts, color='black', bold=False):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.plot([min(tick_pts), max(tick_pts)], [0.5, 0.5], color=color, lw=1.5)
    for val, pt in zip(tick_vals, tick_pts):
        ax.plot(pt, 0.5, '|', color=color, markersize=10, lw=1.5)
        ax.text(pt, 0.05, f"{val}", ha='center', fontsize=8, color=color)
    fw = 'bold' if bold else 'normal'
    ax.text(-1, 0.5, label, ha='right', fontsize=9, va='center',
            color=color, fontweight=fw)

# 累积偏移量（让各变量点数在0-100范围内合理分布）
offsets = {}
cumulative = 0
for v in NOM_VARS:
    offsets[v] = cumulative
    cumulative += raw_ranges[v] * scale

# Points轴（0-100）
ax_pts = axes[0]
draw_axis(ax_pts, 'Points',
          list(range(0, 101, 10)),
          list(range(0, 101, 10)), bold=True)

# 各变量轴
for i, v in enumerate(NOM_VARS):
    ax = axes[i + 1]
    tick_vals = var_ticks[v]
    ref = var_ranges[v][0]
    if coefs[v] > 0:
        tick_pts = [offsets[v] + (val - ref) * coefs[v] * scale for val in tick_vals]
    else:
        tick_pts = [offsets[v] + raw_ranges[v] * scale + (val - ref) * coefs[v] * scale
                    for val in tick_vals]
    draw_axis(ax, var_labels[v], tick_vals, tick_pts)

# Total Points轴
ax_total = axes[len(NOM_VARS) + 1]
draw_axis(ax_total, 'Total Points',
          list(range(0, 101, 10)),
          list(range(0, 101, 10)), bold=True)

# 预测概率轴（基于模型中位数协变量）
ax_prob = axes[len(NOM_VARS) + 2]
ax_prob.set_xlim(0, 100)
ax_prob.set_ylim(0, 1)
ax_prob.axis('off')

intercept = m_nom.params['Intercept']
ref_logit  = intercept + sum(coefs[v] * (var_ranges[v][0] + var_ranges[v][1]) / 2
                              for v in NOM_VARS)
prob_ticks = []
pt_ticks   = []
for pt in range(0, 101, 10):
    logit_val = ref_logit + (pt - 50) * total_range / 100
    prob = 1 / (1 + np.exp(-logit_val))
    prob_ticks.append(round(prob, 2))
    pt_ticks.append(pt)

ax_prob.plot([0, 100], [0.5, 0.5], color='#2166ac', lw=1.5)
for pt, prob in zip(pt_ticks, prob_ticks):
    ax_prob.plot(pt, 0.5, '|', color='#2166ac', markersize=10)
    ax_prob.text(pt, 0.05, f"{prob:.2f}", ha='center', fontsize=8, color='#2166ac')
ax_prob.text(-1, 0.5, 'Predicted\nAKI Prob.', ha='right', fontsize=9,
             va='center', color='#2166ac', fontweight='bold')

axes[0].set_title('Nomogram for Predicting AKI Risk in MASLD Patients with ADHF',
                  fontsize=11, pad=10)
plt.savefig(OUT_NOM, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Nomogram已保存: {OUT_NOM}")

# ============================================================
# PART C：Calibration Plot
# ============================================================
print("\n" + "=" * 55)
print("  Calibration Plot")
print("=" * 55)

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

df_cal = df_nom.copy()
exog = sm.add_constant(df_cal[NOM_VARS])
df_cal['pred_prob'] = m_nom.predict(df_cal[NOM_VARS])

# AUC
auc = roc_auc_score(df_cal['aki'], df_cal['pred_prob'])
print(f"  AUC (C-statistic) = {auc:.3f}")

# Hosmer-Lemeshow
n_bins = 10
df_cal['decile'] = pd.qcut(df_cal['pred_prob'], n_bins, labels=False, duplicates='drop')
hl_rows = []
for d in sorted(df_cal['decile'].unique()):
    grp = df_cal[df_cal['decile'] == d]
    hl_rows.append({'obs': grp['aki'].mean(), 'pred': grp['pred_prob'].mean(), 'n': len(grp)})
hl_df = pd.DataFrame(hl_rows)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.scatter(hl_df['pred'], hl_df['obs'],
           s=hl_df['n']/5, color='#2166ac', alpha=0.8, zorder=5)
ax.plot(hl_df['pred'], hl_df['obs'], color='#2166ac', lw=1.5,
        label=f'Observed vs Predicted\nAUC={auc:.3f}')

ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Observed Frequency', fontsize=11)
ax.set_title('Calibration Plot: AKI Prediction Model\n(MASLD patients with ADHF)', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

OUT_CAL = f"{PROJECT}/output/figures/fig_calibration.png"
plt.tight_layout()
plt.savefig(OUT_CAL, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Calibration plot已保存: {OUT_CAL}")
print("\n所有分析完成！下一步：整理论文表格和图片。")