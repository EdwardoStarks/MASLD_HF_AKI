"""
03_rcs_analysis.py
限制性立方样条（RCS）分析：FIB-4连续值 → AKI风险
输出：
    output/figures/fig_rcs_fib4_aki.png
    output/tables/table3_rcs_data.csv
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW     = f"{PROJECT}/data/raw/cohort_mimic_raw.csv"
OUT_FIG = f"{PROJECT}/output/figures/fig_rcs_fib4_aki.png"
OUT_TAB = f"{PROJECT}/output/tables/table3_rcs_data.csv"

df = pd.read_csv(RAW)
df['aki']        = df['aki_flag']
df['gender_num'] = (df['gender'] == 'M').astype(int)

COVARS_BASE = [
    'age', 'gender_num',
    'dm_t2', 'hypertension', 'dyslipidemia', 'ckd',
    'afib', 'copd_asthma', 'cad',
    'hfref', 'hfpef',
    'egfr', 'sodium_final', 'hemoglobin_final',
]

# ============================================================
# RCS基函数（Harrell 4节点）
# ============================================================
def make_rcs_terms(x, knots):
    """返回DataFrame，含rcs1, rcs2两列（4节点→2个非线性项）"""
    k = knots
    cols = {}
    for i in range(1, len(k) - 1):
        t = (np.maximum(x - k[i-1], 0)**3
             - ((k[-1] - k[i-1]) / (k[-1] - k[-2])) * np.maximum(x - k[-2], 0)**3
             + ((k[-2] - k[i-1]) / (k[-1] - k[-2])) * np.maximum(x - k[-1], 0)**3)
        cols[f'rcs{i}'] = t
    return pd.DataFrame(cols)

# ============================================================
# 数据准备
# ============================================================
df_m = df[df['masld_main'] == 1].copy()
df_m = df_m[df_m['fib4'].notna() & df_m['aki'].notna()]
covs = [c for c in COVARS_BASE if c in df_m.columns]
df_m = df_m.dropna(subset=covs + ['fib4', 'aki']).reset_index(drop=True)

print(f"MASLD组RCS分析 N={len(df_m)}")

# 节点：
knots = np.percentile(df_m['fib4'], [10, 25, 50, 75, 90])  # 5节点，非线性更明显
print(f"RCS节点: {knots.round(2)}")

# 拼入样条项
rcs_df = make_rcs_terms(df_m['fib4'].values, knots)
rcs_cols = rcs_df.columns.tolist()
df_m = pd.concat([df_m, rcs_df], axis=1)

# ============================================================
# 拟合模型
# ============================================================
formula = f"aki ~ fib4 + {' + '.join(rcs_cols)} + {' + '.join(covs)}"
model = smf.logit(formula, data=df_m).fit(disp=0)
print(f"模型收敛: {model.converged}  AIC={model.aic:.1f}")

# ============================================================
# 预测曲线
# ============================================================
fib4_range = np.linspace(
    df_m['fib4'].quantile(0.02),
    df_m['fib4'].quantile(0.90),  # 截到90百分位，去掉稀疏长尾
    200
)

# 协变量固定为中位数
cov_means = {c: df_m[c].median() for c in covs}

pred_rows = []
for fv in fib4_range:
    rcs_v = make_rcs_terms(np.array([fv]), knots)
    row = {'fib4': fv, **cov_means}
    for col in rcs_cols:
        row[col] = rcs_v[col].values[0]
    pred_rows.append(row)

pred_df = pd.DataFrame(pred_rows)

# 严格对齐列顺序（含Intercept）
exog_names = model.model.exog_names  # 第一个是'Intercept'
pred_df['Intercept'] = 1.0
pred_df_sm = pred_df[exog_names]

print(f"模型列数: {len(exog_names)} | 预测矩阵列数: {pred_df_sm.shape[1]}")

# 预测
preds = model.get_prediction(pred_df_sm)
sumf  = preds.summary_frame(alpha=0.05)
print(f"预测列名: {sumf.columns.tolist()}")

pred_df['prob']    = sumf['predicted'].values
pred_df['ci_low']  = sumf['ci_lower'].values
pred_df['ci_high'] = sumf['ci_upper'].values

# ============================================================
# 绘图
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(pred_df['fib4'], pred_df['prob'],
        color='#2166ac', lw=2, label='Adjusted probability')
ax.fill_between(pred_df['fib4'], pred_df['ci_low'], pred_df['ci_high'],
                color='#2166ac', alpha=0.15, label='95% CI')

# 阈值线
ax.axvline(x=1.30, color='gray',    linestyle='--', lw=1,   alpha=0.7)
ax.axvline(x=2.67, color='#d6604d', linestyle='--', lw=1.5, alpha=0.9)

ymin = pred_df['prob'].min() - 0.05
ytext = pred_df['prob'].min() + 0.03
ax.text(1.35, ytext, '1.30', color='gray',    fontsize=8)
ax.text(2.72, ytext, '2.67', color='#d6604d', fontsize=8)

# 底部分布
ax_rug = ax.twinx()
ax_rug.set_yticks([])
ax_rug.set_ylim(0, 1)
rug_y = np.zeros(len(df_m)) + 0.01
ax_rug.eventplot(df_m['fib4'].clip(pred_df['fib4'].min(), pred_df['fib4'].max()),
                 lineoffsets=0.02, linelengths=0.03,
                 linewidths=0.4, colors='gray', alpha=0.3)

ax.set_xlabel('FIB-4 Score', fontsize=11)
ax.set_ylabel('Predicted Probability of AKI', fontsize=11)
ax.set_title('FIB-4 and AKI Risk in MASLD Patients with ADHF\n'
             '(Restricted Cubic Spline, adjusted)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(pred_df['fib4'].min(), pred_df['fib4'].max())
ax.set_ylim(max(0, ymin), min(1, pred_df['prob'].max() + 0.1))

# 样本量注释
n_low  = (df_m['fib4'] <  1.30).sum()
n_mid  = ((df_m['fib4'] >= 1.30) & (df_m['fib4'] < 2.67)).sum()
n_high = (df_m['fib4'] >= 2.67).sum()
ax.text(0.02, 0.06,
        f'FIB-4<1.30: n={n_low}  |  1.30-2.67: n={n_mid}  |  ≥2.67: n={n_high}',
        transform=ax.transAxes, fontsize=7.5, color='dimgray',
        va='bottom', ha='left')

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
plt.close()
print(f"图已保存: {OUT_FIG}")

# ============================================================
# 保存数据
# ============================================================
pred_df[['fib4', 'prob', 'ci_low', 'ci_high']].to_csv(OUT_TAB, index=False)
print(f"数据已保存: {OUT_TAB}")
print("下一步: 运行 04_subgroup_mediation.py")