"""
cohort_eicu.py
eICU外部验证队列构建
输出: data/raw/cohort_eicu_raw.csv
"""

import psycopg2
import pandas as pd
import numpy as np

PROJECT  = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
OUT_PATH = f"{PROJECT}/data/raw/cohort_eicu_raw.csv"

DB_EICU = {
    "host": "localhost", "port": 5432,
    "dbname": "eicu", "user": "postgres",
    "password": "HwJ19980324",
}

SCHEMA = "eicu"

def qry(sql):
    try:
        conn = psycopg2.connect(**DB_EICU)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

# ============================================================
# STEP 1: 心衰队列
# ============================================================
print("STEP 1: 心衰队列...")
df_hf = qry("""
    SELECT DISTINCT
        p.patientunitstayid,
        p.age,
        p.gender,
        p.unitdischargestatus,
        p.hospitaldischargestatus,
        p.unitdischargeoffset,
        p.hospitaladmitoffset,
        p.hospitaldischargeoffset
    FROM eicu.patient p
    WHERE EXISTS (
        SELECT 1 FROM eicu.diagnosis d
        WHERE d.patientunitstayid = p.patientunitstayid
        AND (
            LOWER(d.diagnosisstring) LIKE '%heart failure%'
            OR LOWER(d.diagnosisstring) LIKE '%congestive heart failure%'
        )
    )
    AND p.age NOT IN ('Unknown', '')
    AND p.age ~ '^\d+$'
    AND p.age::integer >= 18
""")
print(f"  心衰队列: {len(df_hf):,}")

# ============================================================
# STEP 2: MASLD识别（实验室辅助）
# ============================================================
print("STEP 2: MASLD识别...")
df_alt = qry("""
    SELECT DISTINCT patientunitstayid
    FROM eicu.lab
    WHERE labname = 'ALT (SGPT)'
    AND labresult > 40
""")

df_excl = qry("""
    SELECT DISTINCT patientunitstayid
    FROM eicu.pasthistory
    WHERE LOWER(pasthistoryvalue) LIKE '%alcohol%'
    OR LOWER(pasthistoryvalue) LIKE '%hepatitis%'
    OR LOWER(pasthistoryvalue) LIKE '%cirrhosis%'
""")

masld_ids = set(df_alt['patientunitstayid']) - set(df_excl['patientunitstayid'])
df_hf['masld_lab'] = df_hf['patientunitstayid'].isin(masld_ids).astype(int)
print(f"  MASLD候选: {df_hf['masld_lab'].sum():,}")

# ============================================================
# STEP 3: 实验室指标（FIB-4所需）
# ============================================================
print("STEP 3: 实验室指标...")
df_labs = qry("""
    SELECT
        patientunitstayid,
        MAX(CASE WHEN labname='ALT (SGPT)'    THEN labresult END) AS alt,
        MAX(CASE WHEN labname='AST (SGOT)'    THEN labresult END) AS ast,
        MIN(CASE WHEN labname='platelets x 1000' THEN labresult END) AS platelets,
        MIN(CASE WHEN labname='albumin'       THEN labresult END) AS albumin,
        MIN(CASE WHEN labname='creatinine'    THEN labresult END) AS creatinine,
        MIN(CASE WHEN labname='sodium'        THEN labresult END) AS sodium,
        MAX(CASE WHEN labname='WBC x 1000'   THEN labresult END) AS wbc,
        MIN(CASE WHEN labname='Hgb'          THEN labresult END) AS hemoglobin
    FROM eicu.lab
    WHERE labname IN (
        'ALT (SGPT)','AST (SGOT)','platelets x 1000','albumin',
        'creatinine','sodium','WBC x 1000','Hgb'
    )
    GROUP BY patientunitstayid
""")
df_hf = df_hf.merge(df_labs, on='patientunitstayid', how='left')

# FIB-4
df_hf['age_num'] = pd.to_numeric(df_hf['age'], errors='coerce')
df_hf['fib4'] = np.where(
    df_hf[['age_num','ast','platelets','alt']].notna().all(axis=1) & (df_hf['platelets'] > 0) & (df_hf['alt'] > 0),
    (df_hf['age_num'] * df_hf['ast']) / (df_hf['platelets'] * np.sqrt(df_hf['alt'])),
    np.nan
)
print(f"FIB-4样本检查: {df_hf['fib4'].describe()}")

# ============================================================
# STEP 4: 结局变量
# ============================================================
print("STEP 4: 结局变量...")

# AKI（肌酐升高>0.3或>1.5倍基线）
df_cr = qry("""
    SELECT
        patientunitstayid,
        MIN(labresult) AS cr_base,
        MAX(labresult) AS cr_peak
    FROM eicu.lab
    WHERE labname = 'creatinine'
    AND labresult > 0
    GROUP BY patientunitstayid
""")
df_hf = df_hf.merge(df_cr, on='patientunitstayid', how='left')
df_hf['aki_flag'] = (
    ((df_hf['cr_peak'] - df_hf['cr_base']) >= 0.3) |
    (df_hf['cr_peak'] >= df_hf['cr_base'] * 1.5)
).astype('Int64')

# 院内死亡
df_hf['hospital_death'] = (
    df_hf['hospitaldischargestatus'].str.lower().str.contains('expired|dead|death', na=False)
).astype(int)

# ============================================================
# STEP 5: 合并症
# ============================================================
print("STEP 5: 合并症...")
df_comorbid = qry("""
    SELECT
        patientunitstayid,
        MAX(CASE WHEN LOWER(pasthistoryvalue) LIKE '%diabetes%' THEN 1 ELSE 0 END) AS dm,
        MAX(CASE WHEN LOWER(pasthistoryvalue) LIKE '%hypertension%' THEN 1 ELSE 0 END) AS hypertension,
        MAX(CASE WHEN LOWER(pasthistoryvalue) LIKE '%renal%' OR
                      LOWER(pasthistoryvalue) LIKE '%kidney%' THEN 1 ELSE 0 END) AS ckd
    FROM eicu.pasthistory
    GROUP BY patientunitstayid
""")
df_hf = df_hf.merge(df_comorbid, on='patientunitstayid', how='left')

# ============================================================
# 保存
# ============================================================
print(f"\n最终队列: {len(df_hf):,}")
print(f"MASLD: {df_hf['masld_lab'].sum():,}")
print(f"FIB-4可计算: {df_hf['fib4'].notna().sum():,}")
print(f"AKI: {df_hf['aki_flag'].sum():,}")
print(f"院内死亡: {df_hf['hospital_death'].sum():,}")

df_hf.to_csv(OUT_PATH, index=False)
print(f"\n已保存: {OUT_PATH}")