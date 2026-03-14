"""
Cohort Building 脚本
输出：
    data/raw/cohort_mimic_raw.csv     MIMIC-IV原始队列
    data/raw/cohort_eicu_raw.csv      eICU原始队列

运行：python cohort_building.py
预计耗时：20-40分钟（labevents分块并行处理）
"""

import psycopg2
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = r"C:\Users\Edward\Desktop\SCI\MASLD_HF_DiureticEfficiency"
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")

DB_MIMIC = {
    "host": "localhost", "port": 5432,
    "dbname": "mimiciv", "user": "postgres",
    "password": "HwJ19980324",
}
DB_EICU = {
    "host": "localhost", "port": 5432,
    "dbname": "eicu", "user": "postgres",
    "password": "HwJ19980324",
}

# labevents分块并行参数
N_THREADS   = 4     # 并行线程数，可根据CPU核数调整
CHUNK_HOURS = 6     # 每块处理入院后多少小时内的数据

_log = []
def p(t=""): print(t); _log.append(str(t))
def sec(t):  p(); p("="*65); p(f"  {t}"); p("="*65)
def sub(t):  p(f"\n  -- {t}"); p("  "+"-"*43)

def qry(sql, db=None):
    cfg = db or DB_MIMIC
    try:
        conn = psycopg2.connect(**cfg)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        p(f"  ERROR: {e}"); return pd.DataFrame()

def save_log():
    log_path = os.path.join(PROJECT_ROOT, "output", "reports",
                            f"cohort_building_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log))
    p(f"  日志已保存 -> {log_path}")

# ============================================================
# 复用CTE片段
# ============================================================
HF_ICU_CTE = """
    hf_icu AS (
        SELECT DISTINCT d.hadm_id, d.subject_id,
               a.admittime, a.dischtime, a.hospital_expire_flag, a.race,
               p.anchor_age, p.gender,
               EXTRACT(EPOCH FROM (a.dischtime-a.admittime))/3600.0 AS los_hours
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        JOIN mimiciv_hosp.patients   p ON d.subject_id = p.subject_id
        JOIN mimiciv_icu.icustays    i ON d.hadm_id = i.hadm_id
        WHERE d.icd_version=10 AND d.icd_code LIKE 'I50%%'
          AND p.anchor_age >= 18
          AND EXTRACT(EPOCH FROM (a.dischtime-a.admittime))/3600.0 >= 48
    )
"""

ICU_BRIDGE_CTE = """
    icu_bridge AS (
        SELECT DISTINCT ON (hadm_id)
               hadm_id, stay_id, intime AS icu_intime, outtime AS icu_outtime
        FROM mimiciv_icu.icustays
        ORDER BY hadm_id, intime
    )
"""

# ============================================================
# MODULE 1: 基础队列 + MASLD三层定义
# ============================================================
def build_base_cohort():
    sec("MODULE 1: 基础队列 + MASLD三层定义")
    t0 = time.time()

    sql = f"""
        WITH {HF_ICU_CTE},
        {ICU_BRIDGE_CTE},

        -- 层1a：本次住院ICD编码
        masld_icd_this AS (
            SELECT DISTINCT hadm_id
            FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version=10 AND icd_code IN ('K7581','K760')
        ),
        -- 层1b：历史住院ICD编码（同一subject_id其他住院有编码）
        masld_icd_ever AS (
            SELECT DISTINCT subject_id
            FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version=10 AND icd_code IN ('K7581','K760')
        ),
        -- 排除标准（酒精性肝病/肝硬化/病毒性肝炎）
        liver_excl AS (
            SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version=10 AND (
                icd_code LIKE 'K70%%' OR icd_code LIKE 'K74%%'
                OR icd_code LIKE 'B18%%' OR icd_code LIKE 'B19%%'
            )
        ),
        -- 层2：实验室辅助定义
        -- ALT>40 + BMI>=25（来自omr表）
        bmi_omr AS (
            SELECT DISTINCT ON (subject_id)
                   subject_id,
                   result_value::numeric AS bmi_val
            FROM mimiciv_hosp.omr
            WHERE result_name = 'BMI (kg/m2)'
              AND result_value ~ '^[0-9]+\.?[0-9]*$'
              AND result_value::numeric BETWEEN 10 AND 80
            ORDER BY subject_id, chartdate DESC
        ),
        lab_alt AS (
            SELECT DISTINCT ON (ib.hadm_id)
                   ib.hadm_id,
                   l.alt_min, l.ast_min,
                   l.albumin_min, l.albumin_max,
                   l.platelets_min,
                   l.abs_neutrophils_min, l.abs_lymphocytes_min,
                   l.creatinine_min, l.creatinine_max,
                   l.sodium_min, l.potassium_min,
                   l.bilirubin_total_min,
                   l.inr_min, l.bun_min,
                   l.wbc_min, l.hemoglobin_min,
                   l.glucose_min, l.bicarbonate_min,
                   l.hematocrit_min
            FROM icu_bridge ib
            JOIN mimiciv_derived.first_day_lab l ON ib.stay_id = l.stay_id
            ORDER BY ib.hadm_id
        ),
        masld_lab AS (
            SELECT DISTINCT h.hadm_id
            FROM hf_icu h
            JOIN lab_alt la ON h.hadm_id = la.hadm_id
            JOIN bmi_omr b  ON h.subject_id = b.subject_id
            LEFT JOIN liver_excl e ON h.hadm_id = e.hadm_id
            WHERE la.alt_min > 40
              AND b.bmi_val >= 25
              AND e.hadm_id IS NULL
        ),

        -- 心衰ICD亚型（HFrEF vs HFpEF）
        hf_type AS (
            SELECT hadm_id,
                   MAX(CASE WHEN icd_code IN ('I5020','I5021','I5022','I5023')
                       THEN 1 ELSE 0 END) AS hfref_flag,
                   MAX(CASE WHEN icd_code IN ('I5030','I5031','I5032','I5033')
                       THEN 1 ELSE 0 END) AS hfpef_flag
            FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version=10 AND icd_code LIKE 'I50%%'
            GROUP BY hadm_id
        ),

        -- 合并症ICD编码
        comorbid AS (
            SELECT hadm_id,
                   MAX(CASE WHEN icd_code LIKE 'E11%%' THEN 1 ELSE 0 END) AS dm_t2,
                   MAX(CASE WHEN icd_code = 'I10'     THEN 1 ELSE 0 END) AS hypertension,
                   MAX(CASE WHEN icd_code LIKE 'E78%%' THEN 1 ELSE 0 END) AS dyslipidemia,
                   MAX(CASE WHEN icd_code LIKE 'E66%%' THEN 1 ELSE 0 END) AS obesity_icd,
                   MAX(CASE WHEN icd_code LIKE 'N18%%' THEN 1 ELSE 0 END) AS ckd,
                   MAX(CASE WHEN icd_code LIKE 'I48%%' THEN 1 ELSE 0 END) AS afib,
                   MAX(CASE WHEN icd_code LIKE 'J44%%'
                             OR icd_code LIKE 'J45%%' THEN 1 ELSE 0 END) AS copd_asthma,
                   MAX(CASE WHEN icd_code LIKE 'I25%%' THEN 1 ELSE 0 END) AS cad
            FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version=10
            GROUP BY hadm_id
        )

        SELECT
            h.hadm_id, h.subject_id,
            h.admittime, h.dischtime,
            h.hospital_expire_flag,
            h.race, h.anchor_age AS age, h.gender,
            ROUND(h.los_hours::numeric/24, 1) AS los_days,
            ib.stay_id, ib.icu_intime, ib.icu_outtime,

            -- MASLD定义标签
            CASE
                WHEN mt.hadm_id IS NOT NULL
                 AND le.hadm_id IS NULL THEN 1 ELSE 0
            END AS masld_icd_this,

            CASE
                WHEN me.subject_id IS NOT NULL
                 AND le.hadm_id IS NULL THEN 1 ELSE 0
            END AS masld_icd_ever,

            CASE
                WHEN ml.hadm_id IS NOT NULL THEN 1 ELSE 0
            END AS masld_lab,

            -- 主定义：三层并集
            CASE
                WHEN (mt.hadm_id IS NOT NULL OR me.subject_id IS NOT NULL
                      OR ml.hadm_id IS NOT NULL)
                 AND le.hadm_id IS NULL THEN 1 ELSE 0
            END AS masld_main,

            -- 心衰类型
            COALESCE(ht.hfref_flag, 0) AS hfref,
            COALESCE(ht.hfpef_flag, 0) AS hfpef,

            -- 合并症
            COALESCE(c.dm_t2, 0)        AS dm_t2,
            COALESCE(c.hypertension, 0) AS hypertension,
            COALESCE(c.dyslipidemia, 0) AS dyslipidemia,
            COALESCE(c.obesity_icd, 0)  AS obesity_icd,
            COALESCE(c.ckd, 0)          AS ckd,
            COALESCE(c.afib, 0)         AS afib,
            COALESCE(c.copd_asthma, 0)  AS copd_asthma,
            COALESCE(c.cad, 0)          AS cad,

            -- 首日实验室指标（来自first_day_lab）
            la.albumin_min, la.albumin_max,
            la.alt_min, la.ast_min,
            la.platelets_min,
            la.abs_neutrophils_min, la.abs_lymphocytes_min,
            la.creatinine_min, la.creatinine_max,
            la.sodium_min, la.potassium_min,
            la.bilirubin_total_min,
            la.inr_min, la.bun_min,
            la.wbc_min, la.hemoglobin_min,
            la.glucose_min, la.bicarbonate_min,
            la.hematocrit_min,

            -- BMI
            b.bmi_val AS bmi

        FROM hf_icu h
        JOIN icu_bridge ib        ON h.hadm_id = ib.hadm_id
        LEFT JOIN masld_icd_this mt ON h.hadm_id = mt.hadm_id
        LEFT JOIN masld_icd_ever me ON h.subject_id = me.subject_id
        LEFT JOIN masld_lab      ml ON h.hadm_id = ml.hadm_id
        LEFT JOIN liver_excl     le ON h.hadm_id = le.hadm_id
        LEFT JOIN hf_type        ht ON h.hadm_id = ht.hadm_id
        LEFT JOIN comorbid        c ON h.hadm_id = c.hadm_id
        LEFT JOIN lab_alt        la ON h.hadm_id = la.hadm_id
        LEFT JOIN bmi_omr         b ON h.subject_id = b.subject_id
        ORDER BY h.hadm_id;
    """

    df = qry(sql)
    p(f"  基础队列: {len(df):,} 例  耗时: {time.time()-t0:.1f}s")
    if df.empty: return df

    # MASLD分布统计
    p(f"  masld_icd_this:  {df['masld_icd_this'].sum():,}")
    p(f"  masld_icd_ever:  {df['masld_icd_ever'].sum():,}")
    p(f"  masld_lab:       {df['masld_lab'].sum():,}")
    p(f"  masld_main（并集）: {df['masld_main'].sum():,}")
    p(f"  非MASLD对照:       {(df['masld_main']==0).sum():,}")

    return df


# ============================================================
# MODULE 2: 利尿剂数据（多线程并行）
# ============================================================
def fetch_diuretics_chunk(hadm_ids_chunk, chunk_idx):
    """单线程：处理一批hadm_id的利尿剂数据"""
    ids_str = ",".join(str(i) for i in hadm_ids_chunk)
    sql = f"""
        SELECT p.hadm_id,
               p.starttime,
               p.drug,
               COALESCE(p.dose_val_rx::numeric, 0) AS dose_mg,
               p.route
        FROM mimiciv_hosp.prescriptions p
        WHERE p.hadm_id IN ({ids_str})
          AND LOWER(p.drug) SIMILAR TO '%%(furosemide|bumetanide|torsemide)%%'
          AND p.starttime IS NOT NULL;
    """
    try:
        conn = psycopg2.connect(**DB_MIMIC)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        p(f"  ERROR chunk {chunk_idx}: {e}")
        return pd.DataFrame()

def build_diuretics(hadm_ids):
    sec("MODULE 2: 利尿剂处方提取（多线程）")
    t0 = time.time()

    # 分块
    chunk_size = max(1, len(hadm_ids) // N_THREADS)
    chunks = [hadm_ids[i:i+chunk_size]
              for i in range(0, len(hadm_ids), chunk_size)]
    p(f"  总住院数: {len(hadm_ids):,}  分{len(chunks)}块  每块约{chunk_size}例")

    results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(fetch_diuretics_chunk, c, i): i
                   for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            df_chunk = fut.result()
            if not df_chunk.empty:
                results.append(df_chunk)
            p(f"  完成块 {futures[fut]+1}/{len(chunks)}"
              f"  当前累计: {sum(len(r) for r in results):,} 条")

    if not results:
        p("  ERROR: 无利尿剂数据"); return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df['starttime'] = pd.to_datetime(df['starttime'])

    # 呋塞米等效剂量换算
    def furo_eq(row):
        drug  = str(row['drug']).lower()
        dose  = float(row['dose_mg']) if pd.notna(row['dose_mg']) else 0
        route = str(row['route']).upper()
        is_iv = any(r in route for r in ['IV','IVPB','INTRAVENOUS','IV DRIP','IVP'])
        if 'furosemide' in drug:
            return dose * (1.0 if is_iv else 0.5)
        elif 'bumetanide' in drug:
            return dose * 40
        elif 'torsemide' in drug:
            return dose * 2
        return 0

    df['furo_eq'] = df.apply(furo_eq, axis=1)
    df['is_iv']   = df['route'].str.upper().isin(
        ['IV','IVPB','INTRAVENOUS','IV DRIP','IVP']).astype(int)

    p(f"  利尿剂处方总记录: {len(df):,}  耗时: {time.time()-t0:.1f}s")
    p(f"  涉及住院次数: {df['hadm_id'].nunique():,}")
    return df


# ============================================================
# MODULE 3: 尿量数据（多线程并行）
# ============================================================
def fetch_urine_chunk(hadm_ids_chunk, chunk_idx):
    """单线程：通过icustays桥接获取一批hadm_id的尿量"""
    ids_str = ",".join(str(i) for i in hadm_ids_chunk)
    sql = f"""
        SELECT ib.hadm_id, u.charttime, u.urineoutput
        FROM mimiciv_derived.urine_output u
        JOIN (
            SELECT DISTINCT ON (hadm_id) hadm_id, stay_id
            FROM mimiciv_icu.icustays
            WHERE hadm_id IN ({ids_str})
            ORDER BY hadm_id, intime
        ) ib ON u.stay_id = ib.stay_id
        WHERE u.urineoutput > 0 AND u.urineoutput < 2000;
    """
    try:
        conn = psycopg2.connect(**DB_MIMIC)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        p(f"  ERROR chunk {chunk_idx}: {e}")
        return pd.DataFrame()

def build_urine(hadm_ids):
    sec("MODULE 3: 尿量数据提取（多线程）")
    t0 = time.time()

    chunk_size = max(1, len(hadm_ids) // N_THREADS)
    chunks = [hadm_ids[i:i+chunk_size]
              for i in range(0, len(hadm_ids), chunk_size)]
    p(f"  总住院数: {len(hadm_ids):,}  分{len(chunks)}块")

    results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(fetch_urine_chunk, c, i): i
                   for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            df_chunk = fut.result()
            if not df_chunk.empty:
                results.append(df_chunk)
            p(f"  完成块 {futures[fut]+1}/{len(chunks)}"
              f"  当前累计: {sum(len(r) for r in results):,} 条")

    if not results:
        p("  ERROR: 无尿量数据"); return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df['charttime'] = pd.to_datetime(df['charttime'])
    p(f"  尿量总记录: {len(df):,}  耗时: {time.time()-t0:.1f}s")
    p(f"  覆盖住院次数: {df['hadm_id'].nunique():,}")
    return df


# ============================================================
# MODULE 4: 非ICU实验室指标（labevents，多线程分块）
# ============================================================
LAB_ITEMIDS = {
    'albumin_lab':    50862,
    'alt_lab':        50861,
    'ast_lab':        50878,
    'platelet_lab':   51265,
    'wbc_lab':        51301,
    'neutrophil_lab': 51256,
    'lymphocyte_lab': 51244,
    'creatinine_lab': 50912,
    'bilirubin_lab':  50885,
    'inr_lab':        51237,
    'sodium_lab':     50983,
    'potassium_lab':  50971,
    'bun_lab':        51006,
    'glucose_lab':    50931,
    'hemoglobin_lab': 51222,
    'bicarbonate_lab':50882,
}
ALL_ITEMIDS = list(LAB_ITEMIDS.values())

def fetch_labevents_chunk(hadm_ids_chunk, chunk_idx):
    """单线程：从labevents取一批患者入院48h内的实验室指标"""
    ids_str = ",".join(str(i) for i in hadm_ids_chunk)
    itemids_str = ",".join(str(i) for i in ALL_ITEMIDS)
    sql = f"""
        SELECT l.hadm_id, l.itemid, l.charttime, l.valuenum
        FROM mimiciv_hosp.labevents l
        JOIN mimiciv_hosp.admissions a ON l.hadm_id = a.hadm_id
        WHERE l.hadm_id IN ({ids_str})
          AND l.itemid IN ({itemids_str})
          AND l.valuenum IS NOT NULL
          AND l.charttime BETWEEN a.admittime
                              AND a.admittime + INTERVAL '48 hours';
    """
    try:
        conn = psycopg2.connect(**DB_MIMIC)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        p(f"  ERROR chunk {chunk_idx}: {e}")
        return pd.DataFrame()

def build_labevents(hadm_ids):
    sec("MODULE 4: 非ICU实验室指标提取（labevents，多线程）")
    p("  注：labevents表19GB，此步骤耗时较长，请耐心等待")
    t0 = time.time()

    chunk_size = max(1, len(hadm_ids) // N_THREADS)
    chunks = [hadm_ids[i:i+chunk_size]
              for i in range(0, len(hadm_ids), chunk_size)]
    p(f"  总住院数: {len(hadm_ids):,}  分{len(chunks)}块  线程数: {N_THREADS}")

    results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(fetch_labevents_chunk, c, i): i
                   for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            df_chunk = fut.result()
            if not df_chunk.empty:
                results.append(df_chunk)
            p(f"  完成块 {futures[fut]+1}/{len(chunks)}"
              f"  当前累计: {sum(len(r) for r in results):,} 条")

    if not results:
        p("  WARNING: labevents无数据"); return pd.DataFrame()

    df_raw = pd.concat(results, ignore_index=True)
    p(f"  labevents原始记录: {len(df_raw):,}  耗时: {time.time()-t0:.1f}s")

    # 每个hadm_id每个itemid取最早一条（入院首次值）
    df_raw['charttime'] = pd.to_datetime(df_raw['charttime'])
    df_first = (df_raw.sort_values('charttime')
                      .groupby(['hadm_id','itemid'])['valuenum']
                      .first().reset_index())

    # 宽表转换
    itemid_to_name = {v: k for k, v in LAB_ITEMIDS.items()}
    df_first['lab_name'] = df_first['itemid'].map(itemid_to_name)
    df_wide = df_first.pivot_table(
        index='hadm_id', columns='lab_name',
        values='valuenum', aggfunc='first'
    ).reset_index()

    p(f"  宽表行数: {len(df_wide):,}  列数: {len(df_wide.columns)}")
    return df_wide


# ============================================================
# MODULE 5: 计算DE值
# ============================================================
def compute_de(base_df, diur_df, urine_df):
    sec("MODULE 5: 计算利尿剂效率（DE）")
    t0 = time.time()

    records = []
    admittime_map = base_df.set_index('hadm_id')['admittime'].to_dict()

    for hadm_id, admittime in admittime_map.items():
        admittime = pd.to_datetime(admittime)
        window_end = admittime + pd.Timedelta(hours=72)

        # 利尿剂：前72h累计等效剂量
        d = diur_df[diur_df['hadm_id'] == hadm_id]
        d = d[(d['starttime'] >= admittime) & (d['starttime'] <= window_end)]
        furo_total = d['furo_eq'].sum()
        is_iv_only = int((d['is_iv'] == 1).all()) if len(d) > 0 else 0

        # 尿量：前72h累计
        u = urine_df[urine_df['hadm_id'] == hadm_id]
        u = u[(u['charttime'] >= admittime) & (u['charttime'] <= window_end)]
        urine_total = u['urineoutput'].sum()

        # DE计算
        de_value = np.nan
        if furo_total >= 40 and urine_total > 0:
            de_value = urine_total / (furo_total / 40.0)

        records.append({
            'hadm_id':        hadm_id,
            'furo_eq_72h':    round(furo_total, 1),
            'urine_ml_72h':   round(urine_total, 0),
            'de_value':       round(de_value, 1) if not np.isnan(de_value) else np.nan,
            'iv_only':        is_iv_only,
            'has_diuretic':   int(furo_total >= 40),
            'has_urine':      int(urine_total > 0),
        })

    df_de = pd.DataFrame(records)

    # 利尿剂抵抗定义
    de_median = df_de['de_value'].median()
    # 只对有DE值的行赋值，无DE值的保持NaN
    df_de['dr_median'] = pd.NA
    df_de.loc[df_de['de_value'].notna(), 'dr_median'] = (
        df_de.loc[df_de['de_value'].notna(), 'de_value'] < de_median
    ).astype('Int64')

    df_de['dr_1400'] = pd.NA
    df_de.loc[df_de['de_value'].notna(), 'dr_1400'] = (
        df_de.loc[df_de['de_value'].notna(), 'de_value'] < 1400
    ).astype('Int64')

    p(f"  DE计算完成  耗时: {time.time()-t0:.1f}s")
    p(f"  有DE值: {df_de['de_value'].notna().sum():,}")
    p(f"  DE中位数: {de_median:.0f} mL/40mg")
    p(f"  利尿剂抵抗(中位数定义): {df_de['dr_median'].sum():,} 例")
    p(f"  利尿剂抵抗(1400定义):   {df_de['dr_1400'].sum():,} 例")
    return df_de


# ============================================================
# MODULE 6: 衍生变量计算
# ============================================================
def compute_derived_vars(df):
    sec("MODULE 6: 衍生变量计算")

    # 优先用first_day_lab列，缺失时用labevents补充列
    def fill_col(df, primary, backup):
        if primary in df.columns and backup in df.columns:
            return df[primary].fillna(df[backup])
        elif primary in df.columns:
            return df[primary]
        elif backup in df.columns:
            return df[backup]
        return np.nan

    df['albumin_final']   = fill_col(df, 'albumin_min',      'albumin_lab')
    df['alt_final']       = fill_col(df, 'alt_min',          'alt_lab')
    df['ast_final']       = fill_col(df, 'ast_min',          'ast_lab')
    df['platelet_final']  = fill_col(df, 'platelets_min',    'platelet_lab')
    df['neutrophil_final']= fill_col(df, 'abs_neutrophils_min','neutrophil_lab')
    df['lymphocyte_final']= fill_col(df, 'abs_lymphocytes_min','lymphocyte_lab')
    df['creatinine_final']= fill_col(df, 'creatinine_min',   'creatinine_lab')
    df['bilirubin_final'] = fill_col(df, 'bilirubin_total_min','bilirubin_lab')
    df['inr_final']       = fill_col(df, 'inr_min',          'inr_lab')
    df['sodium_final']    = fill_col(df, 'sodium_min',        'sodium_lab')
    df['potassium_final'] = fill_col(df, 'potassium_min',     'potassium_lab')
    df['wbc_final']       = fill_col(df, 'wbc_min',           'wbc_lab')
    df['hemoglobin_final']= fill_col(df, 'hemoglobin_min',    'hemoglobin_lab')
    df['bun_final']       = fill_col(df, 'bun_min',           'bun_lab')

    # FIB-4 = 年龄 × AST / (血小板 × √ALT)
    df['fib4'] = np.where(
        df['alt_final'].notna() & df['ast_final'].notna() &
        df['platelet_final'].notna() & (df['platelet_final'] > 0) &
        (df['alt_final'] > 0),
        df['age'] * df['ast_final'] / (df['platelet_final'] * np.sqrt(df['alt_final'])),
        np.nan
    )
    df['fib4'] = df['fib4'].round(3)

    # NLR = 中性粒 / 淋巴细胞
    df['nlr'] = np.where(
        df['neutrophil_final'].notna() & df['lymphocyte_final'].notna() &
        (df['lymphocyte_final'] > 0),
        df['neutrophil_final'] / df['lymphocyte_final'],
        np.nan
    )
    df['nlr'] = df['nlr'].round(2)

    # SII = 血小板 × 中性粒 / 淋巴细胞
    df['sii'] = np.where(
        df['platelet_final'].notna() & df['neutrophil_final'].notna() &
        df['lymphocyte_final'].notna() & (df['lymphocyte_final'] > 0),
        df['platelet_final'] * df['neutrophil_final'] / df['lymphocyte_final'],
        np.nan
    )
    df['sii'] = df['sii'].round(1)

    # eGFR（CKD-EPI 2021，无种族版本）
    def ckd_epi(row):
        cr = row['creatinine_final']
        age = row['age']
        sex = row['gender']
        if pd.isna(cr) or pd.isna(age) or cr <= 0: return np.nan
        kappa = 0.7 if sex == 'F' else 0.9
        alpha = -0.241 if sex == 'F' else -0.302
        cr_k  = cr / kappa
        egfr  = (142 * min(cr_k, 1)**alpha * max(cr_k, 1)**(-1.200)
                 * 0.9938**age)
        if sex == 'F': egfr *= 1.012
        return round(egfr, 1)
    df['egfr'] = df.apply(ckd_epi, axis=1)

    # 种族标准化
    race_map = {
        'WHITE':                    'White',
        'BLACK/AFRICAN AMERICAN':   'Black',
        'HISPANIC/LATINO':          'Hispanic',
        'ASIAN':                    'Asian',
    }
    df['race_group'] = df['race'].str.upper().map(
        lambda x: next((v for k, v in race_map.items() if k in str(x)), 'Other')
    )

    # 统计衍生变量覆盖率
    for var in ['fib4','nlr','sii','egfr','bmi']:
        n = df[var].notna().sum() if var in df.columns else 0
        p(f"  {var:<12} 有值: {n:,} / {len(df):,} ({n/len(df)*100:.1f}%)")

    return df


# ============================================================
# MODULE 7: 结局变量
# ============================================================
def build_outcomes(hadm_ids):
    sec("MODULE 7: 结局变量提取")
    ids_str = ",".join(str(i) for i in hadm_ids)

    # 30天再入院
    sql_readmit = f"""
        WITH index_adm AS (
            SELECT subject_id, hadm_id, dischtime
            FROM mimiciv_hosp.admissions
            WHERE hadm_id IN ({ids_str})
        ),
        next_adm AS (
            SELECT ia.hadm_id AS index_hadm_id,
                   MIN(a2.admittime) AS next_admittime
            FROM index_adm ia
            JOIN mimiciv_hosp.admissions a2
              ON ia.subject_id = a2.subject_id
             AND a2.admittime > ia.dischtime
             AND a2.admittime <= ia.dischtime + INTERVAL '30 days'
            GROUP BY ia.hadm_id
        )
        SELECT ia.hadm_id,
               CASE WHEN na.index_hadm_id IS NOT NULL THEN 1 ELSE 0 END AS readmit_30d,
               EXTRACT(EPOCH FROM (na.next_admittime - ia.dischtime))/86400.0
                   AS days_to_readmit
        FROM index_adm ia
        LEFT JOIN next_adm na ON ia.hadm_id = na.index_hadm_id;
    """

    # AKI结局（kdigo_stages）
    sql_aki = f"""
        SELECT ib.hadm_id,
               MAX(k.aki_stage_creat) AS aki_stage_max,
               MAX(CASE WHEN k.aki_stage_creat >= 1 THEN 1 ELSE 0 END) AS aki_flag
        FROM (
            SELECT DISTINCT ON (hadm_id) hadm_id, stay_id
            FROM mimiciv_icu.icustays
            WHERE hadm_id IN ({ids_str})
            ORDER BY hadm_id, intime
        ) ib
        JOIN mimiciv_derived.kdigo_stages k ON ib.stay_id = k.stay_id
        GROUP BY ib.hadm_id;
    """

    df_r = qry(sql_readmit)
    df_a = qry(sql_aki)

    df_out = df_r.merge(df_a, on='hadm_id', how='left')
    p(f"  结局变量行数: {len(df_out):,}")
    p(f"  30天再入院率: {df_out['readmit_30d'].mean()*100:.1f}%")
    p(f"  AKI发生率:    {df_out['aki_flag'].mean()*100:.1f}%")
    return df_out


# ============================================================
# MODULE 8: 输出最终数据集
# ============================================================
def save_cohort(df, filename):
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    p(f"  已保存: {path}  ({len(df):,} 行 × {len(df.columns)} 列)")


# ============================================================
# 主程序
# ============================================================
def main():
    p("=" * 65)
    p("  MASLD_HF_DiureticEfficiency  Cohort Building")
    p(f"  运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 65)

    # MODULE 1: 基础队列
    base_df = build_base_cohort()
    if base_df.empty: save_log(); return

    hadm_ids = base_df['hadm_id'].tolist()
    p(f"\n  队列规模: {len(hadm_ids):,} 例")
    p(f"  MASLD（主定义）: {base_df['masld_main'].sum():,} 例")

    # MODULE 2: 利尿剂
    diur_df = build_diuretics(hadm_ids)

    # MODULE 3: 尿量
    urine_df = build_urine(hadm_ids)

    # MODULE 4: labevents补充实验室指标
    lab_df = build_labevents(hadm_ids)

    # MODULE 5: DE计算
    de_df = compute_de(base_df, diur_df, urine_df)

    # 合并lab补充列到base_df
    if not lab_df.empty:
        base_df = base_df.merge(lab_df, on='hadm_id', how='left')

    # MODULE 6: 衍生变量
    base_df = compute_derived_vars(base_df)

    # 合并DE
    base_df = base_df.merge(de_df, on='hadm_id', how='left')

    # MODULE 7: 结局变量
    out_df = build_outcomes(hadm_ids)
    base_df = base_df.merge(out_df, on='hadm_id', how='left')

    # MODULE 8: 保存
    sec("MODULE 8: 保存数据集")
    save_cohort(base_df, "cohort_mimic_raw.csv")

    # 数据质量报告
    sub("数据质量概览")
    key_vars = ['masld_main','de_value','albumin_final','fib4',
                'nlr','bmi','egfr','hospital_expire_flag','readmit_30d','aki_flag']
    p(f"  {'变量':<22} {'非空':>8} {'覆盖率':>8}")
    p(f"  {'-'*42}")
    for v in key_vars:
        if v in base_df.columns:
            n = base_df[v].notna().sum()
            p(f"  {v:<22} {n:>8,} {n/len(base_df)*100:>7.1f}%")

    p()
    p("=" * 65)
    p("  Cohort Building 完成")
    p(f"  输出文件: {os.path.join(RAW_DIR, 'cohort_mimic_raw.csv')}")
    p("  下一步: 运行 code/03_analysis/ 统计分析脚本")
    p("=" * 65)
    save_log()


if __name__ == "__main__":
    main()