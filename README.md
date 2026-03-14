# MASLD_HF_AKI

**Hepatic Fibrosis by FIB-4 Predicts Acute Kidney Injury in MASLD
with Acute Decompensated Heart Failure: A Dose-Response Analysis
from Two Critical Care Cohorts**

## Overview

This repository contains all analysis code for the above study,
which uses the MIMIC-IV and eICU Collaborative Research Database
to evaluate the association between FIB-4-defined hepatic fibrosis
and acute kidney injury (AKI) in ICU patients with metabolic
dysfunction-associated steatotic liver disease (MASLD) and acute
decompensated heart failure (ADHF).

## Data Availability

Raw data cannot be shared in accordance with PhysioNet data use
agreements. Access to MIMIC-IV and eICU requires completion of
CITI training and a data use agreement via:
- MIMIC-IV: https://physionet.org/content/mimiciv/
- eICU: https://physionet.org/content/eicu-crd/

## Repository Structure
```
MASLD_HF_AKI/
├── README.md
├── requirements.txt
├── code/
│   ├── 02_cohort/
│   │   ├── cohort_building.py       # MIMIC-IV cohort construction
│   │   └── cohort_eicu.py           # eICU cohort construction
│   └── 03_analysis/
│       ├── 01_descriptive_psm.py    # Baseline statistics + PSM
│       ├── 02_main_regression.py # Primary regression analysis
│       ├── 03_rcs_analysis.py       # Restricted cubic spline
│       ├── 04_subgroup_mediation.py # Subgroup + mediation analysis
│       ├── 05_outcomes_finegray.py  # Competing risk + nomogram
│       └── 06_eicu_validation.py    # External validation (eICU)
└── output/
    └── figures/                     # Representative output figures
```

## Analysis Pipeline

Run scripts in the following order:
```bash
# Step 1: Build cohorts
python code/02_cohort/cohort_building.py
python code/02_cohort/cohort_eicu.py

# Step 2: Run analyses sequentially
python code/03_analysis/01_descriptive_psm.py
python code/03_analysis/02_main_regression.py
python code/03_analysis/03_rcs_analysis.py
python code/03_analysis/04_subgroup_mediation.py
python code/03_analysis/05_outcomes_finegray.py
python code/03_analysis/06_eicu_validation.py
```

## Requirements

Python 3.13. Install dependencies:
```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:

> [Author names]. Hepatic Fibrosis by FIB-4 Predicts Acute Kidney
> Injury in MASLD with Acute Decompensated Heart Failure: A
> Dose-Response Analysis from Two Critical Care Cohorts.
> *European Journal of Heart Failure*, 2026. [DOI to be added]

## License

MIT License. See LICENSE for details.
