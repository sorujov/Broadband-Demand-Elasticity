# Data Preparation Scripts

This directory contains scripts that transform raw/merged data into analysis-ready format for econometric estimation.

## 📋 Overview

**Pipeline Stage:** Stage 2 (after data collection, before analysis)

**Input:** `data/processed/data_merged_with_series.xlsx` (output from data collection)  
**Output:** `data/processed/analysis_ready_data.csv` (ready for regression analysis)

## 📄 Scripts

### `02_prepare_data.py` ⭐ **[MAIN SCRIPT]**

**Purpose:** Create final analysis-ready dataset with all transformations

**Transformations:**
1. **Standardize column names** using `COLUMN_MAPPINGS` from config
2. **Handle missing data** (forward fill + linear interpolation within countries)
3. **Create log transformations** for all continuous variables
4. **Create regional indicators**:
   - `eap_dummy`: Eastern Partnership countries (0/1)
   - `eu_dummy`: EU countries (0/1)
   - Interaction terms: `log_price_x_eap`, `log_price_x_eu`
5. **Create lagged variables** for instrumental variable estimation
6. **Create time trends** (linear time trend, post-COVID dummy)

**Output Variables:**
- `log_fixed_broadband_subs` (dependent variable)
- `log_fixed_broad_price` (primary price measure: % GNI per capita)
- `log_fixed_broad_price_ppp`, `log_fixed_broad_price_usd` (alternative measures)
- `log_gdp_per_capita`, `log_population`, `log_population_density`
- `log_secure_internet_servers`, `regulatory_quality`
- Regional interactions and lagged instruments

**Usage:**
```bash
python code/data_preparation/02_prepare_data.py
```

**Output Location:**
```
data/processed/analysis_ready_data.csv
```

**Validation Checks:**
- ✅ Panel structure: 33 countries × 15 years = 495 observations
- ✅ No missing values in key variables after imputation
- ✅ All required variables present
- ✅ Log transformations properly defined (no -inf or NaN)

---

### `01_analysis.py` [OPTIONAL DIAGNOSTIC]

**Purpose:** Rigorous missing data diagnostics and advanced imputation methods

**When to Use:**
- Publication-grade missing data analysis
- Sensitivity analysis comparing imputation methods
- Generating methodology documentation for papers

**Methods Implemented:**
1. **MCAR Test** (Missing Completely at Random): Little's test
2. **Missing Pattern Analysis**: Heatmaps, co-missingness matrices
3. **Multiple Imputation** (MICE): 5 imputed datasets
4. **EM Algorithm**: Expectation-Maximization imputation
5. **Panel-Specific Methods**: LOCF, linear interpolation, spline
6. **Comparison Framework**: Sensitivity analysis across methods

**Outputs:**
```
data/processed/missing_data_analysis/
├── 01_overall_missingness.xlsx          # Summary statistics
├── 02_missing_by_year.xlsx              # Temporal patterns
├── 03_comissingness_heatmap.png         # Visual diagnostics
├── 04_imputation_method_comparison.xlsx # Method comparison
├── 05_methodology_text_for_paper.txt    # Publication text
├── imputed_data_m1.xlsx                 # MICE dataset 1
├── imputed_data_m2.xlsx                 # MICE dataset 2
├── imputed_data_m3.xlsx                 # MICE dataset 3
├── imputed_data_m4.xlsx                 # MICE dataset 4
└── imputed_data_m5.xlsx                 # MICE dataset 5
```

**Usage:**
```bash
python code/data_preparation/01_analysis.py
```

**Note:** This script is **NOT** part of the main pipeline. It's a diagnostic tool for publication-quality missing data analysis. The main pipeline uses `02_prepare_data.py` which implements simple, transparent imputation (forward fill + interpolation).

---

## 🔄 Pipeline Integration

### Stage 1: Data Collection → Stage 2: **Data Preparation** → Stage 3: Analysis

**Input Requirements:**
- `data/processed/data_merged_with_series.xlsx` must exist
- Created by: `code/data_collection/step4_merge_datasets.py`

**Output Used By:**
- `code/analysis/pre_covid/two_way_fe.py` (pre-COVID analysis)
- `code/analysis/full_sample/two_way_fe_full_sample.py` (full sample)
- `code/analysis/full_sample/covid_diagnostics.py` (diagnostics)

---

## 🛠️ Technical Details

### Missing Data Strategy

**Philosophy:** Transparent, simple, replicable

**Method:**
1. **Sort by country and year** (ensure temporal ordering)
2. **Forward fill within country** (carry last observation forward)
3. **Linear interpolation within country** (fill gaps between observations)
4. **Report before/after** (document missingness rates)

**Affected Variables:**
- `internet_users_pct`: 11 missing (2.2%) → 0 after imputation
- `int_bandwidth`: 157 missing (31.7%) → 0 after imputation
- `fixed_broad_price`: 1 missing (0.2%) → 0 after imputation

### Log Transformations

**Variables transformed:**
- Subscriptions: `fixed_broadband_subs`, `mobile_subs`
- Prices: `fixed_broad_price`, `fixed_broad_price_usd`, `fixed_broad_price_ppp`, `mobile_broad_price`
- Economics: `gdp_per_capita`, `population`, `population_density`
- Infrastructure: `secure_internet_servers`, `int_bandwidth`

**Method:** `log(x + 1e-8)` to avoid log(0)

### Regional Classification

**EU Countries (27):**
Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden

**Eastern Partnership Countries (6):**
Armenia, Azerbaijan, Belarus, Georgia, Moldova, Ukraine

---

## 📊 Output Validation

After running `02_prepare_data.py`, verify:

```python
import pandas as pd

df = pd.read_csv('data/processed/analysis_ready_data.csv')

# Check dimensions
print(f"Observations: {len(df)}")  # Should be 495
print(f"Countries: {df['country'].nunique()}")  # Should be 33
print(f"Years: {df['year'].nunique()}")  # Should be 15

# Check missing values
print(df[['log_fixed_broadband_subs', 'log_fixed_broad_price', 
          'log_gdp_per_capita']].isna().sum())  # Should all be 0

# Check panel structure
print(df.groupby('country')['year'].count().describe())  # All should be 15
```

---

## ⚙️ Configuration

Key parameters defined in `code/utils/config.py`:

```python
# Column mappings (raw → standardized names)
COLUMN_MAPPINGS = {
    'fixed_broadband_subs_i4213tfbb': 'fixed_broadband_subs',
    'fixed_broad_price_gni_pct': 'fixed_broad_price',
    # ... etc
}

# Variables to log-transform
LOG_TRANSFORM_VARS = [
    'fixed_broadband_subs', 'internet_users_pct', 'int_bandwidth',
    'gdp_per_capita', 'population', 'population_density', # ... etc
]

# Country classifications
EU_COUNTRIES = ['AUT', 'BEL', 'BGR', ...]  # 27 countries
EAP_COUNTRIES = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']  # 6 countries
```

---

## 🚨 Troubleshooting

### Issue: Missing input file

**Error:** `FileNotFoundError: data_merged_with_series.xlsx`

**Solution:** Run data collection first:
```bash
python code/data_collection/run_data_collection.py
```

### Issue: Missing values remain

**Check:** Look at which variables still have missing values:
```python
df = pd.read_csv('data/processed/analysis_ready_data.csv')
print(df.isna().sum()[df.isna().sum() > 0])
```

**Note:** Some auxiliary variables may have missing values (e.g., R&D expenditure, tertiary enrollment). Core variables (subscriptions, price, GDP) should have zero missing values.

### Issue: Panel imbalance

**Check:** Some countries may be missing certain years:
```python
df.groupby('country')['year'].count().value_counts()
```

**Expected:** All 33 countries should have 15 observations (2010-2024)

---

## 📚 References

**Missing Data Methods:**
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
- Little, R. J., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- Allison, P. D. (2001). *Missing Data*. Sage Publications.

**Panel Data Imputation:**
- Honaker, J., & King, G. (2010). What to do about missing values in time-series cross-section data. *American Journal of Political Science*, 54(2), 561-581.

---

## 📞 Support

For questions about data preparation:
- Review transformations: Check `02_prepare_data.py` source code
- Validate output: Use validation checks above
- Check configuration: Review `code/utils/config.py`

---

**Last Updated:** December 14, 2025  
**Author:** Samir Orujov, ADA University
