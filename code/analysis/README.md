# Analysis Scripts

This directory contains econometric analysis scripts that estimate broadband price elasticity using two-way fixed effects models with Driscoll-Kraay standard errors.

## 📋 Overview

**Pipeline Stage:** Stage 3 (after data preparation)

**Input:** `data/processed/analysis_ready_data.csv` (output from Stage 2)  
**Output:** 
- `results/regression_output/` (Excel files with regression results)
- `results/figures/` (publication-quality PNG figures, 300 DPI)

## 📊 Estimation Method

### Two-Way Fixed Effects Model

**Specification:**
```
y_it = β₁·Price_it + β₂·(Price_it × EaP_i) + X_it'γ + α_i + δ_t + ε_it
```

Where:
- **y_it**: log(broadband subscriptions per 100) in country *i*, year *t*
- **Price_it**: log(price as % of GNI per capita)
- **EaP_i**: dummy for Eastern Partnership countries
- **X_it**: control variables (GDP, urbanization, education, etc.)
- **α_i**: country fixed effects (time-invariant heterogeneity)
- **δ_t**: year fixed effects (common time shocks)
- **ε_it**: error term

**Standard Errors:** Driscoll-Kraay (1998)
- Robust to heteroskedasticity
- Accounts for serial correlation (autocorrelation within countries)
- Accounts for cross-sectional dependence (common shocks across countries)

**Implementation:**
```python
from linearmodels.panel import PanelOLS

model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)
```

---

## 📄 Scripts

### `pre_covid/two_way_fe.py` ⭐ **[PRE-COVID ANALYSIS]**

**Purpose:** Estimate baseline price elasticity for 2010-2019 period

**Sample:** 330 observations (33 countries × 10 years)

**Analysis:**
1. **Baseline Model** (Full Controls):
   - All available controls (GDP, urbanization, education, regulatory quality, infrastructure, R&D, demographics, macroeconomic)
   - Primary price measure: GNI% (price as % of GNI per capita)
   - Reports EU and EaP elasticities separately

2. **Robustness Check 1: Control Specifications** (8 variations):
   - Full Controls (baseline)
   - Comprehensive (key dimensions)
   - Core (parsimonious: GDP, urban, education)
   - Institutional (governance focus)
   - Infrastructure (digital infrastructure)
   - Demographic (population characteristics)
   - Macroeconomic (economic conditions)
   - Minimal (GDP only)

3. **Robustness Check 2: Price Measurements** (3 alternatives):
   - GNI% (primary): Price as % of GNI per capita
   - PPP: Price in PPP-adjusted dollars
   - USD: Price in nominal USD

**Total Specifications:** 24 (8 controls × 3 price measures)

**Outputs:**
```
results/regression_output/pre_covid_analysis/
├── extended_control_specifications.xlsx  # 8 control specs with GNI%
└── price_robustness_matrix.xlsx          # All 24 specifications
```

**Key Findings:**
- **EaP elasticity**: ε ≈ -0.61*** (highly significant, robust across specs)
- **EU elasticity**: ε ≈ -0.12** (marginally significant)
- **Ratio**: EaP 5.3× more price-elastic than EU
- **Measurement matters**: GNI% yields 100% significance; PPP only 25%

**Usage:**
```bash
python code/analysis/pre_covid/two_way_fe.py
```

---

### `full_sample/two_way_fe_full_sample.py` ⭐ **[FULL SAMPLE WITH COVID]**

**Purpose:** Estimate time-varying elasticity across full 2010-2024 period with COVID interactions

**Sample:** 495 observations (33 countries × 15 years)

**Extended Specification:**
```
y_it = β₁·Price_it + β₂·(Price_it × EaP_i)
       + β₃·(Price_it × COVID_t) + β₄·(Price_it × EaP_i × COVID_t)
       + X_it'γ + α_i + δ_t + ε_it
```

**Time-Varying Elasticities:**
- **EU (Pre-COVID)**: ε = β₁
- **EaP (Pre-COVID)**: ε = β₁ + β₂
- **EU (COVID)**: ε = β₁ + β₃
- **EaP (COVID)**: ε = β₁ + β₂ + β₃ + β₄

**Analysis:**
1. **Baseline Model** (Full Controls + COVID interactions):
   - COVID dummy (2020-2024) absorbed by time fixed effects
   - Price×COVID interaction (change in EU elasticity)
   - Price×EaP×COVID triple interaction (change in regional difference)

2. **Robustness**: Same 24 specifications as pre-COVID analysis

**Outputs:**
```
results/regression_output/full_sample_covid_analysis/
├── extended_control_specifications.xlsx  # 8 control specs with GNI%
└── price_robustness_matrix.xlsx          # All 24 specifications
```

**Key Findings:**
- **Pre-COVID**: EU ε = -0.23***, EaP ε = -0.33** (both price-elastic)
- **COVID**: EU ε = +0.08 (n.s.), EaP ε = +0.09 (n.s.) (price-inelastic)
- **Change**: Δε = +0.31*** (EU), +0.42*** (EaP) (structural break)
- **Interpretation**: Broadband became essential good with near-zero elasticity

**Usage:**
```bash
python code/analysis/full_sample/two_way_fe_full_sample.py
```

---

### `full_sample/covid_diagnostics.py` ⭐ **[DIAGNOSTIC TESTS]**

**Purpose:** Validate COVID effects and test for pre-existing trends

**Diagnostic Tests:**

#### 1. **Price Variation Test**
- Compares within-country price variation pre-COVID vs COVID
- Ensures sufficient identifying variation for FE estimation
- **Result**: COVID period has 72% of pre-COVID variation (adequate)

#### 2. **Sample Composition Test**
- Verifies no compositional changes (same countries in both periods)
- Checks for balanced panel structure
- **Result**: All 33 countries present in both periods (perfect overlap)

#### 3. **Year-by-Year Elasticity**
- Estimates separate elasticity for each year (2015-2024)
- Reveals temporal evolution of price sensitivity
- **Pattern**: Gradual decline from 2015, not sudden 2020 break

#### 4. **Placebo Test** (Critical!)
- Splits pre-COVID period: 2010-2014 (reference) vs 2015-2019 (treatment)
- Tests for pre-existing trend before COVID
- **Result**: Triple interaction significant (p=0.045) → pre-trend detected!

**Outputs:**
```
results/regression_output/full_sample_covid_analysis/
├── year_by_year_elasticities.xlsx       # Annual estimates 2015-2024
└── placebo_test_results.xlsx            # Pre-trend test results

results/figures/covid_diagnostics/
└── diagnostic_covid_effects.png         # Visualization of tests
```

**Critical Finding:**
- ⚠️ **Pre-trend exists**: Elasticity began declining in 2015, not 2020
- ✅ **Revised narrative**: Decade-long transformation (2015-2024), not COVID shock
- 💡 **Implication**: Digital economy expansion reduced price sensitivity gradually

**Usage:**
```bash
python code/analysis/full_sample/covid_diagnostics.py
```

---

### `analysis_visualizations.py` ⭐ **[FIGURE GENERATION]**

**Purpose:** Generate all 6 publication-quality figures from regression outputs

**Figures Generated:**

#### Figure 1: Temporal Evolution (2015-2024)
- Year-by-year elasticity estimates
- Shows three phases: Traditional (2010-14), Transition (2015-19), Essential (2020-24)
- Visualizes gradual decline, not sudden break

#### Figure 2: Robustness Across Control Specifications
- Grouped bar chart: 8 control specifications
- Shows EaP elasticity robust across all specs
- Demonstrates EU elasticity more variable

#### Figure 3: Price Measurement Comparison
- Panel A: Magnitude of elasticities by price measure
- Panel B: Statistical significance rates
- Shows GNI% yields consistent results, PPP problematic

#### Figure 4: COVID Comparison (Disappearance of Elasticity)
- Before/after bars: Pre-COVID vs COVID
- Change arrows showing structural break
- Both regions transition to price-inelastic demand

#### Figure 5: Placebo Test (Pre-Trend Detection)
- Panel A: Three-phase evolution (2010-14, 2015-19, 2020-24)
- Panel B: Placebo test results
- Confirms pre-COVID trend exists (p=0.045)

#### Figure 6: Complete Results Matrix (Heatmap)
- 8 rows (control specs) × 3 columns (price measures)
- Shows EaP elasticity across all 24 specifications
- Color-coded by magnitude with significance stars

**Requirements:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Outputs:**
```
results/figures/analysis_figures/
├── fig1_temporal_evolution.png       # 300 DPI
├── fig2_robustness_specs.png        # 300 DPI
├── fig3_price_measurement.png       # 300 DPI
├── fig4_covid_comparison.png        # 300 DPI
├── fig5_placebo_test.png            # 300 DPI
└── fig6_results_matrix.png          # 300 DPI
```

**Usage:**
```bash
python code/analysis/analysis_visualizations.py
```

**Note:** This script loads Excel outputs from regression scripts. Run analysis scripts first!

---

## 🔄 Execution Order

### Recommended Sequence:

```bash
# 1. Pre-COVID analysis (2010-2019)
python code/analysis/pre_covid/two_way_fe.py

# 2. Full sample analysis (2010-2024)
python code/analysis/full_sample/two_way_fe_full_sample.py

# 3. Diagnostic tests
python code/analysis/full_sample/covid_diagnostics.py

# 4. Generate all figures
python code/analysis/analysis_visualizations.py
```

### Or run complete pipeline:

```bash
python code/main.py
```

---

## 📊 Control Specifications

### Full Controls (Baseline)
```python
controls = [
    'log_gdp_per_capita',           # Economic development
    'urban_population_pct',         # Urbanization
    'education_tertiary_pct',       # Human capital
    'regulatory_quality',           # Institutions
    'log_secure_internet_servers',  # Infrastructure
    'rd_expenditure_pct',          # Innovation
    'log_population_density',      # Demographics
    'inflation_pct',               # Macroeconomic
    'gdp_growth_pct'               # Macroeconomic
]
```

### Comprehensive (Key Dimensions)
- GDP, urban, education, regulatory quality, infrastructure

### Core (Parsimonious)
- GDP, urban, education

### Institutional (Governance)
- GDP, regulatory quality

### Infrastructure (Digital)
- GDP, secure servers, R&D

### Demographic (Population)
- GDP, urban, population density, age dependency

### Macroeconomic (Economic Conditions)
- GDP, growth, inflation

### Minimal
- GDP only

---

## 🔬 Technical Details

### Driscoll-Kraay Standard Errors

**Why use Driscoll-Kraay?**
1. **Heteroskedasticity**: Error variance differs across countries/years
2. **Serial correlation**: Errors correlated within countries over time
3. **Cross-sectional dependence**: Common shocks (COVID, financial crisis) affect all countries

**Parameters:**
```python
cov_type='kernel'        # Kernel-based estimator
kernel='bartlett'        # Bartlett kernel (triangular weights)
bandwidth=3              # Lag truncation (accommodates up to 3-year correlation)
```

**Reference:** Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549-560.

### Fixed Effects

**Country FE (α_i):** Controls for time-invariant country characteristics
- Geography, culture, institutional history, etc.
- Removes all between-country variation
- Identifies from within-country changes over time

**Year FE (δ_t):** Controls for common time shocks
- Global trends, technology changes, financial crises
- Removes all cross-sectional variation
- Identifies from differential country responses to time

**Identification:** Price elasticity identified from within-country deviations from country-specific means, after removing common time trends

### Interaction Terms

**Regional Heterogeneity (Price × EaP):**
- Tests if EaP countries respond differently to price changes than EU
- EaP coefficient = (β₁ + β₂) - β₁ = β₂

**COVID Heterogeneity (Price × COVID):**
- Tests if COVID changed price elasticity for EU countries
- COVID effect on EU = β₃

**Triple Interaction (Price × EaP × COVID):**
- Tests if COVID changed EaP-EU difference
- COVID effect on EaP difference = β₄

---

## 📈 Interpretation Guide

### Elasticity Values

| Range | Interpretation | Example |
|-------|---------------|---------|
| ε < -0.5 | Highly elastic | 10% price drop → >5% subscription increase |
| -0.5 < ε < -0.2 | Moderately elastic | 10% price drop → 2-5% increase |
| -0.2 < ε < -0.1 | Marginally elastic | 10% price drop → 1-2% increase |
| -0.1 < ε < 0.1 | Inelastic | Price changes have minimal effect |
| ε > 0.1 | Perverse (unusual) | Higher price → more subscriptions |

### Significance Levels

| Symbol | p-value | Interpretation |
|--------|---------|----------------|
| *** | p < 0.01 | Highly significant |
| ** | p < 0.05 | Significant |
| * | p < 0.10 | Marginally significant |
| (blank) | p ≥ 0.10 | Not significant |

### Regional Differences

**Pre-COVID (2010-2019):**
- EaP countries: Higher elasticity (ε ≈ -0.61) → price-sensitive
- EU countries: Lower elasticity (ε ≈ -0.12) → less responsive
- **Why?** Income effects stronger in developing markets

**COVID (2020-2024):**
- Both regions: Near-zero elasticity (ε ≈ 0) → price-inelastic
- Convergence of regional differences
- **Why?** Broadband became essential necessity for all

---

## 📋 Output Validation

### Check Regression Results

```python
import pandas as pd

# Load pre-COVID results
df_pre = pd.read_excel('results/regression_output/pre_covid_analysis/extended_control_specifications.xlsx')

# Check baseline
baseline = df_pre[df_pre['specification'] == 'Full Controls (Baseline)']
print(f"EU elasticity: {baseline['eu_elasticity'].iloc[0]:.3f} (p={baseline['eu_pval'].iloc[0]:.3f})")
print(f"EaP elasticity: {baseline['eap_elasticity'].iloc[0]:.3f} (p={baseline['eap_pval'].iloc[0]:.3f})")

# Check robustness
print(f"\nEaP elasticity range: [{df_pre['eap_elasticity'].min():.3f}, {df_pre['eap_elasticity'].max():.3f}]")
print(f"EaP significant count: {(df_pre['eap_pval'] < 0.05).sum()}/8")
```

### Check Figures

All figures should be:
- ✅ 300 DPI (publication-quality)
- ✅ Clear labels and titles
- ✅ Proper legends
- ✅ Consistent color scheme (EU=blue, EaP=purple)
- ✅ File size: ~300-500 KB per figure

---

## 🚨 Troubleshooting

### Issue: "Module linearmodels not found"

**Solution:** Install required package:
```bash
pip install linearmodels
```

### Issue: Regression results differ from README

**Check:**
1. Data file version: `data/processed/analysis_ready_data.csv` up to date?
2. Sample period: Pre-COVID script should use 2010-2019 only
3. Control variables: All required variables present in data?

### Issue: Figures not generated

**Common causes:**
1. Missing Excel outputs (run regression scripts first)
2. Missing dependencies (matplotlib, seaborn)
3. File path issues (check working directory)

**Solution:**
```bash
# Ensure outputs exist
ls results/regression_output/pre_covid_analysis/
ls results/regression_output/full_sample_covid_analysis/

# Run complete pipeline
python code/main.py
```

### Issue: Standard errors too small/large

**Check:**
1. Clustering level: Using Driscoll-Kraay (not clustered by country)?
2. Bandwidth parameter: Should be 3 (accommodates 3-year autocorrelation)
3. Sample size: 330 (pre-COVID) or 495 (full sample)?

---

## 📚 Key References

**Econometric Methods:**
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549-560.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets. *Review of Financial Studies*, 22(1), 435-480.

**Panel Data Analysis:**
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.

**Telecommunications Demand:**
- Grzybowski, L. (2015). Fixed-to-mobile substitution in the European Union. *Telecommunications Policy*, 39(11), 1015-1031.
- Hauge, J. A., & Prieger, J. E. (2010). Demand-side programs to stimulate adoption of broadband. *Information Economics and Policy*, 22(3), 223-239.

---

## 💡 Tips for Publication

### Writing Results

**Do:**
- ✅ Report coefficient, standard error, p-value
- ✅ Interpret magnitude (elasticity × 10% price change)
- ✅ Compare across specifications (robustness)
- ✅ Discuss economic significance, not just statistical

**Don't:**
- ❌ Report t-statistics (use p-values instead)
- ❌ Over-interpret marginally significant results
- ❌ Ignore failed robustness checks
- ❌ Claim causality without proper identification

### Table Format (Example)

```
Table 1: Baseline Results (Pre-COVID: 2010-2019)

                          EU              EaP
Price (% GNI)        -0.115**        -0.609***
                     (0.055)         (0.077)
                     [0.036]         [0.000]

Controls             Yes             Yes
Country FE           Yes             Yes
Year FE              Yes             Yes
N                    330             330
R²                   0.408           0.408

Notes: Two-way fixed effects with Driscoll-Kraay standard errors
in parentheses, p-values in brackets. ***p<0.01, **p<0.05, *p<0.10.
```

---

**Last Updated:** December 14, 2025  
**Author:** Samir Orujov, ADA University
