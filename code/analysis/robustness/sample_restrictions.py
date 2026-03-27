# -*- coding: utf-8 -*-
"""
Sample Restriction Robustness Checks
=====================================
Estimates the main pre-COVID baseline (Full Controls, GNI% price) under four
alternative sample definitions:

  1. Balanced panel only  - restrict to the 2010-2019 pre-COVID years with
                            complete data for ALL 15 years (pre-COVID + COVID).
                            Here we restrict to countries present in all years
                            of the pre-COVID window (2010-2019).
  2. Outlier exclusion    - drop observations with Cook's distance > 4/N.
  3. High-income only     - countries whose average GDP per capita (2010-2019)
                            exceeds the sample median.
  4. Middle-income only   - countries whose average GDP per capita (2010-2019)
                            is below the sample median.

Outputs:
  results/regression_output/robustness/sample_restrictions.xlsx
  manuscript/tables/table8_sample_restrictions.tex
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import warnings
import sys
import io

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR,
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR,
    )

ROBUSTNESS_DIR = RESULTS_REGRESSION / 'robustness'
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONTROLS = [
    'log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
    'regulatory_quality_estimate', 'log_secure_internet_servers',
    'research_development_expenditure', 'log_population_density',
    'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator',
]

print("=" * 80)
print("SAMPLE RESTRICTION ROBUSTNESS CHECKS")
print("=" * 80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

df_full = pd.read_csv(ANALYSIS_READY_FILE)
df_full['year_num'] = df_full['year'].astype(int)
df_pre = df_full[df_full['year_num'] <= 2019].copy()
df_pre['eap_dummy'] = df_pre['country'].isin(EAP_COUNTRIES).astype(float)


# ============================================================================
# CORE ESTIMATION FUNCTION
# ============================================================================

def run_baseline(df_input, label):
    """Run the Full Controls pre-COVID baseline on the provided sample.

    Handles subsamples with no EaP countries by dropping the interaction term
    and reporting EaP elasticity as NaN.
    """
    df_est = df_input.copy()
    df_est['year_dt'] = pd.to_datetime(df_est['year_num'], format='%Y')
    df_est = df_est.set_index(['country', 'year_dt'])
    avail = [c for c in CONTROLS if c in df_est.columns]

    # Only include EaP interaction when EaP countries are present
    has_eap = df_est['eap_dummy'].sum() > 0
    if has_eap:
        df_est['price_x_eap'] = df_est[PRIMARY_PRICE] * df_est['eap_dummy']
        reg_cols = [PRIMARY_PRICE, 'price_x_eap'] + avail
    else:
        reg_cols = [PRIMARY_PRICE] + avail

    df_est = df_est[[PRIMARY_DV] + reg_cols].dropna()

    if len(df_est) < 30:
        print(f"  [{label}] Insufficient obs: {len(df_est)} — skipping")
        return None

    y = df_est[PRIMARY_DV]
    X = df_est[reg_cols]
    res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=3
    )

    b_eu    = res.params[PRIMARY_PRICE]
    se_eu   = res.std_errors[PRIMARY_PRICE]
    pval_eu = res.pvalues[PRIMARY_PRICE]
    n_countries = df_est.index.get_level_values('country').nunique()

    if has_eap:
        b_int    = res.params['price_x_eap']
        se_int   = res.std_errors['price_x_eap']
        eap_b    = b_eu + b_int
        eap_se   = np.sqrt(se_eu**2 + se_int**2 +
                           2 * res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
        eap_pval = 2 * (1 - stats.t.cdf(abs(eap_b / eap_se), df=res.df_resid))
        print(f"\n  [{label}]")
        print(f"    EU:   {b_eu:7.4f}  (p={pval_eu:.3f})")
        print(f"    EaP:  {eap_b:7.4f}  (p={eap_pval:.3f})")
    else:
        eap_b = eap_se = eap_pval = float('nan')
        print(f"\n  [{label}]")
        print(f"    Elasticity (EU only): {b_eu:7.4f}  (p={pval_eu:.3f})")
        print(f"    EaP: N/A — no EaP countries in subsample")

    print(f"    N={res.nobs}, Countries={n_countries}, R²={res.rsquared:.3f}")

    return {
        'restriction':    label,
        'eu_elasticity':  b_eu,
        'eu_se':          se_eu,
        'eu_pval':        pval_eu,
        'eap_elasticity': eap_b,
        'eap_se':         eap_se,
        'eap_pval':       eap_pval,
        'n_obs':          int(res.nobs),
        'n_countries':    n_countries,
        'r_squared':      res.rsquared,
    }


# ============================================================================
# VARIANT 1: FULL BASELINE (reference)
# ============================================================================

print("\n1. FULL BASELINE (reference):")
row_full = run_baseline(df_pre, 'Full pre-COVID sample')
results = [row_full] if row_full else []

# ============================================================================
# VARIANT 2: BALANCED PANEL ONLY
# ============================================================================

print("\n2. BALANCED PANEL (complete data for all 10 pre-COVID years):")
avail_controls = [c for c in CONTROLS if c in df_pre.columns]
vars_needed = [PRIMARY_DV, PRIMARY_PRICE] + avail_controls
df_check = df_pre.dropna(subset=vars_needed)
year_counts = df_check.groupby('country')['year_num'].count()
full_years = year_counts[year_counts == (df_pre['year_num'].nunique())].index
df_balanced = df_pre[df_pre['country'].isin(full_years)].copy()
print(f"   Countries with complete pre-COVID data: {len(full_years)} "
      f"({year_counts.max()} obs each)")
row_balanced = run_baseline(df_balanced, 'Balanced panel only')
if row_balanced:
    results.append(row_balanced)

# ============================================================================
# VARIANT 3: OUTLIER EXCLUSION (Cook's distance > 4/N)
# ============================================================================

print("\n3. OUTLIER EXCLUSION (Cook's distance > 4/N):")
# Approximate Cook's distance via OLS leverage and residuals
df_ols = df_pre.dropna(subset=[PRIMARY_DV, PRIMARY_PRICE, 'eap_dummy'] +
                       [c for c in CONTROLS if c in df_pre.columns])
df_ols = df_ols.copy()
df_ols['price_x_eap'] = df_ols[PRIMARY_PRICE] * df_ols['eap_dummy']
avail = [c for c in CONTROLS if c in df_ols.columns]

# Demean for within-estimator Cook's distance approximation
df_ols['_y'] = df_ols.groupby('country')[PRIMARY_DV].transform(
    lambda x: x - x.mean()) - df_ols.groupby('year_num')[PRIMARY_DV].transform(
    lambda x: x - x.mean()) + df_ols[PRIMARY_DV].mean()
feature_cols = [PRIMARY_PRICE, 'price_x_eap'] + avail
for col in feature_cols + ['_y']:
    df_ols[f'_dm_{col}'] = (
        df_ols.groupby('country')[col].transform(lambda x: x - x.mean())
        - df_ols.groupby('year_num')[col].transform(lambda x: x - x.mean())
        + df_ols[col].mean()
    )

X_dm = df_ols[[f'_dm_{c}' for c in feature_cols]].values
y_dm = df_ols['_dm__y'].values
k    = X_dm.shape[1]
n_obs = len(y_dm)

try:
    from numpy.linalg import lstsq, pinv
    beta, _, _, _ = lstsq(X_dm, y_dm, rcond=None)
    residuals = y_dm - X_dm @ beta
    H = X_dm @ pinv(X_dm.T @ X_dm) @ X_dm.T
    h_ii = np.diag(H)
    mse = np.mean(residuals**2)
    cook_d = (residuals**2 * h_ii) / (k * mse * (1 - h_ii + 1e-12)**2)
    threshold = 4.0 / n_obs
    keep_mask = cook_d <= threshold
    n_dropped = (~keep_mask).sum()
    print(f"   Threshold = 4/N = {threshold:.4f}; dropping {n_dropped} observations")
    df_ols['_keep'] = keep_mask
    df_no_outlier = df_pre.copy()
    df_no_outlier = df_no_outlier.merge(
        df_ols[['country', 'year_num', '_keep']],
        on=['country', 'year_num'],
        how='left'
    )
    df_no_outlier = df_no_outlier[df_no_outlier['_keep'].fillna(True)].drop(
        columns=['_keep']
    )
except Exception as e:
    print(f"   Cook's distance computation failed ({e}); using full sample")
    df_no_outlier = df_pre.copy()

row_no_outlier = run_baseline(df_no_outlier, 'Excluding outliers (Cook\'s d > 4/N)')
if row_no_outlier:
    results.append(row_no_outlier)

# ============================================================================
# VARIANT 4: HIGH-INCOME COUNTRIES (above median GDP/capita, 2010-2019)
# ============================================================================

print("\n4. HIGH-INCOME COUNTRIES (above median GDP per capita, 2010-2019):")
if 'log_gdp_per_capita' in df_pre.columns:
    country_mean_gdp = (df_pre.groupby('country')['log_gdp_per_capita']
                        .mean().dropna())
    gdp_median = country_mean_gdp.median()
    high_income_countries = country_mean_gdp[
        country_mean_gdp >= gdp_median].index.tolist()
    print(f"   Median log GDP/cap = {gdp_median:.3f} (~${np.exp(gdp_median):,.0f})")
    print(f"   High-income countries: {len(high_income_countries)}")
    df_high = df_pre[df_pre['country'].isin(high_income_countries)].copy()
    df_high['eap_dummy'] = df_high['country'].isin(EAP_COUNTRIES).astype(float)
    row_high = run_baseline(df_high, 'High-income (above median GDP/cap)')
    if row_high:
        results.append(row_high)
else:
    print("   log_gdp_per_capita not found in data — skipping")
    row_high = None
    gdp_median = None

# ============================================================================
# VARIANT 5: MIDDLE-INCOME COUNTRIES (below median GDP/capita, 2010-2019)
# ============================================================================

print("\n5. MIDDLE/LOWER-INCOME COUNTRIES (below median GDP per capita, 2010-2019):")
if 'log_gdp_per_capita' in df_pre.columns and gdp_median is not None:
    low_income_countries = country_mean_gdp[
        country_mean_gdp < gdp_median].index.tolist()
    print(f"   Middle/lower-income countries: {len(low_income_countries)}")
    df_low = df_pre[df_pre['country'].isin(low_income_countries)].copy()
    df_low['eap_dummy'] = df_low['country'].isin(EAP_COUNTRIES).astype(float)
    row_low = run_baseline(df_low, 'Middle/lower-income (below median GDP/cap)')
    if row_low:
        results.append(row_low)

# ============================================================================
# SAVE EXCEL
# ============================================================================

results_df = pd.DataFrame(results)
out_xlsx = ROBUSTNESS_DIR / 'sample_restrictions.xlsx'
results_df.to_excel(out_xlsx, index=False)
print(f"\n[OK] Excel results saved → {out_xlsx}")

# ============================================================================
# GENERATE LATEX TABLE 8
# ============================================================================

def fmt(coef, se, pval):
    if pd.isna(coef):
        return r'\multicolumn{1}{c}{---}', r'\multicolumn{1}{c}{}'
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    if stars:
        return f'${coef:.3f}^{{{stars}}}$', f'({se:.3f})'
    return f'${coef:.3f}$', f'({se:.3f})'


# Row labels (short form for table)
ROW_LABELS = {
    'Full pre-COVID sample':                     'Full sample (baseline)',
    'Balanced panel only':                       'Balanced panel only',
    "Excluding outliers (Cook's d > 4/N)":       r'Excl.\ outliers (Cook\'s $d>4/N$)',
    'High-income (above median GDP/cap)':        'High-income countries only',
    'Middle/lower-income (below median GDP/cap)':'Middle/lower-income only',
}

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Sample Restriction Robustness Checks (Pre-COVID Baseline, 2010--2019)}',
    r'\label{tab:sample_restrictions}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{EU Elasticity} & \multicolumn{2}{c}{EaP Elasticity} & & \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Sample Restriction & Coef. & SE & Coef. & SE & N & Countries \\',
    r'\midrule',
]

for _, row in results_df.iterrows():
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    label = ROW_LABELS.get(row['restriction'], row['restriction'])
    lines.append(f'{label} & {c_eu} & {s_eu} & {c_eap} & {s_eap} '
                 f'& {int(row["n_obs"])} & {int(row["n_countries"])} \\\\')
    lines.append(f' & & {s_eu} & & {s_eap} & & \\\\')

# Remove duplicate SE rows — show coef + SE stacked (standard format)
# Rebuild properly: each row is coef row then SE row
lines_clean = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Sample Restriction Robustness Checks (Pre-COVID Baseline, 2010--2019)}',
    r'\label{tab:sample_restrictions}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{EU Elasticity} & \multicolumn{2}{c}{EaP Elasticity} & & \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Sample Restriction & Coef. & SE & Coef. & SE & N & Countries \\',
    r'\midrule',
]

is_first = True
for _, row in results_df.iterrows():
    if not is_first:
        lines_clean.append(r'\addlinespace[4pt]')
    is_first = False
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    label = ROW_LABELS.get(row['restriction'], row['restriction'])
    lines_clean.append(
        f'{label} & {c_eu} & {c_eap} & {int(row["n_obs"])} '
        f'& {int(row["n_countries"])} & & \\\\'
    )
    lines_clean.append(f' & {s_eu} & {s_eap} & & & & \\\\')

# Fix column structure (6 cols: label, EU coef, EU SE, EaP coef, EaP SE, N, countries)
lines_final = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Sample Restriction Robustness Checks (Pre-COVID Baseline, Full Controls,',
    r'    GNI\%-price). Driscoll--Kraay standard errors in parentheses.',
    r'    Significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}',
    r'\label{tab:sample_restrictions}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{EU} & \multicolumn{2}{c}{EaP} & & \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Sample Restriction & $\hat\varepsilon$ & SE & $\hat\varepsilon$ & SE & $N$ & Countries \\',
    r'\midrule',
]

is_first = True
for _, row in results_df.iterrows():
    if not is_first:
        lines_final.append(r'\addlinespace[4pt]')
    is_first = False
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    label = ROW_LABELS.get(row['restriction'], row['restriction'])
    lines_final.append(
        f'{label} & {c_eu} & {s_eu} & {c_eap} & {s_eap} '
        f'& {int(row["n_obs"])} & {int(row["n_countries"])} \\\\'
    )

lines_final += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\begin{tablenotes}',
    r'\scriptsize',
    r'\item \textit{Notes:} All models estimated with two-way fixed effects (country and year)',
    r'    and Driscoll--Kraay standard errors (bandwidth = 3). Price variable is log fixed',
    r'    broadband price as \% of GNI per capita. High/middle-income split uses median',
    r'    country-level GDP per capita (constant 2015 USD) averaged over 2010--2019.',
    r'    Cook\'s distance threshold is $4/N$ where $N$ is the number of observations.',
    r'\end{tablenotes}',
    r'\end{minipage}',
    r'\end{table}',
]

tex_content = '\n'.join(lines_final)
out_tex = MANUSCRIPT_TABLES_DIR / 'table8_sample_restrictions.tex'
out_tex.write_text(tex_content, encoding='utf-8')
print(f"[OK] LaTeX table saved → {out_tex}")

print("\n" + "=" * 80)
print("SAMPLE RESTRICTION ROBUSTNESS — DONE")
print("=" * 80)
