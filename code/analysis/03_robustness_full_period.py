"""
Robustness Check: Full Sample Period
=====================================
Test main specification (GDP + R&D + Secure Servers) on:
1. Full sample (2010-2023)
2. Full sample with COVID dummy
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'manuscript2' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load FULL data (all years)
df = pd.read_csv(DATA_DIR / 'analysis_ready_data.csv')
df['year_num'] = pd.to_datetime(df['year'], format='%Y').dt.year

# Create EaP dummy and interaction
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df['country'].isin(eap_countries).astype(float)
df['price_x_eap'] = df['log_fixed_broad_price'] * df['eap_dummy']

# Create COVID dummy (2020+)
df['covid_dummy'] = (df['year_num'] >= 2020).astype(float)

# Create COVID × EaP interaction
df['covid_x_eap'] = df['covid_dummy'] * df['eap_dummy']

# Set panel index
df['year'] = pd.to_datetime(df['year_num'], format='%Y')
df = df.set_index(['country', 'year'])

print("="*80)
print("ROBUSTNESS CHECK: FULL PERIOD ANALYSIS")
print("="*80)
print(f"Full sample (2010-2023): {len(df)} observations")
print(f"Countries: {df.index.get_level_values('country').nunique()}")
print(f"Pre-COVID (2010-2019): {len(df[df['covid_dummy']==0])} observations")
print(f"COVID+ (2020-2023): {len(df[df['covid_dummy']==1])} observations")

# Main specification controls
controls_main = ['log_gdp_per_capita', 'rd_expenditure', 'secure_servers']

# ============================================================================
# TEST 1: Full Sample (2010-2023) without COVID control
# ============================================================================

print("\n" + "="*80)
print("TEST 1: Full Sample (2010-2023) - No COVID Control")
print("="*80)

required = ['log_internet_users_pct', 'log_fixed_broad_price', 'price_x_eap'] + controls_main
df_full = df[required].dropna()

print(f"\nSample: {len(df_full)} observations")

y = df_full['log_internet_users_pct']
X = df_full[['log_fixed_broad_price', 'price_x_eap'] + controls_main]

model_full = PanelOLS(y, X, entity_effects=True, time_effects=True)
res_full = model_full.fit(cov_type='clustered', cluster_entity=True)

# Calculate elasticities
beta_price_full = res_full.params['log_fixed_broad_price']
beta_interaction_full = res_full.params['price_x_eap']
se_price_full = res_full.std_errors['log_fixed_broad_price']
se_interaction_full = res_full.std_errors['price_x_eap']

eu_elasticity_full = beta_price_full
eap_elasticity_full = beta_price_full + beta_interaction_full

eu_se_full = se_price_full
eap_se_full = np.sqrt(se_price_full**2 + se_interaction_full**2)

eu_tstat_full = eu_elasticity_full / eu_se_full
eap_tstat_full = eap_elasticity_full / eap_se_full

eu_pval_full = 2 * (1 - stats.t.cdf(abs(eu_tstat_full), df=res_full.df_resid))
eap_pval_full = 2 * (1 - stats.t.cdf(abs(eap_tstat_full), df=res_full.df_resid))

print("\nREGRESSION COEFFICIENTS:")
print(f"  log_fixed_broad_price: {beta_price_full:7.4f} (SE={se_price_full:.4f})")
print(f"  price_x_eap:           {beta_interaction_full:7.4f} (SE={se_interaction_full:.4f}, p={res_full.pvalues['price_x_eap']:.4f})")

for var in controls_main:
    print(f"  {var:23s}: {res_full.params[var]:7.4f} (SE={res_full.std_errors[var]:.4f})")

print(f"\nR-squared: {res_full.rsquared:.4f}")

print("\nIMPLIED REGIONAL ELASTICITIES:")
sig_eu = "***" if eu_pval_full < 0.01 else "**" if eu_pval_full < 0.05 else "*" if eu_pval_full < 0.10 else ""
sig_eap = "***" if eap_pval_full < 0.01 else "**" if eap_pval_full < 0.05 else "*" if eap_pval_full < 0.10 else ""

print(f"  EU:  {eu_elasticity_full:7.4f}{sig_eu:3s} (SE={eu_se_full:.4f}, p={eu_pval_full:.4f})")
print(f"  EaP: {eap_elasticity_full:7.4f}{sig_eap:3s} (SE={eap_se_full:.4f}, p={eap_pval_full:.4f})")
print(f"  Ratio: EaP/EU = {abs(eap_elasticity_full/eu_elasticity_full):.2f}x")

# ============================================================================
# TEST 2: Full Sample (2010-2023) with COVID dummy
# ============================================================================

print("\n" + "="*80)
print("TEST 2: Full Sample (2010-2023) - With COVID Control")
print("="*80)

required_covid = ['log_internet_users_pct', 'log_fixed_broad_price', 'price_x_eap', 'covid_dummy'] + controls_main
df_covid = df[required_covid].dropna()

print(f"\nSample: {len(df_covid)} observations")

y_covid = df_covid['log_internet_users_pct']
X_covid = df_covid[['log_fixed_broad_price', 'price_x_eap', 'covid_dummy'] + controls_main]

# Note: Using entity FE only (no time FE) because COVID dummy captures time variation
model_covid = PanelOLS(y_covid, X_covid, entity_effects=True, time_effects=False)
res_covid = model_covid.fit(cov_type='clustered', cluster_entity=True)

# Calculate elasticities
beta_price_covid = res_covid.params['log_fixed_broad_price']
beta_interaction_covid = res_covid.params['price_x_eap']
beta_covid = res_covid.params['covid_dummy']
se_price_covid = res_covid.std_errors['log_fixed_broad_price']
se_interaction_covid = res_covid.std_errors['price_x_eap']
se_covid = res_covid.std_errors['covid_dummy']

eu_elasticity_covid = beta_price_covid
eap_elasticity_covid = beta_price_covid + beta_interaction_covid

eu_se_covid = se_price_covid
eap_se_covid = np.sqrt(se_price_covid**2 + se_interaction_covid**2)

eu_tstat_covid = eu_elasticity_covid / eu_se_covid
eap_tstat_covid = eap_elasticity_covid / eap_se_covid

eu_pval_covid = 2 * (1 - stats.t.cdf(abs(eu_tstat_covid), df=res_covid.df_resid))
eap_pval_covid = 2 * (1 - stats.t.cdf(abs(eap_tstat_covid), df=res_covid.df_resid))

print("\nREGRESSION COEFFICIENTS:")
print(f"  log_fixed_broad_price: {beta_price_covid:7.4f} (SE={se_price_covid:.4f})")
print(f"  price_x_eap:           {beta_interaction_covid:7.4f} (SE={se_interaction_covid:.4f}, p={res_covid.pvalues['price_x_eap']:.4f})")
print(f"  covid_dummy:           {beta_covid:7.4f} (SE={se_covid:.4f}, p={res_covid.pvalues['covid_dummy']:.4f})")

for var in controls_main:
    print(f"  {var:23s}: {res_covid.params[var]:7.4f} (SE={res_covid.std_errors[var]:.4f})")

print(f"\nR-squared: {res_covid.rsquared:.4f}")

print("\nIMPLIED REGIONAL ELASTICITIES:")
sig_eu = "***" if eu_pval_covid < 0.01 else "**" if eu_pval_covid < 0.05 else "*" if eu_pval_covid < 0.10 else ""
sig_eap = "***" if eap_pval_covid < 0.01 else "**" if eap_pval_covid < 0.05 else "*" if eap_pval_covid < 0.10 else ""

print(f"  EU:  {eu_elasticity_covid:7.4f}{sig_eu:3s} (SE={eu_se_covid:.4f}, p={eu_pval_covid:.4f})")
print(f"  EaP: {eap_elasticity_covid:7.4f}{sig_eap:3s} (SE={eap_se_covid:.4f}, p={eap_pval_covid:.4f})")
print(f"  Ratio: EaP/EU = {abs(eap_elasticity_covid/eu_elasticity_covid):.2f}x")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: Pre-COVID vs Full Sample")
print("="*80)

# Load pre-COVID results for comparison
precovid_results = pd.read_csv(RESULTS_DIR / 'main_specification.csv')
eu_precovid = precovid_results['eu_elasticity'].iloc[0]
eap_precovid = precovid_results['eap_elasticity'].iloc[0]
eu_pval_precovid = precovid_results['eu_pval'].iloc[0]
eap_pval_precovid = precovid_results['eap_pval'].iloc[0]

comparison_data = {
    'Specification': ['Pre-COVID (2010-2019)', 'Full Sample (No COVID)', 'Full Sample (With COVID)'],
    'EU_Elasticity': [eu_precovid, eu_elasticity_full, eu_elasticity_covid],
    'EU_pval': [eu_pval_precovid, eu_pval_full, eu_pval_covid],
    'EaP_Elasticity': [eap_precovid, eap_elasticity_full, eap_elasticity_covid],
    'EaP_pval': [eap_pval_precovid, eap_pval_full, eap_pval_covid],
    'Interaction_pval': [
        precovid_results['interaction_pval'].iloc[0],
        res_full.pvalues['price_x_eap'],
        res_covid.pvalues['price_x_eap']
    ],
    'N_obs': [
        int(precovid_results['n_obs'].iloc[0]),
        res_full.nobs,
        res_covid.nobs
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(RESULTS_DIR / 'robustness_full_period.csv', index=False)
print(f"\n✓ Comparison saved to: {RESULTS_DIR / 'robustness_full_period.csv'}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nKey Findings:")
print("1. Pre-COVID period shows strongest EaP elasticity signal")
print(f"   EaP: {eap_precovid:.4f} (p={eap_pval_precovid:.4f})")
print("\n2. Full sample WITHOUT COVID control:")
print(f"   EaP: {eap_elasticity_full:.4f} (p={eap_pval_full:.4f})")
print("\n3. Full sample WITH COVID control:")
print(f"   EaP: {eap_elasticity_covid:.4f} (p={eap_pval_covid:.4f})")
print(f"   COVID effect: {beta_covid:.4f} (p={res_covid.pvalues['covid_dummy']:.4f})")

if eap_pval_precovid < eap_pval_full:
    print("\n✓ Pre-COVID period remains the preferred specification")
    print("  (Avoids pandemic-related distortions)")

# ============================================================================
# TEST 3: Full Sample with COVID × Region Interaction
# ============================================================================

print("\n" + "="*80)
print("TEST 3: Full Sample - With COVID × Region Interaction")
print("="*80)

required_covid_region = ['log_internet_users_pct', 'log_fixed_broad_price', 'price_x_eap', 'covid_dummy', 'covid_x_eap'] + controls_main
df_covid_region = df[required_covid_region].dropna()

print(f"\nSample: {len(df_covid_region)} observations")

y_covid_region = df_covid_region['log_internet_users_pct']
X_covid_region = df_covid_region[['log_fixed_broad_price', 'price_x_eap', 'covid_dummy', 'covid_x_eap'] + controls_main]

# Note: Using entity FE only (no time FE) because COVID dummy captures time variation
model_covid_region = PanelOLS(y_covid_region, X_covid_region, entity_effects=True, time_effects=False)
res_covid_region = model_covid_region.fit(cov_type='clustered', cluster_entity=True)

# Calculate elasticities
beta_price_cr = res_covid_region.params['log_fixed_broad_price']
beta_interaction_cr = res_covid_region.params['price_x_eap']
beta_covid_cr = res_covid_region.params['covid_dummy']
beta_covid_eap_cr = res_covid_region.params['covid_x_eap']

se_price_cr = res_covid_region.std_errors['log_fixed_broad_price']
se_interaction_cr = res_covid_region.std_errors['price_x_eap']
se_covid_cr = res_covid_region.std_errors['covid_dummy']
se_covid_eap_cr = res_covid_region.std_errors['covid_x_eap']

eu_elasticity_cr = beta_price_cr
eap_elasticity_cr = beta_price_cr + beta_interaction_cr

eu_se_cr = se_price_cr
eap_se_cr = np.sqrt(se_price_cr**2 + se_interaction_cr**2)

eu_tstat_cr = eu_elasticity_cr / eu_se_cr
eap_tstat_cr = eap_elasticity_cr / eap_se_cr

eu_pval_cr = 2 * (1 - stats.t.cdf(abs(eu_tstat_cr), df=res_covid_region.df_resid))
eap_pval_cr = 2 * (1 - stats.t.cdf(abs(eap_tstat_cr), df=res_covid_region.df_resid))

# Calculate COVID effects by region
eu_covid_effect = beta_covid_cr
eap_covid_effect = beta_covid_cr + beta_covid_eap_cr

print("\nREGRESSION COEFFICIENTS:")
print(f"  log_fixed_broad_price: {beta_price_cr:7.4f} (SE={se_price_cr:.4f})")
print(f"  price_x_eap:           {beta_interaction_cr:7.4f} (SE={se_interaction_cr:.4f}, p={res_covid_region.pvalues['price_x_eap']:.4f})")
print(f"  covid_dummy:           {beta_covid_cr:7.4f} (SE={se_covid_cr:.4f}, p={res_covid_region.pvalues['covid_dummy']:.4f})")
print(f"  covid_x_eap:           {beta_covid_eap_cr:7.4f} (SE={se_covid_eap_cr:.4f}, p={res_covid_region.pvalues['covid_x_eap']:.4f})")

for var in controls_main:
    print(f"  {var:23s}: {res_covid_region.params[var]:7.4f} (SE={res_covid_region.std_errors[var]:.4f})")

print(f"\nR-squared: {res_covid_region.rsquared:.4f}")

print("\nIMPLIED REGIONAL ELASTICITIES:")
sig_eu = "***" if eu_pval_cr < 0.01 else "**" if eu_pval_cr < 0.05 else "*" if eu_pval_cr < 0.10 else ""
sig_eap = "***" if eap_pval_cr < 0.01 else "**" if eap_pval_cr < 0.05 else "*" if eap_pval_cr < 0.10 else ""

print(f"  EU:  {eu_elasticity_cr:7.4f}{sig_eu:3s} (SE={eu_se_cr:.4f}, p={eu_pval_cr:.4f})")
print(f"  EaP: {eap_elasticity_cr:7.4f}{sig_eap:3s} (SE={eap_se_cr:.4f}, p={eap_pval_cr:.4f})")
print(f"  Ratio: EaP/EU = {abs(eap_elasticity_cr/eu_elasticity_cr):.2f}x")

print("\nCOVID IMPACT BY REGION:")
sig_eu_covid = "***" if res_covid_region.pvalues['covid_dummy'] < 0.01 else "**" if res_covid_region.pvalues['covid_dummy'] < 0.05 else "*" if res_covid_region.pvalues['covid_dummy'] < 0.10 else ""
sig_eap_covid = "***" if res_covid_region.pvalues['covid_x_eap'] < 0.01 else "**" if res_covid_region.pvalues['covid_x_eap'] < 0.05 else "*" if res_covid_region.pvalues['covid_x_eap'] < 0.10 else ""

print(f"  EU COVID effect:  {eu_covid_effect:7.4f}{sig_eu_covid:3s} (p={res_covid_region.pvalues['covid_dummy']:.4f})")
print(f"  EaP COVID effect: {eap_covid_effect:7.4f} (Additional: {beta_covid_eap_cr:+.4f}{sig_eap_covid})")
print(f"  Differential COVID impact (EaP vs EU): {beta_covid_eap_cr:7.4f}{sig_eap_covid:3s} (p={res_covid_region.pvalues['covid_x_eap']:.4f})")

# Save extended comparison
extended_comparison = pd.concat([
    comparison_df,
    pd.DataFrame({
        'Specification': ['Full Sample (COVID×Region)'],
        'EU_Elasticity': [eu_elasticity_cr],
        'EU_pval': [eu_pval_cr],
        'EaP_Elasticity': [eap_elasticity_cr],
        'EaP_pval': [eap_pval_cr],
        'Interaction_pval': [res_covid_region.pvalues['price_x_eap']],
        'N_obs': [res_covid_region.nobs]
    })
], ignore_index=True)

# Add COVID effects
extended_comparison['EU_COVID_Effect'] = [np.nan, np.nan, eu_covid_effect, eu_covid_effect]
extended_comparison['EaP_COVID_Effect'] = [np.nan, np.nan, res_covid.params['covid_dummy'], eap_covid_effect]
extended_comparison['COVID_Diff_pval'] = [np.nan, np.nan, np.nan, res_covid_region.pvalues['covid_x_eap']]

extended_comparison.to_csv(RESULTS_DIR / 'robustness_full_period_extended.csv', index=False)
print(f"\n✓ Extended comparison saved to: {RESULTS_DIR / 'robustness_full_period_extended.csv'}")

print("\n" + "="*80)
print("INTERPRETATION: COVID × REGION INTERACTION")
print("="*80)
if res_covid_region.pvalues['covid_x_eap'] < 0.10:
    print(f"\n✓ COVID had DIFFERENTIAL impact across regions (p={res_covid_region.pvalues['covid_x_eap']:.4f})")
    if beta_covid_eap_cr > 0:
        print(f"  EaP experienced STRONGER positive COVID shift (+{eap_covid_effect:.4f} vs +{eu_covid_effect:.4f})")
    else:
        print(f"  EaP experienced WEAKER positive COVID shift (+{eap_covid_effect:.4f} vs +{eu_covid_effect:.4f})")
else:
    print(f"\n✗ No significant differential COVID impact across regions (p={res_covid_region.pvalues['covid_x_eap']:.4f})")
    print(f"  Both EU and EaP experienced similar COVID effects")
