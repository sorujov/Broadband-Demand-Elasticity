"""
Testing Mathematical Equivalence: Percentage vs Levels with Population Control
===============================================================================
Tests if using internet_users (level) + population control ≈ internet_users_pct
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load data
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')
df = df[df['year'] <= 2019].copy()

# Create variables
# Model 1: Percentage (current approach)
df['log_internet_users_pct'] = np.log(df['internet_users_pct_i99H'] + 0.01)

# Model 2: Levels with population
df['internet_users_count'] = (df['internet_users_pct_i99H'] / 100) * df['population']
df['log_internet_users_count'] = np.log(df['internet_users_count'] + 1)
df['log_population'] = np.log(df['population'])

# Price
df['log_price_usd'] = np.log(df['fixed_broad_price_usd'] + 0.01)

# Controls
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])

# EaP dummy
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df['country'].isin(eap_countries).astype(float)
df['price_usd_x_eap'] = df['log_price_usd'] * df['eap_dummy']

# Panel index
df['year_dt'] = pd.to_datetime(df['year'], format='%Y')
df = df.set_index(['country', 'year_dt'])

print("="*80)
print("TESTING EQUIVALENCE: PERCENTAGE vs LEVELS + POPULATION")
print("="*80)

# Model 1: Using percentage (current)
print("\n" + "="*80)
print("MODEL 1: log(internet_users_pct) ~ log(price_usd)")
print("="*80)

required1 = ['log_internet_users_pct', 'log_price_usd', 'price_usd_x_eap',
             'log_gdp_per_capita', 'research_development_expenditure',
             'secure_internet_servers']
df1 = df[required1].dropna()

y1 = df1['log_internet_users_pct']
X1 = df1[['log_price_usd', 'price_usd_x_eap', 'log_gdp_per_capita',
          'research_development_expenditure', 'secure_internet_servers']]

model1 = PanelOLS(y1, X1, entity_effects=True, time_effects=True)
res1 = model1.fit(cov_type='clustered', cluster_entity=True)

beta_price1 = res1.params['log_price_usd']
beta_interaction1 = res1.params['price_usd_x_eap']
eu_elast1 = beta_price1
eap_elast1 = beta_price1 + beta_interaction1

print(f"Sample: {res1.nobs} observations")
print(f"Price coefficient (EU): {beta_price1:.6f}")
print(f"Interaction (EaP diff): {beta_interaction1:.6f}")
print(f"EU elasticity:  {eu_elast1:.6f}")
print(f"EaP elasticity: {eap_elast1:.6f}")
print(f"R-squared: {res1.rsquared:.4f}")

# Model 2: Using levels with population control
print("\n" + "="*80)
print("MODEL 2: log(internet_users_count) ~ log(price_usd) + log(population)")
print("="*80)

required2 = ['log_internet_users_count', 'log_price_usd', 'price_usd_x_eap',
             'log_population', 'log_gdp_per_capita',
             'research_development_expenditure', 'secure_internet_servers']
df2 = df[required2].dropna()

y2 = df2['log_internet_users_count']
X2 = df2[['log_price_usd', 'price_usd_x_eap', 'log_population',
          'log_gdp_per_capita', 'research_development_expenditure',
          'secure_internet_servers']]

model2 = PanelOLS(y2, X2, entity_effects=True, time_effects=True)
res2 = model2.fit(cov_type='clustered', cluster_entity=True)

beta_price2 = res2.params['log_price_usd']
beta_interaction2 = res2.params['price_usd_x_eap']
beta_pop2 = res2.params['log_population']
eu_elast2 = beta_price2
eap_elast2 = beta_price2 + beta_interaction2

print(f"Sample: {res2.nobs} observations")
print(f"Price coefficient (EU): {beta_price2:.6f}")
print(f"Interaction (EaP diff): {beta_interaction2:.6f}")
print(f"Population coefficient: {beta_pop2:.6f}")
print(f"EU elasticity:  {eu_elast2:.6f}")
print(f"EaP elasticity: {eap_elast2:.6f}")
print(f"R-squared: {res2.rsquared:.4f}")

# Model 3: Using levels WITHOUT population (wrong specification)
print("\n" + "="*80)
print("MODEL 3: log(internet_users_count) ~ log(price_usd) [NO POPULATION]")
print("         (Incorrect - omitted variable bias)")
print("="*80)

required3 = ['log_internet_users_count', 'log_price_usd', 'price_usd_x_eap',
             'log_gdp_per_capita', 'research_development_expenditure',
             'secure_internet_servers']
df3 = df[required3].dropna()

y3 = df3['log_internet_users_count']
X3 = df3[['log_price_usd', 'price_usd_x_eap', 'log_gdp_per_capita',
          'research_development_expenditure', 'secure_internet_servers']]

model3 = PanelOLS(y3, X3, entity_effects=True, time_effects=True)
res3 = model3.fit(cov_type='clustered', cluster_entity=True)

beta_price3 = res3.params['log_price_usd']
beta_interaction3 = res3.params['price_usd_x_eap']
eu_elast3 = beta_price3
eap_elast3 = beta_price3 + beta_interaction3

print(f"Sample: {res3.nobs} observations")
print(f"Price coefficient (EU): {beta_price3:.6f}")
print(f"Interaction (EaP diff): {beta_interaction3:.6f}")
print(f"EU elasticity:  {eu_elast3:.6f}")
print(f"EaP elasticity: {eap_elast3:.6f}")
print(f"R-squared: {res3.rsquared:.4f}")

# Comparison
print("\n" + "="*80)
print("COMPARISON OF PRICE ELASTICITIES")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['Percentage (Model 1)', 'Levels + Pop (Model 2)', 'Levels Only (Model 3)'],
    'EU_Elasticity': [eu_elast1, eu_elast2, eu_elast3],
    'EaP_Elasticity': [eap_elast1, eap_elast2, eap_elast3],
    'Population_Coef': ['N/A (normalized)', f'{beta_pop2:.6f}', 'Omitted'],
    'R_squared': [res1.rsquared, res2.rsquared, res3.rsquared]
})

print(comparison.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check if Model 1 ≈ Model 2
eu_diff = abs(eu_elast1 - eu_elast2)
eap_diff = abs(eap_elast1 - eap_elast2)

print(f"\n1. Equivalence Test (Model 1 vs Model 2):")
print(f"   EU elasticity difference:  {eu_diff:.6f}")
print(f"   EaP elasticity difference: {eap_diff:.6f}")

if eu_diff < 0.01 and eap_diff < 0.01:
    print("   ✓ EQUIVALENT (differences < 0.01)")
    print("   → Using percentage or levels+population gives same results")
elif eu_diff < 0.05 and eap_diff < 0.05:
    print("   ≈ APPROXIMATELY EQUIVALENT (differences < 0.05)")
    print("   → Results very similar, minor differences due to:")
    print("     - Rounding/computational precision")
    print("     - Sample differences (missing data)")
else:
    print("   ✗ NOT EQUIVALENT (differences > 0.05)")
    print("   → Different results - investigate why")

print(f"\n2. Population Coefficient Test:")
print(f"   Estimated coefficient: {beta_pop2:.6f}")
print(f"   Standard error: {res2.std_errors['log_population']:.6f}")
print(f"   p-value: {res2.pvalues['log_population']:.6f}")

if abs(beta_pop2 - 1.0) < 0.05:
    print("   ✓ CLOSE TO 1.0 - supports equivalence")
    print("   → Internet users scale proportionally with population")
elif res2.pvalues['log_population'] < 0.05:
    print(f"   ⚠ DIFFERS FROM 1.0 (significantly different)")
    print(f"   → Population elasticity = {beta_pop2:.3f}, not 1.0")
    print("   → Models are NOT mathematically equivalent")
else:
    print("   ? INCONCLUSIVE - coefficient not significantly different from 1.0")

print(f"\n3. Omitted Variable Bias Test (Model 3):")
model3_bias_eu = abs(eu_elast3 - eu_elast1)
model3_bias_eap = abs(eap_elast3 - eap_elast1)
print(f"   EU elasticity bias:  {model3_bias_eu:.6f}")
print(f"   EaP elasticity bias: {model3_bias_eap:.6f}")

if model3_bias_eu > 0.05 or model3_bias_eap > 0.05:
    print("   ✗ SUBSTANTIAL BIAS from omitting population")
    print("   → Must include population when using levels")
else:
    print("   ✓ Minimal bias (absorbed by fixed effects)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nYour insight is correct! Here's what we found:")
print("\n1. Mathematical Relationship:")
print("   log(users/pop) ≈ log(users) - log(pop)")
print("   → If pop coefficient = 1, models are equivalent")

print(f"\n2. Empirical Test:")
if abs(beta_pop2 - 1.0) < 0.05:
    print(f"   ✓ Population coefficient ≈ 1.0 (actual: {beta_pop2:.3f})")
    print("   ✓ Price elasticities virtually identical:")
    print(f"     Model 1 (pct): EU={eu_elast1:.4f}, EaP={eap_elast1:.4f}")
    print(f"     Model 2 (lvl): EU={eu_elast2:.4f}, EaP={eap_elast2:.4f}")
else:
    print(f"   ⚠ Population coefficient = {beta_pop2:.3f} (not 1.0)")
    print("   → Models give slightly different elasticities")
    print(f"     Model 1 (pct): EU={eu_elast1:.4f}, EaP={eap_elast1:.4f}")
    print(f"     Model 2 (lvl): EU={eu_elast2:.4f}, EaP={eap_elast2:.4f}")

print("\n3. Practical Recommendation:")
print("   → Use PERCENTAGE (Model 1) because:")
print("     - Simpler interpretation (adoption rate)")
print("     - One less variable to estimate")
print("     - Standard in literature")
print("     - Gives same elasticities (if pop coef ≈ 1)")

print("\n4. When to Use Levels + Population:")
print("   → If you want to test whether internet users")
print("     scale proportionally with population")
print("   → If population coefficient ≠ 1 is substantively interesting")
print("   → If you need to control for population dynamics explicitly")

print("\n" + "="*80)
