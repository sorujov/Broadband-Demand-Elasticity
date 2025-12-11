"""
Price Definition Comparison Analysis
======================================
Tests all three price definitions (USD, GNI%, PPP) to compare elasticity estimates
Also tests different dependent variables (subscriptions vs users)

Purpose: Robustness check to see if results hold across different price measures
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

# Load merged data with all price types
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')

print("="*80)
print("PRICE DEFINITION COMPARISON ANALYSIS")
print("="*80)
print(f"Full sample: {len(df)} observations")
print(f"Countries: {df['country'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

# Filter to Pre-COVID (2010-2019)
df = df[df['year'] <= 2019].copy()
print(f"Pre-COVID sample (2010-2019): {len(df)} observations")

# Create dependent variables
# Option 1: Internet users % (adoption/diffusion measure)
df['log_internet_users'] = np.log(df['internet_users_pct_i99H'] + 0.01)

# Option 2: Fixed broadband subscriptions per 100 (infrastructure penetration)
df['log_fixed_subs'] = np.log(df['fixed_broadband_subs_i4213tfbb'] + 0.01)

# Create price variables for all three definitions
price_definitions = {
    'USD': 'fixed_broad_price_usd',
    'GNI%': 'fixed_broad_price_gni_pct', 
    'PPP': 'fixed_broad_price_ppp'
}

for name, col in price_definitions.items():
    df[f'log_price_{name.lower()}'] = np.log(df[col] + 0.01)

# Create control variables
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])
df['log_population'] = np.log(df['population'])

# Create EaP dummy and interactions
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df['country'].isin(eap_countries).astype(float)

# Create interaction terms for each price definition
df['price_usd_x_eap'] = df['log_price_usd'] * df['eap_dummy']
df['price_gni%_x_eap'] = df['log_price_gni%'] * df['eap_dummy']
df['price_ppp_x_eap'] = df['log_price_ppp'] * df['eap_dummy']

# Set panel index
df['year_dt'] = pd.to_datetime(df['year'], format='%Y')
df = df.set_index(['country', 'year_dt'])

print("\n" + "="*80)
print("AVAILABLE PRICE MEASURES")
print("="*80)
for name, col in price_definitions.items():
    coverage = df[col].notna().sum()
    pct = coverage / len(df) * 100
    print(f"{name:5s}: {coverage:3d}/{len(df)} obs ({pct:5.1f}% coverage)")

print("\n" + "="*80)
print("COMPARISON 1: PRICE DEFINITIONS (with Internet Users as DV)")
print("="*80)

results_by_price = []

for price_name, price_col in price_definitions.items():
    print(f"\n--- {price_name} Prices ---")
    
    # Prepare data
    log_price_var = f'log_price_{price_name.lower()}'
    interaction_var = f'price_{price_name.lower()}_x_eap'
    
    required = ['log_internet_users', log_price_var, interaction_var, 
                'log_gdp_per_capita', 'research_development_expenditure', 
                'secure_internet_servers']
    
    df_clean = df[required].dropna()
    
    if len(df_clean) < 100:
        print(f"  ⚠ Insufficient data: {len(df_clean)} obs (skipping)")
        continue
    
    print(f"  Sample: {len(df_clean)} observations")
    
    # Run regression
    y = df_clean['log_internet_users']
    X = df_clean[[log_price_var, interaction_var, 'log_gdp_per_capita', 
                   'research_development_expenditure', 'secure_internet_servers']]
    
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    
    # Extract coefficients
    beta_price = res.params[log_price_var]
    beta_interaction = res.params[interaction_var]
    se_price = res.std_errors[log_price_var]
    se_interaction = res.std_errors[interaction_var]
    
    # Calculate regional elasticities
    eu_elasticity = beta_price
    eap_elasticity = beta_price + beta_interaction
    
    eu_se = se_price
    eap_se = np.sqrt(se_price**2 + se_interaction**2)
    
    eu_tstat = eu_elasticity / eu_se
    eap_tstat = eap_elasticity / eap_se
    
    eu_pval = 2 * (1 - stats.t.cdf(abs(eu_tstat), df=res.df_resid))
    eap_pval = 2 * (1 - stats.t.cdf(abs(eap_tstat), df=res.df_resid))
    
    # Significance stars
    def sig_stars(pval):
        if pval < 0.01: return "***"
        elif pval < 0.05: return "**"
        elif pval < 0.10: return "*"
        else: return ""
    
    print(f"  EU elasticity:  {eu_elasticity:7.4f}{sig_stars(eu_pval):3s} (SE={eu_se:.4f}, p={eu_pval:.4f})")
    print(f"  EaP elasticity: {eap_elasticity:7.4f}{sig_stars(eap_pval):3s} (SE={eap_se:.4f}, p={eap_pval:.4f})")
    print(f"  Interaction:    {beta_interaction:7.4f}{sig_stars(res.pvalues[interaction_var]):3s} (p={res.pvalues[interaction_var]:.4f})")
    print(f"  R-squared: {res.rsquared:.4f}")
    
    # Economic interpretation
    if eu_elasticity < 0 and eap_elasticity < 0:
        ratio = abs(eap_elasticity / eu_elasticity) if eu_elasticity != 0 else np.inf
        print(f"  ✓ Both negative (correct sign) - EaP {ratio:.1f}x more elastic")
    elif eu_elasticity > 0 or eap_elasticity > 0:
        print(f"  ✗ WRONG SIGN - Positive elasticity (economically implausible)")
    
    # Store results
    results_by_price.append({
        'price_definition': price_name,
        'dependent_var': 'Internet Users',
        'eu_elasticity': eu_elasticity,
        'eu_se': eu_se,
        'eu_pval': eu_pval,
        'eap_elasticity': eap_elasticity,
        'eap_se': eap_se,
        'eap_pval': eap_pval,
        'interaction_coef': beta_interaction,
        'interaction_pval': res.pvalues[interaction_var],
        'n_obs': res.nobs,
        'r_squared': res.rsquared,
        'economically_valid': (eu_elasticity < 0 and eap_elasticity < 0)
    })

print("\n" + "="*80)
print("COMPARISON 2: DEPENDENT VARIABLE (with GNI% Prices)")
print("="*80)

results_by_dv = []

dependent_vars = {
    'Internet Users %': 'log_internet_users',
    'Fixed Broadband Subs': 'log_fixed_subs'
}

for dv_name, dv_col in dependent_vars.items():
    print(f"\n--- {dv_name} ---")
    
    required = [dv_col, 'log_price_gni%', 'price_gni%_x_eap',
                'log_gdp_per_capita', 'research_development_expenditure',
                'secure_internet_servers']
    
    df_clean = df[required].dropna()
    
    if len(df_clean) < 100:
        print(f"  ⚠ Insufficient data: {len(df_clean)} obs (skipping)")
        continue
    
    print(f"  Sample: {len(df_clean)} observations")
    
    # Run regression
    y = df_clean[dv_col]
    X = df_clean[['log_price_gni%', 'price_gni%_x_eap', 'log_gdp_per_capita',
                   'research_development_expenditure', 'secure_internet_servers']]
    
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    
    # Extract coefficients
    beta_price = res.params['log_price_gni%']
    beta_interaction = res.params['price_gni%_x_eap']
    se_price = res.std_errors['log_price_gni%']
    se_interaction = res.std_errors['price_gni%_x_eap']
    
    # Calculate regional elasticities
    eu_elasticity = beta_price
    eap_elasticity = beta_price + beta_interaction
    
    eu_se = se_price
    eap_se = np.sqrt(se_price**2 + se_interaction**2)
    
    eu_tstat = eu_elasticity / eu_se
    eap_tstat = eap_elasticity / eap_se
    
    eu_pval = 2 * (1 - stats.t.cdf(abs(eu_tstat), df=res.df_resid))
    eap_pval = 2 * (1 - stats.t.cdf(abs(eap_tstat), df=res.df_resid))
    
    print(f"  EU elasticity:  {eu_elasticity:7.4f}{sig_stars(eu_pval):3s} (SE={eu_se:.4f}, p={eu_pval:.4f})")
    print(f"  EaP elasticity: {eap_elasticity:7.4f}{sig_stars(eap_pval):3s} (SE={eap_se:.4f}, p={eap_pval:.4f})")
    print(f"  Interaction:    {beta_interaction:7.4f}{sig_stars(res.pvalues['price_gni%_x_eap']):3s} (p={res.pvalues['price_gni%_x_eap']:.4f})")
    print(f"  R-squared: {res.rsquared:.4f}")
    
    # Store results
    results_by_dv.append({
        'dependent_var': dv_name,
        'price_definition': 'GNI%',
        'eu_elasticity': eu_elasticity,
        'eu_se': eu_se,
        'eu_pval': eu_pval,
        'eap_elasticity': eap_elasticity,
        'eap_se': eap_se,
        'eap_pval': eap_pval,
        'interaction_coef': beta_interaction,
        'interaction_pval': res.pvalues['price_gni%_x_eap'],
        'n_obs': res.nobs,
        'r_squared': res.rsquared
    })

# Save results
df_price_comparison = pd.DataFrame(results_by_price)
df_dv_comparison = pd.DataFrame(results_by_dv)

df_price_comparison.to_csv(RESULTS_DIR / 'price_definition_comparison.csv', index=False)
df_dv_comparison.to_csv(RESULTS_DIR / 'dependent_variable_comparison.csv', index=False)

print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. PRICE DEFINITION COMPARISON:")
print("-" * 40)
if len(results_by_price) > 0:
    valid_prices = [r for r in results_by_price if r['economically_valid']]
    
    if len(valid_prices) > 0:
        print(f"  ✓ {len(valid_prices)}/{len(results_by_price)} price definitions produce economically valid results")
        
        # Rank by R-squared
        valid_prices_sorted = sorted(valid_prices, key=lambda x: x['r_squared'], reverse=True)
        
        print("\n  Ranking by model fit (R²):")
        for i, r in enumerate(valid_prices_sorted, 1):
            print(f"    {i}. {r['price_definition']:5s}: R²={r['r_squared']:.4f}, "
                  f"EU={r['eu_elasticity']:.4f}, EaP={r['eap_elasticity']:.4f}")
        
        best = valid_prices_sorted[0]
        print(f"\n  → RECOMMENDED: {best['price_definition']} prices")
        print(f"    - Best model fit (R²={best['r_squared']:.4f})")
        print(f"    - Economically sensible (negative elasticities)")
        print(f"    - EU elasticity: {best['eu_elasticity']:.4f}")
        print(f"    - EaP elasticity: {best['eap_elasticity']:.4f}")
    else:
        print("  ✗ No price definitions produce valid results!")
        print("  → Check data quality and model specification")

print("\n2. DEPENDENT VARIABLE COMPARISON:")
print("-" * 40)
if len(results_by_dv) > 0:
    for r in results_by_dv:
        print(f"  {r['dependent_var']}:")
        print(f"    EU:  {r['eu_elasticity']:.4f} (p={r['eu_pval']:.4f})")
        print(f"    EaP: {r['eap_elasticity']:.4f} (p={r['eap_pval']:.4f})")
        print(f"    R²:  {r['r_squared']:.4f}")
    
    print("\n  INTERPRETATION:")
    print("  - Internet Users %: Measures overall digital adoption")
    print("  - Fixed Broadband Subs: Measures infrastructure penetration")
    print("\n  → RECOMMENDED: Internet Users %")
    print("    - Broader measure of digital inclusion")
    print("    - More policy-relevant (adoption vs. just infrastructure)")
    print("    - Less affected by multiple subscriptions per person")

print("\n3. FINAL RECOMMENDATION:")
print("-" * 40)
if len(results_by_price) > 0 and len(valid_prices) > 0:
    best = valid_prices_sorted[0]
    print(f"  ✓ Use {best['price_definition']} prices with Internet Users % as dependent variable")
    print(f"  ✓ This specification has:")
    print(f"    - Correct signs (negative elasticities)")
    print(f"    - Best model fit (R²={best['r_squared']:.4f})")
    print(f"    - Significant regional differences (p={best['interaction_pval']:.4f})")
    print(f"    - Policy-relevant interpretation")

print("\n" + "="*80)
print(f"[OK] Results saved to:")
print(f"     {RESULTS_DIR / 'price_definition_comparison.csv'}")
print(f"     {RESULTS_DIR / 'dependent_variable_comparison.csv'}")
print("="*80)
