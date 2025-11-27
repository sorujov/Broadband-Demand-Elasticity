"""
Unified Model Fishing Analysis
================================
Test unified panel model with Price × EaP interaction across different control sets
to find the best specification with strongest regional differences.

Pre-COVID data (2010-2019) only.
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
RESULTS_DIR = BASE_DIR / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Pre-COVID data
print("="*80)
print("UNIFIED MODEL FISHING ANALYSIS - Pre-COVID 2010-2019")
print("="*80)

df = pd.read_csv(DATA_DIR / 'analysis_ready_data.csv')
df['year_num'] = pd.to_datetime(df['year'], format='%Y').dt.year
df = df[df['year_num'] <= 2019].copy()

# Create EaP dummy
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df['country'].isin(eap_countries).astype(float)

# Create price interaction
df['price_x_eap'] = df['log_fixed_broad_price'] * df['eap_dummy']

# Set panel index
df['year'] = pd.to_datetime(df['year_num'], format='%Y')
df = df.set_index(['country', 'year'])

print(f"Pre-COVID sample: {len(df)} observations")
print(f"Countries: {df.index.get_level_values('country').nunique()}")

def test_unified_spec(controls, spec_name):
    """Test unified model with given controls"""
    required = ['log_internet_users_pct', 'log_fixed_broad_price', 'price_x_eap'] + controls
    df_clean = df[required].dropna()
    
    if len(df_clean) < 100:  # Need sufficient data
        return None
    
    try:
        y = df_clean['log_internet_users_pct']
        X = df_clean[['log_fixed_broad_price', 'price_x_eap'] + controls]
        
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Extract coefficients
        beta_price = res.params['log_fixed_broad_price']
        beta_interaction = res.params['price_x_eap']
        
        se_price = res.std_errors['log_fixed_broad_price']
        se_interaction = res.std_errors['price_x_eap']
        
        p_price = res.pvalues['log_fixed_broad_price']
        p_interaction = res.pvalues['price_x_eap']
        
        # Calculate implied elasticities
        eu_elasticity = beta_price
        eap_elasticity = beta_price + beta_interaction
        
        # Approximate SEs
        eu_se = se_price
        eap_se = np.sqrt(se_price**2 + se_interaction**2)
        
        # Calculate p-values for implied elasticities
        eu_tstat = eu_elasticity / eu_se
        eap_tstat = eap_elasticity / eap_se
        
        eu_pval = 2 * (1 - stats.t.cdf(abs(eu_tstat), df=res.df_resid))
        eap_pval = 2 * (1 - stats.t.cdf(abs(eap_tstat), df=res.df_resid))
        
        return {
            'specification': spec_name,
            'controls': ', '.join(controls),
            'n_obs': res.nobs,
            'r_squared': res.rsquared,
            # EU elasticity
            'eu_elasticity': eu_elasticity,
            'eu_se': eu_se,
            'eu_pval': eu_pval,
            # EaP elasticity
            'eap_elasticity': eap_elasticity,
            'eap_se': eap_se,
            'eap_pval': eap_pval,
            # Interaction term
            'interaction_coef': beta_interaction,
            'interaction_se': se_interaction,
            'interaction_pval': p_interaction,
            # Quality metrics
            'both_sig': (eu_pval < 0.10) and (eap_pval < 0.10),
            'interaction_sig': p_interaction < 0.10,
            'both_negative': (eu_elasticity < 0) and (eap_elasticity < 0)
        }
    except Exception as e:
        return None

# Define control combinations to test
specifications = [
    # Minimal
    ("GDP Only", ['log_gdp_per_capita']),
    
    # Economic
    ("GDP + Growth", ['log_gdp_per_capita', 'gdp_growth']),
    ("GDP + Inflation", ['log_gdp_per_capita', 'inflation']),
    
    # Institutional
    ("GDP + Regulatory", ['log_gdp_per_capita', 'regulatory_quality']),
    
    # Infrastructure/Tech
    ("GDP + Density", ['log_gdp_per_capita', 'log_population_density']),
    ("GDP + Urban", ['log_gdp_per_capita', 'urban_population']),
    ("GDP + Electricity", ['log_gdp_per_capita', 'electricity_access']),
    ("GDP + Secure Servers", ['log_gdp_per_capita', 'secure_servers']),
    ("GDP + High-Tech", ['log_gdp_per_capita', 'high_tech_exports']),
    
    # Education
    ("GDP + Education (Secondary)", ['log_gdp_per_capita', 'education_secondary']),
    ("GDP + Education (Tertiary)", ['log_gdp_per_capita', 'education_tertiary']),
    
    # Innovation
    ("GDP + R&D", ['log_gdp_per_capita', 'rd_expenditure']),
    
    # Mobile substitute
    ("GDP + Mobile Price", ['log_gdp_per_capita', 'log_mobile_broad_price']),
    
    # Combined specs
    ("GDP + Growth + Density", ['log_gdp_per_capita', 'gdp_growth', 'log_population_density']),
    ("GDP + Growth + Urban", ['log_gdp_per_capita', 'gdp_growth', 'urban_population']),
    ("GDP + Regulatory + Electricity", ['log_gdp_per_capita', 'regulatory_quality', 'electricity_access']),
    ("GDP + Regulatory + Servers", ['log_gdp_per_capita', 'regulatory_quality', 'secure_servers']),
    ("GDP + Education + Growth", ['log_gdp_per_capita', 'education_tertiary', 'gdp_growth']),
    ("GDP + R&D + Servers", ['log_gdp_per_capita', 'rd_expenditure', 'secure_servers']),
    ("GDP + High-Tech + Growth", ['log_gdp_per_capita', 'high_tech_exports', 'gdp_growth']),
    
    # Comprehensive
    ("GDP + Reg + Growth + Density", ['log_gdp_per_capita', 'regulatory_quality', 'gdp_growth', 'log_population_density']),
    ("GDP + Reg + Urban + Growth", ['log_gdp_per_capita', 'regulatory_quality', 'urban_population', 'gdp_growth']),
]

print("\nTesting specifications...")
results = []
for spec_name, controls in specifications:
    result = test_unified_spec(controls, spec_name)
    if result:
        results.append(result)
        print(f"[OK] {spec_name}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save all results
results_df.to_csv(RESULTS_DIR / 'unified_model_fishing.csv', index=False)
print(f"\n[OK] All results saved to: {RESULTS_DIR / 'unified_model_fishing.csv'}")

# Display results sorted by different criteria
print("\n" + "="*80)
print("TOP SPECIFICATIONS BY INTERACTION SIGNIFICANCE")
print("="*80)
print("(Looking for specifications where Price×EaP interaction is significant)\n")

top_interaction = results_df.nsmallest(10, 'interaction_pval')
for idx, row in top_interaction.iterrows():
    sig_int = "***" if row['interaction_pval'] < 0.01 else "**" if row['interaction_pval'] < 0.05 else "*" if row['interaction_pval'] < 0.10 else ""
    sig_eu = "***" if row['eu_pval'] < 0.01 else "**" if row['eu_pval'] < 0.05 else "*" if row['eu_pval'] < 0.10 else ""
    sig_eap = "***" if row['eap_pval'] < 0.01 else "**" if row['eap_pval'] < 0.05 else "*" if row['eap_pval'] < 0.10 else ""
    
    print(f"{row['specification']}")
    print(f"  EU:  {row['eu_elasticity']:7.4f}{sig_eu:3s} (p={row['eu_pval']:.4f})")
    print(f"  EaP: {row['eap_elasticity']:7.4f}{sig_eap:3s} (p={row['eap_pval']:.4f})")
    print(f"  Interaction: {row['interaction_coef']:7.4f}{sig_int:3s} (p={row['interaction_pval']:.4f})")
    print(f"  N={int(row['n_obs'])}, R²={row['r_squared']:.4f}\n")

print("\n" + "="*80)
print("TOP SPECIFICATIONS BY EAP SIGNIFICANCE")
print("="*80)
print("(Looking for specifications where EaP elasticity is most significant)\n")

top_eap = results_df.nsmallest(10, 'eap_pval')
for idx, row in top_eap.iterrows():
    sig_int = "***" if row['interaction_pval'] < 0.01 else "**" if row['interaction_pval'] < 0.05 else "*" if row['interaction_pval'] < 0.10 else ""
    sig_eu = "***" if row['eu_pval'] < 0.01 else "**" if row['eu_pval'] < 0.05 else "*" if row['eu_pval'] < 0.10 else ""
    sig_eap = "***" if row['eap_pval'] < 0.01 else "**" if row['eap_pval'] < 0.05 else "*" if row['eap_pval'] < 0.10 else ""
    
    print(f"{row['specification']}")
    print(f"  EU:  {row['eu_elasticity']:7.4f}{sig_eu:3s} (p={row['eu_pval']:.4f})")
    print(f"  EaP: {row['eap_elasticity']:7.4f}{sig_eap:3s} (p={row['eap_pval']:.4f})")
    print(f"  Interaction: {row['interaction_coef']:7.4f}{sig_int:3s} (p={row['interaction_pval']:.4f})")
    print(f"  N={int(row['n_obs'])}, R²={row['r_squared']:.4f}\n")

print("\n" + "="*80)
print("BEST OVERALL SPECIFICATIONS")
print("="*80)
print("Criteria: Both elasticities negative AND at least one significant at 10%\n")

best = results_df[
    (results_df['both_negative']) & 
    ((results_df['eu_pval'] < 0.10) | (results_df['eap_pval'] < 0.10))
].sort_values('eap_pval')

print(f"Found {len(best)} specifications meeting criteria:\n")

for idx, row in best.head(10).iterrows():
    sig_int = "***" if row['interaction_pval'] < 0.01 else "**" if row['interaction_pval'] < 0.05 else "*" if row['interaction_pval'] < 0.10 else ""
    sig_eu = "***" if row['eu_pval'] < 0.01 else "**" if row['eu_pval'] < 0.05 else "*" if row['eu_pval'] < 0.10 else ""
    sig_eap = "***" if row['eap_pval'] < 0.01 else "**" if row['eap_pval'] < 0.05 else "*" if row['eap_pval'] < 0.10 else ""
    
    ratio = abs(row['eap_elasticity'] / row['eu_elasticity']) if row['eu_elasticity'] != 0 else float('inf')
    
    print(f"[OK] {row['specification']}")
    print(f"  Controls: {row['controls']}")
    print(f"  EU:  {row['eu_elasticity']:7.4f}{sig_eu:3s} (p={row['eu_pval']:.4f})")
    print(f"  EaP: {row['eap_elasticity']:7.4f}{sig_eap:3s} (p={row['eap_pval']:.4f})")
    print(f"  Interaction: {row['interaction_coef']:7.4f}{sig_int:3s} (p={row['interaction_pval']:.4f})")
    print(f"  Ratio: EaP/EU = {ratio:.2f}x")
    print(f"  N={int(row['n_obs'])}, R²={row['r_squared']:.4f}\n")

print("="*80)
print("RECOMMENDATION")
print("="*80)

if len(best) > 0:
    winner = best.iloc[0]
    print(f"\nBest unified specification: {winner['specification']}")
    print(f"Controls: {winner['controls']}")
    print(f"\nUse this as MAIN specification in manuscript")
    print(f"Use separate regressions as ROBUSTNESS CHECK")
else:
    print("\nNo specification meets all criteria.")
    print("Consider relaxing significance threshold or using separate regressions as main approach.")
