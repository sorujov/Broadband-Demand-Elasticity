# -*- coding: utf-8 -*-
"""
Extended Analysis: Control Specifications and Non-Stationarity Diagnostics
===========================================================================
Runs extended robustness checks with various control specifications and
tests for non-stationarity issues using first-difference models.
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import sys
import io

warnings.filterwarnings('ignore')

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE
    )

RESULTS_DIR = RESULTS_REGRESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create subfolders for organized results
FULL_SAMPLE_DIR = RESULTS_DIR / 'full_sample_covid_analysis'
FULL_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Load FULL dataset (including COVID period)
print("=" * 80)
print("LOADING DATA - FULL SAMPLE (2010-2024)")
print("=" * 80)

df = pd.read_csv(ANALYSIS_READY_FILE)
print(f"\nLoaded: {ANALYSIS_READY_FILE}")
print(f"  • Full dataset: {len(df)} observations")
print(f"  • Years: {df['year'].min()}-{df['year'].max()}")

# Create year variable
df['year_num'] = df['year'].astype(int)

# Create COVID dummy (1 for years >= 2020)
df['covid_dummy'] = (df['year_num'] >= 2020).astype(float)

# Count observations by period
n_pre_covid = (df['covid_dummy'] == 0).sum()
n_covid = (df['covid_dummy'] == 1).sum()
print(f"\n  • Pre-COVID (2010-2019): {n_pre_covid} observations")
print(f"  • COVID period (2020-2024): {n_covid} observations")

# Create regional variables
df['eap_dummy'] = df['country'].isin(EAP_COUNTRIES).astype(float)

# Create interaction terms (COVID dummy absorbed by time FE, so only use interactions)
df['price_x_eap'] = df[PRIMARY_PRICE] * df['eap_dummy']
df['price_x_covid'] = df[PRIMARY_PRICE] * df['covid_dummy']
df['price_x_eap_x_covid'] = df[PRIMARY_PRICE] * df['eap_dummy'] * df['covid_dummy']

# Set panel index
df['year_dt'] = pd.to_datetime(df['year_num'], format='%Y')
df = df.set_index(['country', 'year_dt'])

# ============================================================================
# BASELINE MODEL (for reference comparison)
# ============================================================================

print("\n" + "=" * 80)
print("BASELINE MODEL ESTIMATION (with COVID dummy and interactions)")
print("=" * 80)

controls_baseline = ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                     'regulatory_quality_estimate', 'log_secure_internet_servers', 
                     'research_development_expenditure', 'population_ages_15_64',
                     'gdp_growth', 'inflation_gdp_deflator', 'log_population_density']
required_baseline = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap', 
                     'price_x_covid', 'price_x_eap_x_covid'] + controls_baseline
df_baseline = df[required_baseline].dropna()

print(f"\nBaseline specification (Full Controls + COVID interactions):")
print(f"  • Controls: GDP, urban, education, regulatory, infrastructure, R&D, demographics, macro")
print(f"  • COVID terms: price×COVID, price×EaP×COVID (dummy absorbed by time FE)")
print(f"  • N observations: {len(df_baseline)}")
print(f"  • N countries: {df_baseline.index.get_level_values('country').nunique()}")

y_base = df_baseline[PRIMARY_DV]
X_base = df_baseline[[PRIMARY_PRICE, 'price_x_eap', 
                       'price_x_covid', 'price_x_eap_x_covid'] + controls_baseline]

model_baseline = PanelOLS(y_base, X_base, entity_effects=True, time_effects=True)
res_baseline = model_baseline.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

# Extract coefficients
beta_price = res_baseline.params[PRIMARY_PRICE]
beta_eap_interact = res_baseline.params['price_x_eap']
beta_price_covid = res_baseline.params['price_x_covid']
beta_triple = res_baseline.params['price_x_eap_x_covid']

# Calculate elasticities for each period×region
# Pre-COVID (covid=0):
eu_pre = beta_price
eap_pre = beta_price + beta_eap_interact

# COVID period (covid=1):
eu_covid = beta_price + beta_price_covid
eap_covid = beta_price + beta_eap_interact + beta_price_covid + beta_triple

# Standard errors (using delta method)
se_price = res_baseline.std_errors[PRIMARY_PRICE]
se_eap_int = res_baseline.std_errors['price_x_eap']
se_price_covid = res_baseline.std_errors['price_x_covid']
se_triple = res_baseline.std_errors['price_x_eap_x_covid']

eu_pre_se = se_price
eap_pre_se = np.sqrt(se_price**2 + se_eap_int**2)
eu_covid_se = np.sqrt(se_price**2 + se_price_covid**2)
eap_covid_se = np.sqrt(se_price**2 + se_eap_int**2 + se_price_covid**2 + se_triple**2)

# P-values
eu_pre_pval = 2 * (1 - stats.t.cdf(abs(eu_pre/eu_pre_se), df=res_baseline.df_resid))
eap_pre_pval = 2 * (1 - stats.t.cdf(abs(eap_pre/eap_pre_se), df=res_baseline.df_resid))
eu_covid_pval = 2 * (1 - stats.t.cdf(abs(eu_covid/eu_covid_se), df=res_baseline.df_resid))
eap_covid_pval = 2 * (1 - stats.t.cdf(abs(eap_covid/eap_covid_se), df=res_baseline.df_resid))

# Significance stars
sig_eu_pre = "***" if eu_pre_pval < 0.01 else "**" if eu_pre_pval < 0.05 else "*" if eu_pre_pval < 0.10 else ""
sig_eap_pre = "***" if eap_pre_pval < 0.01 else "**" if eap_pre_pval < 0.05 else "*" if eap_pre_pval < 0.10 else ""
sig_eu_covid = "***" if eu_covid_pval < 0.01 else "**" if eu_covid_pval < 0.05 else "*" if eu_covid_pval < 0.10 else ""
sig_eap_covid = "***" if eap_covid_pval < 0.01 else "**" if eap_covid_pval < 0.05 else "*" if eap_covid_pval < 0.10 else ""

print(f"\nBaseline Results (by Period × Region):")
print(f"\n  PRE-COVID (2010-2019):")
print(f"    • EU elasticity:  {eu_pre:7.4f}{sig_eu_pre:3s} (SE={eu_pre_se:.4f}, p={eu_pre_pval:.4f})")
print(f"    • EaP elasticity: {eap_pre:7.4f}{sig_eap_pre:3s} (SE={eap_pre_se:.4f}, p={eap_pre_pval:.4f})")
print(f"    • Ratio (EaP/EU): {abs(eap_pre/eu_pre):.2f}x")

print(f"\n  COVID PERIOD (2020-2024):")
print(f"    • EU elasticity:  {eu_covid:7.4f}{sig_eu_covid:3s} (SE={eu_covid_se:.4f}, p={eu_covid_pval:.4f})")
print(f"    • EaP elasticity: {eap_covid:7.4f}{sig_eap_covid:3s} (SE={eap_covid_se:.4f}, p={eap_covid_pval:.4f})")
print(f"    • Ratio (EaP/EU): {abs(eap_covid/eu_covid):.2f}x")

print(f"\n  COVID EFFECTS (Interaction Terms):")
sig_price_covid = "***" if res_baseline.pvalues['price_x_covid'] < 0.01 else "**" if res_baseline.pvalues['price_x_covid'] < 0.05 else "*" if res_baseline.pvalues['price_x_covid'] < 0.10 else ""
sig_triple = "***" if res_baseline.pvalues['price_x_eap_x_covid'] < 0.01 else "**" if res_baseline.pvalues['price_x_eap_x_covid'] < 0.05 else "*" if res_baseline.pvalues['price_x_eap_x_covid'] < 0.10 else ""

print(f"    • Price×COVID:           {beta_price_covid:7.4f}{sig_price_covid:3s} (p={res_baseline.pvalues['price_x_covid']:.4f})")
print(f"    • Price×EaP×COVID:       {beta_triple:7.4f}{sig_triple:3s} (p={res_baseline.pvalues['price_x_eap_x_covid']:.4f})")

print(f"\n  MODEL FIT:")
print(f"    • R²={res_baseline.rsquared:.4f}")

# ============================================================================
# CODE ADDITION: Extended Control Specifications
# ============================================================================

print("\n" + "=" * 80)
print("EXTENDED ANALYSIS: Alternative Control Specifications")
print("=" * 80)

# Define comprehensive control specifications
CONTROL_SPECS = {
    'Full Controls (Baseline)': {
        'controls': ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                     'regulatory_quality_estimate', 'log_secure_internet_servers', 
                     'research_development_expenditure', 'population_ages_15_64',
                     'gdp_growth', 'inflation_gdp_deflator', 'log_population_density'],
        'description': 'Baseline: All available controls (kitchen sink)'
    },

    'Comprehensive': {
        'controls': ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                     'regulatory_quality_estimate', 'log_secure_internet_servers'],
        'description': 'Key dimensions: income, human capital, institutions, infrastructure'
    },

    'Core': {
        'controls': ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct'],
        'description': 'Parsimonious: Income, urbanization, human capital'
    },

    'Institutional': {
        'controls': ['log_gdp_per_capita', 'regulatory_quality_estimate'],
        'description': 'Governance and regulatory quality'
    },

    'Infrastructure': {
        'controls': ['log_gdp_per_capita', 'log_secure_internet_servers', 'research_development_expenditure'],
        'description': 'Digital infrastructure and innovation'
    },

    'Demographic': {
        'controls': ['log_gdp_per_capita', 'urban_population_pct', 'log_population', 
                     'population_ages_15_64'],
        'description': 'Population characteristics'
    },

    'Macroeconomic': {
        'controls': ['log_gdp_per_capita', 'gdp_growth', 'inflation_gdp_deflator'],
        'description': 'Macroeconomic conditions'
    },

    'Minimal': {
        'controls': ['log_gdp_per_capita'],
        'description': 'Minimal controls (GDP only)'
    }
}

extended_results = []

for spec_name, spec_info in CONTROL_SPECS.items():
    print(f"\n{spec_name}: {spec_info['description']}")

    controls = spec_info['controls']
    required = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap', 
                'price_x_covid', 'price_x_eap_x_covid'] + controls

    # Check if all variables exist
    available = [col for col in required if col in df.columns]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"  ⚠ Skipping: Missing variables {missing}")
        continue

    df_spec = df[available].dropna()

    if len(df_spec) < 100:
        print(f"  ⚠ Skipping: Only {len(df_spec)} obs after dropna")
        continue

    # Estimate model
    y = df_spec[PRIMARY_DV]
    X = df_spec[[PRIMARY_PRICE, 'price_x_eap', 
                  'price_x_covid', 'price_x_eap_x_covid'] + [c for c in controls if c in df_spec.columns]]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

    # Extract coefficients
    beta_price = res.params[PRIMARY_PRICE]
    beta_eap_int = res.params['price_x_eap']
    beta_price_covid = res.params['price_x_covid']
    beta_triple = res.params['price_x_eap_x_covid']

    # Calculate elasticities by period×region
    eu_pre = beta_price
    eap_pre = beta_price + beta_eap_int
    eu_covid = beta_price + beta_price_covid
    eap_covid = beta_price + beta_eap_int + beta_price_covid + beta_triple

    # Standard errors
    se_price = res.std_errors[PRIMARY_PRICE]
    se_eap_int = res.std_errors['price_x_eap']
    se_price_covid = res.std_errors['price_x_covid']
    se_triple = res.std_errors['price_x_eap_x_covid']

    eu_pre_se = se_price
    eap_pre_se = np.sqrt(se_price**2 + se_eap_int**2)
    eu_covid_se = np.sqrt(se_price**2 + se_price_covid**2)
    eap_covid_se = np.sqrt(se_price**2 + se_eap_int**2 + se_price_covid**2 + se_triple**2)

    # P-values
    eu_pre_pval = 2 * (1 - stats.t.cdf(abs(eu_pre/eu_pre_se), df=res.df_resid))
    eap_pre_pval = 2 * (1 - stats.t.cdf(abs(eap_pre/eap_pre_se), df=res.df_resid))
    eu_covid_pval = 2 * (1 - stats.t.cdf(abs(eu_covid/eu_covid_se), df=res.df_resid))
    eap_covid_pval = 2 * (1 - stats.t.cdf(abs(eap_covid/eap_covid_se), df=res.df_resid))

    # Format output
    sig_eu_pre = "***" if eu_pre_pval < 0.01 else "**" if eu_pre_pval < 0.05 else "*" if eu_pre_pval < 0.10 else ""
    sig_eap_pre = "***" if eap_pre_pval < 0.01 else "**" if eap_pre_pval < 0.05 else "*" if eap_pre_pval < 0.10 else ""
    sig_eu_covid = "***" if eu_covid_pval < 0.01 else "**" if eu_covid_pval < 0.05 else "*" if eu_covid_pval < 0.10 else ""
    sig_eap_covid = "***" if eap_covid_pval < 0.01 else "**" if eap_covid_pval < 0.05 else "*" if eap_covid_pval < 0.10 else ""

    print(f"\n  PRE-COVID:")
    print(f"    EU:  {eu_pre:7.4f}{sig_eu_pre:3s} (p={eu_pre_pval:.3f})")
    print(f"    EaP: {eap_pre:7.4f}{sig_eap_pre:3s} (p={eap_pre_pval:.3f})")
    print(f"    Ratio: {abs(eap_pre/eu_pre):.2f}x")
    
    print(f"\n  COVID:")
    print(f"    EU:  {eu_covid:7.4f}{sig_eu_covid:3s} (p={eu_covid_pval:.3f})")
    print(f"    EaP: {eap_covid:7.4f}{sig_eap_covid:3s} (p={eap_covid_pval:.3f})")
    print(f"    Ratio: {abs(eap_covid/eu_covid):.2f}x")
    
    print(f"\n  N={res.nobs}, R²={res.rsquared:.4f}")

    # Store results
    extended_results.append({
        'specification': spec_name,
        'description': spec_info['description'],
        'controls': ', '.join([c for c in controls if c in df_spec.columns]),
        'n_controls': len([c for c in controls if c in df_spec.columns]),
        'eu_pre_elasticity': eu_pre,
        'eu_pre_se': eu_pre_se,
        'eu_pre_pval': eu_pre_pval,
        'eap_pre_elasticity': eap_pre,
        'eap_pre_se': eap_pre_se,
        'eap_pre_pval': eap_pre_pval,
        'eu_covid_elasticity': eu_covid,
        'eu_covid_se': eu_covid_se,
        'eu_covid_pval': eu_covid_pval,
        'eap_covid_elasticity': eap_covid,
        'eap_covid_se': eap_covid_se,
        'eap_covid_pval': eap_covid_pval,
        'price_x_covid_coef': beta_price_covid,
        'price_x_covid_pval': res.pvalues['price_x_covid'],
        'triple_interaction_coef': beta_triple,
        'triple_interaction_pval': res.pvalues['price_x_eap_x_covid'],
        'n_obs': res.nobs,
        'r_squared': res.rsquared
    })

# Save extended results
extended_df = pd.DataFrame(extended_results)
extended_df.to_excel(FULL_SAMPLE_DIR / 'extended_control_specifications.xlsx', index=False)

print(f"\n[OK] Extended specifications saved: full_sample_covid_analysis/extended_control_specifications.xlsx")
print(f"  • Total specifications tested: {len(extended_results)}")
print(f"  • Full sample (2010-2024) with COVID interactions")


# ============================================================================
# ROBUSTNESS: All Control Specs × Alternative Price Measures
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE ROBUSTNESS: Control Specs × Price Definitions")
print("=" * 80)

# Define alternative price measures (use PRIMARY_DV for subscriptions)
PRICE_MEASURES = [
    {'name': 'GNI%', 'var': 'log_fixed_broad_price', 'desc': 'Price as % of GNI per capita'},
    {'name': 'PPP', 'var': 'log_fixed_broad_price_ppp', 'desc': 'Price in PPP dollars'},
    {'name': 'USD', 'var': 'log_fixed_broad_price_usd', 'desc': 'Price in nominal USD'}
]

print(f"\nTesting {len(CONTROL_SPECS)} control specifications")
print(f"  × {len(PRICE_MEASURES)} price measures")  
print(f"  × COVID interactions (dummy + price×COVID + price×EaP×COVID)")
print(f"  × 1 subscription measure (PRIMARY_DV: {PRIMARY_DV})")
print(f"  = {len(CONTROL_SPECS) * len(PRICE_MEASURES)} total specifications")

comprehensive_results = []
spec_counter = 0

print(f"\n{'='*80}")
print("RESULTS BY PRICE MEASURE (with COVID interactions)")
print(f"{'='*80}")

for price_def in PRICE_MEASURES:
    print(f"\n{'-'*80}")
    print(f"PRICE MEASURE: {price_def['name']} - {price_def['desc']}")
    print(f"{'-'*80}")
    
    for control_name, control_info in CONTROL_SPECS.items():
        spec_counter += 1
        
        # Create interaction terms
        interaction_name = f"price_x_eap_{price_def['name']}"
        price_covid_name = f"price_x_covid_{price_def['name']}"
        triple_name = f"price_x_eap_x_covid_{price_def['name']}"
        
        df[interaction_name] = df[price_def['var']] * df['eap_dummy']
        df[price_covid_name] = df[price_def['var']] * df['covid_dummy']
        df[triple_name] = df[price_def['var']] * df['eap_dummy'] * df['covid_dummy']
        
        # Select required variables
        controls = control_info['controls']
        required = [PRIMARY_DV, price_def['var'], interaction_name,
                    price_covid_name, triple_name] + controls
        
        # Check if all variables exist
        available = [col for col in required if col in df.columns]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            continue
        
        df_spec = df[available].dropna()
        
        if len(df_spec) < 100:
            continue
        
        try:
            # Estimate model
            y = df_spec[PRIMARY_DV]
            X = df_spec[[price_def['var'], interaction_name,
                          price_covid_name, triple_name] + [c for c in controls if c in df_spec.columns]]
            
            model = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)
            
            # Extract coefficients
            beta_price = res.params[price_def['var']]
            beta_eap_int = res.params[interaction_name]
            beta_price_covid = res.params[price_covid_name]
            beta_triple = res.params[triple_name]
            
            # Calculate elasticities by period×region
            eu_pre = beta_price
            eap_pre = beta_price + beta_eap_int
            eu_covid = beta_price + beta_price_covid
            eap_covid = beta_price + beta_eap_int + beta_price_covid + beta_triple
            
            # Standard errors
            se_price = res.std_errors[price_def['var']]
            se_eap_int = res.std_errors[interaction_name]
            se_price_covid = res.std_errors[price_covid_name]
            se_triple = res.std_errors[triple_name]
            
            eu_pre_se = se_price
            eap_pre_se = np.sqrt(se_price**2 + se_eap_int**2)
            eu_covid_se = np.sqrt(se_price**2 + se_price_covid**2)
            eap_covid_se = np.sqrt(se_price**2 + se_eap_int**2 + se_price_covid**2 + se_triple**2)
            
            # P-values
            eu_pre_pval = 2 * (1 - stats.t.cdf(abs(eu_pre/eu_pre_se), df=res.df_resid))
            eap_pre_pval = 2 * (1 - stats.t.cdf(abs(eap_pre/eap_pre_se), df=res.df_resid))
            eu_covid_pval = 2 * (1 - stats.t.cdf(abs(eu_covid/eu_covid_se), df=res.df_resid))
            eap_covid_pval = 2 * (1 - stats.t.cdf(abs(eap_covid/eap_covid_se), df=res.df_resid))
            
            # Format significance
            sig_eu_pre = "***" if eu_pre_pval < 0.01 else "**" if eu_pre_pval < 0.05 else "*" if eu_pre_pval < 0.10 else ""
            sig_eap_pre = "***" if eap_pre_pval < 0.01 else "**" if eap_pre_pval < 0.05 else "*" if eap_pre_pval < 0.10 else ""
            sig_eu_covid = "***" if eu_covid_pval < 0.01 else "**" if eu_covid_pval < 0.05 else "*" if eu_covid_pval < 0.10 else ""
            sig_eap_covid = "***" if eap_covid_pval < 0.01 else "**" if eap_covid_pval < 0.05 else "*" if eap_covid_pval < 0.10 else ""
            
            # Print results
            print(f"\n  [{control_name}]")
            print(f"    PRE-COVID:")
            print(f"      EU:  {eu_pre:7.4f}{sig_eu_pre:3s} (SE={eu_pre_se:.4f}, p={eu_pre_pval:.3f})")
            print(f"      EaP: {eap_pre:7.4f}{sig_eap_pre:3s} (SE={eap_pre_se:.4f}, p={eap_pre_pval:.3f})")
            print(f"      Ratio: {abs(eap_pre/eu_pre):.2f}x")
            print(f"    COVID:")
            print(f"      EU:  {eu_covid:7.4f}{sig_eu_covid:3s} (SE={eu_covid_se:.4f}, p={eu_covid_pval:.3f})")
            print(f"      EaP: {eap_covid:7.4f}{sig_eap_covid:3s} (SE={eap_covid_se:.4f}, p={eap_covid_pval:.3f})")
            print(f"      Ratio: {abs(eap_covid/eu_covid):.2f}x")
            print(f"    R²={res.rsquared:.4f}")
            
            # Store results
            comprehensive_results.append({
                'spec_id': spec_counter,
                'control_spec': control_name,
                'price_measure': price_def['name'],
                'price_description': price_def['desc'],
                'controls': ', '.join([c for c in controls if c in df_spec.columns]),
                'n_controls': len([c for c in controls if c in df_spec.columns]),
                'eu_pre_elasticity': eu_pre,
                'eu_pre_se': eu_pre_se,
                'eu_pre_pval': eu_pre_pval,
                'eap_pre_elasticity': eap_pre,
                'eap_pre_se': eap_pre_se,
                'eap_pre_pval': eap_pre_pval,
                'eu_covid_elasticity': eu_covid,
                'eu_covid_se': eu_covid_se,
                'eu_covid_pval': eu_covid_pval,
                'eap_covid_elasticity': eap_covid,
                'eap_covid_se': eap_covid_se,
                'eap_covid_pval': eap_covid_pval,
                'ratio_pre': abs(eap_pre/eu_pre) if eu_pre != 0 else np.nan,
                'ratio_covid': abs(eap_covid/eu_covid) if eu_covid != 0 else np.nan,
                'price_x_covid_coef': beta_price_covid,
                'price_x_covid_pval': res.pvalues[price_covid_name],
                'triple_interaction_coef': beta_triple,
                'triple_interaction_pval': res.pvalues[triple_name],
                'n_obs': res.nobs,
                'r_squared': res.rsquared
            })
                
        except Exception as e:
            print(f"\n  [{control_name}] ERROR: {str(e)[:50]}")
            continue

# Save comprehensive results
comprehensive_df = pd.DataFrame(comprehensive_results)
comprehensive_df.to_excel(FULL_SAMPLE_DIR / 'price_robustness_matrix.xlsx', index=False)

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"\n[OK] Price robustness matrix saved: full_sample_covid_analysis/price_robustness_matrix.xlsx")
print(f"  • Total specifications successfully estimated: {len(comprehensive_results)}")
print(f"  • Control specifications: {comprehensive_df['control_spec'].nunique()}")
print(f"  • Price measures: {comprehensive_df['price_measure'].nunique()}")

# Summary by price measure
print(f"\n[SUMMARY BY PRICE MEASURE]")
for price_name in comprehensive_df['price_measure'].unique():
    subset = comprehensive_df[comprehensive_df['price_measure'] == price_name]
    n_sig_eap_pre = (subset['eap_pre_pval'] < 0.05).sum()
    n_sig_eap_covid = (subset['eap_covid_pval'] < 0.05).sum()
    n_sig_triple = (subset['triple_interaction_pval'] < 0.05).sum()
    
    print(f"\n  {price_name}:")
    print(f"    • Specifications: {len(subset)}")
    print(f"\n    PRE-COVID (2010-2019):")
    print(f"      • EU elasticity range: [{subset['eu_pre_elasticity'].min():.3f}, {subset['eu_pre_elasticity'].max():.3f}]")
    print(f"      • EaP elasticity range: [{subset['eap_pre_elasticity'].min():.3f}, {subset['eap_pre_elasticity'].max():.3f}]")
    print(f"      • Ratio range: [{subset['ratio_pre'].min():.2f}x, {subset['ratio_pre'].max():.2f}x]")
    print(f"      • EaP significant (p<0.05): {n_sig_eap_pre}/{len(subset)} ({n_sig_eap_pre/len(subset)*100:.1f}%)")
    
    print(f"\n    COVID PERIOD (2020-2024):")
    print(f"      • EU elasticity range: [{subset['eu_covid_elasticity'].min():.3f}, {subset['eu_covid_elasticity'].max():.3f}]")
    print(f"      • EaP elasticity range: [{subset['eap_covid_elasticity'].min():.3f}, {subset['eap_covid_elasticity'].max():.3f}]")
    print(f"      • Ratio range: [{subset['ratio_covid'].min():.2f}x, {subset['ratio_covid'].max():.2f}x]")
    print(f"      • EaP significant (p<0.05): {n_sig_eap_covid}/{len(subset)} ({n_sig_eap_covid/len(subset)*100:.1f}%)")
    
    print(f"\n    COVID EFFECTS:")
    print(f"      • Triple interaction significant: {n_sig_triple}/{len(subset)} ({n_sig_triple/len(subset)*100:.1f}%)")
    print(f"      • Mean R²: {subset['r_squared'].mean():.3f} (range: [{subset['r_squared'].min():.3f}, {subset['r_squared'].max():.3f}])")

# Overall summary
print(f"\n[OVERALL SUMMARY]")
print(f"  • Total specifications: {len(comprehensive_results)}")
print(f"\n  PRE-COVID:")
print(f"    • EU elasticity range: [{comprehensive_df['eu_pre_elasticity'].min():.3f}, {comprehensive_df['eu_pre_elasticity'].max():.3f}]")
print(f"    • EaP elasticity range: [{comprehensive_df['eap_pre_elasticity'].min():.3f}, {comprehensive_df['eap_pre_elasticity'].max():.3f}]")
print(f"    • Ratio range: [{comprehensive_df['ratio_pre'].min():.2f}x, {comprehensive_df['ratio_pre'].max():.2f}x]")
print(f"    • EaP significant (p<0.05): {(comprehensive_df['eap_pre_pval'] < 0.05).sum()}/{len(comprehensive_df)} ({(comprehensive_df['eap_pre_pval'] < 0.05).sum()/len(comprehensive_df)*100:.1f}%)")

print(f"\n  COVID:")
print(f"    • EU elasticity range: [{comprehensive_df['eu_covid_elasticity'].min():.3f}, {comprehensive_df['eu_covid_elasticity'].max():.3f}]")
print(f"    • EaP elasticity range: [{comprehensive_df['eap_covid_elasticity'].min():.3f}, {comprehensive_df['eap_covid_elasticity'].max():.3f}]")
print(f"    • Ratio range: [{comprehensive_df['ratio_covid'].min():.2f}x, {comprehensive_df['ratio_covid'].max():.2f}x]")
print(f"    • EaP significant (p<0.05): {(comprehensive_df['eap_covid_pval'] < 0.05).sum()}/{len(comprehensive_df)} ({(comprehensive_df['eap_covid_pval'] < 0.05).sum()/len(comprehensive_df)*100:.1f}%)")

print(f"\n  COVID EFFECTS:")
print(f"    • Triple interaction significant: {(comprehensive_df['triple_interaction_pval'] < 0.05).sum()}/{len(comprehensive_df)} ({(comprehensive_df['triple_interaction_pval'] < 0.05).sum()/len(comprehensive_df)*100:.1f}%)")

print(f"\n  MODEL FIT:")
print(f"    • R² range: [{comprehensive_df['r_squared'].min():.3f}, {comprehensive_df['r_squared'].max():.3f}]")

print("\n" + "=" * 80)
print("✓ COMPREHENSIVE ANALYSIS COMPLETE (Full Sample with COVID Interactions)")
print("=" * 80)
print(f"\nModel specification:")
print(f"  • Time FE absorb COVID dummy (constant within year)")
print(f"  • Price×COVID: Change in EU price elasticity during pandemic")
print(f"  • Price×EaP×COVID: Change in regional difference during pandemic")
print(f"\nElasticity calculation:")
print(f"  • EU (Pre-COVID) = β_price")
print(f"  • EaP (Pre-COVID) = β_price + β_price×EaP")
print(f"  • EU (COVID) = β_price + β_price×COVID")
print(f"  • EaP (COVID) = β_price + β_price×EaP + β_price×COVID + β_price×EaP×COVID")
