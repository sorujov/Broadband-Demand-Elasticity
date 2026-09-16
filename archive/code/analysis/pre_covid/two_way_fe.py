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
PRE_COVID_DIR = RESULTS_DIR / 'pre_covid_analysis'
PRE_COVID_DIR.mkdir(parents=True, exist_ok=True)

# Load and filter data (Pre-COVID 2010-2019)
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df_full = pd.read_csv(ANALYSIS_READY_FILE)
print(f"\nLoaded: {ANALYSIS_READY_FILE}")
print(f"  • Full dataset: {len(df_full)} observations")
print(f"  • Years: {df_full['year'].min()}-{df_full['year'].max()}")

# Filter to pre-COVID
df_full['year_num'] = df_full['year'].astype(int)
df = df_full[df_full['year_num'] <= 2019].copy()
print(f"\n  • Pre-COVID (2010-2019): {len(df)} observations")

# Create regional variables
df['eap_dummy'] = df['country'].isin(EAP_COUNTRIES).astype(float)
df['price_x_eap'] = df[PRIMARY_PRICE] * df['eap_dummy']

# Set panel index
df['year_dt'] = pd.to_datetime(df['year_num'], format='%Y')
df = df.set_index(['country', 'year_dt'])

# ============================================================================
# BASELINE MODEL (for reference comparison)
# ============================================================================

print("\n" + "=" * 80)
print("BASELINE MODEL ESTIMATION")
print("=" * 80)

controls_baseline = ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                     'regulatory_quality_estimate', 'log_secure_internet_servers', 
                     'research_development_expenditure', 'population_ages_15_64',
                     'gdp_growth', 'inflation_gdp_deflator', 'log_population_density']
required_baseline = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + controls_baseline
df_baseline = df[required_baseline].dropna()

print(f"\nBaseline specification (Full Controls):")
print(f"  • Controls: GDP, urban, education, regulatory, infrastructure, R&D, demographics, macro")
print(f"  • N observations: {len(df_baseline)}")
print(f"  • N countries: {df_baseline.index.get_level_values('country').nunique()}")

y_base = df_baseline[PRIMARY_DV]
X_base = df_baseline[[PRIMARY_PRICE, 'price_x_eap'] + controls_baseline]

model_baseline = PanelOLS(y_base, X_base, entity_effects=True, time_effects=True)
res_baseline = model_baseline.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

beta_price = res_baseline.params[PRIMARY_PRICE]
beta_interaction = res_baseline.params['price_x_eap']
se_price = res_baseline.std_errors[PRIMARY_PRICE]
se_interaction = res_baseline.std_errors['price_x_eap']

eu_elasticity = beta_price
eap_elasticity = beta_price + beta_interaction
eu_se = se_price
eap_se = np.sqrt(se_price**2 + se_interaction**2 + 2 * res_baseline.cov.loc[PRIMARY_PRICE, 'price_x_eap'])

eu_pval = 2 * (1 - stats.t.cdf(abs(eu_elasticity/eu_se), df=res_baseline.df_resid))
eap_pval = 2 * (1 - stats.t.cdf(abs(eap_elasticity/eap_se), df=res_baseline.df_resid))

sig_eu = "***" if eu_pval < 0.01 else "**" if eu_pval < 0.05 else "*" if eu_pval < 0.10 else ""
sig_eap = "***" if eap_pval < 0.01 else "**" if eap_pval < 0.05 else "*" if eap_pval < 0.10 else ""

print(f"\nBaseline Results:")
print(f"  • EU elasticity:  {eu_elasticity:7.4f}{sig_eu:3s} (p={eu_pval:.4f})")
print(f"  • EaP elasticity: {eap_elasticity:7.4f}{sig_eap:3s} (p={eap_pval:.4f})")
print(f"  • Interaction: {beta_interaction:7.4f} (p={res_baseline.pvalues['price_x_eap']:.4f})")
print(f"  • R²={res_baseline.rsquared:.4f}")

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
    required = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + controls

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
    X = df_spec[[PRIMARY_PRICE, 'price_x_eap'] + [c for c in controls if c in df_spec.columns]]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

    # Extract elasticities
    beta_price = res.params[PRIMARY_PRICE]
    beta_interaction = res.params['price_x_eap']
    se_price = res.std_errors[PRIMARY_PRICE]
    se_interaction = res.std_errors['price_x_eap']

    eu_elast = beta_price
    eap_elast = beta_price + beta_interaction
    eu_se = se_price
    eap_se = np.sqrt(se_price**2 + se_interaction**2 + 2 * res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])

    eu_pval = 2 * (1 - stats.t.cdf(abs(eu_elast/eu_se), df=res.df_resid))
    eap_pval = 2 * (1 - stats.t.cdf(abs(eap_elast/eap_se), df=res.df_resid))

    # Format output
    sig_eu = "***" if eu_pval < 0.01 else "**" if eu_pval < 0.05 else "*" if eu_pval < 0.10 else ""
    sig_eap = "***" if eap_pval < 0.01 else "**" if eap_pval < 0.05 else "*" if eap_pval < 0.10 else ""

    print(f"  EU:  {eu_elast:7.4f}{sig_eu:3s} (p={eu_pval:.3f})")
    print(f"  EaP: {eap_elast:7.4f}{sig_eap:3s} (p={eap_pval:.3f})")
    print(f"  Ratio: {abs(eap_elast/eu_elast):.2f}x, N={res.nobs}, R²={res.rsquared:.4f}")

    # Store results
    gdp_col = 'log_gdp_per_capita'
    extended_results.append({
        'specification': spec_name,
        'description': spec_info['description'],
        'controls': ', '.join([c for c in controls if c in df_spec.columns]),
        'n_controls': len([c for c in controls if c in df_spec.columns]),
        'eu_elasticity': eu_elast,
        'eu_se': eu_se,
        'eu_pval': eu_pval,
        'eap_elasticity': eap_elast,
        'eap_se': eap_se,
        'eap_pval': eap_pval,
        'interaction_coef': beta_interaction,
        'interaction_pval': res.pvalues['price_x_eap'],
        'gdp_coef': res.params[gdp_col] if gdp_col in res.params.index else np.nan,
        'gdp_pval': res.pvalues[gdp_col] if gdp_col in res.pvalues.index else np.nan,
        'n_obs': res.nobs,
        'r_squared': res.rsquared,
        'df_resid': res.df_resid
    })

# Save extended results
extended_df = pd.DataFrame(extended_results)
extended_df.to_excel(PRE_COVID_DIR / 'extended_control_specifications.xlsx', index=False)

print(f"\n[OK] Extended specifications saved: pre_covid_analysis/extended_control_specifications.xlsx")
print(f"  • Total specifications tested: {len(extended_results)}")
print(f"  • All use pre-COVID 2010-2019 consistently")


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
print(f"  × 1 subscription measure (PRIMARY_DV: {PRIMARY_DV})")
print(f"  = {len(CONTROL_SPECS) * len(PRICE_MEASURES)} total specifications")

comprehensive_results = []
spec_counter = 0

print(f"\n{'='*80}")
print("RESULTS BY PRICE MEASURE")
print(f"{'='*80}")

for price_def in PRICE_MEASURES:
    print(f"\n{'-'*80}")
    print(f"PRICE MEASURE: {price_def['name']} - {price_def['desc']}")
    print(f"{'-'*80}")
    
    for control_name, control_info in CONTROL_SPECS.items():
        spec_counter += 1
        
        # Create interaction term
        interaction_name = f"price_x_eap_{price_def['name']}"
        df[interaction_name] = df[price_def['var']] * df['eap_dummy']
        
        # Select required variables
        controls = control_info['controls']
        required = [PRIMARY_DV, price_def['var'], interaction_name] + controls
        
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
            X = df_spec[[price_def['var'], interaction_name] + [c for c in controls if c in df_spec.columns]]
            
            model = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)
            
            # Extract elasticities
            beta_price = res.params[price_def['var']]
            beta_interaction = res.params[interaction_name]
            se_price = res.std_errors[price_def['var']]
            se_interaction = res.std_errors[interaction_name]
            
            eu_elast = beta_price
            eap_elast = beta_price + beta_interaction
            eu_se = se_price
            eap_se = np.sqrt(se_price**2 + se_interaction**2 + 2 * res.cov.loc[price_def['var'], interaction_name])
            
            eu_pval = 2 * (1 - stats.t.cdf(abs(eu_elast/eu_se), df=res.df_resid))
            eap_pval = 2 * (1 - stats.t.cdf(abs(eap_elast/eap_se), df=res.df_resid))
            
            # Format significance
            sig_eu = "***" if eu_pval < 0.01 else "**" if eu_pval < 0.05 else "*" if eu_pval < 0.10 else ""
            sig_eap = "***" if eap_pval < 0.01 else "**" if eap_pval < 0.05 else "*" if eap_pval < 0.10 else ""
            
            # Print results
            print(f"\n  [{control_name}]")
            print(f"    EU:  {eu_elast:7.4f}{sig_eu:3s} (SE={eu_se:.4f}, p={eu_pval:.3f})")
            print(f"    EaP: {eap_elast:7.4f}{sig_eap:3s} (SE={eap_se:.4f}, p={eap_pval:.3f})")
            print(f"    Ratio: {abs(eap_elast/eu_elast):.2f}x, R²={res.rsquared:.4f}")
            
            # Store results
            comprehensive_results.append({
                'spec_id': spec_counter,
                'control_spec': control_name,
                'price_measure': price_def['name'],
                'price_description': price_def['desc'],
                'controls': ', '.join([c for c in controls if c in df_spec.columns]),
                'n_controls': len([c for c in controls if c in df_spec.columns]),
                'eu_elasticity': eu_elast,
                'eu_se': eu_se,
                'eu_pval': eu_pval,
                'eap_elasticity': eap_elast,
                'eap_se': eap_se,
                'eap_pval': eap_pval,
                'ratio': abs(eap_elast/eu_elast) if eu_elast != 0 else np.nan,
                'interaction_coef': beta_interaction,
                'interaction_pval': res.pvalues[interaction_name],
                'n_obs': res.nobs,
                'r_squared': res.rsquared,
                'df_resid': res.df_resid
            })
                
        except Exception as e:
            print(f"\n  [{control_name}] ERROR: {str(e)[:50]}")
            continue

# Save comprehensive results
comprehensive_df = pd.DataFrame(comprehensive_results)
comprehensive_df.to_excel(PRE_COVID_DIR / 'price_robustness_matrix.xlsx', index=False)

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"\n[OK] Price robustness matrix saved: pre_covid_analysis/price_robustness_matrix.xlsx")
print(f"  • Total specifications successfully estimated: {len(comprehensive_results)}")
print(f"  • Control specifications: {comprehensive_df['control_spec'].nunique()}")
print(f"  • Price measures: {comprehensive_df['price_measure'].nunique()}")

# Summary by price measure
print(f"\n[SUMMARY BY PRICE MEASURE]")
for price_name in comprehensive_df['price_measure'].unique():
    subset = comprehensive_df[comprehensive_df['price_measure'] == price_name]
    n_sig_eap = (subset['eap_pval'] < 0.05).sum()
    print(f"\n  {price_name}:")
    print(f"    • Specifications: {len(subset)}")
    print(f"    • EU elasticity range: [{subset['eu_elasticity'].min():.3f}, {subset['eu_elasticity'].max():.3f}]")
    print(f"    • EaP elasticity range: [{subset['eap_elasticity'].min():.3f}, {subset['eap_elasticity'].max():.3f}]")
    print(f"    • Ratio range: [{subset['ratio'].min():.2f}x, {subset['ratio'].max():.2f}x]")
    print(f"    • EaP significant (p<0.05): {n_sig_eap}/{len(subset)} ({n_sig_eap/len(subset)*100:.1f}%)")
    print(f"    • Mean R²: {subset['r_squared'].mean():.3f} (range: [{subset['r_squared'].min():.3f}, {subset['r_squared'].max():.3f}])")

# Overall summary
print(f"\n[OVERALL SUMMARY]")
print(f"  • Total specifications: {len(comprehensive_results)}")
print(f"  • EU elasticity range: [{comprehensive_df['eu_elasticity'].min():.3f}, {comprehensive_df['eu_elasticity'].max():.3f}]")
print(f"  • EaP elasticity range: [{comprehensive_df['eap_elasticity'].min():.3f}, {comprehensive_df['eap_elasticity'].max():.3f}]")
print(f"  • Ratio range: [{comprehensive_df['ratio'].min():.2f}x, {comprehensive_df['ratio'].max():.2f}x]")
print(f"  • EaP significant (p<0.05): {(comprehensive_df['eap_pval'] < 0.05).sum()}/{len(comprehensive_df)} ({(comprehensive_df['eap_pval'] < 0.05).sum()/len(comprehensive_df)*100:.1f}%)")
print(f"  • R² range: [{comprehensive_df['r_squared'].min():.3f}, {comprehensive_df['r_squared'].max():.3f}]")

print("\n" + "=" * 80)
print("✓ COMPREHENSIVE PRICE ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)


# ============================================================================
# GENERATE LATEX TABLES → manuscript/tables/
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING LATEX TABLES")
print("=" * 80)

try:
    from code.utils.config import MANUSCRIPT_TABLES_DIR
except (ImportError, ModuleNotFoundError):
    try:
        from utils.config import MANUSCRIPT_TABLES_DIR
    except ImportError:
        MANUSCRIPT_TABLES_DIR = BASE_DIR / 'manuscript' / 'tables'
MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Sequential build-up specs for Table 1 (7 columns matching the paper)
TABLE_SPECS = [
    ('GDP Only',  ['log_gdp_per_capita']),
    ('+ Socio',   ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct']),
    ('+ Instit.', ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                   'regulatory_quality_estimate']),
    ('+ Infra.',  ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                   'regulatory_quality_estimate', 'log_secure_internet_servers',
                   'research_development_expenditure']),
    ('+ Demog.',  ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                   'regulatory_quality_estimate', 'log_secure_internet_servers',
                   'research_development_expenditure', 'log_population_density',
                   'population_ages_15_64']),
    ('+ All',     ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                   'regulatory_quality_estimate', 'log_secure_internet_servers',
                   'research_development_expenditure', 'log_population_density',
                   'population_ages_15_64']),
    ('Full',      ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                   'regulatory_quality_estimate', 'log_secure_internet_servers',
                   'research_development_expenditure', 'log_population_density',
                   'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator']),
]

CTRL_DISPLAY = [
    ('log_gdp_per_capita',               'Log(GDP per capita)'),
    ('urban_population_pct',             'Urban population (\\%)'),
    ('education_tertiary_pct',           'Tertiary enrollment (\\%)'),
    ('regulatory_quality_estimate',      'Regulatory quality'),
    ('log_secure_internet_servers',      'Log(Secure servers)'),
    ('research_development_expenditure', 'R\\&D (\\% GDP)'),
    ('log_population_density',           'Log(Population density)'),
    ('population_ages_15_64',            'Working-age pop.\\ (15--64, \\%)'),
]


def fmt_coef(coef, se, pval):
    """Format coefficient with significance stars for LaTeX."""
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    if stars:
        return f'${coef:.2f}^{{{stars}}}$', f'({se:.2f})'
    return f'${coef:.2f}$', f'({se:.2f})'


# Run Table 1 regressions
table1_res = []
for spec_label, controls in TABLE_SPECS:
    avail = [c for c in controls if c in df.columns]
    required = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + avail
    df_t = df[required].dropna()
    if len(df_t) < 50:
        table1_res.append(None)
        continue
    y_t = df_t[PRIMARY_DV]
    X_t = df_t[[PRIMARY_PRICE, 'price_x_eap'] + avail]
    res_t = PanelOLS(y_t, X_t, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=3)
    table1_res.append(res_t)

# Build LaTeX for Table 1
n_cols = len(TABLE_SPECS)
col_labels = [s[0] for s in TABLE_SPECS]
col_nums = [f'({i+1})' for i in range(n_cols)]

lines = []
lines += [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Baseline Two-Way Fixed Effects Estimates: Pre-COVID Period (2010--2019)}',
    r'\label{tab:baseline}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\setlength{\tabcolsep}{3pt}',
    r'\renewcommand{\arraystretch}{0.85}',
    f'\\begin{{tabular}}{{@{{}}l{"c" * n_cols}@{{}}}}',
    r'\toprule',
    f'& \\multicolumn{{{n_cols}}}{{c}}{{Dependent Variable: Log(Subscriptions per 100)}} \\\\',
    f'\\cmidrule(lr){{2-{n_cols+1}}}',
    '& ' + ' & '.join(col_nums) + ' \\\\',
    '& ' + ' & '.join(col_labels) + ' \\\\',
    r'\midrule',
    r'\textbf{Panel A: Price Elasticity} \\',
]

def add_row(lines, label, coefs, ses):
    lines.append(label + ' & ' + ' & '.join(coefs) + ' \\\\')
    lines.append('& ' + ' & '.join(ses) + ' \\\\[2pt]')

# Log(Price)
coefs, ses = [], []
for res in table1_res:
    if res is None:
        coefs.append(''); ses.append(''); continue
    c, s = fmt_coef(res.params[PRIMARY_PRICE], res.std_errors[PRIMARY_PRICE],
                    res.pvalues[PRIMARY_PRICE])
    coefs.append(c); ses.append(s)
add_row(lines, 'Log(Price)', coefs, ses)

# Log(Price) × EaP
coefs, ses = [], []
for res in table1_res:
    if res is None:
        coefs.append(''); ses.append(''); continue
    c, s = fmt_coef(res.params['price_x_eap'], res.std_errors['price_x_eap'],
                    res.pvalues['price_x_eap'])
    coefs.append(c); ses.append(s)
add_row(lines, 'Log(Price) $\\times$ EaP', coefs, ses)

# Implied EaP elasticity
coefs, ses = [], []
for res in table1_res:
    if res is None:
        coefs.append(''); ses.append(''); continue
    b1 = res.params[PRIMARY_PRICE]; b2 = res.params['price_x_eap']
    s1 = res.std_errors[PRIMARY_PRICE]; s2 = res.std_errors['price_x_eap']
    eap_b = b1 + b2
    eap_se = np.sqrt(s1**2 + s2**2 + 2 * res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
    eap_pval = 2 * (1 - stats.t.cdf(abs(eap_b / eap_se), df=res.df_resid))
    c, s = fmt_coef(eap_b, eap_se, eap_pval)
    coefs.append(f'\\textit{{{c}}}'); ses.append(f'\\textit{{{s}}}')
add_row(lines, '\\textit{Implied EaP elasticity}', coefs, ses)

lines += [r'\midrule', r'\textbf{Panel B: Control Variables} \\']

# Control variable rows
for ctrl_var, ctrl_name in CTRL_DISPLAY:
    coefs, ses = [], []
    for idx, (_, controls) in enumerate(TABLE_SPECS):
        res = table1_res[idx]
        if res is None or ctrl_var not in controls or ctrl_var not in res.params.index:
            coefs.append(''); ses.append('')
        else:
            c, s = fmt_coef(res.params[ctrl_var], res.std_errors[ctrl_var],
                            res.pvalues[ctrl_var])
            coefs.append(c); ses.append(s)
    add_row(lines, ctrl_name, coefs, ses)

lines += [r'\midrule', r'\textbf{Panel C: Model Statistics} \\']
lines.append('Country fixed effects & ' + ' & '.join(['Yes'] * n_cols) + ' \\\\')
lines.append('Year fixed effects & ' + ' & '.join(['Yes'] * n_cols) + ' \\\\')
lines.append('Observations & ' + ' & '.join(
    [str(int(r.nobs)) if r is not None else '' for r in table1_res]) + ' \\\\')
lines.append('Countries & ' + ' & '.join(['33'] * n_cols) + ' \\\\')
lines.append('R-squared & ' + ' & '.join(
    [f'{r.rsquared:.2f}' if r is not None else '' for r in table1_res]) + ' \\\\')
lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\par\vspace{2pt}',
    r'\scriptsize',
    r'\textit{Notes:} Dependent variable is log fixed broadband subscriptions per 100 inhabitants.',
    r'Price measured as \% of GNI per capita. EaP dummy for Eastern Partnership countries.',
    r'Columns~(6)--(7) include all non-macro controls; (7) adds GDP growth and inflation (coefficients not shown).',
    r'Country and year FE. Driscoll--Kraay SEs (bandwidth$\,=\,$3). Qualitatively unchanged with clustered or robust SEs.',
    r'$^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01.',
    r'\end{minipage}',
    r'\end{table}',
]

table1_path = MANUSCRIPT_TABLES_DIR / 'table1_baseline.tex'
with open(table1_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\n[OK] Table 1 written → {table1_path}")

# --- Table 3: Price Robustness Matrix ---

def build_price_robustness_table(comp_df, out_path):
    """Generate table3_price_robustness.tex from comprehensive results."""
    price_names = comp_df['price_measure'].unique().tolist()
    ctrl_names = comp_df['control_spec'].unique().tolist()

    header_cols = ['Price Measure', 'Control Spec', 'EU Elast.', 'EU SE', 'EU p',
                   'EaP Elast.', 'EaP SE', 'EaP p', 'N', 'R$^2$']
    n_c = len(header_cols)

    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{Price Elasticity Robustness: Alternative Price Measures and Control Specifications (Pre-COVID)}',
        r'\label{tab:price_robustness}',
        r'\begin{minipage}{\textwidth}',
        r'\begin{adjustbox}{width=\textwidth}',
        r'\scriptsize',
        f'\\begin{{tabular}}{{ll{"c" * (n_c - 2)}}}',
        r'\toprule',
        ' & '.join(header_cols) + r' \\',
        r'\midrule',
    ]

    for pm in price_names:
        pm_tex = pm.replace('%', r'\%')
        lines.append(f'\\multicolumn{{{n_c}}}{{l}}{{\\textit{{Price: {pm_tex}}}}} \\\\')
        subset = comp_df[comp_df['price_measure'] == pm]
        for _, row in subset.iterrows():
            eu_stars = '***' if row['eu_pval'] < 0.01 else '**' if row['eu_pval'] < 0.05 else '*' if row['eu_pval'] < 0.10 else ''
            eap_stars = '***' if row['eap_pval'] < 0.01 else '**' if row['eap_pval'] < 0.05 else '*' if row['eap_pval'] < 0.10 else ''
            eu_str = f"${row['eu_elasticity']:.3f}^{{{eu_stars}}}$" if eu_stars else f"${row['eu_elasticity']:.3f}$"
            eap_str = f"${row['eap_elasticity']:.3f}^{{{eap_stars}}}$" if eap_stars else f"${row['eap_elasticity']:.3f}$"
            data_row = [
                '', row['control_spec'],
                eu_str, f"({row['eu_se']:.3f})", f"{row['eu_pval']:.3f}",
                eap_str, f"({row['eap_se']:.3f})", f"{row['eap_pval']:.3f}",
                str(int(row['n_obs'])), f"{row['r_squared']:.2f}"
            ]
            lines.append(' & '.join(data_row) + r' \\')
        lines.append(r'\addlinespace')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{adjustbox}',
        r'\par\vspace{4pt}',
        r'\scriptsize',
        r'\textit{Notes:} Each row is a separate regression. All specifications include country',
        r'and year fixed effects. Driscoll--Kraay standard errors in parentheses.',
        r'$^{*}$ p $<$ 0.10, $^{**}$ p $<$ 0.05, $^{***}$ p $<$ 0.01.',
        r'\end{minipage}',
        r'\end{table}',
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

table3_path = MANUSCRIPT_TABLES_DIR / 'table3_price_robustness.tex'
build_price_robustness_table(comprehensive_df, table3_path)
print(f"[OK] Table 3 written → {table3_path}")

print("\n" + "=" * 80)
print("✓ LATEX TABLES GENERATED")
print("=" * 80)
