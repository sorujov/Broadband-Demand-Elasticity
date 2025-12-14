# -*- coding: utf-8 -*-
"""
COVID Analysis Diagnostics
==========================
Run these tests to verify whether COVID eliminated price elasticity
or if there are data/specification issues.
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import sys
import io

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
FIGURES_DIR = RESULTS_DIR.parent / 'figures' / 'covid_diagnostics'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(ANALYSIS_READY_FILE)
df['year_num'] = df['year'].astype(int)
df['eap_dummy'] = df['country'].isin(EAP_COUNTRIES).astype(float)
df['price_x_eap'] = df[PRIMARY_PRICE] * df['eap_dummy']

print("="*80)
print("COVID ANALYSIS DIAGNOSTICS")
print("="*80)

# ============================================================================
# DIAGNOSTIC 1: Price Variation Comparison
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 1: Price Variation (Pre-COVID vs COVID)")
print("="*80)

df_pre = df[df['year_num'] <= 2019].copy()
df_covid = df[df['year_num'] >= 2020].copy()

print(f"\n[OVERALL PRICE VARIATION]")
print(f"Pre-COVID (2010-2019):")
print(f"  • N observations: {len(df_pre)}")
print(f"  • Mean price: {df_pre[PRIMARY_PRICE].mean():.3f}")
print(f"  • SD price: {df_pre[PRIMARY_PRICE].std():.3f}")
print(f"  • Range: [{df_pre[PRIMARY_PRICE].min():.3f}, {df_pre[PRIMARY_PRICE].max():.3f}]")

print(f"\nCOVID (2020-2024):")
print(f"  • N observations: {len(df_covid)}")
print(f"  • Mean price: {df_covid[PRIMARY_PRICE].mean():.3f}")
print(f"  • SD price: {df_covid[PRIMARY_PRICE].std():.3f}")
print(f"  • Range: [{df_covid[PRIMARY_PRICE].min():.3f}, {df_covid[PRIMARY_PRICE].max():.3f}]")

# Within-country variation (critical for FE models)
print(f"\n[WITHIN-COUNTRY PRICE VARIATION]")
within_pre = df_pre.groupby('country')[PRIMARY_PRICE].std()
within_covid = df_covid.groupby('country')[PRIMARY_PRICE].std()

print(f"Pre-COVID:")
print(f"  • Mean within-SD: {within_pre.mean():.3f}")
print(f"  • Median within-SD: {within_pre.median():.3f}")
print(f"  • Countries with SD > 0.1: {(within_pre > 0.1).sum()}/{len(within_pre)}")

print(f"\nCOVID:")
print(f"  • Mean within-SD: {within_covid.mean():.3f}")
print(f"  • Median within-SD: {within_covid.median():.3f}")
print(f"  • Countries with SD > 0.1: {(within_covid > 0.1).sum()}/{len(within_covid)}")

ratio_variation = within_covid.mean() / within_pre.mean()
print(f"\n⚠️ ASSESSMENT:")
if ratio_variation < 0.5:
    print(f"   COVID within-variation is {ratio_variation:.1%} of pre-COVID")
    print(f"   → MUCH LESS variation during COVID (explains imprecision)")
elif ratio_variation < 0.8:
    print(f"   COVID within-variation is {ratio_variation:.1%} of pre-COVID")
    print(f"   → Somewhat less variation (contributes to imprecision)")
else:
    print(f"   COVID within-variation is {ratio_variation:.1%} of pre-COVID")
    print(f"   → Similar variation (rules out this explanation)")

# ============================================================================
# DIAGNOSTIC 2: Sample Composition
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 2: Sample Composition")
print("="*80)

countries_pre = set(df_pre['country'].unique())
countries_covid = set(df_covid['country'].unique())
countries_both = countries_pre.intersection(countries_covid)

print(f"\n[COUNTRY COVERAGE]")
print(f"  • Pre-COVID countries: {len(countries_pre)}")
print(f"  • COVID countries: {len(countries_covid)}")
print(f"  • Overlap: {len(countries_both)}")
print(f"  • Only pre-COVID: {len(countries_pre - countries_covid)}")
print(f"  • Only COVID: {len(countries_covid - countries_pre)}")

if len(countries_both) < len(countries_pre):
    print(f"\n⚠️ WARNING: Sample composition changed!")
    print(f"   Missing in COVID: {countries_pre - countries_covid}")

print(f"\n[OBSERVATIONS PER COUNTRY]")
obs_pre = df_pre.groupby('country').size()
obs_covid = df_covid.groupby('country').size()

print(f"Pre-COVID:")
print(f"  • Mean: {obs_pre.mean():.1f}")
print(f"  • Range: [{obs_pre.min()}, {obs_pre.max()}]")

print(f"\nCOVID:")
print(f"  • Mean: {obs_covid.mean():.1f}")
print(f"  • Range: [{obs_covid.min()}, {obs_covid.max()}]")

# ============================================================================
# DIAGNOSTIC 3: Year-by-Year Elasticities
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 3: Year-by-Year Price Elasticity")
print("="*80)

df['year_dt'] = pd.to_datetime(df['year_num'], format='%Y')
df = df.set_index(['country', 'year_dt'])

# Controls
controls = ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct']

# Run model with year-specific interactions (instead of COVID dummy)
year_results = []

for year in range(2015, 2025):  # 2015-2024
    # Create year-specific interactions
    df[f'price_x_year{year}'] = df[PRIMARY_PRICE] * (df['year_num'] == year).astype(float)
    df[f'price_x_eap_x_year{year}'] = df[PRIMARY_PRICE] * df['eap_dummy'] * (df['year_num'] == year).astype(float)

# Prepare data
required = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + controls
for year in range(2015, 2025):
    required += [f'price_x_year{year}', f'price_x_eap_x_year{year}']

df_diag = df[required].dropna()
y = df_diag[PRIMARY_DV]
X_vars = [PRIMARY_PRICE, 'price_x_eap'] + controls
for year in range(2015, 2025):
    X_vars += [f'price_x_year{year}', f'price_x_eap_x_year{year}']
X = df_diag[X_vars]

model = PanelOLS(y, X, entity_effects=True, time_effects=True)
res = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

# Extract year-by-year elasticities
print(f"\n[YEAR-BY-YEAR ELASTICITIES]")
print(f"\nYear   EU Elasticity    EaP Elasticity   p(EU)   p(EaP)")
print(f"-" * 65)

beta_base = res.params[PRIMARY_PRICE]
beta_eap = res.params['price_x_eap']

# Baseline (reference: years not in 2015-2024, i.e., 2010-2014)
eu_base = beta_base
eap_base = beta_base + beta_eap
print(f"2010-14 (ref)  {eu_base:7.3f}         {eap_base:7.3f}")

for year in range(2015, 2025):
    try:
        beta_year = res.params[f'price_x_year{year}']
        beta_triple = res.params[f'price_x_eap_x_year{year}']

        eu_year = beta_base + beta_year
        eap_year = beta_base + beta_eap + beta_year + beta_triple

        # Standard errors
        se_year = res.std_errors[f'price_x_year{year}']
        se_triple = res.std_errors[f'price_x_eap_x_year{year}']

        se_eu = np.sqrt(res.std_errors[PRIMARY_PRICE]**2 + se_year**2)
        se_eap = np.sqrt(res.std_errors[PRIMARY_PRICE]**2 + res.std_errors['price_x_eap']**2 + 
                         se_year**2 + se_triple**2)

        p_eu = 2 * (1 - stats.t.cdf(abs(eu_year/se_eu), df=res.df_resid))
        p_eap = 2 * (1 - stats.t.cdf(abs(eap_year/se_eap), df=res.df_resid))

        sig_eu = "***" if p_eu < 0.01 else "**" if p_eu < 0.05 else "*" if p_eu < 0.10 else ""
        sig_eap = "***" if p_eap < 0.01 else "**" if p_eap < 0.05 else "*" if p_eap < 0.10 else ""

        covid_marker = " [COVID]" if year >= 2020 else ""
        print(f"{year}{covid_marker:9s}  {eu_year:7.3f}{sig_eu:3s}      {eap_year:7.3f}{sig_eap:3s}    {p_eu:.3f}   {p_eap:.3f}")

        year_results.append({
            'year': year,
            'eu_elasticity': eu_year,
            'eap_elasticity': eap_year,
            'eu_pval': p_eu,
            'eap_pval': p_eap
        })
    except:
        print(f"{year}         [No data]")

print(f"\n⚠️ ASSESSMENT:")
pre_covid_eap = [r['eap_elasticity'] for r in year_results if r['year'] < 2020]
covid_eap = [r['eap_elasticity'] for r in year_results if r['year'] >= 2020]

if len(pre_covid_eap) > 0 and len(covid_eap) > 0:
    mean_pre = np.mean(pre_covid_eap)
    mean_covid = np.mean(covid_eap)
    print(f"   EaP elasticity pre-COVID mean: {mean_pre:.3f}")
    print(f"   EaP elasticity COVID mean: {mean_covid:.3f}")
    print(f"   Change: {mean_covid - mean_pre:+.3f}")

    if mean_covid > -0.1:
        print(f"   → Elasticity essentially ZERO during COVID (confirms finding)")
    elif abs(mean_covid) < abs(mean_pre) / 2:
        print(f"   → Elasticity substantially REDUCED during COVID")

# Save year-by-year results
df_year_results = pd.DataFrame(year_results)
COVID_RESULTS_DIR = RESULTS_DIR / 'full_sample_covid_analysis'
COVID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_year_results.to_excel(COVID_RESULTS_DIR / 'year_by_year_elasticities.xlsx', index=False)
print(f"\n✅ Year-by-year results saved: {COVID_RESULTS_DIR / 'year_by_year_elasticities.xlsx'}")

# ============================================================================
# DIAGNOSTIC 4: Placebo Test (Pre-COVID Split)
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 4: Placebo Test (2010-2014 vs 2015-2019)")
print("="*80)

df_placebo = df[df['year_num'] <= 2019].copy()
df_placebo['placebo_covid'] = (df_placebo['year_num'] >= 2015).astype(float)

# Reset index for placebo test
df_placebo = df_placebo.reset_index()
df_placebo['year_dt_placebo'] = pd.to_datetime(df_placebo['year_num'], format='%Y')
df_placebo = df_placebo.set_index(['country', 'year_dt_placebo'])

# Create placebo interactions
df_placebo['price_x_placebo'] = df_placebo[PRIMARY_PRICE] * df_placebo['placebo_covid']
df_placebo['price_x_eap_x_placebo'] = df_placebo[PRIMARY_PRICE] * df_placebo['eap_dummy'] * df_placebo['placebo_covid']

# Estimate placebo model
required_placebo = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap', 
                    'price_x_placebo', 'price_x_eap_x_placebo'] + controls
df_placebo_clean = df_placebo[required_placebo].dropna()

y_placebo = df_placebo_clean[PRIMARY_DV]
X_placebo = df_placebo_clean[[PRIMARY_PRICE, 'price_x_eap', 
                               'price_x_placebo', 'price_x_eap_x_placebo'] + controls]

model_placebo = PanelOLS(y_placebo, X_placebo, entity_effects=True, time_effects=True)
res_placebo = model_placebo.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)

beta_placebo = res_placebo.params['price_x_placebo']
p_placebo = res_placebo.pvalues['price_x_placebo']

beta_triple_placebo = res_placebo.params['price_x_eap_x_placebo']
p_triple_placebo = res_placebo.pvalues['price_x_eap_x_placebo']

print(f"\n[PLACEBO RESULTS]")
print(f"Price×Placebo (EU effect):     {beta_placebo:7.3f} (p={p_placebo:.3f})")
print(f"Price×EaP×Placebo (Triple):    {beta_triple_placebo:7.3f} (p={p_triple_placebo:.3f})")

print(f"\n⚠️ ASSESSMENT:")
if p_placebo > 0.10:
    print(f"   ✅ NO spurious time trend (placebo NOT significant)")
    print(f"   → COVID effect is likely REAL, not model artifact")
else:
    print(f"   ⚠️ WARNING: Placebo IS significant!")
    print(f"   → Suggests pre-existing time trend or model issue")
    print(f"   → COVID effect may be spurious")

# Save placebo test results
placebo_results = pd.DataFrame([
    {
        'test_type': 'EU Effect (2015-19 vs 2010-14)',
        'coefficient': beta_placebo,
        'pvalue': p_placebo,
        'interpretation': 'Pass' if p_placebo > 0.10 else 'Fail'
    },
    {
        'test_type': 'EaP Difference (Triple Interaction)',
        'coefficient': beta_triple_placebo,
        'pvalue': p_triple_placebo,
        'interpretation': 'Pass' if p_triple_placebo > 0.10 else 'Fail'
    }
])
placebo_results.to_excel(COVID_RESULTS_DIR / 'placebo_test_results.xlsx', index=False)
print(f"\n✅ Placebo test results saved: {COVID_RESULTS_DIR / 'placebo_test_results.xlsx'}")

# ============================================================================
# DIAGNOSTIC 5: Visualization
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 5: Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Year-by-year elasticities
years_plot = [r['year'] for r in year_results]
eu_elast_plot = [r['eu_elasticity'] for r in year_results]
eap_elast_plot = [r['eap_elasticity'] for r in year_results]

axes[0, 0].plot(years_plot, eu_elast_plot, 'o-', label='EU', markersize=8, linewidth=2)
axes[0, 0].plot(years_plot, eap_elast_plot, 's-', label='EaP', markersize=8, linewidth=2)
axes[0, 0].axvline(2019.5, color='red', linestyle='--', linewidth=2, label='COVID Start', alpha=0.7)
axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Price Elasticity', fontsize=11)
axes[0, 0].set_title('A. Evolution of Price Elasticity Over Time', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Panel B: Price distribution comparison
axes[0, 1].hist(df_pre[PRIMARY_PRICE].dropna(), bins=25, alpha=0.6, label='Pre-COVID', 
                color='blue', edgecolor='black')
axes[0, 1].hist(df_covid[PRIMARY_PRICE].dropna(), bins=25, alpha=0.6, label='COVID', 
                color='red', edgecolor='black')
axes[0, 1].set_xlabel('log(Price as % GNI)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('B. Price Distribution: Pre-COVID vs COVID', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Panel C: Within-country price variation
countries = sorted(within_pre.index)
x_pos = np.arange(len(countries))

axes[1, 0].bar(x_pos - 0.2, [within_pre.get(c, 0) for c in countries], 0.4, 
               label='Pre-COVID', alpha=0.7, color='blue')
axes[1, 0].bar(x_pos + 0.2, [within_covid.get(c, 0) for c in countries], 0.4, 
               label='COVID', alpha=0.7, color='red')
axes[1, 0].set_xlabel('Country', fontsize=11)
axes[1, 0].set_ylabel('Within-Country Price SD', fontsize=11)
axes[1, 0].set_title('C. Within-Country Price Variation by Country', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(countries, rotation=90, fontsize=7)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Panel D: Subscription growth
df_subs = df.reset_index()
df_subs_mean = df_subs.groupby('year_num')[PRIMARY_DV].mean()

axes[1, 1].plot(df_subs_mean.index, df_subs_mean.values, 'o-', linewidth=2, markersize=8, color='purple')
axes[1, 1].axvline(2019.5, color='red', linestyle='--', linewidth=2, label='COVID Start', alpha=0.7)
axes[1, 1].set_xlabel('Year', fontsize=11)
axes[1, 1].set_ylabel('log(Subscriptions)', fontsize=11)
axes[1, 1].set_title('D. Mean Broadband Subscriptions Over Time', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'covid_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Diagnostic plots saved: {FIGURES_DIR / 'covid_diagnostics.png'}")

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

print(f"\n[SUMMARY OF DIAGNOSTICS]\n")

# Assessment 1: Variation
if ratio_variation < 0.6:
    assessment_var = "⚠️ CAUTION: Much less price variation during COVID"
    confidence_var = "Low confidence in COVID estimates"
else:
    assessment_var = "✅ Adequate price variation during COVID"
    confidence_var = "Variation not a major concern"

# Assessment 2: Placebo
if p_placebo > 0.10:
    assessment_placebo = "✅ Placebo test PASSED (no pre-trend)"
    confidence_placebo = "COVID effect likely real"
else:
    assessment_placebo = "⚠️ Placebo test FAILED (pre-trend exists)"
    confidence_placebo = "COVID effect may be spurious"

# Assessment 3: Pattern
pattern_clear = all(r['eap_elasticity'] > -0.15 for r in year_results if r['year'] >= 2020)
if pattern_clear:
    assessment_pattern = "✅ Clear break at 2020 (elasticity collapses)"
    confidence_pattern = "Pattern consistent with COVID shock"
else:
    assessment_pattern = "⚠️ No clear break (gradual change)"
    confidence_pattern = "Pattern suggests pre-existing trend"

print(f"1. Price Variation:    {assessment_var}")
print(f"   → {confidence_var}\n")

print(f"2. Placebo Test:       {assessment_placebo}")
print(f"   → {confidence_placebo}\n")

print(f"3. Temporal Pattern:   {assessment_pattern}")
print(f"   → {confidence_pattern}\n")

# Overall recommendation
passed = sum([ratio_variation >= 0.6, p_placebo > 0.10, pattern_clear])

print(f"\n[OVERALL RECOMMENDATION]\n")

if passed >= 2:
    print(f"✅ LIKELY REAL PHENOMENON (passed {passed}/3 diagnostics)")
    print(f"\nRecommendation:")
    print(f"   • Feature COVID analysis as KEY FINDING")
    print(f"   • Interpret as structural break (broadband became essential)")
    print(f"   • Policy implication: price interventions less effective for necessities")
    print(f"   • Frame as natural experiment")
else:
    print(f"⚠️ UNCERTAIN (passed only {passed}/3 diagnostics)")
    print(f"\nRecommendation:")
    print(f"   • Report COVID analysis as ROBUSTNESS CHECK (appendix)")
    print(f"   • Focus main analysis on pre-COVID period")
    print(f"   • Acknowledge limitations (data quality, short time series)")
    print(f"   • Do NOT make strong claims about COVID effects")

print(f"\n" + "="*80)
print(f"DIAGNOSTICS COMPLETE")
print(f"="*80)
