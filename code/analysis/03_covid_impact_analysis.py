"""
COVID-19 Regional Impact Analysis
==================================
Examines how COVID-19 pandemic affected EU vs EaP regions differently
Tests for structural breaks and regional heterogeneity during pandemic period
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'manuscript2' / 'tables'
FIGURES_DIR = BASE_DIR / 'figures' / 'descriptive'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load full data (including COVID period)
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')

print("="*80)
print("COVID-19 REGIONAL IMPACT ANALYSIS")
print("="*80)
print(f"Full sample: {len(df)} observations")
print(f"Countries: {df['country'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

# Create variables
df['log_internet_users'] = np.log(df['internet_users_pct_i99H'] + 0.01)
df['log_price_gni'] = np.log(df['fixed_broad_price_gni_pct'] + 0.01)
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])

# Create COVID dummy (2020+)
df['covid_period'] = (df['year'] >= 2020).astype(float)

# Create EaP dummy
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df['country'].isin(eap_countries).astype(float)
df['eu_dummy'] = (~df['country'].isin(eap_countries)).astype(float)

# Create interactions
df['covid_x_eap'] = df['covid_period'] * df['eap_dummy']
df['covid_x_eu'] = df['covid_period'] * df['eu_dummy']
df['price_x_eap'] = df['log_price_gni'] * df['eap_dummy']

# Set panel index
df['year_dt'] = pd.to_datetime(df['year'], format='%Y')
df = df.set_index(['country', 'year_dt'])

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS: COVID vs PRE-COVID")
print("="*80)

# Split data
df_precovid = df[df['covid_period'] == 0].copy()
df_covid = df[df['covid_period'] == 1].copy()

print("\nPre-COVID (2010-2019):")
print(f"  Observations: {len(df_precovid)}")
print(f"  EU observations: {(df_precovid['eu_dummy'] == 1).sum()}")
print(f"  EaP observations: {(df_precovid['eap_dummy'] == 1).sum()}")

print("\nCOVID Period (2020-2024):")
print(f"  Observations: {len(df_covid)}")
print(f"  EU observations: {(df_covid['eu_dummy'] == 1).sum()}")
print(f"  EaP observations: {(df_covid['eap_dummy'] == 1).sum()}")

# Calculate mean adoption rates
print("\n" + "-"*80)
print("Mean Internet Adoption Rates (%)")
print("-"*80)

for region, dummy in [('EU', 'eu_dummy'), ('EaP', 'eap_dummy')]:
    precovid_mean = df_precovid[df_precovid[dummy] == 1]['internet_users_pct_i99H'].mean()
    covid_mean = df_covid[df_covid[dummy] == 1]['internet_users_pct_i99H'].mean()
    change = covid_mean - precovid_mean
    pct_change = (change / precovid_mean) * 100
    
    print(f"\n{region}:")
    print(f"  Pre-COVID (2010-2019): {precovid_mean:.1f}%")
    print(f"  COVID (2020-2024):     {covid_mean:.1f}%")
    print(f"  Absolute change:       {change:+.1f} percentage points")
    print(f"  Relative change:       {pct_change:+.1f}%")

print("\n" + "="*80)
print("MODEL 1: COVID EFFECT WITHOUT REGIONAL HETEROGENEITY")
print("="*80)

required1 = ['log_internet_users', 'log_price_gni', 'covid_period',
             'log_gdp_per_capita', 'research_development_expenditure',
             'secure_internet_servers']
df1 = df[required1].dropna()

print(f"\nSample: {len(df1)} observations")

y1 = df1['log_internet_users']
X1 = df1[['log_price_gni', 'covid_period', 'log_gdp_per_capita',
          'research_development_expenditure', 'secure_internet_servers']]

model1 = PanelOLS(y1, X1, entity_effects=True, time_effects=False)  # No time FE to identify COVID
res1 = model1.fit(cov_type='clustered', cluster_entity=True)

covid_coef1 = res1.params['covid_period']
covid_se1 = res1.std_errors['covid_period']
covid_pval1 = res1.pvalues['covid_period']

def sig_stars(pval):
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    else: return ""

print(f"\nCOVID coefficient (pooled): {covid_coef1:.6f}{sig_stars(covid_pval1)} (SE={covid_se1:.6f}, p={covid_pval1:.4f})")
print(f"Interpretation: COVID increased internet adoption by {(np.exp(covid_coef1)-1)*100:.2f}% on average")

print("\n" + "="*80)
print("MODEL 2: COVID EFFECT WITH REGIONAL HETEROGENEITY")
print("="*80)

required2 = ['log_internet_users', 'log_price_gni', 'covid_x_eu', 'covid_x_eap',
             'log_gdp_per_capita', 'research_development_expenditure',
             'secure_internet_servers']
df2 = df[required2].dropna()

print(f"\nSample: {len(df2)} observations")

y2 = df2['log_internet_users']
X2 = df2[['log_price_gni', 'covid_x_eu', 'covid_x_eap', 'log_gdp_per_capita',
          'research_development_expenditure', 'secure_internet_servers']]

model2 = PanelOLS(y2, X2, entity_effects=True, time_effects=False)
res2 = model2.fit(cov_type='clustered', cluster_entity=True)

covid_eu = res2.params['covid_x_eu']
covid_eap = res2.params['covid_x_eap']
se_eu = res2.std_errors['covid_x_eu']
se_eap = res2.std_errors['covid_x_eap']
pval_eu = res2.pvalues['covid_x_eu']
pval_eap = res2.pvalues['covid_x_eap']

print(f"\nCOVID effect in EU:  {covid_eu:.6f}{sig_stars(pval_eu)} (SE={se_eu:.6f}, p={pval_eu:.4f})")
print(f"  → Internet adoption increased by {(np.exp(covid_eu)-1)*100:.2f}% in EU")

print(f"\nCOVID effect in EaP: {covid_eap:.6f}{sig_stars(pval_eap)} (SE={se_eap:.6f}, p={pval_eap:.4f})")
print(f"  → Internet adoption increased by {(np.exp(covid_eap)-1)*100:.2f}% in EaP")

# Test if effects are different
diff = covid_eap - covid_eu
diff_se = np.sqrt(se_eap**2 + se_eu**2)
diff_tstat = diff / diff_se
diff_pval = 2 * (1 - stats.t.cdf(abs(diff_tstat), df=res2.df_resid))

print(f"\nDifference (EaP - EU): {diff:.6f}{sig_stars(diff_pval)} (SE={diff_se:.6f}, p={diff_pval:.4f})")

if diff > 0 and diff_pval < 0.05:
    print(f"  ✓ EaP experienced SIGNIFICANTLY LARGER COVID boost")
    print(f"    EaP adoption increased {(np.exp(covid_eap)-1)*100:.2f}% vs EU's {(np.exp(covid_eu)-1)*100:.2f}%")
elif diff < 0 and diff_pval < 0.05:
    print(f"  ✓ EU experienced SIGNIFICANTLY LARGER COVID boost")
elif diff_pval >= 0.05:
    print(f"  ✗ No significant difference between regions (p={diff_pval:.4f})")
    print(f"    Both regions experienced similar COVID effects")

print("\n" + "="*80)
print("MODEL 3: FULL SPECIFICATION (COVID + PRICE INTERACTIONS)")
print("="*80)

required3 = ['log_internet_users', 'log_price_gni', 'price_x_eap',
             'covid_x_eu', 'covid_x_eap',
             'log_gdp_per_capita', 'research_development_expenditure',
             'secure_internet_servers']
df3 = df[required3].dropna()

print(f"\nSample: {len(df3)} observations")

y3 = df3['log_internet_users']
X3 = df3[['log_price_gni', 'price_x_eap', 'covid_x_eu', 'covid_x_eap',
          'log_gdp_per_capita', 'research_development_expenditure',
          'secure_internet_servers']]

model3 = PanelOLS(y3, X3, entity_effects=True, time_effects=False)
res3 = model3.fit(cov_type='clustered', cluster_entity=True)

print("\nAll Coefficients:")
for var in ['log_price_gni', 'price_x_eap', 'covid_x_eu', 'covid_x_eap']:
    coef = res3.params[var]
    se = res3.std_errors[var]
    pval = res3.pvalues[var]
    print(f"  {var:25s}: {coef:8.6f}{sig_stars(pval)} (SE={se:.6f}, p={pval:.4f})")

print(f"\nR-squared: {res3.rsquared:.4f}")

print("\n" + "="*80)
print("VISUALIZATION: COVID IMPACT BY REGION")
print("="*80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Time series by region
ax1 = axes[0, 0]
for region, dummy in [('EU', 'eu_dummy'), ('EaP', 'eap_dummy')]:
    df_region = df[df[dummy] == 1].reset_index()
    yearly_mean = df_region.groupby('year')['internet_users_pct_i99H'].mean()
    ax1.plot(yearly_mean.index, yearly_mean.values, marker='o', label=region, linewidth=2)

ax1.axvline(x=2019.5, color='red', linestyle='--', alpha=0.7, label='COVID Start')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Internet Users (%)', fontsize=11)
ax1.set_title('Internet Adoption Over Time by Region', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Pre-COVID vs COVID comparison
ax2 = axes[0, 1]
regions = ['EU', 'EaP']
precovid_means = []
covid_means = []

for region, dummy in [('EU', 'eu_dummy'), ('EaP', 'eap_dummy')]:
    precovid = df_precovid[df_precovid[dummy] == 1]['internet_users_pct_i99H'].mean()
    covid = df_covid[df_covid[dummy] == 1]['internet_users_pct_i99H'].mean()
    precovid_means.append(precovid)
    covid_means.append(covid)

x = np.arange(len(regions))
width = 0.35
ax2.bar(x - width/2, precovid_means, width, label='Pre-COVID', alpha=0.8)
ax2.bar(x + width/2, covid_means, width, label='COVID', alpha=0.8)
ax2.set_ylabel('Internet Users (%)', fontsize=11)
ax2.set_title('Mean Adoption: Pre-COVID vs COVID', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(regions)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (p, c) in enumerate(zip(precovid_means, covid_means)):
    ax2.text(i - width/2, p + 1, f'{p:.1f}%', ha='center', fontsize=9)
    ax2.text(i + width/2, c + 1, f'{c:.1f}%', ha='center', fontsize=9)

# 3. COVID effect coefficients
ax3 = axes[1, 0]
effects = [covid_eu, covid_eap]
errors = [se_eu, se_eap]
bars = ax3.bar(regions, effects, yerr=errors, capsize=10, alpha=0.7, 
               color=['#1f77b4', '#ff7f0e'])
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_ylabel('COVID Effect (log points)', fontsize=11)
ax3.set_title('Estimated COVID Effect by Region\n(from regression)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add significance stars
for i, (effect, pval) in enumerate(zip(effects, [pval_eu, pval_eap])):
    stars = sig_stars(pval)
    ax3.text(i, effect + errors[i] + 0.01, stars, ha='center', fontsize=14)

# 4. Percentage change
ax4 = axes[1, 1]
pct_changes = [(np.exp(covid_eu)-1)*100, (np.exp(covid_eap)-1)*100]
bars = ax4.bar(regions, pct_changes, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.set_ylabel('% Change in Adoption', fontsize=11)
ax4.set_title('COVID Impact on Internet Adoption\n(% change)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, pct in enumerate(pct_changes):
    ax4.text(i, pct + 0.5, f'{pct:+.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'covid_regional_impact.png', dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved visualization: {FIGURES_DIR / 'covid_regional_impact.png'}")

print("\n" + "="*80)
print("SUMMARY AND INTERPRETATION")
print("="*80)

print("\n1. OVERALL COVID EFFECT:")
print(f"   Pooled estimate: {(np.exp(covid_coef1)-1)*100:+.2f}% increase in adoption")
print(f"   Statistical significance: {sig_stars(covid_pval1)} (p={covid_pval1:.4f})")

print("\n2. REGIONAL HETEROGENEITY:")
print(f"   EU effect:  {(np.exp(covid_eu)-1)*100:+.2f}% {sig_stars(pval_eu)}")
print(f"   EaP effect: {(np.exp(covid_eap)-1)*100:+.2f}% {sig_stars(pval_eap)}")
print(f"   Difference: {(np.exp(diff)-1)*100:+.2f}% {sig_stars(diff_pval)} (p={diff_pval:.4f})")

print("\n3. MECHANISMS:")
if covid_eap > covid_eu and diff_pval < 0.10:
    print("   ✓ EaP countries caught up during COVID")
    print("   Possible reasons:")
    print("     - Lower baseline adoption → more room to grow")
    print("     - Remote work/education became essential")
    print("     - Government digitalization initiatives")
    print("     - Infrastructure investments accelerated")
elif covid_eu > covid_eap and diff_pval < 0.10:
    print("   ✓ EU countries benefited more from COVID digitalization")
    print("   Possible reasons:")
    print("     - Better digital infrastructure")
    print("     - Higher remote work adoption")
    print("     - Stronger institutional response")
else:
    print("   ≈ Both regions experienced similar COVID effects")
    print("   Possible reasons:")
    print("     - Pandemic was global shock affecting all equally")
    print("     - Digital divide persists despite COVID push")
    print("     - Structural factors more important than crisis response")

print("\n4. POLICY IMPLICATIONS:")
print("   - COVID accelerated digital transformation globally")
print("   - Price elasticity patterns likely stable pre/post COVID")
print("   - Consider COVID dummy in full-period regressions")
print("   - Pre-COVID period (2010-2019) may be more stable for elasticity estimation")

print("\n" + "="*80)
print(f"[OK] Analysis complete. Results show {sig_stars(diff_pval)} regional difference in COVID impact")
print("="*80)

# Save results
results_df = pd.DataFrame({
    'region': ['EU', 'EaP', 'Difference'],
    'covid_effect_log': [covid_eu, covid_eap, diff],
    'covid_effect_pct': [(np.exp(covid_eu)-1)*100, (np.exp(covid_eap)-1)*100, (np.exp(diff)-1)*100],
    'std_error': [se_eu, se_eap, diff_se],
    'p_value': [pval_eu, pval_eap, diff_pval],
    'significant': [sig_stars(pval_eu), sig_stars(pval_eap), sig_stars(diff_pval)]
})

results_df.to_csv(RESULTS_DIR / 'covid_regional_impact.csv', index=False)
print(f"[OK] Saved results: {RESULTS_DIR / 'covid_regional_impact.csv'}")
