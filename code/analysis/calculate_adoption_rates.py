"""
Calculate Adoption Rates and Regional Means
============================================
Shows all available adoption metrics and how to calculate regional averages
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load data
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')

print("="*80)
print("AVAILABLE ADOPTION RATE VARIABLES")
print("="*80)

# Find all adoption-related columns
adoption_cols = [col for col in df.columns if any(x in col.lower() for x in 
                 ['internet', 'broadband', 'subs', 'user', 'penetration', 'pct', 'per_100'])]

print(f"\nFound {len(adoption_cols)} adoption-related variables:\n")
for col in sorted(adoption_cols):
    non_missing = df[col].notna().sum()
    coverage = (non_missing / len(df)) * 100
    print(f"  {col:50s} | {non_missing:3d} obs ({coverage:5.1f}%)")

print("\n" + "="*80)
print("PRIMARY ADOPTION METRICS")
print("="*80)

# Define EaP countries
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['region'] = df['country'].apply(lambda x: 'EaP' if x in eap_countries else 'EU')

# Key metrics
metrics = {
    'Internet Users (%)': 'internet_users_pct_i99H',
    'Fixed Broadband Subs (per 100)': 'fixed_broadband_subs_i4213tfbb',
    'Mobile Broadband Subs (per 100)': 'mobile_broadband_subs_i4213tmb',
    'Active Mobile Broadband (per 100)': 'active_mobile_broadband_subs_i46',
}

for metric_name, col_name in metrics.items():
    if col_name in df.columns:
        print(f"\n{metric_name}:")
        print(f"  Variable: {col_name}")
        print(f"  Coverage: {df[col_name].notna().sum()} / {len(df)} ({df[col_name].notna().mean()*100:.1f}%)")
        print(f"  Range: {df[col_name].min():.1f} - {df[col_name].max():.1f}")
        print(f"  Mean: {df[col_name].mean():.1f}")

print("\n" + "="*80)
print("REGIONAL MEANS: OVERALL (2010-2024)")
print("="*80)

for metric_name, col_name in metrics.items():
    if col_name in df.columns:
        print(f"\n{metric_name}:")
        for region in ['EU', 'EaP']:
            mean_val = df[df['region'] == region][col_name].mean()
            median_val = df[df['region'] == region][col_name].median()
            std_val = df[df['region'] == region][col_name].std()
            n_obs = df[df['region'] == region][col_name].notna().sum()
            print(f"  {region:3s}: Mean={mean_val:6.2f}, Median={median_val:6.2f}, SD={std_val:6.2f} (n={n_obs})")

print("\n" + "="*80)
print("REGIONAL MEANS BY TIME PERIOD")
print("="*80)

periods = [
    ('Full Period', 2010, 2024),
    ('Pre-COVID', 2010, 2019),
    ('COVID Era', 2020, 2024),
    ('Early (2010-2014)', 2010, 2014),
    ('Mid (2015-2019)', 2015, 2019),
    ('Recent (2020-2024)', 2020, 2024),
]

# Focus on main metric: Internet Users %
main_metric = 'internet_users_pct_i99H'

if main_metric in df.columns:
    print(f"\nInternet Users (%) by Period and Region:")
    print("-" * 80)
    print(f"{'Period':20s} | {'EU Mean':>10s} | {'EaP Mean':>10s} | {'Difference':>10s} | {'Gap':>6s}")
    print("-" * 80)
    
    for period_name, start_year, end_year in periods:
        df_period = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        eu_mean = df_period[df_period['region'] == 'EU'][main_metric].mean()
        eap_mean = df_period[df_period['region'] == 'EaP'][main_metric].mean()
        diff = eu_mean - eap_mean
        
        print(f"{period_name:20s} | {eu_mean:10.2f} | {eap_mean:10.2f} | {diff:10.2f} | {diff:6.1f}pp")

print("\n" + "="*80)
print("YEAR-BY-YEAR EVOLUTION")
print("="*80)

if main_metric in df.columns:
    print(f"\nInternet Users (%) - Annual Averages:")
    print("-" * 60)
    print(f"{'Year':>6s} | {'EU':>8s} | {'EaP':>8s} | {'Gap':>8s}")
    print("-" * 60)
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        eu_mean = df_year[df_year['region'] == 'EU'][main_metric].mean()
        eap_mean = df_year[df_year['region'] == 'EaP'][main_metric].mean()
        gap = eu_mean - eap_mean
        
        covid_marker = " [COVID]" if year >= 2020 else ""
        print(f"{year:6d} | {eu_mean:8.2f} | {eap_mean:8.2f} | {gap:8.2f}{covid_marker}")

print("\n" + "="*80)
print("GROWTH RATES (YEAR-OVER-YEAR)")
print("="*80)

if main_metric in df.columns:
    print(f"\nInternet Users (%) - YoY Growth:")
    print("-" * 80)
    print(f"{'Year':>6s} | {'EU Growth':>10s} | {'EaP Growth':>10s} | {'EU pp':>8s} | {'EaP pp':>8s}")
    print("-" * 80)
    
    years = sorted(df['year'].unique())
    for i in range(1, len(years)):
        year = years[i]
        prev_year = years[i-1]
        
        df_year = df[df['year'] == year]
        df_prev = df[df['year'] == prev_year]
        
        eu_curr = df_year[df_year['region'] == 'EU'][main_metric].mean()
        eu_prev = df_prev[df_prev['region'] == 'EU'][main_metric].mean()
        eap_curr = df_year[df_year['region'] == 'EaP'][main_metric].mean()
        eap_prev = df_prev[df_prev['region'] == 'EaP'][main_metric].mean()
        
        eu_growth_pct = ((eu_curr - eu_prev) / eu_prev) * 100
        eap_growth_pct = ((eap_curr - eap_prev) / eap_prev) * 100
        eu_growth_pp = eu_curr - eu_prev
        eap_growth_pp = eap_curr - eap_prev
        
        covid_marker = " [COVID START]" if year == 2020 else ""
        print(f"{year:6d} | {eu_growth_pct:9.2f}% | {eap_growth_pct:9.2f}% | {eu_growth_pp:7.2f} | {eap_growth_pp:7.2f}{covid_marker}")

print("\n" + "="*80)
print("HOW TO CALCULATE REGIONAL MEANS")
print("="*80)

print("""
METHOD 1: Simple Mean (used above)
----------------------------------
df['region'] = df['country'].apply(lambda x: 'EaP' if x in eap_countries else 'EU')
regional_mean = df[df['region'] == 'EU']['internet_users_pct_i99H'].mean()

This treats each country-year observation equally.

METHOD 2: Country-level Mean (then average)
--------------------------------------------
country_means = df.groupby(['country', 'region'])['internet_users_pct_i99H'].mean()
regional_mean = country_means.groupby('region').mean()

This gives each country equal weight regardless of data availability.

METHOD 3: Weighted by Population
---------------------------------
# Weight by population to get true regional adoption
df['internet_users_count'] = df['internet_users_pct_i99H'] * df['population'] / 100
regional_total_users = df[df['region'] == 'EU']['internet_users_count'].sum()
regional_total_pop = df[df['region'] == 'EU']['population'].sum()
regional_mean = (regional_total_users / regional_total_pop) * 100

This accounts for country size - better for policy analysis.

RECOMMENDATION:
---------------
For your paper, use METHOD 1 (simple mean) for descriptive statistics
to show typical country experience. Use METHOD 3 (population-weighted)
for policy implications about regional populations.
""")

print("\n" + "="*80)
print("SUMMARY STATISTICS TABLE")
print("="*80)

summary_data = []
for region in ['EU', 'EaP']:
    df_region = df[df['region'] == region]
    
    row = {
        'Region': region,
        'Countries': df_region['country'].nunique(),
        'Observations': len(df_region),
        'Mean Internet Users (%)': df_region[main_metric].mean(),
        'SD': df_region[main_metric].std(),
        'Min': df_region[main_metric].min(),
        'Max': df_region[main_metric].max(),
        '2010 Mean': df_region[df_region['year'] == 2010][main_metric].mean(),
        '2024 Mean': df_region[df_region['year'] == 2024][main_metric].mean(),
        'Growth (pp)': df_region[df_region['year'] == 2024][main_metric].mean() - 
                       df_region[df_region['year'] == 2010][main_metric].mean()
    }
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save
output_path = BASE_DIR / 'manuscript2' / 'tables' / 'regional_adoption_summary.csv'
summary_df.to_csv(output_path, index=False)
print(f"\n[OK] Saved summary: {output_path}")

print("\n" + "="*80)
print("[OK] Analysis complete")
print("="*80)
