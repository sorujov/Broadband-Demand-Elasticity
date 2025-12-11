"""
Quick adapter to prepare data for 02_main_analysis.py
Creates the expected column names and transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('data/processed')
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')

print("Preparing data for main analysis...")
print(f"Loaded: {len(df)} rows")

# Use GNI% prices as the main price measure (most economically sensible)
df['fixed_broad_price'] = df['fixed_broad_price_gni_pct']
df['mobile_broad_price'] = df['mobile_broad_price_gni_pct']

# Create log transformations
df['log_internet_users_pct'] = np.log(df['internet_users_pct_i99H'] + 0.01)  # +0.01 to handle zeros
df['log_fixed_broad_price'] = np.log(df['fixed_broad_price'] + 0.01)
df['log_mobile_broad_price'] = np.log(df['mobile_broad_price'] + 0.01)
df['log_fixed_broadband_subs'] = np.log(df['fixed_broadband_subs_i4213tfbb'] + 0.01)
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])

# Rename columns to expected names
rename_map = {
    'research_development_expenditure': 'rd_expenditure',
    'regulatory_quality_estimate': 'regulatory_quality',
    'education_tertiary_pct': 'education_tertiary',
    'urban_population_pct': 'urban_population'
}

for old_name, new_name in rename_map.items():
    if old_name in df.columns:
        df[new_name] = df[old_name]

# Keep relevant columns
keep_cols = [
    'country', 'year',
    'log_internet_users_pct', 'log_fixed_broad_price', 'log_mobile_broad_price',
    'log_fixed_broadband_subs', 'log_gdp_per_capita',
    'gdp_per_capita', 'gdp_growth', 'population',
    'rd_expenditure', 'secure_internet_servers', 'regulatory_quality',
    'education_tertiary', 'urban_population',
    'ict_goods_exports', 'electricity_access_pct'
]

# Filter to columns that exist
keep_cols = [c for c in keep_cols if c in df.columns]
df_out = df[keep_cols].copy()

# Add secure_servers alias
if 'secure_internet_servers' in df_out.columns:
    df_out['secure_servers'] = df_out['secure_internet_servers']

# Save as CSV for the main analysis script
df_out.to_csv(DATA_DIR / 'analysis_ready_data.csv', index=False)

print(f"Saved: {len(df_out)} rows, {len(df_out.columns)} columns")
print("\nColumns included:")
for col in sorted(df_out.columns):
    print(f"  - {col}")
