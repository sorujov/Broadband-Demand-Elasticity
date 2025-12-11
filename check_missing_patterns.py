import pandas as pd
import numpy as np

df = pd.read_excel('data/processed/data_merged_with_series.xlsx')

demand_vars = [
    'fixed_broadband_subs_i4213tfbb',
    'internet_users_pct_i99H',
    'mobile_subs_i271'
]

price_vars = [
    'fixed_broad_price_usd',
    'fixed_broad_price_gni_pct',
    'fixed_broad_price_ppp',
    'mobile_broad_price_usd',
    'mobile_broad_price_gni_pct',
    'mobile_broad_price_ppp'
]

key_vars = demand_vars + price_vars

print("="*80)
print("MISSING DATA CHECK FOR CO-MISSINGNESS HEATMAP")
print("="*80)

print("\nVariables with missing data:")
for i, var in enumerate(key_vars, 1):
    missing = df[var].isnull().sum()
    pct = missing / len(df) * 100
    status = "✓ Has missing" if missing > 0 else "✗ No missing (empty in heatmap)"
    print(f"{i}. {var}")
    print(f"   {status}: {missing} missing ({pct:.1f}%)")

# Create missing indicator matrix to see correlations
missing_indicators = df[key_vars].isnull().astype(int)
comissing_corr = missing_indicators.corr()

print("\n" + "="*80)
print("CO-MISSINGNESS CORRELATION MATRIX SHAPE")
print("="*80)
print(f"Expected: {len(key_vars)}x{len(key_vars)}")
print(f"Actual: {comissing_corr.shape}")

# Check for variables with no missing data (will show as NaN correlations)
print("\n" + "="*80)
print("VARIABLES WITH NO VARIANCE IN MISSING (All present or all missing)")
print("="*80)
for var in key_vars:
    missing_count = df[var].isnull().sum()
    if missing_count == 0 or missing_count == len(df):
        print(f"  - {var}: {missing_count}/{len(df)} missing (causes empty row/col)")
