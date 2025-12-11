import pandas as pd
import numpy as np

# Load the data
df = pd.read_excel('data/processed/data_merged_with_series.xlsx')

# Define key variables (from the script)
demand_vars = [
    'fixed_broadband_subs_i4213tfbb',
    'internet_users_pct_i99H',
    'mobile_subs_i271'
]

price_vars = [
    'fixed_broad_price_gni_pct',
    'mobile_broad_price_gni_pct'
]

key_vars = demand_vars + price_vars
available_key_vars = [v for v in key_vars if v in df.columns]

print("="*80)
print("CO-MISSINGNESS ANALYSIS CHECK")
print("="*80)

print(f"\nKey variables being analyzed: {len(available_key_vars)}")
for i, var in enumerate(available_key_vars, 1):
    missing = df[var].isnull().sum()
    pct = (missing / len(df) * 100)
    print(f"  {i}. {var}")
    print(f"     Missing: {missing} ({pct:.1f}%)")

# Create missing indicator matrix
missing_indicators = df[available_key_vars].isnull().astype(int)

# Compute correlation
comissing_corr = missing_indicators.corr()

print("\n" + "="*80)
print("CO-MISSINGNESS CORRELATION MATRIX")
print("="*80)
print("\nShape:", comissing_corr.shape)
print("\nFull matrix:")
print(comissing_corr.to_string())

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

# Check if we have the full 5x5 matrix
expected_size = len(available_key_vars)
actual_size = comissing_corr.shape[0]

if actual_size == expected_size:
    print(f"✓ Complete {expected_size}x{expected_size} matrix generated!")
else:
    print(f"✗ Incomplete matrix: Expected {expected_size}x{expected_size}, got {actual_size}x{actual_size}")

# Check which pairs have high co-missingness
print("\nHigh co-missingness pairs (correlation > 0.3):")
for i in range(len(comissing_corr)):
    for j in range(i+1, len(comissing_corr)):
        corr_val = comissing_corr.iloc[i, j]
        if abs(corr_val) > 0.3:
            var1 = available_key_vars[i]
            var2 = available_key_vars[j]
            print(f"  - {var1.split('_')[0]} <-> {var2.split('_')[0]}: {corr_val:.3f}")
