"""
Compare calculated PPP prices with ITU's PPP series for 2018-2023
to validate our conversion methodology
"""

import pandas as pd
import numpy as np

print("="*80)
print("COMPARING CALCULATED PPP vs ITU PPP PRICES (2018-2023)")
print("="*80)

# Load our calculated prices
df_calc = pd.read_csv('data/raw/itu_prices_converted.csv')

# Load ITU's direct PPP prices (from new code)
df_itu = pd.read_csv('data/raw/itu_fixed_broad_price.csv')
df_itu_ppp = df_itu[df_itu['seriesCode']=='i154_FBB5_PPP'].copy()
df_itu_ppp = df_itu_ppp.rename(columns={
    'country_iso3': 'country',
    'dataYear': 'year',
    'dataValue': 'itu_ppp'
})

# Also get ITU's GNI and USD series for comparison
df_itu_gni = df_itu[df_itu['seriesCode']=='i154_FBB5_GNI'].copy()
df_itu_gni = df_itu_gni.rename(columns={'dataValue': 'itu_gni'})

df_itu_usd = df_itu[df_itu['seriesCode']=='i154_FBB5$'].copy()
df_itu_usd = df_itu_usd.rename(columns={'dataValue': 'itu_usd'})

# Merge all data
df_comp = df_calc[['country', 'year', 'entityName', 'price_pct_gni', 'price_usd', 'price_ppp']].copy()
df_comp = df_comp.merge(
    df_itu_ppp[['country', 'year', 'itu_ppp']], 
    on=['country', 'year'], 
    how='inner'
)
df_comp = df_comp.merge(
    df_itu_gni[['country_iso3', 'dataYear', 'itu_gni']], 
    left_on=['country', 'year'],
    right_on=['country_iso3', 'dataYear'],
    how='left'
).drop(columns=['country_iso3', 'dataYear'])

df_comp = df_comp.merge(
    df_itu_usd[['country_iso3', 'dataYear', 'itu_usd']], 
    left_on=['country', 'year'],
    right_on=['country_iso3', 'dataYear'],
    how='left'
).drop(columns=['country_iso3', 'dataYear'])

# Calculate differences
df_comp['ppp_diff'] = df_comp['price_ppp'] - df_comp['itu_ppp']
df_comp['ppp_diff_pct'] = (df_comp['ppp_diff'] / df_comp['itu_ppp']) * 100

df_comp['gni_diff'] = df_comp['price_pct_gni'] - df_comp['itu_gni']
df_comp['gni_diff_pct'] = (df_comp['gni_diff'] / df_comp['itu_gni']) * 100

print(f"\nTotal comparisons: {len(df_comp)}")
print(f"Countries: {df_comp['country'].nunique()}")
print(f"Years: {df_comp['year'].min()}-{df_comp['year'].max()}")

print("\n" + "="*80)
print("GNI SERIES COMPARISON (% GNI per capita)")
print("="*80)
print(f"\nOur calculated (from old code) vs ITU's new code:")
print(f"  Mean difference: {df_comp['gni_diff'].mean():.2f} percentage points")
print(f"  Mean % difference: {df_comp['gni_diff_pct'].mean():.2f}%")
print(f"  Median % difference: {df_comp['gni_diff_pct'].median():.2f}%")
print(f"  Std dev % difference: {df_comp['gni_diff_pct'].std():.2f}%")
print(f"  Max absolute % difference: {df_comp['gni_diff_pct'].abs().max():.2f}%")

print("\n" + "="*80)
print("PPP SERIES COMPARISON")
print("="*80)
print(f"\nOur calculated vs ITU's PPP series:")
print(f"  Mean difference: {df_comp['ppp_diff'].mean():.2f} PPP$")
print(f"  Mean % difference: {df_comp['ppp_diff_pct'].mean():.2f}%")
print(f"  Median % difference: {df_comp['ppp_diff_pct'].median():.2f}%")
print(f"  Std dev % difference: {df_comp['ppp_diff_pct'].std():.2f}%")
print(f"  Max absolute % difference: {df_comp['ppp_diff_pct'].abs().max():.2f}%")

# Distribution of differences
print("\n" + "="*80)
print("DISTRIBUTION OF % DIFFERENCES")
print("="*80)

print("\nPPP differences by range:")
ranges = [
    (0, 5, "Very close (0-5%)"),
    (5, 10, "Close (5-10%)"),
    (10, 20, "Moderate (10-20%)"),
    (20, 50, "Large (20-50%)"),
    (50, 100, "Very large (50-100%)"),
]

for low, high, label in ranges:
    count = ((df_comp['ppp_diff_pct'].abs() >= low) & (df_comp['ppp_diff_pct'].abs() < high)).sum()
    pct = (count / len(df_comp)) * 100
    print(f"  {label}: {count} obs ({pct:.1f}%)")

count_100plus = (df_comp['ppp_diff_pct'].abs() >= 100).sum()
pct_100plus = (count_100plus / len(df_comp)) * 100
print(f"  Extreme (100%+): {count_100plus} obs ({pct_100plus:.1f}%)")

# Sample countries comparison
print("\n" + "="*80)
print("SAMPLE: ARMENIA (ARM) - Full Time Series")
print("="*80)

sample = df_comp[df_comp['country']=='ARM'].sort_values('year')
print("\nGNI Series (% GNI per capita):")
print(sample[['year', 'price_pct_gni', 'itu_gni', 'gni_diff_pct']].to_string(index=False))

print("\nPPP Series:")
print(sample[['year', 'price_ppp', 'itu_ppp', 'ppp_diff_pct']].to_string(index=False))

# Another sample
print("\n" + "="*80)
print("SAMPLE: AUSTRIA (AUT) - Full Time Series")
print("="*80)

sample2 = df_comp[df_comp['country']=='AUT'].sort_values('year')
print("\nGNI Series (% GNI per capita):")
print(sample2[['year', 'price_pct_gni', 'itu_gni', 'gni_diff_pct']].to_string(index=False))

print("\nPPP Series:")
print(sample2[['year', 'price_ppp', 'itu_ppp', 'ppp_diff_pct']].to_string(index=False))

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_ppp = df_comp['price_ppp'].corr(df_comp['itu_ppp'])
corr_gni = df_comp['price_pct_gni'].corr(df_comp['itu_gni'])

print(f"\nCorrelation between our values and ITU's:")
print(f"  GNI series: {corr_gni:.4f}")
print(f"  PPP series: {corr_ppp:.4f}")

# Scatter plot data for visualization
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

gni_close = (df_comp['gni_diff_pct'].abs() < 5).sum()
ppp_close = (df_comp['ppp_diff_pct'].abs() < 20).sum()

print(f"\n1. GNI Series Validation:")
print(f"   - {gni_close}/{len(df_comp)} observations ({gni_close/len(df_comp)*100:.1f}%) differ by <5%")
print(f"   - Very high agreement suggests our data source is compatible")

print(f"\n2. PPP Series Validation:")
print(f"   - {ppp_close}/{len(df_comp)} observations ({ppp_close/len(df_comp)*100:.1f}%) differ by <20%")
print(f"   - Moderate agreement suggests different PPP conversion methodology")

print(f"\n3. Recommendation:")
if df_comp['gni_diff_pct'].abs().mean() < 5 and df_comp['ppp_diff_pct'].abs().mean() < 30:
    print("   ✓ Differences are reasonable - our conversion is valid")
    print("   ✓ Can use calculated PPP for 2010-2017 to fill gaps")
    print("   ✓ For 2018-2023, consider using ITU's direct PPP values")
else:
    print("   ⚠ Large systematic differences detected")
    print("   → Recommend using GNI series (% GNI per capita) for full period")
    print("   → GNI already accounts for affordability without PPP adjustment")

# Save comparison
df_comp.to_csv('data/raw/price_comparison_itu_vs_calculated.csv', index=False)
print(f"\n✓ Saved detailed comparison to: data/raw/price_comparison_itu_vs_calculated.csv")
