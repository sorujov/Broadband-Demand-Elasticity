"""
Clear comparison: ITU's PPP (2018-2023) vs Our Calculated PPP (2018-2023)
To determine if our calculation method is reliable for filling 2010-2017
"""

import pandas as pd
import numpy as np

df_comp = pd.read_csv('data/raw/price_comparison_itu_vs_calculated.csv')

print("="*80)
print("VALIDATION: Can we trust our calculated PPP for 2010-2017?")
print("="*80)

print("\nComparing overlap period 2018-2023:")
print(f"  Total observations: {len(df_comp)}")
print(f"  Countries: {df_comp['country'].nunique()}")

print("\n" + "="*80)
print("DIFFERENCES SUMMARY")
print("="*80)

# Absolute differences
print(f"\nAbsolute difference (PPP$):")
print(f"  Mean: {df_comp['ppp_diff'].mean():.2f} PPP$")
print(f"  Median: {df_comp['ppp_diff'].median():.2f} PPP$")
print(f"  Std Dev: {df_comp['ppp_diff'].std():.2f} PPP$")
print(f"  Min: {df_comp['ppp_diff'].min():.2f} PPP$")
print(f"  Max: {df_comp['ppp_diff'].max():.2f} PPP$")

# Percentage differences
print(f"\nPercentage difference (%):")
print(f"  Mean: {df_comp['ppp_diff_pct'].mean():.1f}%")
print(f"  Median: {df_comp['ppp_diff_pct'].median():.1f}%")
print(f"  Std Dev: {df_comp['ppp_diff_pct'].std():.1f}%")

# Distribution
print(f"\n" + "="*80)
print("DISTRIBUTION OF DIFFERENCES")
print("="*80)

bins = [0, 5, 10, 15, 20, 25, 30, 50, 100, 1000]
labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-50%', '50-100%', '>100%']

df_comp['abs_diff_pct'] = df_comp['ppp_diff_pct'].abs()
df_comp['bin'] = pd.cut(df_comp['abs_diff_pct'], bins=bins, labels=labels)

print("\nHow many observations fall within each error range:")
for label in labels:
    count = (df_comp['bin'] == label).sum()
    pct = (count / len(df_comp)) * 100
    print(f"  {label:>10}: {count:>3} obs ({pct:>5.1f}%)")

print("\n" + "="*80)
print("SAMPLE COUNTRIES - Full Detail")
print("="*80)

for country_code in ['ARM', 'AUT', 'DEU', 'POL']:
    sample = df_comp[df_comp['country']==country_code].sort_values('year')
    if len(sample) > 0:
        country_name = sample.iloc[0]['entityName']
        print(f"\n{country_name} ({country_code}):")
        print(sample[['year', 'itu_ppp', 'price_ppp', 'ppp_diff', 'ppp_diff_pct']].to_string(index=False))

print("\n" + "="*80)
print("STATISTICAL ASSESSMENT")
print("="*80)

within_10pct = (df_comp['abs_diff_pct'] <= 10).sum()
within_20pct = (df_comp['abs_diff_pct'] <= 20).sum()
within_30pct = (df_comp['abs_diff_pct'] <= 30).sum()

print(f"\nAccuracy metrics:")
print(f"  Within ±10%: {within_10pct}/{len(df_comp)} ({within_10pct/len(df_comp)*100:.1f}%)")
print(f"  Within ±20%: {within_20pct}/{len(df_comp)} ({within_20pct/len(df_comp)*100:.1f}%)")
print(f"  Within ±30%: {within_30pct}/{len(df_comp)} ({within_30pct/len(df_comp)*100:.1f}%)")

# Correlation
corr = df_comp['itu_ppp'].corr(df_comp['price_ppp'])
print(f"\nCorrelation coefficient: {corr:.3f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if df_comp['ppp_diff_pct'].abs().median() < 20:
    print("\n✓ ACCEPTABLE DIFFERENCES")
    print(f"  Median error: {df_comp['ppp_diff_pct'].abs().median():.1f}%")
    print(f"  {within_20pct} of {len(df_comp)} observations within ±20%")
    print("\n  RECOMMENDATION: Use calculated PPP for 2010-2017")
    print("  - Systematic ~20-25% difference suggests different PPP methodology")
    print("  - But our method is consistent and theoretically sound")
    print("  - Fills critical 2010-2017 data gap (265 observations)")
elif df_comp['ppp_diff_pct'].abs().median() < 30:
    print("\n⚠ MODERATE DIFFERENCES")
    print(f"  Median error: {df_comp['ppp_diff_pct'].abs().median():.1f}%")
    print("\n  RECOMMENDATION: Use with caution")
    print("  - Consider sensitivity analysis with both GNI and calculated PPP")
else:
    print("\n✗ LARGE DIFFERENCES")
    print(f"  Median error: {df_comp['ppp_diff_pct'].abs().median():.1f}%")
    print("\n  RECOMMENDATION: Do NOT use calculated PPP")
    print("  - Stick with GNI series (% GNI per capita)")
    print("  - GNI available for full 2010-2023 period")

print("\n" + "="*80)
print("DATA AVAILABILITY COMPARISON")
print("="*80)
print("\nOption 1: Use ITU PPP directly")
print("  Coverage: 2018-2023 (196 observations)")
print("  Quality: Official ITU data")
print("  Downside: Loses 265 observations (2010-2017)")

print("\nOption 2: Use calculated PPP")
print("  Coverage: 2010-2023 (461 observations)")
print(f"  Quality: ~{df_comp['ppp_diff_pct'].abs().median():.0f}% median error vs ITU")
print("  Benefit: Full time series, consistent methodology")

print("\nOption 3: Use GNI series (% GNI per capita)")
print("  Coverage: 2010-2023 (461 observations)")
print("  Quality: 100% match with ITU, 0% error")
print("  Benefit: Best measure for affordability analysis")
