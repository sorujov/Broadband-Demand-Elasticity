"""
Create comprehensive comparison Excel file with all price data and conversion factors
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("CREATING COMPREHENSIVE PRICE COMPARISON DATASET")
print("="*80)

# Load all data
print("\nLoading data...")
df_calc = pd.read_csv('data/raw/itu_prices_converted.csv')
df_itu = pd.read_csv('data/raw/itu_fixed_broad_price.csv')
df_conv = pd.read_csv('data/raw/worldbank_conversion_factors.csv')

# Pivot ITU data to wide format (one row per country-year)
print("Reshaping ITU data...")
itu_wide = df_itu.pivot_table(
    index=['country_iso3', 'dataYear', 'entityName'],
    columns='seriesCode',
    values='dataValue',
    aggfunc='first'
).reset_index()

itu_wide.columns.name = None
itu_wide = itu_wide.rename(columns={
    'country_iso3': 'country',
    'dataYear': 'year',
    'entityName': 'country_name',
    'i154_FBB5$': 'itu_usd',
    'i154_FBB5_GNI': 'itu_gni_pct',
    'i154_FBB5_PPP': 'itu_ppp'
})

# Merge calculated prices with ITU data
print("Merging datasets...")
df_combined = df_calc[['country', 'year', 'entityName', 'price_pct_gni', 'price_usd', 'price_ppp']].copy()

df_combined = df_combined.merge(
    itu_wide[['country', 'year', 'itu_usd', 'itu_gni_pct', 'itu_ppp']],
    on=['country', 'year'],
    how='outer'
)

df_combined = df_combined.merge(
    df_conv,
    on=['country', 'year'],
    how='left'
)

# Calculate differences
df_combined['ppp_difference'] = df_combined['price_ppp'] - df_combined['itu_ppp']
df_combined['ppp_diff_pct'] = (df_combined['ppp_difference'] / df_combined['itu_ppp']) * 100

df_combined['gni_difference'] = df_combined['price_pct_gni'] - df_combined['itu_gni_pct']
df_combined['gni_diff_pct'] = (df_combined['gni_difference'] / df_combined['itu_gni_pct']) * 100

df_combined['usd_difference'] = df_combined['price_usd'] - df_combined['itu_usd']
df_combined['usd_diff_pct'] = (df_combined['usd_difference'] / df_combined['itu_usd']) * 100

# Sort by country and year
df_combined = df_combined.sort_values(['country', 'year'])

# Rename columns for clarity
df_combined = df_combined.rename(columns={
    'entityName': 'country_name',
    'price_pct_gni': 'calculated_gni_pct',
    'price_usd': 'calculated_usd',
    'price_ppp': 'calculated_ppp'
})

# Reorder columns logically
columns_order = [
    'country', 'country_name', 'year',
    # Conversion factors
    'gni_per_capita_current_usd',
    'ppp_conversion_factor',
    'official_exchange_rate',
    'gdp_deflator',
    'inflation_consumer_prices',
    # ITU official values
    'itu_gni_pct',
    'itu_usd',
    'itu_ppp',
    # Our calculated values
    'calculated_gni_pct',
    'calculated_usd',
    'calculated_ppp',
    # Differences
    'gni_difference',
    'gni_diff_pct',
    'usd_difference',
    'usd_diff_pct',
    'ppp_difference',
    'ppp_diff_pct'
]

df_final = df_combined[columns_order]

# Save to Excel with formatting
output_file = Path('data/processed/price_comparison_full.xlsx')

print(f"\nSaving to Excel: {output_file}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Main data sheet
    df_final.to_excel(writer, sheet_name='Full_Comparison', index=False)
    
    # Summary statistics sheet
    summary_data = {
        'Metric': [
            'Total Observations',
            'Countries',
            'Years',
            'Year Range',
            '',
            'PPP Comparison (2018-2023 overlap)',
            'Observations with both values',
            'Mean PPP difference (PPP$)',
            'Median PPP difference (%)',
            'Observations within ±10%',
            'Observations within ±20%',
            'Observations within ±30%',
            '',
            'GNI Comparison (2018-2023 overlap)',
            'Mean GNI difference (% points)',
            'Median GNI difference (%)',
            '',
            'Data Coverage',
            'ITU PPP available (2018-2023)',
            'Calculated PPP available (2010-2023)',
            'ITU GNI available (2018-2023)',
            'Calculated GNI available (2010-2023)'
        ],
        'Value': [
            len(df_final),
            df_final['country'].nunique(),
            df_final['year'].nunique(),
            f"{df_final['year'].min()}-{df_final['year'].max()}",
            '',
            '',
            df_final['itu_ppp'].notna().sum(),
            f"{df_final['ppp_difference'].mean():.2f}",
            f"{df_final['ppp_diff_pct'].median():.2f}",
            f"{(df_final['ppp_diff_pct'].abs() <= 10).sum()}",
            f"{(df_final['ppp_diff_pct'].abs() <= 20).sum()}",
            f"{(df_final['ppp_diff_pct'].abs() <= 30).sum()}",
            '',
            '',
            f"{df_final['gni_difference'].mean():.4f}",
            f"{df_final['gni_diff_pct'].median():.2f}",
            '',
            '',
            f"{df_final['itu_ppp'].notna().sum()}",
            f"{df_final['calculated_ppp'].notna().sum()}",
            f"{df_final['itu_gni_pct'].notna().sum()}",
            f"{df_final['calculated_gni_pct'].notna().sum()}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # By country summary
    country_summary = df_final.groupby('country').agg({
        'year': ['min', 'max', 'count'],
        'itu_ppp': 'count',
        'calculated_ppp': 'count',
        'ppp_diff_pct': ['mean', 'median'],
        'gni_diff_pct': ['mean', 'median']
    }).reset_index()
    
    country_summary.columns = [
        'country', 'year_min', 'year_max', 'total_obs',
        'itu_ppp_count', 'calc_ppp_count',
        'ppp_diff_mean_pct', 'ppp_diff_median_pct',
        'gni_diff_mean_pct', 'gni_diff_median_pct'
    ]
    
    country_summary.to_excel(writer, sheet_name='By_Country', index=False)

print(f"✓ Saved: {output_file}")
print(f"\nFile contains {len(df_final)} rows × {len(df_final.columns)} columns")
print(f"\nSheets:")
print(f"  1. Full_Comparison - All data side by side")
print(f"  2. Summary - Key statistics")
print(f"  3. By_Country - Country-level summaries")

print("\n" + "="*80)
print("COLUMN GUIDE")
print("="*80)
print("""
CONVERSION FACTORS (from World Bank):
  - gni_per_capita_current_usd: GNI per capita in current USD
  - ppp_conversion_factor: PPP conversion factor (LCU per international $)
  - official_exchange_rate: Exchange rate (LCU per USD)
  - gdp_deflator: GDP deflator index
  - inflation_consumer_prices: Annual inflation (%)

ITU OFFICIAL VALUES (from code 33331, 2018-2023 only):
  - itu_gni_pct: Price as % GNI per capita
  - itu_usd: Monthly price in current USD
  - itu_ppp: Monthly price in PPP$ (ITU's calculation)

OUR CALCULATED VALUES (from code 34616 + WB conversions, 2010-2023):
  - calculated_gni_pct: Price as % GNI per capita (from old code)
  - calculated_usd: Monthly price in USD (our calculation)
  - calculated_ppp: Monthly price in PPP$ (our calculation)

DIFFERENCES:
  - *_difference: Absolute difference (calculated - ITU)
  - *_diff_pct: Percentage difference relative to ITU value
""")

print("\n" + "="*80)
print("KEY FINDINGS IN THE DATA")
print("="*80)

overlap = df_final[(df_final['itu_ppp'].notna()) & (df_final['calculated_ppp'].notna())]
print(f"\nOverlap period (2018-2023):")
print(f"  Observations: {len(overlap)}")
print(f"  Median PPP difference: {overlap['ppp_diff_pct'].median():.1f}%")
print(f"  Within ±20%: {(overlap['ppp_diff_pct'].abs() <= 20).sum()} ({(overlap['ppp_diff_pct'].abs() <= 20).sum()/len(overlap)*100:.1f}%)")

historical = df_final[(df_final['year'] < 2018) & (df_final['calculated_ppp'].notna())]
print(f"\nHistorical period (2010-2017) - calculated PPP only:")
print(f"  Observations: {len(historical)}")
print(f"  Countries: {historical['country'].nunique()}")
print(f"  This fills the gap where ITU has no PPP data")

print("\n✓ Open the Excel file to explore the data in detail!")
