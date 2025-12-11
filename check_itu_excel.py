"""
Check ITU Excel file for our countries and compare with API data
"""

import pandas as pd
import requests
from io import BytesIO
import sys
sys.path.insert(0, 'code')
from utils.config import EU_COUNTRIES, EAP_COUNTRIES, COUNTRY_NAMES

# Download Excel file
url = 'https://www.itu.int/en/ITU-D/Statistics/Documents/publications/prices2024/ITU_ICTPriceBaskets_2008-2024.xlsx'
print("Downloading ITU Excel file...")
response = requests.get(url, timeout=30)
excel_file = BytesIO(response.content)

# Read economies sheet
df = pd.read_excel(excel_file, sheet_name='economies_2008-2024')

print(f"\nTotal data shape: {df.shape}")
print(f"\nAvailable baskets:")
print(df['basket_combined_simplified'].unique())
print(f"\nAvailable units:")
print(df['Unit'].unique())

# Filter to fixed broadband
df_fbb = df[df['basket_combined_simplified'] == 'Fixed-broadband basket'].copy()

print(f"\n\nFIXED BROADBAND DATA:")
print(f"Total observations: {len(df_fbb)}")
print(f"Countries: {df_fbb['IsoCode'].nunique()}")
print(f"Units available: {df_fbb['Unit'].unique()}")

# Check our countries
countries = EU_COUNTRIES + EAP_COUNTRIES
df_our = df_fbb[df_fbb['IsoCode'].isin(countries)].copy()

print(f"\n\nOUR COUNTRIES ({len(countries)} countries):")
print(f"Total rows: {len(df_our)}")
print(f"Countries found: {df_our['IsoCode'].nunique()}")

# Reshape to long format
year_cols = [col for col in df.columns if isinstance(col, int)]
df_long = df_our.melt(
    id_vars=['IsoCode', 'Economy', 'basket_combined_simplified', 'Unit', 'Code'],
    value_vars=year_cols,
    var_name='year',
    value_name='value'
)

# Pivot to have one row per country-year
df_wide = df_long.pivot_table(
    index=['IsoCode', 'Economy', 'year'],
    columns='Unit',
    values='value',
    aggfunc='first'
).reset_index()

# Filter to 2010-2023
df_wide = df_wide[(df_wide['year'] >= 2010) & (df_wide['year'] <= 2023)]

print(f"\n\nRESTRUCTURED DATA (2010-2023):")
print(f"Shape: {df_wide.shape}")
print(f"Columns: {list(df_wide.columns)}")

print(f"\n\nDATA COVERAGE BY UNIT:")
for unit in ['GNIpc', 'PPP', 'USD']:
    if unit in df_wide.columns:
        non_null = df_wide[unit].notna().sum()
        pct = non_null / len(df_wide) * 100
        print(f"  {unit}: {non_null}/{len(df_wide)} ({pct:.1f}%)")

print(f"\n\nCOVERAGE BY YEAR:")
coverage = df_wide.groupby('year').agg({
    'GNIpc': lambda x: x.notna().sum(),
    'PPP': lambda x: x.notna().sum(),
    'USD': lambda x: x.notna().sum()
})
print(coverage.to_string())

# Show sample for Azerbaijan
print(f"\n\nAZERBAIJAN SAMPLE:")
aze = df_wide[df_wide['IsoCode'] == 'AZE'].sort_values('year')
print(aze[['year', 'GNIpc', 'PPP', 'USD']].to_string())

# Compare with API data
print(f"\n\nCOMPARISON WITH API DATA:")
df_api = pd.read_csv('data/raw/itu_fixed_broad_price.csv')
df_api = df_api[df_api['seriesCode'] == 'i154_FBB5_PPP']

print(f"API data (code 33331, 2018-2023): {len(df_api)} observations")
print(f"Excel data (2010-2023, PPP): {df_wide['PPP'].notna().sum()} observations")

# Check if values match for 2018-2023
print(f"\nAPI columns: {df_api.columns.tolist()}")
if 'entityCode' in df_api.columns:
    df_api_merge = df_api[['entityCode', 'dataYear', 'dataValue']].copy()
    df_api_merge.columns = ['IsoCode', 'year', 'api_ppp']
    df_compare = df_wide.merge(df_api_merge, on=['IsoCode', 'year'], how='inner')
else:
    df_compare = pd.DataFrame()  # Empty if no match

if len(df_compare) > 0:
    df_compare['diff'] = df_compare['PPP'] - df_compare['api_ppp']
    df_compare['diff_pct'] = (df_compare['diff'] / df_compare['api_ppp'] * 100).abs()
    
    print(f"\nOverlap comparison ({len(df_compare)} observations):")
    print(f"  Mean absolute difference: {df_compare['diff'].abs().mean():.4f}")
    print(f"  Median absolute difference: {df_compare['diff'].abs().median():.4f}")
    print(f"  Mean % difference: {df_compare['diff_pct'].mean():.2f}%")
    print(f"  Max % difference: {df_compare['diff_pct'].max():.2f}%")
    
    print(f"\n  Sample (Azerbaijan 2018-2020):")
    aze_comp = df_compare[(df_compare['IsoCode'] == 'AZE') & (df_compare['year'] >= 2018) & (df_compare['year'] <= 2020)]
    print(aze_comp[['year', 'PPP', 'api_ppp', 'diff', 'diff_pct']].to_string())

print(f"\n\nRECOMMENDATION:")
print("="*80)
if df_wide['PPP'].notna().sum() > df_api.shape[0]:
    print("✓ Excel file has MORE PPP data than API!")
    print(f"  Excel: {df_wide['PPP'].notna().sum()} observations (2010-2023)")
    print(f"  API: {len(df_api)} observations (2018-2023)")
    print("\n  ACTION: Use Excel file as primary data source")
    print("  BENEFIT: Get full 2010-2023 coverage without needing to calculate PPP")
else:
    print("  Excel and API have similar coverage")
