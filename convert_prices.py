"""
Convert ITU broadband prices using World Bank conversion factors

Strategy:
1. We have ITU prices as % GNI per capita from old code (2010-2023)
2. We have GNI per capita in current USD from World Bank
3. We have PPP conversion factors from World Bank

Calculations:
- Price in USD = (Price as % GNI) × (GNI per capita in USD) / 100
- Price in PPP$ = (Price in USD) / (PPP conversion factor) × (Official exchange rate)

This fills the 2010-2017 gap in PPP price data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("ITU PRICE CONVERSION - FILLING HISTORICAL PPP DATA")
print("="*80)

# Load data
print("\nLoading data...")
df_old = pd.read_csv('data/raw/itu_fixed_broad_price.csv')
df_conv = pd.read_csv('data/raw/worldbank_conversion_factors.csv')

print(f"  ITU prices: {df_old.shape}")
print(f"  Conversion factors: {df_conv.shape}")

# Download old code data (2010-2023 with GNI series)
print("\nDownloading historical data from old code 34616...")
import requests
import zipfile
from io import BytesIO

response = requests.get('https://api.datahub.itu.int/v2/data/download/byid/34616/iscollection/false', timeout=30)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    with z.open(z.namelist()[0]) as f:
        df_old_full = pd.read_csv(f)

# Filter to our countries and years
import sys
sys.path.insert(0, 'code')
from utils.config import EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR, COUNTRY_NAMES

countries = EU_COUNTRIES + EAP_COUNTRIES
country_names_list = [COUNTRY_NAMES[code] for code in countries]

df_old_full = df_old_full[
    (df_old_full['entityName'].isin(country_names_list)) &
    (df_old_full['dataYear'] >= START_YEAR) &
    (df_old_full['dataYear'] <= END_YEAR)
].copy()

# Add ISO3 codes
name_to_iso3 = {v: k for k, v in COUNTRY_NAMES.items()}
df_old_full['country'] = df_old_full['entityName'].map(name_to_iso3)
df_old_full = df_old_full.rename(columns={'dataYear': 'year', 'dataValue': 'price_pct_gni'})

print(f"  Old code data: {df_old_full.shape} ({df_old_full['year'].min()}-{df_old_full['year'].max()})")

# Merge with conversion factors
print("\nMerging with World Bank conversion factors...")
df_merged = df_old_full.merge(df_conv, on=['country', 'year'], how='left')

print(f"  Merged data: {df_merged.shape}")

# Calculate prices in different units
print("\nCalculating converted prices...")

# 1. Price in current USD (monthly cost)
df_merged['price_usd'] = (df_merged['price_pct_gni'] * df_merged['gni_per_capita_current_usd']) / (100 * 12)

# 2. Price in PPP$ 
# PPP conversion factor is LCU per international $
# We want: Price in PPP$ = (Price in USD) × (Official exchange rate) / (PPP conversion factor)
df_merged['price_ppp'] = (df_merged['price_usd'] * df_merged['official_exchange_rate']) / df_merged['ppp_conversion_factor']

print(f"  ✓ Calculated price_usd (monthly cost in current USD)")
print(f"  ✓ Calculated price_ppp (monthly cost in PPP$)")

# Keep relevant columns
df_converted = df_merged[[
    'country', 'year', 'entityName',
    'price_pct_gni', 'price_usd', 'price_ppp',
    'gni_per_capita_current_usd', 'ppp_conversion_factor'
]].copy()

# Check coverage
print("\n" + "="*80)
print("COVERAGE CHECK")
print("="*80)

print(f"\nTotal observations: {len(df_converted)}")
print(f"Countries: {df_converted['country'].nunique()}")
print(f"Years: {df_converted['year'].min()}-{df_converted['year'].max()}")

print(f"\nMissing values:")
print(f"  price_pct_gni: {df_converted['price_pct_gni'].isna().sum()} ({df_converted['price_pct_gni'].isna().sum()/len(df_converted)*100:.1f}%)")
print(f"  price_usd: {df_converted['price_usd'].isna().sum()} ({df_converted['price_usd'].isna().sum()/len(df_converted)*100:.1f}%)")
print(f"  price_ppp: {df_converted['price_ppp'].isna().sum()} ({df_converted['price_ppp'].isna().sum()/len(df_converted)*100:.1f}%)")

# Sample data
print("\n" + "="*80)
print("SAMPLE DATA - Armenia 2010-2023")
print("="*80)
sample = df_converted[df_converted['country']=='ARM'][['year', 'price_pct_gni', 'price_usd', 'price_ppp']]
print(sample.to_string(index=False))

# Save
output_file = Path('data/raw/itu_prices_converted.csv')
df_converted.to_csv(output_file, index=False)

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)
print(f"✓ Saved: {output_file}")
print(f"  {len(df_converted)} observations")
print(f"  {df_converted['country'].nunique()} countries")
print(f"  {df_converted['year'].min()}-{df_converted['year'].max()}")

print("\nNow you have:")
print("  • price_pct_gni: Original % GNI (2010-2023)")
print("  • price_usd: Monthly cost in current USD (2010-2023)")  
print("  • price_ppp: Monthly cost in PPP$ (2010-2023)")
print("\nThis fills the 2010-2017 gap in PPP prices!")
