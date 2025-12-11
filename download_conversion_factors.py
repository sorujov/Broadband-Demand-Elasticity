"""
Download additional World Bank indicators for price conversion:
- GNI per capita (Atlas method - in current USD)
- PPP conversion factor (GDP to market exchange rate)
- Official exchange rate (LCU per USD)

This allows us to:
1. Convert ITU prices from % GNI to absolute USD
2. Convert USD to PPP-adjusted prices
3. Fill missing 2010-2017 PPP data
"""

import wbgapi as wb
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
code_path = project_root / 'code'
sys.path.insert(0, str(code_path))

from utils.config import EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR

# Countries
countries = EU_COUNTRIES + EAP_COUNTRIES

# Additional indicators for price conversion
conversion_indicators = {
    'NY.GNP.PCAP.CD': 'gni_per_capita_current_usd',  # GNI per capita, Atlas method (current US$)
    'PA.NUS.PPP': 'ppp_conversion_factor',            # PPP conversion factor, GDP (LCU per international $)
    'PA.NUS.FCRF': 'official_exchange_rate',          # Official exchange rate (LCU per US$, period average)
    'NY.GDP.DEFL.ZS': 'gdp_deflator',                 # GDP deflator (base year varies by country)
    'FP.CPI.TOTL.ZG': 'inflation_consumer_prices',    # Inflation, consumer prices (annual %)
}

print("="*80)
print("DOWNLOADING WORLD BANK CONVERSION FACTORS")
print("="*80)
print(f"Countries: {len(countries)}")
print(f"Years: {START_YEAR}-{END_YEAR}")
print(f"Indicators: {len(conversion_indicators)}")

all_data = []

for wb_code, var_name in conversion_indicators.items():
    print(f"\n[{var_name}]")
    print(f"  Code: {wb_code}")
    
    try:
        # Download data
        data = wb.data.DataFrame(
            wb_code,
            countries,
            time=range(START_YEAR, END_YEAR + 1),
            skipBlanks=True
        )
        
        if not data.empty:
            # Reshape from wide to long
            data = data.reset_index()
            
            # Get year columns (they should be like 'YR2010', 'YR2011', etc.)
            year_cols = [col for col in data.columns if col.startswith('YR')]
            
            data = data.melt(
                id_vars=['economy'],
                value_vars=year_cols,
                var_name='year',
                value_name=var_name
            )
            data['year'] = data['year'].str.replace('YR', '').astype(int)
            data = data.rename(columns={'economy': 'country'})
            
            all_data.append(data)
            print(f"  ✓ Downloaded: {len(data)} observations")
        else:
            print(f"  ✗ No data available")
            
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")

# Merge all indicators
if all_data:
    df_conversions = all_data[0]
    for df in all_data[1:]:
        df_conversions = df_conversions.merge(df, on=['country', 'year'], how='outer')
    
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    
    output_file = Path('data/raw/worldbank_conversion_factors.csv')
    df_conversions.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {df_conversions.shape}")
    print(f"  Countries: {df_conversions['country'].nunique()}")
    print(f"  Years: {df_conversions['year'].min()}-{df_conversions['year'].max()}")
    
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(df_conversions.head(10))
    
    print("\n" + "="*80)
    print("COVERAGE CHECK")
    print("="*80)
    print("Missing values per indicator:")
    for col in df_conversions.columns:
        if col not in ['country', 'year']:
            missing = df_conversions[col].isna().sum()
            pct = (missing / len(df_conversions)) * 100
            print(f"  {col}: {missing} ({pct:.1f}%)")

else:
    print("\n✗ No data downloaded")
    sys.exit(1)

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print("\nNext step: Create price conversion script to:")
print("1. Convert ITU % GNI prices to absolute USD")
print("2. Convert USD to PPP-adjusted prices")
print("3. Fill 2010-2017 gaps in PPP series")
