"""
Explain Azerbaijan's broadband price calculation
"""

import pandas as pd

# Load the comparison data
df = pd.read_excel('data/processed/price_comparison_full.xlsx', sheet_name='Full_Comparison')

# Filter Azerbaijan
aze = df[df['country'] == 'AZE'].sort_values('year')

print("="*80)
print("AZERBAIJAN FIXED BROADBAND PRICE - DETAILED EXPLANATION")
print("="*80)

print("\n1. WHAT IS 'PRICE AS % GNI PER CAPITA'?")
print("-" * 80)
print("This is the monthly fixed broadband price as a percentage of ANNUAL GNI per capita.")
print("GNI = Gross National Income (similar to GDP)")
print("It's in NOMINAL terms (current USD), not real/inflation-adjusted.")
print("\nExample: If GNI per capita = $4,000/year and price = 1%, then:")
print("  Monthly price = (1% × $4,000) / 12 months = $3.33/month")

print("\n\n2. FORMULA TO CALCULATE PPP PRICE:")
print("-" * 80)
print("Step 1: Convert % GNI to USD monthly price")
print("  Price_USD = (Price_%GNI × GNI_per_capita_USD) / (100 × 12)")
print("\nStep 2: Convert USD to PPP$")
print("  Price_PPP = (Price_USD × Exchange_Rate) / PPP_Conversion_Factor")
print("\nWhere:")
print("  - PPP conversion factor = Local Currency Units (LCU) per international $")
print("  - Exchange rate = LCU per USD")

print("\n\n3. AZERBAIJAN DATA (2018 EXAMPLE):")
print("-" * 80)

row = aze[aze['year'] == 2018].iloc[0]

print(f"\nInput data:")
print(f"  GNI per capita (current USD): ${row['gni_per_capita_current_usd']:,.2f}")
print(f"  Price as % GNI: {row['calculated_gni_pct']:.4f}%")
print(f"  PPP conversion factor: {row['ppp_conversion_factor']:.4f} AZN per international $")
print(f"  Official exchange rate: {row['official_exchange_rate']:.4f} AZN per USD")

print(f"\nStep-by-step calculation:")
print(f"  Step 1: Price in USD")
print(f"    = ({row['calculated_gni_pct']:.4f}% × ${row['gni_per_capita_current_usd']:,.2f}) / (100 × 12)")
print(f"    = ${row['calculated_gni_pct'] * row['gni_per_capita_current_usd'] / 100:.2f} / 12")
print(f"    = ${row['calculated_usd']:.2f} per month")

print(f"\n  Step 2: Price in PPP$")
print(f"    = (${row['calculated_usd']:.2f} × {row['official_exchange_rate']:.4f}) / {row['ppp_conversion_factor']:.4f}")
print(f"    = {row['calculated_usd'] * row['official_exchange_rate']:.2f} AZN / {row['ppp_conversion_factor']:.4f}")
print(f"    = ${row['calculated_ppp']:.2f} PPP$ per month")

print(f"\nComparison with ITU official:")
print(f"  Calculated PPP: ${row['calculated_ppp']:.2f}")
print(f"  ITU official PPP: ${row['itu_ppp']:.2f}")
print(f"  Difference: ${abs(row['calculated_ppp'] - row['itu_ppp']):.2f} ({row['ppp_diff_pct']:.1f}%)")

print("\n\n4. FULL TIME SERIES FOR AZERBAIJAN:")
print("-" * 80)
print(f"{'Year':<6} {'GNI/cap':<10} {'%GNI':<8} {'USD':<8} {'Calc_PPP':<10} {'ITU_PPP':<10} {'Diff%':<8}")
print("-" * 80)

for _, row in aze.iterrows():
    year = int(row['year'])
    gni = row['gni_per_capita_current_usd']
    pct_gni = row['calculated_gni_pct']
    usd = row['calculated_usd']
    calc_ppp = row['calculated_ppp']
    itu_ppp = row['itu_ppp']
    
    if pd.notna(itu_ppp):
        diff = (calc_ppp - itu_ppp) / itu_ppp * 100
        print(f"{year:<6} ${gni:>8,.0f} {pct_gni:>6.3f}% ${usd:>6.2f} ${calc_ppp:>8.2f} ${itu_ppp:>8.2f}  {diff:>6.1f}%")
    else:
        print(f"{year:<6} ${gni:>8,.0f} {pct_gni:>6.3f}% ${usd:>6.2f} ${calc_ppp:>8.2f} {'N/A':>10} {'N/A':>8}")

print("\n\n5. KEY INSIGHTS:")
print("-" * 80)
print(f"✓ Azerbaijan has {len(aze)} observations (2010-2023)")
print(f"✓ ITU official PPP only available for 2018-2023 ({aze['itu_ppp'].notna().sum()} obs)")
print(f"✓ Calculated PPP fills 2010-2017 gap ({aze['itu_ppp'].isna().sum()} obs)")
print(f"✓ For 2018-2023: Average difference is {aze[aze['itu_ppp'].notna()]['ppp_diff_pct'].mean():.1f}%")

print("\n\n6. WHAT SEEMS STRANGE?")
print("-" * 80)
print("The % GNI values (0.78-0.86%) seem reasonable:")
print("  - ITU/UN target: <2% for affordability")
print("  - Azerbaijan at ~0.8% is quite affordable")
print("  - Corresponds to $3-6 USD per month depending on year")
print("\nIf something looks wrong, please specify which values and why!")

print("\n" + "="*80)
