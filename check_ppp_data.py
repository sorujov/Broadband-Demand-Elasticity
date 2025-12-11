"""Check PPP data coverage in raw and processed files"""
import pandas as pd

print("="*80)
print("RAW DATA CHECK")
print("="*80)

df_raw = pd.read_csv('data/raw/itu_fixed_broad_price.csv')
print(f"Raw data shape: {df_raw.shape}")
print(f"\nSeries counts:")
print(df_raw['seriesCode'].value_counts())

print(f"\nYear range per series:")
for series in df_raw['seriesCode'].unique():
    df_series = df_raw[df_raw['seriesCode']==series]
    years = df_series['dataYear'].agg(['min','max','count'])
    countries = df_series['country_iso3'].nunique()
    print(f"{series}: {int(years['min'])}-{int(years['max'])} ({int(years['count'])} obs, {countries} countries)")

print(f"\nPPP series sample (first 10 rows):")
df_ppp = df_raw[df_raw['seriesCode']=='i154_FBB5_PPP']
print(df_ppp[['country_iso3', 'entityName', 'dataYear', 'dataValue']].head(10))

print("\n" + "="*80)
print("PROCESSED DATA CHECK")
print("="*80)

df_proc = pd.read_excel('data/interim/itu_fixed_broad_price_processed.xlsx')
print(f"Processed data shape: {df_proc.shape}")
print(f"\nSeries codes:")
print(df_proc['series_code'].value_counts())

print(f"\nYear range for PPP series:")
df_ppp_proc = df_proc[df_proc['series_code']=='i154_FBB5_PPP']
years_proc = df_ppp_proc['year'].agg(['min','max','count'])
print(f"PPP: {int(years_proc['min'])}-{int(years_proc['max'])} ({int(years_proc['count'])} obs)")

print("\n" + "="*80)
print("FINAL MERGED DATA CHECK")
print("="*80)

df_merged = pd.read_excel('data/processed/data_merged_with_series.xlsx')
print(f"Merged data shape: {df_merged.shape}")
print(f"\nPPP column stats:")
ppp_col = 'fixed_broad_price_i154_FBB5_PPP'
print(f"Non-null values: {df_merged[ppp_col].notna().sum()}")
print(f"Year range with PPP data:")
df_ppp_merged = df_merged[df_merged[ppp_col].notna()]
years_merged = df_ppp_merged['year'].agg(['min','max','count'])
print(f"{int(years_merged['min'])}-{int(years_merged['max'])} ({int(years_merged['count'])} obs)")
print(f"\nCountries with PPP data: {df_ppp_merged['country'].nunique()}")
