"""Compare old vs new ITU price codes"""
import requests
import pandas as pd
from io import BytesIO
import zipfile

print("="*80)
print("OLD CODE 34616 (Fixed-broadband Internet basket - Time Series)")
print("="*80)

response_old = requests.get('https://api.datahub.itu.int/v2/data/download/byid/34616/iscollection/false', timeout=30)
with zipfile.ZipFile(BytesIO(response_old.content)) as z:
    with z.open(z.namelist()[0]) as f:
        df_old = pd.read_csv(f)

print(f"Total rows: {len(df_old)}")
print(f"Series codes: {df_old['seriesCode'].unique()}")
print(f"Year range: {df_old['dataYear'].min()} - {df_old['dataYear'].max()}")
print(f"\nSeries breakdown:")
print(df_old['seriesCode'].value_counts())

print("\n" + "="*80)
print("NEW CODE 33331 (Fixed-broadband Internet 5GB)")
print("="*80)

response_new = requests.get('https://api.datahub.itu.int/v2/data/download/byid/33331/iscollection/false', timeout=30)
with zipfile.ZipFile(BytesIO(response_new.content)) as z:
    with z.open(z.namelist()[0]) as f:
        df_new = pd.read_csv(f)

print(f"Total rows: {len(df_new)}")
print(f"Series codes: {df_new['seriesCode'].unique()}")
print(f"Year range: {df_new['dataYear'].min()} - {df_new['dataYear'].max()}")
print(f"\nSeries breakdown:")
for series in df_new['seriesCode'].unique():
    df_s = df_new[df_new['seriesCode']==series]
    years = f"{df_s['dataYear'].min()}-{df_s['dataYear'].max()}"
    print(f"  {series}: {len(df_s)} rows ({years})")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nOLD CODE (34616):")
print(f"  - Only GNI series (% GNI per capita)")
print(f"  - Years: 2008-2023 (longer history)")
print(f"  - Mixed basket: 1GB (2008-2017) + 5GB (2018-2023)")
print(f"  - NO PPP or USD series")

print(f"\nNEW CODE (33331):")
print(f"  - THREE series: USD, GNI, PPP")
print(f"  - Years: 2018-2023 only (shorter history)")
print(f"  - Consistent 5GB basket")
print(f"  - HAS PPP series!")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("ITU changed their methodology in 2018:")
print("  - Standardized on 5GB plans")
print("  - Added PPP and USD series")
print("  - PPP data ONLY exists from 2018 onwards (not historical)")
print("\nThis is ITU's data limitation, not a code error.")
print("For full 2010-2023 coverage, use GNI series (available in both codes).")
