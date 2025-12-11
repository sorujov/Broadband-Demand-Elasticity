"""Test new ITU API codes for price data"""
import requests
import pandas as pd
import zipfile
from io import BytesIO

# Test new fixed broadband code 33331
print("Testing code 33331 (Fixed-broadband Internet 5GB)...")
response = requests.get('https://api.datahub.itu.int/v2/data/download/byid/33331/iscollection/false', timeout=30)
print(f"Status: {response.status_code}")

with zipfile.ZipFile(BytesIO(response.content)) as z:
    csv_files = z.namelist()
    print(f"Files in ZIP: {csv_files}")
    
    with z.open(csv_files[0]) as f:
        df = pd.read_csv(f)
        
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()[:15]}")
print(f"\nUnique series codes:")
print(df['seriesCode'].value_counts())
print(f"\nSample data (first 3 rows):")
print(df[['entityName', 'dataYear', 'seriesCode', 'seriesName', 'dataValue']].head(3))

# Compare with old code 34616
print("\n" + "="*80)
print("Testing OLD code 34616 (Fixed-broadband Internet basket)...")
response2 = requests.get('https://api.datahub.itu.int/v2/data/download/byid/34616/iscollection/false', timeout=30)

with zipfile.ZipFile(BytesIO(response2.content)) as z:
    csv_files2 = z.namelist()
    with z.open(csv_files2[0]) as f:
        df2 = pd.read_csv(f)

print(f"\nShape: {df2.shape}")
print(f"\nUnique series codes:")
print(df2['seriesCode'].value_counts())
print(f"\nSample data (first 3 rows):")
print(df2[['entityName', 'dataYear', 'seriesCode', 'seriesName', 'dataValue']].head(3))

# Test mobile broadband code 36056
print("\n" + "="*80)
print("Testing code 36056 (Data-only mobile broadband basket 5 GB)...")
response3 = requests.get('https://api.datahub.itu.int/v2/data/download/byid/36056/iscollection/false', timeout=30)

with zipfile.ZipFile(BytesIO(response3.content)) as z:
    csv_files3 = z.namelist()
    with z.open(csv_files3[0]) as f:
        df3 = pd.read_csv(f)

print(f"\nShape: {df3.shape}")
print(f"\nUnique series codes:")
print(df3['seriesCode'].value_counts())
