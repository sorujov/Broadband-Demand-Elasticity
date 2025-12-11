import pandas as pd

df = pd.read_excel('data/processed/data_merged_with_series.xlsx')

print("="*80)
print("DEMAND/SUBSCRIPTION COLUMNS")
print("="*80)
demand_cols = [c for c in df.columns if 'subs' in c.lower() or 'users' in c.lower()]
for col in demand_cols:
    missing = df[col].isnull().sum()
    pct = (missing / len(df) * 100)
    print(f"  - {col}")
    print(f"    Missing: {missing} ({pct:.1f}%)")

print("\n" + "="*80)
print("PRICE COLUMNS")
print("="*80)
price_cols = [c for c in df.columns if 'price' in c.lower()]
for col in price_cols:
    missing = df[col].isnull().sum()
    pct = (missing / len(df) * 100)
    print(f"  - {col}")
    print(f"    Missing: {missing} ({pct:.1f}%)")
