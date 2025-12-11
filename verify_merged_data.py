import pandas as pd

df = pd.read_excel('data/processed/data_merged_with_series.xlsx', engine='openpyxl')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns[:10])}...')
print(f'Countries: {df["country"].nunique()}')
print(f'Years: {df["year"].min()}-{df["year"].max()}')
print(f'\nFirst 3 rows:')
print(df.head(3))
