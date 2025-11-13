"""
================================================================================
Data Merging Script
================================================================================
Purpose: Merge ITU and World Bank data into a single analysis dataset
Author: Samir Orujov
Date: November 13, 2025
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from code.utils.config
try:
    from code.utils.config import (
        DATA_RAW, DATA_INTERIM, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, COUNTRY_NAMES, ALL_COUNTRIES, get_region
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, DATA_INTERIM, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, COUNTRY_NAMES, ALL_COUNTRIES, get_region
    )

class DataMerger:
    """Merge ITU and World Bank datasets."""

    def __init__(self):
        # ITU data files
        self.itu_files = {
            'fixed_broad_price': DATA_RAW / 'itu_fixed_broad_price.csv',
            'mobile_broad_price': DATA_RAW / 'itu_mobile_broad_price.csv',
            'fixed_broadband_subs': DATA_RAW / 'itu_fixed_broadband_subs.csv',
            'internet_users_pct': DATA_RAW / 'itu_internet_users_pct.csv',
            'int_bandwidth': DATA_RAW / 'itu_int_bandwidth.csv',
            'mobile_subs': DATA_RAW / 'itu_mobile_subs.csv',
        }
        self.wb_file = DATA_RAW / 'worldbank_data.csv'

    def load_data(self):
        """Load ITU and World Bank data."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        # Load ITU data files
        print(f"\nLoading ITU data files...")
        itu_dfs = {}
        for name, filepath in self.itu_files.items():
            if filepath.exists():
                df = pd.read_csv(filepath)
                itu_dfs[name] = df
                print(f"  ✓ {name}: {df.shape[0]} rows × {df.shape[1]} columns")
            else:
                print(f"  ⚠ {name}: File not found - {filepath}")
        
        if len(itu_dfs) == 0:
            print(f"\n  ✗ No ITU data files found!")
            print(f"  Please run: python code\\data_collection\\01_download_itu_data.py")
            return None, None
        
        # Merge ITU dataframes
        print(f"\nMerging {len(itu_dfs)} ITU datasets...")
        df_itu = None
        for name, df in itu_dfs.items():
            # Standardize column names and select relevant columns
            df_clean = df[['country_iso3', 'dataYear', 'dataValue']].copy()
            df_clean.columns = ['country', 'year', name]
            
            if df_itu is None:
                df_itu = df_clean
            else:
                # Merge on country and year
                df_itu = df_itu.merge(df_clean, on=['country', 'year'], how='outer')
        
        print(f"  ✓ Merged ITU data: {df_itu.shape[0]} rows × {df_itu.shape[1]} columns")

        # Load World Bank data
        print(f"\nLoading World Bank data from: {self.wb_file}")
        if self.wb_file.exists():
            df_wb = pd.read_csv(self.wb_file)
            print(f"  ✓ Loaded: {df_wb.shape[0]} rows × {df_wb.shape[1]} columns")
            print(f"  Columns: {list(df_wb.columns[:10])}..." if len(df_wb.columns) > 10 else list(df_wb.columns))
        else:
            print(f"  ✗ File not found: {self.wb_file}")
            print(f"\n  Please run: python 02_download_worldbank_data.py")
            return None, None

        return df_itu, df_wb

    def standardize_columns(self, df_itu, df_wb):
        """Standardize country and year columns."""
        print("\n" + "="*80)
        print("STANDARDIZING COLUMNS")
        print("="*80)

        # Ensure both have 'country' and 'year' columns
        if 'country' not in df_itu.columns:
            print("  ✗ ITU data missing 'country' column")
            return None, None

        if 'country' not in df_wb.columns:
            print("  ✗ World Bank data missing 'country' column")
            return None, None

        # Ensure year columns are integer
        df_itu['year'] = pd.to_numeric(df_itu['year'], errors='coerce').astype('Int64')
        df_wb['year'] = pd.to_numeric(df_wb['year'], errors='coerce').astype('Int64')

        # Remove rows with missing years
        df_itu = df_itu.dropna(subset=['year'])
        df_wb = df_wb.dropna(subset=['year'])

        print(f"  ✓ ITU data: {len(df_itu)} rows")
        print(f"  ✓ World Bank data: {len(df_wb)} rows")

        return df_itu, df_wb

    def merge_datasets(self, df_itu, df_wb):
        """Merge ITU and World Bank datasets."""
        print("\n" + "="*80)
        print("MERGING DATASETS")
        print("="*80)

        # Check countries in each dataset
        itu_countries = set(df_itu['country'].unique())
        wb_countries = set(df_wb['country'].unique())

        print(f"\nCountries in ITU data: {len(itu_countries)}")
        print(f"Countries in World Bank data: {len(wb_countries)}")
        print(f"Common countries: {len(itu_countries & wb_countries)}")

        if len(itu_countries & wb_countries) == 0:
            print("\n⚠ WARNING: No common countries found!")
            print("  ITU countries:", sorted(list(itu_countries))[:5])
            print("  WB countries:", sorted(list(wb_countries))[:5])

        # Merge on country and year
        print("\nMerging on: country, year")
        df_merged = df_itu.merge(
            df_wb,
            on=['country', 'year'],
            how='outer',
            indicator=True
        )

        # Check merge results
        merge_status = df_merged['_merge'].value_counts()
        print("\nMerge results:")
        print(f"  Both datasets: {merge_status.get('both', 0)}")
        print(f"  Only ITU: {merge_status.get('left_only', 0)}")
        print(f"  Only World Bank: {merge_status.get('right_only', 0)}")

        # Drop merge indicator
        df_merged = df_merged.drop('_merge', axis=1)

        # Add region indicator
        df_merged['region'] = df_merged['country'].apply(get_region)

        # Filter to analysis period and countries
        df_merged = df_merged[
            (df_merged['year'] >= START_YEAR) & 
            (df_merged['year'] <= END_YEAR) &
            (df_merged['country'].isin(ALL_COUNTRIES))
        ]

        print(f"\n✓ Merged dataset: {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")
        print(f"  Countries: {df_merged['country'].nunique()}")
        print(f"  Years: {df_merged['year'].min()}-{df_merged['year'].max()}")
        print(f"  Regions: {df_merged['region'].value_counts().to_dict()}")

        return df_merged

    def add_date_column(self, df):
        """Add date column for easier handling."""
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
        return df

    def save_merged_data(self, df):
        """Save merged dataset."""
        output_file = DATA_INTERIM / 'data_merged.csv'
        df.to_csv(output_file, index=False)

        print("\n" + "="*80)
        print("SAVING MERGED DATA")
        print("="*80)
        print(f"\n✓ Saved to: {output_file}")
        print(f"  - {df.shape[0]} observations")
        print(f"  - {df.shape[1]} variables")
        print(f"  - {df['country'].nunique()} countries")
        print(f"  - {df['year'].nunique()} years ({df['year'].min()}-{df['year'].max()})")

        # Missing data summary
        print("\n" + "="*80)
        print("MISSING DATA SUMMARY")
        print("="*80)

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_df = pd.DataFrame({
            'Variable': missing.index,
            'Missing': missing.values,
            'Pct': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Pct', ascending=False)

        if len(missing_df) > 0:
            print(f"\nVariables with missing data (top 15):")
            print(missing_df.head(15).to_string(index=False))
        else:
            print("\n✓ No missing data!")

        # Show sample data
        print("\n" + "="*80)
        print("SAMPLE DATA (first 3 rows)")
        print("="*80)
        print(df[['country', 'year', 'region']].head(3).to_string())

        return output_file


def main():
    """Main execution function."""
    print("="*80)
    print("DATA MERGING SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize merger
        merger = DataMerger()

        # Load data
        df_itu, df_wb = merger.load_data()

        if df_itu is None or df_wb is None:
            print("\n✗ Cannot proceed without data files.")
            print("\nRequired files:")
            print(f"  1. {merger.itu_file}")
            print(f"  2. {merger.wb_file}")
            return

        # Standardize columns
        df_itu, df_wb = merger.standardize_columns(df_itu, df_wb)

        if df_itu is None or df_wb is None:
            print("\n✗ Column standardization failed.")
            return

        # Merge datasets
        df_merged = merger.merge_datasets(df_itu, df_wb)

        # Add date column
        df_merged = merger.add_date_column(df_merged)

        # Save merged data
        merger.save_merged_data(df_merged)

        print("\n" + "="*80)
        print("MERGE COMPLETE ✓")
        print("="*80)
        print("\nNext step: Run 04_prepare_data.py for data cleaning")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
