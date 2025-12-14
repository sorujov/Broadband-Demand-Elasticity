# code/data_collection/03_merge_data.py (UPDATED)
"""
================================================================================
Data Merging Script (UPDATED)
================================================================================
Purpose: Merge processed ITU and World Bank data with series metadata preserved
Author: Samir Orujov  
Date: November 20, 2025 (Updated)

IMPORTANT: This script now works with PROCESSED data from 01.5_process_data.py
           Series metadata (codes, units) are preserved for analysis
           NA values are PRESERVED (not filled) - missing data remains missing
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

try:
    from code.utils.config import (
        DATA_RAW, DATA_INTERIM, DATA_PROCESSED,
        EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, COUNTRY_NAMES, ALL_COUNTRIES, get_region
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, DATA_INTERIM, DATA_PROCESSED,
        EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, COUNTRY_NAMES, ALL_COUNTRIES, get_region
    )


class DataMerger:
    """Merge processed ITU and World Bank datasets with metadata."""

    def __init__(self):
        # Processed ITU files (from 01.5_process_data.py)
        self.itu_processed_files = list(DATA_INTERIM.glob('itu_*_processed.xlsx'))
        self.wb_processed_file = DATA_INTERIM / 'worldbank_processed.xlsx'
        self.catalog_file = DATA_INTERIM / 'data_catalog.xlsx'

        # Ensure output directory exists
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    def load_processed_itu_data(self):
        """Load all processed ITU data files."""
        print("="*80)
        print("LOADING PROCESSED ITU DATA")
        print("="*80)

        if len(self.itu_processed_files) == 0:
            print("\n[ERROR] No processed ITU files found!")
            print("   Please run: python code/data_collection/01.5_process_data.py")
            return None

        print(f"\nFound {len(self.itu_processed_files)} processed ITU files")

        # Separate price and non-price data
        price_dfs = []
        non_price_dfs = []

        for filepath in self.itu_processed_files:
            df = pd.read_excel(filepath, engine='openpyxl')
            var_name = filepath.stem.replace('itu_', '').replace('_processed', '')
            
            # Check if this is price data (has USD/GNI%/PPP columns)
            price_cols = [c for c in df.columns if c.endswith('_usd') or c.endswith('_gni_pct') or c.endswith('_ppp')]
            
            if price_cols:
                # Price data - already in wide format
                price_dfs.append(df)
                print(f"  [OK] {var_name}: {len(df)} obs (PRICE DATA - wide format)")
            else:
                # Non-price data - in long format
                non_price_dfs.append(df)
                n_series = df['series_code'].nunique()
                print(f"  [OK] {var_name}: {len(df)} obs, {n_series} series")

        return price_dfs, non_price_dfs

    def load_processed_wb_data(self):
        """Load processed World Bank data."""
        print("\n" + "="*80)
        print("LOADING PROCESSED WORLD BANK DATA")
        print("="*80)

        if not self.wb_processed_file.exists():
            print(f"\n[ERROR] File not found: {self.wb_processed_file}")
            print("   Please run: python code/data_collection/01.5_process_data.py")
            return None

        df_wb = pd.read_excel(self.wb_processed_file, engine='openpyxl')

        print(f"\nLoaded: {df_wb.shape[0]} rows × {df_wb.shape[1]} columns")
        print(f"  Variables: {df_wb['variable'].nunique()}")
        print(f"  Countries: {df_wb['country'].nunique()}")
        print(f"  Years: {df_wb['year'].min()}-{df_wb['year'].max()}")

        return df_wb

    def create_wide_format_itu(self, price_dfs, non_price_dfs):
        """
        Convert ITU data to wide format.
        Price data is already wide; non-price data needs pivoting.
        """
        print("\n" + "="*80)
        print("CONVERTING ITU DATA TO WIDE FORMAT")
        print("="*80)

        all_wide_dfs = []
        series_metadata = []

        # Process price data (already wide format)
        if price_dfs:
            print("\nProcessing price data (already wide):")
            for df in price_dfs:
                var_name = df['variable'].iloc[0]
                # Select only country, year, and price columns
                price_cols = ['country', 'year'] + [c for c in df.columns if c.endswith('_usd') or c.endswith('_gni_pct') or c.endswith('_ppp')]
                df_wide = df[price_cols].copy()
                all_wide_dfs.append(df_wide)
                print(f"  - {var_name}: {df_wide.shape[1]-2} price columns")
                
                # Add metadata for each price type
                for suffix, unit in [('_usd', 'USD'), ('_gni_pct', 'GNI%'), ('_ppp', 'PPP')]:
                    series_metadata.append({
                        'variable': var_name,
                        'series_code': suffix.replace('_', ''),
                        'series_units': unit
                    })

        # Process non-price data (needs pivoting)
        if non_price_dfs:
            print("\nProcessing non-price data (pivoting):")
            df_non_price = pd.concat(non_price_dfs, ignore_index=True)
            
            # Get unique series
            series_list = df_non_price[['variable', 'series_code', 'series_units']].drop_duplicates()
            print(f"  Found {len(series_list)} series to pivot")

            # Pivot each series
            for idx, row in series_list.iterrows():
                var_name = row['variable']
                series_code = row['series_code']
                col_name = f"{var_name}_{series_code}"

                df_series = df_non_price[
                    (df_non_price['variable'] == var_name) & 
                    (df_non_price['series_code'] == series_code)
                ].copy()

                # The value column is named after the variable
                df_pivot = df_series[['country', 'year', var_name]].copy()
                df_pivot = df_pivot.rename(columns={var_name: col_name})

                all_wide_dfs.append(df_pivot)
                print(f"    - {col_name} ({row['series_units']}): {len(df_series)} obs")
                
                # Add metadata
                series_metadata.append({
                    'variable': var_name,
                    'series_code': series_code,
                    'series_units': row['series_units']
                })

        # Merge all wide dataframes
        print("\nMerging all ITU data...")
        df_wide = all_wide_dfs[0]
        for df_next in all_wide_dfs[1:]:
            df_wide = df_wide.merge(df_next, on=['country', 'year'], how='outer')

        print(f"\n[OK] Wide format: {df_wide.shape[0]} rows × {df_wide.shape[1]} columns")

        # Convert metadata to DataFrame
        series_metadata_df = pd.DataFrame(series_metadata)

        return df_wide, series_metadata_df

    def create_wide_format_wb(self, df_wb):
        """
        Convert WB long format to wide format.
        """
        print("\n" + "="*80)
        print("CONVERTING WORLD BANK DATA TO WIDE FORMAT")
        print("="*80)

        # Pivot WB data (preserve NAs - do NOT fill)
        df_wide = df_wb.pivot_table(
            index=['country', 'year'],
            columns='variable',
            values='value',
            aggfunc='first',
            fill_value=None  # Explicitly preserve NAs
        ).reset_index()
        
        print(f"\n[!] NA values preserved (not filled)")

        print(f"\n[OK] Wide format: {df_wide.shape[0]} rows × {df_wide.shape[1]} columns")
        print(f"  Variables: {len(df_wide.columns) - 2}")  # -2 for country, year

        return df_wide

    def merge_datasets(self, df_itu, df_wb):
        """Merge ITU and World Bank wide-format datasets."""
        print("\n" + "="*80)
        print("MERGING DATASETS")
        print("="*80)

        # Check countries
        itu_countries = set(df_itu['country'].unique())
        wb_countries = set(df_wb['country'].unique())

        print(f"\nCountries:")
        print(f"  ITU: {len(itu_countries)}")
        print(f"  World Bank: {len(wb_countries)}")
        print(f"  Common: {len(itu_countries & wb_countries)}")

        # Merge
        df_merged = df_itu.merge(
            df_wb,
            on=['country', 'year'],
            how='outer',
            indicator=True
        )

        # Check merge results
        merge_status = df_merged['_merge'].value_counts()
        print(f"\nMerge results:")
        print(f"  Both datasets: {merge_status.get('both', 0)}")
        print(f"  Only ITU: {merge_status.get('left_only', 0)}")
        print(f"  Only World Bank: {merge_status.get('right_only', 0)}")

        df_merged = df_merged.drop('_merge', axis=1)

        # Verify NA values are preserved (not filled)
        na_count = df_merged.isnull().sum().sum()
        print(f"\n[OK] NA values preserved: {na_count:,} missing values in merged data")
        print("  (This is expected - missing data is NOT filled)")

        # Add region
        df_merged['region'] = df_merged['country'].apply(get_region)

        # Filter to analysis period and countries
        df_merged = df_merged[
            (df_merged['year'] >= START_YEAR) &
            (df_merged['year'] <= END_YEAR) &
            (df_merged['country'].isin(ALL_COUNTRIES))
        ]

        print(f"\n[OK] Final dataset: {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")
        print(f"  Countries: {df_merged['country'].nunique()}")
        print(f"  Years: {df_merged['year'].min()}-{df_merged['year'].max()}")
        print(f"  Regions: {df_merged['region'].value_counts().to_dict()}")

        return df_merged

    def create_series_reference(self, series_metadata):
        """
        Create a reference table for ITU series selection.
        This helps users choose which series to use in regression.
        """
        print("\n" + "="*80)
        print("CREATING SERIES REFERENCE TABLE")
        print("="*80)

        # Load catalog for additional info
        if self.catalog_file.exists():
            catalog = pd.read_excel(self.catalog_file, engine='openpyxl')
            catalog_itu = catalog[catalog['data_source'] == 'ITU']

            # Merge with series metadata
            reference = series_metadata.merge(
                catalog_itu,
                left_on=['variable', 'series_code'],
                right_on=['variable', 'series_code'],
                how='left'
            )
        else:
            reference = series_metadata.copy()

        # Add column name for easy reference
        reference['column_name'] = reference['variable'] + '_' + reference['series_code']

        # Save reference table
        reference_file = DATA_PROCESSED / 'itu_series_reference.xlsx'
        reference.to_excel(reference_file, index=False, engine='openpyxl')

        print(f"\n[OK] Saved: {reference_file.name}")
        print(f"  {len(reference)} ITU series documented")

        # Print user-friendly guide
        print("\n" + "="*80)
        print("ITU SERIES GUIDE FOR REGRESSION ANALYSIS")
        print("="*80)

        for var in reference['variable'].unique():
            print(f"\n{var.upper()}:")
            var_series = reference[reference['variable'] == var]
            for idx, row in var_series.iterrows():
                print(f"  - {row['column_name']}")
                print(f"    Units: {row['series_units']}")
                print(f"    Code: {row['series_code']}")

        return reference

    def save_merged_data(self, df, series_reference):
        """Save merged dataset and documentation."""

        # Main merged dataset
        output_file = DATA_PROCESSED / 'data_merged_with_series.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')

        print("\n" + "="*80)
        print("SAVING MERGED DATA")
        print("="*80)
        print(f"\n[!] IMPORTANT: NA values are preserved (NOT filled)")
        print(f"[OK] Saved: {output_file}")
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
            print(f"\nVariables with missing data (top 20):")
            print(missing_df.head(20).to_string(index=False))
        else:
            print("\n[OK] No missing data!")

        # Save missing data report
        missing_file = DATA_PROCESSED / 'missing_data_report.xlsx'
        missing_df.to_excel(missing_file, index=False, engine='openpyxl')
        print(f"\n[OK] Missing data report: {missing_file.name}")

        return output_file


def main():
    """Main execution function."""
    print("="*80)
    print("DATA MERGING SCRIPT (UPDATED)")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis version preserves series metadata for proper variable selection.\n")

    try:
        merger = DataMerger()

        # Load processed data
        result = merger.load_processed_itu_data()
        df_wb = merger.load_processed_wb_data()

        if result is None or df_wb is None:
            print("\n[ERROR] Cannot proceed without processed data files.")
            print("\nRequired: Run python code/data_collection/01.5_process_data.py first")
            sys.exit(1)

        price_dfs, non_price_dfs = result

        # Convert to wide format
        df_itu_wide, series_metadata = merger.create_wide_format_itu(price_dfs, non_price_dfs)
        df_wb_wide = merger.create_wide_format_wb(df_wb)

        # Merge datasets
        df_merged = merger.merge_datasets(df_itu_wide, df_wb_wide)

        # Create series reference
        series_reference = merger.create_series_reference(series_metadata)

        # Save
        merger.save_merged_data(df_merged, series_reference)

        print("\n" + "="*80)
        print("MERGE COMPLETE [OK]")
        print("="*80)
        print(f"\nOutput files in: {DATA_PROCESSED}")
        print("\nIMPORTANT: Check 'itu_series_reference.csv' to see which series")
        print("           to use for your regression analysis!")

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
