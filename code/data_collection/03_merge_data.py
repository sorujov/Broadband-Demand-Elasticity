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
        self.itu_processed_files = list(DATA_INTERIM.glob('itu_*_processed.csv'))
        self.wb_processed_file = DATA_INTERIM / 'worldbank_processed.csv'
        self.catalog_file = DATA_INTERIM / 'data_catalog.csv'

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

        all_series = []

        for filepath in self.itu_processed_files:
            df = pd.read_csv(filepath)
            all_series.append(df)

            var_name = filepath.stem.replace('itu_', '').replace('_processed', '')
            n_series = df['series_code'].nunique()
            print(f"  [OK] {var_name}: {len(df)} obs, {n_series} series")

        # Combine all ITU data
        df_itu = pd.concat(all_series, ignore_index=True)

        print(f"\n[OK] Combined ITU data: {df_itu.shape[0]} rows × {df_itu.shape[1]} columns")
        print(f"  Total series: {df_itu['series_code'].nunique()}")
        print(f"  Variables: {df_itu['variable'].nunique()}")

        return df_itu

    def load_processed_wb_data(self):
        """Load processed World Bank data."""
        print("\n" + "="*80)
        print("LOADING PROCESSED WORLD BANK DATA")
        print("="*80)

        if not self.wb_processed_file.exists():
            print(f"\n[ERROR] File not found: {self.wb_processed_file}")
            print("   Please run: python code/data_collection/01.5_process_data.py")
            return None

        df_wb = pd.read_csv(self.wb_processed_file)

        print(f"\nLoaded: {df_wb.shape[0]} rows × {df_wb.shape[1]} columns")
        print(f"  Variables: {df_wb['variable'].nunique()}")
        print(f"  Countries: {df_wb['country'].nunique()}")
        print(f"  Years: {df_wb['year'].min()}-{df_wb['year'].max()}")

        return df_wb

    def create_wide_format_itu(self, df_itu):
        """
        Convert ITU long format to wide format.
        Each series becomes a separate column.
        """
        print("\n" + "="*80)
        print("CONVERTING ITU DATA TO WIDE FORMAT")
        print("="*80)

        # Get all unique series with their metadata
        series_metadata = df_itu[['variable', 'series_code', 'series_name', 'series_units']].drop_duplicates()

        print(f"\nProcessing {len(series_metadata)} ITU series:")
        for idx, row in series_metadata.iterrows():
            print(f"  • {row['variable']}_{row['series_code']} ({row['series_units']})")

        # Pivot each variable-series combination
        wide_dfs = []

        for idx, meta in series_metadata.iterrows():
            var_name = meta['variable']
            series_code = meta['series_code']
            col_name = f"{var_name}_{series_code}"

            # Filter to this series
            df_series = df_itu[
                (df_itu['variable'] == var_name) & 
                (df_itu['series_code'] == series_code)
            ].copy()

            # Select relevant columns
            df_pivot = df_series[['country', 'year', col_name]].copy()

            wide_dfs.append(df_pivot)

        # Merge all series into one wide dataframe
        df_wide = wide_dfs[0]
        for df_next in wide_dfs[1:]:
            df_wide = df_wide.merge(df_next, on=['country', 'year'], how='outer')

        print(f"\n[OK] Wide format: {df_wide.shape[0]} rows × {df_wide.shape[1]} columns")

        return df_wide, series_metadata

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
        
        print(f"\n⚠ NA values preserved (not filled)")

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
            catalog = pd.read_csv(self.catalog_file)
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
        reference_file = DATA_PROCESSED / 'itu_series_reference.csv'
        reference.to_csv(reference_file, index=False)

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
                print(f"  • {row['column_name']}")
                print(f"    Units: {row['series_units']}")
                print(f"    Code: {row['series_code']}")

        return reference

    def save_merged_data(self, df, series_reference):
        """Save merged dataset and documentation."""

        # Main merged dataset
        output_file = DATA_PROCESSED / 'data_merged_with_series.csv'
        df.to_csv(output_file, index=False)

        print("\n" + "="*80)
        print("SAVING MERGED DATA")
        print("="*80)
        print(f"\n⚠ IMPORTANT: NA values are preserved (NOT filled)")
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
        missing_file = DATA_PROCESSED / 'missing_data_report.csv'
        missing_df.to_csv(missing_file, index=False)
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
        df_itu = merger.load_processed_itu_data()
        df_wb = merger.load_processed_wb_data()

        if df_itu is None or df_wb is None:
            print("\n[ERROR] Cannot proceed without processed data files.")
            print("\nRequired: Run python code/data_collection/01.5_process_data.py first")
            return

        # Convert to wide format
        df_itu_wide, series_metadata = merger.create_wide_format_itu(df_itu)
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


if __name__ == "__main__":
    main()
