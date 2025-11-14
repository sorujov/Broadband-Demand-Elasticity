# code/data_preparation/04_prepare_data.py (UPDATED)
"""
================================================================================
Data Preparation Script (UPDATED)
================================================================================
Purpose: Clean, impute, and prepare analysis-ready dataset
Author: Samir Orujov
Date: November 14, 2025 (Updated)

UPDATED: Now compatible with series-metadata-preserved data structure
         Allows selection of specific ITU series for analysis

This script:
1. Loads merged data with series metadata
2. Selects appropriate series for analysis
3. Handles missing data systematically
4. Creates variable transformations
5. Generates lagged variables and interactions
6. Saves analysis-ready dataset
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from code.utils.config
try:
    from code.utils.config import (
        DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, get_region
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, get_region
    )


class DataPreparator:
    """Prepare analysis-ready dataset with series selection."""

    def __init__(self, series_config=None):
        """
        Initialize with series selection configuration.

        Parameters:
        -----------
        series_config : dict, optional
            Dictionary specifying which series to use for each variable.
            Example:
            {
                'fixed_broad_price': 'i154_FBB_ts_PPP',
                'mobile_broad_price': 'i456_MBB_ts_PPP'
            }

            If None, uses default (PPP series for prices)
        """
        # Input/output files
        self.input_file = DATA_PROCESSED / 'data_merged_with_series.csv'
        self.series_ref_file = DATA_PROCESSED / 'itu_series_reference.csv'
        self.output_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'

        # Default series configuration (PPP-adjusted for prices)
        self.default_series = {
            'fixed_broad_price': 'i154_FBB_ts_PPP',
            'mobile_broad_price': 'i456_MBB_ts_PPP',  # Adjust if needed
        }

        # Use provided config or defaults
        self.series_config = series_config if series_config else self.default_series

        # Store original series columns for reference
        self.selected_series = {}

    def load_data(self):
        """Load merged dataset with series metadata."""
        print("="*80)
        print("LOADING MERGED DATA WITH SERIES METADATA")
        print("="*80)

        if not self.input_file.exists():
            print(f"\n✗ File not found: {self.input_file}")
            print("\nPlease run: python code/data_collection/03_merge_data.py first")
            return None

        df = pd.read_csv(self.input_file)

        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")

        # Load series reference if available
        if self.series_ref_file.exists():
            series_ref = pd.read_csv(self.series_ref_file)
            print(f"\n✓ Loaded series reference: {len(series_ref)} series documented")

        return df

    def select_and_rename_series(self, df):
        """
        Select specific ITU series and rename to standard variable names.
        This makes the rest of the analysis code work seamlessly.
        """
        print("\n" + "="*80)
        print("SERIES SELECTION AND STANDARDIZATION")
        print("="*80)

        print("\nSelected series for analysis:")

        for var_base, series_code in self.series_config.items():
            # Construct the full column name
            full_col_name = f"{var_base}_{series_code}"

            if full_col_name in df.columns:
                # Store the original column name for documentation
                self.selected_series[var_base] = {
                    'series_code': series_code,
                    'original_column': full_col_name
                }

                # Rename to standard name for analysis
                df[var_base] = df[full_col_name]

                print(f"  ✓ {var_base}")
                print(f"    Series: {series_code}")
                print(f"    Column: {full_col_name}")
                print(f"    → Renamed to: {var_base}")
            else:
                print(f"  ⚠ {var_base}: Series {series_code} not found")
                print(f"    Looking for column: {full_col_name}")

        # Document series selection
        self._save_series_selection_log()

        return df

    def _save_series_selection_log(self):
        """Save a log of which series were selected."""
        log_file = DATA_PROCESSED / 'series_selection_log.txt'

        with open(log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ITU SERIES SELECTION LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Selected series for this analysis:\n")
            f.write("-"*80 + "\n")

            for var_base, info in self.selected_series.items():
                f.write(f"\nVariable: {var_base}\n")
                f.write(f"  Series Code: {info['series_code']}\n")
                f.write(f"  Original Column: {info['original_column']}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("IMPORTANT: Document these choices in your paper methodology!\n")
            f.write("="*80 + "\n")

        print(f"\n✓ Series selection log saved: {log_file.name}")

    def rename_other_variables(self, df):
        """Rename other variables for easier handling."""
        print("\n" + "="*80)
        print("RENAMING OTHER VARIABLES")
        print("="*80)

        # Map ITU/WB variable names to simpler analysis names
        rename_dict = {}

        # Check for different possible column name patterns
        possible_renames = {
            # ITU variables (check for series-specific names)
            'int_bandwidth': 'bandwidth_use',
            'fixed_broadband_subs': 'bb_subs_per100',
            'internet_users_pct': 'internet_users_pct',
            'mobile_subs': 'mobile_subs_per100',
        }

        # Only rename columns that exist and aren't already renamed
        for old_name, new_name in possible_renames.items():
            # Check for exact match
            if old_name in df.columns and old_name != new_name:
                rename_dict[old_name] = new_name
            # Check for series-specific versions (take first available)
            else:
                matching_cols = [col for col in df.columns if col.startswith(f"{old_name}_")]
                if matching_cols:
                    # Use the first matching series
                    rename_dict[matching_cols[0]] = new_name

        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"\n✓ Renamed {len(rename_dict)} variables")
            for old, new in rename_dict.items():
                print(f"  {old} → {new}")
        else:
            print("\n✓ No additional renaming needed")

        return df

    def filter_time_period(self, df):
        """Focus on period with price data (2010-2023)."""
        print("\n" + "="*80)
        print("TIME PERIOD RESTRICTION")
        print("="*80)

        df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()

        print(f"\n✓ Restricted to {START_YEAR}-{END_YEAR}")
        print(f"  Observations: {len(df)}")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Years per country: {len(df) / df['country'].nunique():.1f} average")

        return df

    def create_transformations(self, df):
        """Create log transformations and derived variables."""
        print("\n" + "="*80)
        print("VARIABLE TRANSFORMATIONS")
        print("="*80)

        # Log transformations (with safety checks)
        log_vars = {
            'bandwidth_use': 'log_bandwidth_use',
            'bb_subs_per100': 'log_bb_subs',
            'fixed_broad_price': 'log_price',
            'gdp_per_capita': 'log_gdp_pc',
            'population': 'log_population'
        }

        created = []
        for orig, log_name in log_vars.items():
            if orig in df.columns:
                # Add small constant to avoid log(0)
                df[log_name] = np.log(df[orig].replace(0, np.nan) + 0.1)
                created.append(log_name)

        print(f"\n✓ Created {len(created)} log variables")
        for var in created:
            print(f"  {var}")

        # Growth rates
        df = df.sort_values(['country', 'year'])

        if 'gdp_per_capita' in df.columns:
            df['gdp_growth'] = df.groupby('country')['gdp_per_capita'].pct_change() * 100
            print("\n✓ Created gdp_growth")

        if 'fixed_broad_price' in df.columns:
            df['price_growth'] = df.groupby('country')['fixed_broad_price'].pct_change() * 100
            print("✓ Created price_growth")

        # Regional interactions
        if 'is_eap' not in df.columns:
            df['is_eap'] = (df['region'] == 'EaP').astype(int)

        if 'fixed_broad_price' in df.columns:
            df['price_x_eap'] = df['fixed_broad_price'] * df['is_eap']
            print("✓ Created price_x_eap")

        if 'log_price' in df.columns:
            df['log_price_x_eap'] = df['log_price'] * df['is_eap']
            print("✓ Created log_price_x_eap")

        # Time trends
        df['time_trend'] = df['year'] - START_YEAR
        df['time_trend_sq'] = df['time_trend'] ** 2
        print("\n✓ Created time trend variables")

        return df

    def create_lags(self, df):
        """Create lagged variables for dynamic models."""
        print("\n" + "="*80)
        print("LAGGED VARIABLES")
        print("="*80)

        df = df.sort_values(['country', 'year'])

        lag_vars = ['bb_subs_per100', 'bandwidth_use', 'gdp_per_capita', 'fixed_broad_price']

        created = []
        for var in lag_vars:
            if var in df.columns:
                df[f'{var}_lag1'] = df.groupby('country')[var].shift(1)
                created.append(f'{var}_lag1')

        print(f"\n✓ Created {len(created)} lagged variables")
        for var in created:
            print(f"  {var}")

        return df

    def handle_missing_data(self, df):
        """Handle missing data systematically."""
        print("\n" + "="*80)
        print("MISSING DATA HANDLING")
        print("="*80)

        # Identify variables to impute
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        impute_vars = [col for col in numeric_cols if col not in ['year', 'is_eap', 'time_trend', 'time_trend_sq']]

        print(f"\nVariables to check: {len(impute_vars)}")

        # Strategy 1: Forward/backward fill within countries
        time_series_vars = [col for col in impute_vars if any(x in col for x in
                           ['bandwidth', 'subs', 'price', 'internet', 'mobile', 'population'])]

        print(f"\nStrategy 1: Time-series interpolation ({len(time_series_vars)} variables)")
        df = df.sort_values(['country', 'year'])

        for var in time_series_vars:
            if var in df.columns:
                missing_before = df[var].isna().sum()
                if missing_before > 0:
                    df[var] = df.groupby('country')[var].fillna(method='ffill').fillna(method='bfill')
                    missing_after = df[var].isna().sum()
                    filled = missing_before - missing_after
                    if filled > 0:
                        print(f"  {var}: filled {filled}/{missing_before}")

        # Strategy 2: Regional mean imputation
        regional_vars = [col for col in impute_vars if any(x in col for x in
                        ['education', 'labor', 'regulatory'])]

        if regional_vars:
            print(f"\nStrategy 2: Regional mean imputation ({len(regional_vars)} variables)")
            for var in regional_vars:
                if var in df.columns:
                    missing_before = df[var].isna().sum()
                    if missing_before > 0:
                        regional_means = df.groupby(['region', 'year'])[var].transform('mean')
                        df[var] = df[var].fillna(regional_means)
                        missing_after = df[var].isna().sum()
                        filled = missing_before - missing_after
                        if filled > 0:
                            print(f"  {var}: filled {filled}/{missing_before}")

        return df

    def generate_summary_stats(self, df):
        """Generate summary statistics by region."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS BY REGION")
        print("="*80)

        key_vars = []
        for var in ['fixed_broad_price', 'bandwidth_use', 'bb_subs_per100', 'gdp_per_capita']:
            if var in df.columns:
                key_vars.append(var)

        if not key_vars:
            print("\n⚠ No key variables found for summary")
            return df

        print("\nEU Countries:")
        if 'EU' in df['region'].values:
            print(df[df['region'] == 'EU'][key_vars].describe().round(2))

        print("\nEastern Partnership Countries:")
        if 'EaP' in df['region'].values:
            print(df[df['region'] == 'EaP'][key_vars].describe().round(2))

        return df

    def save_clean_data(self, df):
        """Save analysis-ready dataset."""
        print("\n" + "="*80)
        print("SAVING CLEAN DATASET")
        print("="*80)

        # Save main dataset
        df.to_csv(self.output_file, index=False)

        print(f"\n✓ Saved: {self.output_file}")
        print(f"  - {df.shape[0]} observations")
        print(f"  - {df.shape[1]} variables")
        print(f"  - {df['country'].nunique()} countries")
        print(f"  - {df['year'].min()}-{df['year'].max()}")

        # Save descriptive stats
        stats_file = DATA_PROCESSED / 'descriptive_stats_by_region.csv'
        key_vars = [col for col in ['fixed_broad_price', 'bandwidth_use', 'bb_subs_per100',
                                      'gdp_per_capita', 'internet_users_pct'] if col in df.columns]

        if key_vars and 'region' in df.columns:
            stats = df.groupby('region')[key_vars].describe()
            stats.to_csv(stats_file)
            print(f"\n✓ Saved descriptive stats: {stats_file}")

        # Final missing data report
        print("\n" + "="*80)
        print("FINAL DATA QUALITY")
        print("="*80)

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            print(f"\nVariables with remaining missing data:")
            missing_pct = (missing / len(df) * 100).round(1)
            for var, count in missing.items():
                print(f"  {var}: {count} ({missing_pct[var]}%)")
        else:
            print("\n✓ No missing data in final dataset!")

        print(f"\nComplete observations: {df.dropna().shape[0]}/{len(df)}")

        return self.output_file


def main():
    """Main execution function."""
    print("="*80)
    print("DATA PREPARATION SCRIPT (UPDATED)")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Series configuration for main analysis
    # IMPORTANT: Document these choices in your paper!
    series_config = {
        'fixed_broad_price': 'i154_FBB_ts_PPP',  # PPP-adjusted (recommended)
        # 'fixed_broad_price': 'i154_FBB_ts_GNI',  # Alternative for robustness
        # 'fixed_broad_price': 'i154_FBB_ts$',  # Alternative for robustness
    }

    print("\n" + "="*80)
    print("SERIES CONFIGURATION")
    print("="*80)
    print("\nThis run uses:")
    for var, series in series_config.items():
        print(f"  {var}: {series}")
    print("\nTo use alternative series for robustness checks,")
    print("edit the series_config dictionary in main().")
    print("="*80)

    try:
        # Initialize preparator with series configuration
        preparator = DataPreparator(series_config=series_config)

        # Load data
        df = preparator.load_data()
        if df is None:
            return

        # Select and rename series
        df = preparator.select_and_rename_series(df)

        # Rename other variables
        df = preparator.rename_other_variables(df)

        # Filter time period
        df = preparator.filter_time_period(df)

        # Create transformations
        df = preparator.create_transformations(df)

        # Create lags
        df = preparator.create_lags(df)

        # Handle missing data
        df = preparator.handle_missing_data(df)

        # Generate summary stats
        df = preparator.generate_summary_stats(df)

        # Save clean data
        preparator.save_clean_data(df)

        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE ✓")
        print("="*80)
        print("\nYour data is now ready for econometric analysis!")
        print("\nIMPORTANT: Check series_selection_log.txt to document")
        print("           which ITU series you used in your paper!")
        print("\nNext steps:")
        print("  → Descriptive statistics and visualization")
        print("  → Baseline regression models")
        print("  → IV/2SLS estimation")
        print("  → Robustness checks with alternative series")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
