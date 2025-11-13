# code/data_preparation/04_prepare_data.py
"""
================================================================================
Data Preparation Script
================================================================================
Purpose: Clean, impute, and prepare analysis-ready dataset
Author: Samir Orujov
Date: November 13, 2025

This script:
1. Loads merged data
2. Handles missing data systematically
3. Creates variable transformations
4. Generates lagged variables and interactions
5. Saves analysis-ready dataset
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
        DATA_INTERIM, DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, get_region
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_INTERIM, DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, get_region
    )

class DataPreparator:
    """Prepare analysis-ready dataset."""

    def __init__(self):
        self.input_file = DATA_INTERIM / 'data_merged.csv'
        self.output_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'

    def load_data(self):
        """Load merged dataset."""
        print("="*80)
        print("LOADING MERGED DATA")
        print("="*80)

        if not self.input_file.exists():
            print(f"\n✗ File not found: {self.input_file}")
            print("\nPlease run: python 03_merge_data.py first")
            return None

        df = pd.read_csv(self.input_file)
        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")

        return df

    def rename_variables(self, df):
        """Rename variables for easier handling."""
        print("\n" + "="*80)
        print("RENAMING VARIABLES")
        print("="*80)

        # Create renaming dictionary for common ITU/WB variables
        rename_dict = {}

        # ITU variables (if they exist)
        possible_renames = {
            'int_band_use': 'bandwidth_use',
            'fixed_broad_basket_USD': 'price_fixed_bb',
            'mobile_broad_basket_USD': 'price_mobile_bb',
            'fixed_broadband_subs': 'bb_subs_per100',
            'internet_users_pct': 'internet_users_pct',
            'mobile_subs': 'mobile_subs_per100',
        }

        # Only rename columns that exist
        for old_name, new_name in possible_renames.items():
            if old_name in df.columns and old_name != new_name:
                rename_dict[old_name] = new_name

        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"\n✓ Renamed {len(rename_dict)} variables")
            for old, new in rename_dict.items():
                print(f"  {old} → {new}")
        else:
            print("\n✓ No renaming needed")

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
            'int_bandwidth': 'log_bandwidth_use',
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

        lag_vars = ['bb_subs_per100', 'int_bandwidth', 'gdp_per_capita', 'fixed_broad_price']
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
        for var in ['price_fixed_bb', 'bandwidth_use', 'bb_subs_per100', 'gdp_per_capita']:
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

        key_vars = [col for col in ['price_fixed_bb', 'bandwidth_use', 'bb_subs_per100', 
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
    print("DATA PREPARATION SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize preparator
        preparator = DataPreparator()

        # Load data
        df = preparator.load_data()
        if df is None:
            return

        # Rename variables
        df = preparator.rename_variables(df)

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
        print("\nNext steps:")
        print("  → Descriptive statistics and visualization")
        print("  → Baseline regression models")
        print("  → IV/2SLS estimation")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
