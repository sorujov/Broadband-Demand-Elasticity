"""
================================================================================
Data Preparation Script
================================================================================
Transforms merged data into analysis-ready format.

Pipeline Position: Stage 2 (after data_collection, before analysis)

Input:  data/processed/data_merged_with_series.xlsx
Output: data/processed/analysis_ready_data.csv

Transformations:
1. Standardize column names using COLUMN_MAPPINGS from config
2. Handle missing data (forward fill + linear interpolation)
3. Create log transformations for key variables
4. Create regional indicators and interaction terms
5. Create lagged price variables for IV estimation

================================================================================
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Import config - try multiple approaches for robustness
try:
    from code.utils.config import (
        DATA_MERGED_FILE, ANALYSIS_READY_FILE, COLUMN_MAPPINGS,
        LOG_TRANSFORM_VARS, EU_COUNTRIES, EAP_COUNTRIES
    )
except ModuleNotFoundError:
    # Alternative import when running from different directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_MERGED_FILE, ANALYSIS_READY_FILE, COLUMN_MAPPINGS,
        LOG_TRANSFORM_VARS, EU_COUNTRIES, EAP_COUNTRIES
    )


def load_data():
    """Load merged data from Excel file."""
    print(f"Loading data from: {DATA_MERGED_FILE}")
    df = pd.read_excel(DATA_MERGED_FILE, engine='openpyxl')
    print(f"  Loaded {len(df)} observations, {len(df.columns)} columns")
    return df


def standardize_columns(df):
    """Apply column name mappings from config."""
    print("\nStandardizing column names...")

    # Apply mappings where source column exists
    rename_dict = {}
    for old_name, new_name in COLUMN_MAPPINGS.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
            print(f"  {old_name} -> {new_name}")

    df = df.rename(columns=rename_dict)
    return df


def handle_missing_data(df):
    """
    Handle missing data using simple interpolation.

    Strategy:
    1. Sort by country and year
    2. Forward fill within each country (carry last observation)
    3. Linear interpolation for remaining gaps within country
    4. Report missing data before and after
    """
    print("\nHandling missing data...")

    # Define key variables to check
    key_vars = [
        'fixed_broadband_subs', 'internet_users_pct', 'int_bandwidth',
        'fixed_broad_price', 'gdp_per_capita', 'population'
    ]

    # Report initial missing data
    print("\n  Missing data BEFORE imputation:")
    for var in key_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            pct = 100 * missing / len(df)
            print(f"    {var}: {missing} ({pct:.1f}%)")

    # Sort for time-series operations
    df = df.sort_values(['country', 'year']).reset_index(drop=True)

    # Get numeric columns for imputation (exclude country, year, region)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')

    # Apply forward fill then interpolation within each country
    for col in numeric_cols:
        # Forward fill within country
        df[col] = df.groupby('country')[col].transform(
            lambda x: x.ffill()
        )
        # Linear interpolation for remaining gaps
        df[col] = df.groupby('country')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

    # Report final missing data
    print("\n  Missing data AFTER imputation:")
    for var in key_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            pct = 100 * missing / len(df)
            print(f"    {var}: {missing} ({pct:.1f}%)")

    return df


def create_log_transforms(df):
    """Create log transformations for specified variables."""
    print("\nCreating log transformations...")

    for var in LOG_TRANSFORM_VARS:
        if var in df.columns:
            # Add small constant to handle zeros
            log_var = f'log_{var}'
            df[log_var] = np.log(df[var] + 1)
            print(f"  Created {log_var}")

    return df


def create_regional_indicators(df):
    """Create regional indicator and interaction terms."""
    print("\nCreating regional indicators...")

    # Create EaP indicator (1 for EaP countries, 0 for EU)
    df['eap'] = df['country'].isin(EAP_COUNTRIES).astype(int)
    df['eu'] = df['country'].isin(EU_COUNTRIES).astype(int)

    print(f"  EaP countries: {df['eap'].sum()} observations")
    print(f"  EU countries: {df['eu'].sum()} observations")

    # Create price-region interactions
    if 'log_fixed_broad_price' in df.columns:
        df['log_price_x_eap'] = df['log_fixed_broad_price'] * df['eap']
        df['log_price_x_eu'] = df['log_fixed_broad_price'] * df['eu']
        print("  Created log_price_x_eap, log_price_x_eu interactions")

    return df


def create_lagged_variables(df):
    """Create lagged price variables for IV estimation."""
    print("\nCreating lagged variables...")

    df = df.sort_values(['country', 'year']).reset_index(drop=True)

    # Lagged prices (potential instruments)
    lag_vars = ['log_fixed_broad_price', 'log_mobile_broad_price']

    for var in lag_vars:
        if var in df.columns:
            lag1_var = f'{var}_lag1'
            df[lag1_var] = df.groupby('country')[var].shift(1)
            print(f"  Created {lag1_var}")

    return df


def create_time_trends(df):
    """Create time trend variables."""
    print("\nCreating time trends...")

    # Time trend (years since start)
    min_year = df['year'].min()
    df['time_trend'] = df['year'] - min_year
    df['time_trend_sq'] = df['time_trend'] ** 2

    # COVID indicator (2020+)
    df['post_covid'] = (df['year'] >= 2020).astype(int)

    print(f"  Time trend: {min_year} = 0")
    print(f"  Post-COVID observations: {df['post_covid'].sum()}")

    return df


def validate_output(df):
    """Validate the prepared dataset."""
    print("\n" + "="*60)
    print("OUTPUT VALIDATION")
    print("="*60)

    # Check key variables exist
    required_vars = [
        'country', 'year', 'region', 'eap',
        'log_fixed_broadband_subs',  # Primary DV
        'log_fixed_broad_price',      # Primary price
        'log_gdp_per_capita',         # Key control
    ]

    missing_vars = [v for v in required_vars if v not in df.columns]
    if missing_vars:
        print(f"  WARNING: Missing required variables: {missing_vars}")
    else:
        print("  All required variables present")

    # Check panel structure
    n_countries = df['country'].nunique()
    n_years = df['year'].nunique()
    print(f"\n  Panel structure:")
    print(f"    Countries: {n_countries}")
    print(f"    Years: {df['year'].min()} - {df['year'].max()} ({n_years} years)")
    print(f"    Total observations: {len(df)}")

    # Check for missing values in key variables
    print(f"\n  Missing values in key variables:")
    for var in ['log_fixed_broadband_subs', 'log_fixed_broad_price', 'log_gdp_per_capita']:
        if var in df.columns:
            missing = df[var].isna().sum()
            print(f"    {var}: {missing}")

    return df


def save_data(df):
    """Save prepared data to CSV."""
    print(f"\nSaving to: {ANALYSIS_READY_FILE}")
    df.to_csv(ANALYSIS_READY_FILE, index=False)
    print(f"  Saved {len(df)} observations, {len(df.columns)} columns")


def main():
    """Main data preparation pipeline."""
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)

    # Load data
    df = load_data()

    # Apply transformations
    df = standardize_columns(df)
    df = handle_missing_data(df)
    df = create_log_transforms(df)
    df = create_regional_indicators(df)
    df = create_lagged_variables(df)
    df = create_time_trends(df)

    # Validate and save
    df = validate_output(df)
    save_data(df)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)

    return df


if __name__ == "__main__":
    main()
