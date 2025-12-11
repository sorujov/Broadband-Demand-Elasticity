# code/data_preparation/04_prepare_data_FINAL.py

"""
================================================================================
Data Preparation Script (FINAL VERSION - Nov 20, 2025)
================================================================================
Purpose: Clean, impute, and prepare analysis-ready dataset

Author: Samir Orujov
Date: November 20, 2025

CRITICAL CHANGES:
- DROPPED: All bandwidth variables (int_bandwidth_*) due to systematic unavailability
  confirmed by ITU for European countries (72% missing, unreliable forward fill)
- FORWARD FILL: Only for variables with <10% missing (education, R&D, ICT exports)
- NO FILL: Variables with >10% missing are kept as NA for robustness checks
- FOCUS: Fixed/mobile subscriptions and internet users as demand measures

This script:
1. Loads merged data with series metadata
2. Selects appropriate series for analysis
3. Handles missing data systematically (forward fill <10% only)
4. Creates variable transformations (logs, growth rates)
5. Generates lagged variables and interactions
6. Saves analysis-ready dataset

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import (
        DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES


class DataPreparation:
    """Prepare analysis-ready dataset for price elasticity estimation."""

    def __init__(self, input_file=None, output_file=None):
        """Initialize with file paths."""
        if input_file is None:
            self.input_file = DATA_PROCESSED / 'data_merged_with_series.xlsx'
        else:
            self.input_file = input_file

        if output_file is None:
            self.output_file = DATA_PROCESSED / 'analysis_ready_data.xlsx'
        else:
            self.output_file = output_file

        self.df = None

        # Series selection configuration (BANDWIDTH REMOVED)
        self.series_config = {
            # Dependent variables (demand measures)
            'fixed_broadband_subs': 'fixed_broadband_subs_i4213tfbb',  # per 100 inhabitants
            'internet_users_pct': 'internet_users_pct_i99H',           # % of population
            'mobile_subs': 'mobile_subs_i271',                         # per 100 inhabitants

            # Key independent variable (price)
            'fixed_broad_price': 'fixed_broad_price_i154_FBB_ts_GNI',  # GNI-adjusted (% of GNI per capita)
            'mobile_broad_price': 'mobile_broad_price_i271mb_ts_GNI',  # GNI-adjusted (% of GNI per capita)

            # NOTE: Bandwidth variables DROPPED due to systematic unavailability
            # int_bandwidth_* variables not included (72% missing, unreliable)

            # Control variables (economic)
            'gdp_per_capita': 'gdp_per_capita',
            'gdp_growth': 'gdp_growth',
            'inflation': 'inflation_gdp_deflator',

            # Control variables (infrastructure/technology)
            'electricity_access': 'access_to_electricity',
            'electric_consumption': 'electric_power_consumption',
            'secure_servers': 'secure_internet_servers',

            # Control variables (human capital) - FORWARD FILL <10%
            'education_secondary': 'education_secondary_pct',
            'education_tertiary': 'education_tertiary_pct',
            'labor_advanced_education': 'labor_force_advanced_education',

            # Control variables (innovation) - FORWARD FILL <10%
            'rd_expenditure': 'research_development_expenditure',
            'high_tech_exports': 'high_tech_exports',
            'ict_exports': 'ict_goods_exports',

            # Control variables (institutional)
            'regulatory_quality': 'regulatory_quality_estimate',

            # Control variables (demographics)
            'population': 'population',
            'population_density': 'population_density',
            'urban_population': 'urban_population_pct',
            'working_age_pop': 'population_ages_15_64',
            'wage_workers': 'wage_salaried_workers'
        }

        # Variables to apply forward fill (ONLY <10% missing)
        self.forward_fill_vars = [
            'education_secondary', 'education_tertiary', 
            'labor_advanced_education', 'rd_expenditure',
            'high_tech_exports', 'ict_exports'
        ]

    def load_data(self):
        """Load merged data."""
        print("="*80)
        print("LOADING MERGED DATA")
        print("="*80)

        self.df = pd.read_excel(self.input_file, engine='openpyxl')
        print(f"\n[OK] Loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"  Countries: {self.df['country'].nunique()}")
        print(f"  Years: {self.df['year'].min()}-{self.df['year'].max()}")

        return self

    def select_series(self):
        """Select appropriate series from available data."""
        print("\n" + "="*80)
        print("SELECTING SERIES")
        print("="*80)

        # Create mapping of new names to original column names
        available_cols = []
        missing_cols = []

        for new_name, original_name in self.series_config.items():
            if original_name in self.df.columns:
                available_cols.append((new_name, original_name))
            else:
                missing_cols.append(f"  ⚠ {new_name} ({original_name})")

        # Rename selected columns
        rename_dict = {orig: new for new, orig in available_cols}
        keep_cols = ['country', 'year'] + [orig for _, orig in available_cols]

        self.df = self.df[keep_cols].rename(columns=rename_dict)

        print(f"\n[OK] Selected {len(available_cols)} variables")

        if missing_cols:
            print(f"\n⚠ Missing {len(missing_cols)} variables:")
            for col in missing_cols:
                print(col)

        # Add region indicators
        self.df['is_eu'] = self.df['country'].isin(EU_COUNTRIES).astype(int)
        self.df['is_eap'] = self.df['country'].isin(EAP_COUNTRIES).astype(int)

        print(f"\n[OK] Added region indicators:")
        print(f"  EU countries: {self.df['is_eu'].sum() // self.df.groupby('country').ngroups}")
        print(f"  EaP countries: {self.df['is_eap'].sum() // self.df.groupby('country').ngroups}")

        return self

    def handle_missing_data(self):
        """Handle missing data with forward fill for specific variables only."""
        print("\n" + "="*80)
        print("HANDLING MISSING DATA")
        print("="*80)

        # Report missingness before imputation
        print("\nMissing values BEFORE imputation:")
        missing_before = self.df.isnull().sum()
        missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

        for var, count in missing_before.items():
            pct = (count / len(self.df)) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")

        # Apply forward fill ONLY to specified variables (<10% missing)
        print(f"\n[OK] Applying FORWARD FILL to {len(self.forward_fill_vars)} variables:")

        for var in self.forward_fill_vars:
            if var in self.df.columns:
                before_missing = self.df[var].isnull().sum()

                # Forward fill within each country
                self.df[var] = self.df.groupby('country')[var].fillna(method='ffill')

                after_missing = self.df[var].isnull().sum()
                filled_count = before_missing - after_missing

                if filled_count > 0:
                    print(f"  {var}: filled {filled_count} values")

        # Report missingness after forward fill
        print("\nMissing values AFTER forward fill:")
        missing_after = self.df.isnull().sum()
        missing_after = missing_after[missing_after > 0].sort_values(ascending=False)

        if len(missing_after) > 0:
            for var, count in missing_after.items():
                pct = (count / len(self.df)) * 100
                print(f"  {var}: {count} ({pct:.1f}%)")
        else:
            print("  No missing values remain")

        print("\n⚠ IMPORTANT: Variables with >10% missing are kept as NA")
        print("  These will be handled in regression models (listwise deletion)")

        return self

    def create_transformations(self):
        """Create log transformations and growth rates."""
        print("\n" + "="*80)
        print("CREATING TRANSFORMATIONS")
        print("="*80)

        # Log transformations for key variables
        log_vars = [
            'fixed_broadband_subs', 'internet_users_pct', 'mobile_subs',
            'fixed_broad_price', 'mobile_broad_price',
            'gdp_per_capita', 'population', 'population_density'
        ]

        for var in log_vars:
            if var in self.df.columns:
                # Add small constant to avoid log(0)
                self.df[f'log_{var}'] = np.log(self.df[var] + 1)
                print(f"  [OK] Created log_{var}")

        # Growth rates for demand variables
        growth_vars = ['fixed_broadband_subs', 'internet_users_pct', 'mobile_subs']

        for var in growth_vars:
            if var in self.df.columns:
                self.df[f'{var}_growth'] = self.df.groupby('country')[var].pct_change() * 100
                print(f"  [OK] Created {var}_growth")

        # Price changes
        if 'fixed_broad_price' in self.df.columns:
            self.df['price_change'] = self.df.groupby('country')['fixed_broad_price'].pct_change() * 100
            print(f"  [OK] Created price_change")

        return self

    def create_lags(self):
        """Create lagged variables for IV estimation."""
        print("\n" + "="*80)
        print("CREATING LAGGED VARIABLES")
        print("="*80)

        lag_vars = [
            'fixed_broad_price', 'mobile_broad_price',
            'gdp_per_capita', 'regulatory_quality'
        ]

        for var in lag_vars:
            if var in self.df.columns:
                # Create 1-year lag
                self.df[f'{var}_lag1'] = self.df.groupby('country')[var].shift(1)
                print(f"  [OK] Created {var}_lag1")

        return self

    def create_interactions(self):
        """Create interaction terms for regional analysis."""
        print("\n" + "="*80)
        print("CREATING INTERACTION TERMS")
        print("="*80)

        # Price × Region interactions
        if 'fixed_broad_price' in self.df.columns:
            self.df['price_x_eu'] = self.df['fixed_broad_price'] * self.df['is_eu']
            self.df['price_x_eap'] = self.df['fixed_broad_price'] * self.df['is_eap']
            print("  [OK] Created price_x_eu, price_x_eap")

        # Log price × Region interactions
        if 'log_fixed_broad_price' in self.df.columns:
            self.df['log_price_x_eu'] = self.df['log_fixed_broad_price'] * self.df['is_eu']
            self.df['log_price_x_eap'] = self.df['log_fixed_broad_price'] * self.df['is_eap']
            print("  [OK] Created log_price_x_eu, log_price_x_eap")

        return self

    def save_data(self):
        """Save analysis-ready dataset."""
        print("\n" + "="*80)
        print("SAVING ANALYSIS-READY DATA")
        print("="*80)

        self.df.to_excel(self.output_file, engine='openpyxl', index=False)

        print(f"\n[OK] Saved: {self.output_file}")
        print(f"  Rows: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")
        print(f"  Countries: {self.df['country'].nunique()}")
        print(f"  Years: {self.df['year'].min()}-{self.df['year'].max()}")

        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        key_vars = ['fixed_broadband_subs', 'internet_users_pct', 
                    'fixed_broad_price', 'gdp_per_capita']

        if all(v in self.df.columns for v in key_vars):
            print("\nKey variables:")
            print(self.df[key_vars].describe().round(2))

        return self

    def run(self):
        """Execute full data preparation pipeline."""
        print("="*80)
        print("DATA PREPARATION PIPELINE - FINAL VERSION")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        (self
         .load_data()
         .select_series()
         .handle_missing_data()
         .create_transformations()
         .create_lags()
         .create_interactions()
         .save_data())

        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE [OK]")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run 05_descriptive_stats.py")
        print("  2. Run 06_baseline_regression.py")
        print("  3. Run 07_iv_estimation.py")
        print("  4. Run 08_robustness_checks.py")

        return self.df


# Main execution
if __name__ == "__main__":
    prep = DataPreparation()
    df = prep.run()
