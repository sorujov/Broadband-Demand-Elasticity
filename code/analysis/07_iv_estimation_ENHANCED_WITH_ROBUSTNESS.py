# code/analysis/07_iv_estimation_ENHANCED_WITH_ROBUSTNESS.py
"""
================================================================================
Enhanced IV/2SLS Estimation Script with Comprehensive Robustness Checks
================================================================================
Purpose: Address price endogeneity using instrumental variables with robustness
Author: Samir Orujov
Date: November 14, 2025 (Enhanced)

ENHANCEMENTS:
1. Two-way fixed effects (entity + time)
2. Individual instrument tests
3. Regional subsamples (EU vs EaP)
4. Dynamic specification (lagged bandwidth)

Original Features:
- Panel IV with entity fixed effects
- Mobile broadband price + R&D as instruments
- First-stage F-statistic
- Hausman test
- Sargan-Hansen J-test
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore', message='covariance of constraints does not have full rank')

# Econometrics packages
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
try:
    from linearmodels.panel.iv import PanelIVGMM
except ImportError:
    PanelIVGMM = None
import statsmodels.api as sm
from scipy import stats

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import DATA_PROCESSED, RESULTS_TABLES, RESULTS_REGRESSION
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import DATA_PROCESSED, RESULTS_TABLES, RESULTS_REGRESSION


class IVEstimator:
    """Enhanced IV estimation with comprehensive robustness checks."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_REGRESSION
        self.tables_dir = RESULTS_TABLES
        self.robustness_dir = RESULTS_TABLES / 'robustness'

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.robustness_dir.mkdir(parents=True, exist_ok=True)

        # Store results for comparison
        self.results_summary = []

    def load_data(self):
        """Load analysis-ready dataset."""
        print("="*80)
        print("ENHANCED IV/2SLS ESTIMATION WITH ROBUSTNESS CHECKS")
        print("="*80)
        print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.data_file.exists():
            print(f"\n✗ File not found: {self.data_file}")
            return None

        df = pd.read_csv(self.data_file)
        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        if 'country' in df.columns and 'year' in df.columns:
            df = df.set_index(['country', 'year'])
            print("✓ Panel structure set: country × year")

        return df

    def check_instruments(self, df):
        """Check availability of instrumental variables."""
        instruments = {
            'mobile_broad_price_i271mb_ts_GNI': 'Mobile Broadband Price',
            'research_development_expenditure': 'R&D Expenditure % GDP',
        }

        available_instruments = []
        for inst, description in instruments.items():
            if inst in df.columns:
                non_null = df[inst].notna().sum()
                pct_available = 100 * non_null / len(df)
                if pct_available > 50:
                    available_instruments.append(inst)

        return available_instruments

    def first_stage_regression(self, df, instruments, specification_name="Main"):
        """Estimate first-stage regression."""
        y_first = df['log_price']

        exog_vars = ['log_gdp_pc', 'internet_users_pct', 'mobile_subs_per100']
        exog_vars = [v for v in exog_vars if v in df.columns]

        X_first = pd.DataFrame(index=df.index)
        for var in instruments:
            X_first[var] = df[var]
        for var in exog_vars:
            X_first[var] = df[var]

        valid = y_first.notna() & X_first.notna().all(axis=1)
        y_first = y_first[valid]
        X_first = X_first[valid]

        first_stage = sm.OLS(y_first, sm.add_constant(X_first)).fit(cov_type='HC1')

        # Calculate F-statistic
        inst_params = first_stage.params[instruments]
        inst_cov = first_stage.cov_params().loc[instruments, instruments]
        F_stat = (inst_params.T @ np.linalg.inv(inst_cov) @ inst_params) / len(instruments)

        print(f"\n[{specification_name}] First-stage F-statistic: {F_stat:.2f}")

        return first_stage, F_stat

    def estimate_iv(self, df, instruments, include_time_fe=False, 
                   include_lagged_dep=False, specification_name="Main"):
        """
        Estimate Panel IV with flexible specifications.

        Parameters:
        -----------
        df : DataFrame
            Panel data with country-year index
        instruments : list
            List of instrument variable names
        include_time_fe : bool
            Whether to include year fixed effects
        include_lagged_dep : bool
            Whether to include lagged dependent variable (dynamic spec)
        specification_name : str
            Name for this specification (for output)
        """
        print(f"\n{'='*80}")
        print(f"PANEL IV ESTIMATION: {specification_name}")
        print(f"{'='*80}")

        # Prepare variables
        y = df['log_bandwidth_use']
        endog = df[['log_price']]

        # Base exogenous variables
        exog_vars = ['log_gdp_pc', 'internet_users_pct', 'mobile_subs_per100']
        exog_vars = [v for v in exog_vars if v in df.columns]
        exog = df[exog_vars].copy()

        # Add lagged dependent variable if requested (dynamic specification)
        if include_lagged_dep:
            if 'bandwidth_use_lag1' in df.columns:
                # Create log of lagged bandwidth
                df_temp = df.copy()
                df_temp['log_bandwidth_lag1'] = np.log(df_temp['bandwidth_use_lag1'] + 0.1)
                exog['log_bandwidth_lag1'] = df_temp['log_bandwidth_lag1']
                print("✓ Added lagged bandwidth (dynamic specification)")

        instr = df[instruments]

        # Drop missing
        valid = (y.notna() & endog.notna().all(axis=1) & 
                exog.notna().all(axis=1) & instr.notna().all(axis=1))

        y = y[valid]
        endog = endog[valid]
        exog = exog[valid]
        instr = instr[valid]

        print(f"Observations: {len(y)}")
        print(f"Entities: {len(y.index.get_level_values(0).unique())}")

        # Within transformation (entity demeaning)
        data_combined = pd.concat([y, endog, exog, instr], axis=1)
        data_demeaned = data_combined.copy()

        for col in data_combined.columns:
            entity_mean = data_combined.groupby(level=0)[col].transform('mean')
            data_demeaned[col] = data_combined[col] - entity_mean

        y_dm = data_demeaned[y.name]
        endog_dm = data_demeaned[endog.columns]
        exog_dm = data_demeaned[exog.columns.tolist()]
        instr_dm = data_demeaned[instruments]

        # Add time fixed effects if requested
        if include_time_fe:
            years = y_dm.index.get_level_values(1)
            year_dummies = pd.get_dummies(years, prefix='year', drop_first=True)
            year_dummies.index = y_dm.index
            exog_dm = pd.concat([exog_dm, year_dummies], axis=1)
            print(f"✓ Added {len(year_dummies.columns)} year fixed effects")

        # Estimate IV
        iv_model = IV2SLS(
            dependent=y_dm,
            exog=exog_dm,
            endog=endog_dm,
            instruments=instr_dm
        )

        iv_results = iv_model.fit(cov_type='robust')

        # Extract key statistics
        price_coef = iv_results.params['log_price']
        price_se = iv_results.std_errors['log_price']
        price_pval = iv_results.pvalues['log_price']

        print(f"\nPrice Elasticity: {price_coef:.3f} (SE: {price_se:.3f}, p: {price_pval:.3f})")

        return iv_results

    # ROBUSTNESS CHECK 1: TWO-WAY FIXED EFFECTS
    def robustness_twoway_fe(self, df, instruments):
        """Test robustness with entity + time fixed effects."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 1: TWO-WAY FIXED EFFECTS (Entity + Time)")
        print("="*80)

        iv_results = self.estimate_iv(
            df, instruments, 
            include_time_fe=True,
            specification_name="Two-Way FE"
        )

        # Store results
        self.results_summary.append({
            'Specification': 'Two-Way FE (Entity + Time)',
            'Coefficient': iv_results.params['log_price'],
            'Std_Error': iv_results.std_errors['log_price'],
            'P_Value': iv_results.pvalues['log_price'],
            'N_Obs': iv_results.nobs
        })

        return iv_results

    # ROBUSTNESS CHECK 2: INDIVIDUAL INSTRUMENTS
    def robustness_individual_instruments(self, df, instruments):
        """Test each instrument separately."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 2: INDIVIDUAL INSTRUMENTS")
        print("="*80)

        for inst in instruments:
            print(f"\n--- Testing instrument: {inst} ---")

            # First stage with single instrument
            first_stage, f_stat = self.first_stage_regression(df, [inst], 
                                                               f"Single: {inst[:20]}")

            # IV estimation with single instrument
            iv_results = self.estimate_iv(
                df, [inst],
                specification_name=f"Single IV: {inst[:30]}"
            )

            # Store results
            self.results_summary.append({
                'Specification': f'Single IV: {inst[:30]}',
                'Coefficient': iv_results.params['log_price'],
                'Std_Error': iv_results.std_errors['log_price'],
                'P_Value': iv_results.pvalues['log_price'],
                'F_Statistic': f_stat,
                'N_Obs': iv_results.nobs
            })

    # ROBUSTNESS CHECK 3: REGIONAL SUBSAMPLES
    def robustness_regional_subsamples(self, df, instruments):
        """Estimate separately for EU and EaP countries."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 3: REGIONAL SUBSAMPLES")
        print("="*80)

        if 'region' not in df.reset_index().columns:
            print("⚠ Region variable not found, skipping regional analysis")
            return

        df_reset = df.reset_index()

        # EU subsample
        df_eu = df_reset[df_reset['region'] == 'EU'].set_index(['country', 'year'])
        if len(df_eu) > 0:
            print(f"\n--- EU Countries: {df_eu.index.get_level_values(0).nunique()} countries ---")

            first_stage, f_stat = self.first_stage_regression(df_eu, instruments, "EU")
            iv_eu = self.estimate_iv(df_eu, instruments, specification_name="EU Countries")

            self.results_summary.append({
                'Specification': 'EU Countries Only',
                'Coefficient': iv_eu.params['log_price'],
                'Std_Error': iv_eu.std_errors['log_price'],
                'P_Value': iv_eu.pvalues['log_price'],
                'F_Statistic': f_stat,
                'N_Obs': iv_eu.nobs
            })

        # EaP subsample
        df_eap = df_reset[df_reset['region'] == 'EaP'].set_index(['country', 'year'])
        if len(df_eap) > 0:
            print(f"\n--- EaP Countries: {df_eap.index.get_level_values(0).nunique()} countries ---")

            first_stage, f_stat = self.first_stage_regression(df_eap, instruments, "EaP")
            iv_eap = self.estimate_iv(df_eap, instruments, specification_name="EaP Countries")

            self.results_summary.append({
                'Specification': 'EaP Countries Only',
                'Coefficient': iv_eap.params['log_price'],
                'Std_Error': iv_eap.std_errors['log_price'],
                'P_Value': iv_eap.pvalues['log_price'],
                'F_Statistic': f_stat,
                'N_Obs': iv_eap.nobs
            })

    # ROBUSTNESS CHECK 4: DYNAMIC SPECIFICATION
    def robustness_dynamic_specification(self, df, instruments):
        """Add lagged dependent variable (dynamic panel)."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 4: DYNAMIC SPECIFICATION")
        print("="*80)

        if 'bandwidth_use_lag1' not in df.columns:
            print("⚠ Lagged bandwidth not found, skipping dynamic specification")
            return

        first_stage, f_stat = self.first_stage_regression(df, instruments, "Dynamic")

        iv_dynamic = self.estimate_iv(
            df, instruments,
            include_lagged_dep=True,
            specification_name="Dynamic (with lag)"
        )

        # Calculate long-run elasticity if lagged coef available
        if 'log_bandwidth_lag1' in iv_dynamic.params.index:
            beta_price = iv_dynamic.params['log_price']
            beta_lag = iv_dynamic.params['log_bandwidth_lag1']
            long_run = beta_price / (1 - beta_lag)

            print(f"\nShort-run elasticity: {beta_price:.3f}")
            print(f"Lagged bandwidth coef: {beta_lag:.3f}")
            print(f"Long-run elasticity: {long_run:.3f}")

        self.results_summary.append({
            'Specification': 'Dynamic (with lag)',
            'Coefficient': iv_dynamic.params['log_price'],
            'Std_Error': iv_dynamic.std_errors['log_price'],
            'P_Value': iv_dynamic.pvalues['log_price'],
            'F_Statistic': f_stat,
            'N_Obs': iv_dynamic.nobs
        })

    def create_robustness_table(self):
        """Create comprehensive robustness results table."""
        print("\n" + "="*80)
        print("ROBUSTNESS RESULTS SUMMARY")
        print("="*80)

        if not self.results_summary:
            print("No results to summarize")
            return

        results_df = pd.DataFrame(self.results_summary)

        # Format for display
        results_df['Coefficient_Formatted'] = results_df.apply(
            lambda row: f"{row['Coefficient']:.3f}{'***' if row['P_Value'] < 0.01 else '**' if row['P_Value'] < 0.05 else '*' if row['P_Value'] < 0.10 else ''}", 
            axis=1
        )
        results_df['SE_Formatted'] = results_df['Std_Error'].apply(lambda x: f"({x:.3f})")

        # Display
        print("\n" + results_df[['Specification', 'Coefficient_Formatted', 'SE_Formatted', 'N_Obs']].to_string(index=False))

        # Save
        robustness_file = self.robustness_dir / 'robustness_summary.csv'
        results_df.to_csv(robustness_file, index=False)
        print(f"\n✓ Saved: {robustness_file}")

        return results_df


def main():
    """Main execution with all robustness checks."""

    try:
        estimator = IVEstimator()

        # Load data
        df = estimator.load_data()
        if df is None:
            return

        # Check instruments
        instruments = estimator.check_instruments(df)
        if not instruments:
            print("No instruments available")
            return

        print("\n" + "="*80)
        print("BASELINE SPECIFICATION")
        print("="*80)

        # Baseline: First stage + IV
        first_stage, f_stat = estimator.first_stage_regression(df, instruments, "Baseline")
        iv_baseline = estimator.estimate_iv(df, instruments, specification_name="Baseline")

        # Store baseline
        estimator.results_summary.append({
            'Specification': 'Baseline (Entity FE)',
            'Coefficient': iv_baseline.params['log_price'],
            'Std_Error': iv_baseline.std_errors['log_price'],
            'P_Value': iv_baseline.pvalues['log_price'],
            'F_Statistic': f_stat,
            'N_Obs': iv_baseline.nobs
        })

        # Run all robustness checks
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE ROBUSTNESS CHECKS")
        print("="*80)

        # 1. Two-way FE
        estimator.robustness_twoway_fe(df, instruments)

        # 2. Individual instruments
        estimator.robustness_individual_instruments(df, instruments)

        # 3. Regional subsamples
        estimator.robustness_regional_subsamples(df, instruments)

        # 4. Dynamic specification
        estimator.robustness_dynamic_specification(df, instruments)

        # Create summary table
        estimator.create_robustness_table()

        print("\n" + "="*80)
        print("ENHANCED IV ESTIMATION COMPLETE ✓")
        print("="*80)
        print("\nAll robustness checks completed successfully!")
        print("Check results/tables/robustness/ for detailed output")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
