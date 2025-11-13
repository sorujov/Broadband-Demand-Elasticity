# code/analysis/07_iv_estimation.py
"""
================================================================================
IV/2SLS Estimation Script
================================================================================
Purpose: Address price endogeneity using instrumental variables
Author: Samir Orujov
Date: November 13, 2025

Instruments:
- Regulatory quality (supply-side shifter)
- Mobile broadband price (substitute price)

Tests:
- First-stage F-statistic (weak instruments)
- Hausman test (endogeneity)
- Sargan/Hansen J-test (overidentification)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Econometrics packages
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from scipy import stats

# Add parent directory to path
# sys.path.append(str(Path(__file__).parent.parent))
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from code.utils.config
try:
    from code.utils.config import (
        DATA_PROCESSED, RESULTS_REGRESSION, RESULTS_TABLES
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_PROCESSED, RESULTS_REGRESSION, RESULTS_TABLES
    )
class IVEstimation:
    """Instrumental variable estimation for price elasticity."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_REGRESSION
        self.tables_dir = RESULTS_TABLES

    def load_data(self):
        """Load clean dataset."""
        print("="*80)
        print("LOADING DATA FOR IV ESTIMATION")
        print("="*80)

        if not self.data_file.exists():
            print(f"\n✗ File not found: {self.data_file}")
            return None

        df = pd.read_csv(self.data_file)
        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Set multi-index for panel data
        if 'country' in df.columns and 'year' in df.columns:
            df = df.set_index(['country', 'year'])
            print("✓ Panel structure set: country × year")

        return df

    def check_instruments(self, df):
        """Check availability of instrumental variables."""
        print("\n" + "="*80)
        print("INSTRUMENT AVAILABILITY CHECK")
        print("="*80)

        instruments = {
            'regulatory_quality_estimate': 'Regulatory Quality (supply shifter)',
            'mobile_broad_price': 'Mobile Broadband Price (substitute)',
            'log_price': 'Log Price (endogenous variable)'
        }

        available_instruments = []

        for var, description in instruments.items():
            if var in df.columns:
                non_missing = df[var].notna().sum()
                print(f"  ✓ {var}: {description}")
                print(f"    Available: {non_missing} observations")
                available_instruments.append(var)
            else:
                print(f"  ✗ {var}: NOT FOUND")

        return available_instruments

    def first_stage_regression(self, df, instruments, dependent_var='log_bandwidth_use'):
        """Estimate first-stage regression."""
        print("\n" + "="*80)
        print("FIRST-STAGE REGRESSION")
        print("="*80)
        print("Dependent variable: log_price (endogenous)")
        print(f"Instruments: {instruments}")

        # Prepare variables
        exog_vars = [col for col in ['log_gdp_pc', 'internet_users_pct'] 
                     if col in df.columns]

        all_vars = ['log_price'] + instruments + exog_vars + [dependent_var]
        model_data = df[all_vars].dropna()

        print(f"\nObservations: {len(model_data)}")

        # First stage: log_price ~ instruments + controls
        X = sm.add_constant(model_data[instruments + exog_vars])
        y = model_data['log_price']

        first_stage = sm.OLS(y, X).fit(cov_type='HC1')

        print("\n" + first_stage.summary().as_text())

        # Test instrument strength (F-statistic)
        # Joint F-test for instruments
        instrument_params = [instruments[i] for i in range(len(instruments)) 
                           if instruments[i] in first_stage.params.index]

        if len(instrument_params) > 0:
            f_test_result = first_stage.f_test(instrument_params)
            # Extract F-statistic correctly
            if hasattr(f_test_result.fvalue, '__getitem__'):
                f_stat = f_test_result.fvalue[0][0] if isinstance(f_test_result.fvalue[0], (list, tuple, np.ndarray)) else f_test_result.fvalue[0]
            else:
                f_stat = f_test_result.fvalue
                
            print(f"\n{'='*60}")
            print(f"WEAK INSTRUMENT TEST")
            print(f"{'='*60}")
            print(f"F-statistic for instruments: {f_stat:.2f}")

            # Rule of thumb: F > 10 indicates strong instruments
            if f_stat > 10:
                print("✓ Instruments appear STRONG (F > 10)")
            else:
                print("⚠ WARNING: Instruments may be WEAK (F < 10)")

        return first_stage, model_data

    def estimate_iv2sls(self, df, instruments, dependent_var='log_bandwidth_use'):
        """Estimate IV/2SLS model."""
        print("\n" + "="*80)
        print("IV/2SLS ESTIMATION")
        print("="*80)
        print(f"Dependent variable: {dependent_var}")
        print(f"Endogenous: log_price")
        print(f"Instruments: {instruments}")

        # Prepare variables
        exog_vars = [col for col in ['log_gdp_pc', 'internet_users_pct'] 
                     if col in df.columns]

        all_vars = [dependent_var, 'log_price'] + instruments + exog_vars
        model_data = df[all_vars].dropna()

        print(f"\nObservations: {len(model_data)}")
        print(f"Entities: {model_data.index.get_level_values(0).nunique()}")

        # Reset index for IV2SLS
        model_data_reset = model_data.reset_index()

        # Prepare formula components
        # IV2SLS expects: exog = included exogenous vars, instruments = excluded instruments only
        endog = model_data_reset[['log_price']]
        exog = sm.add_constant(model_data_reset[exog_vars])
        instr = model_data_reset[instruments]  # Only excluded instruments, not all instruments

        # Estimate IV/2SLS
        iv_model = IV2SLS(
            dependent=model_data_reset[dependent_var],
            exog=exog,
            endog=endog,
            instruments=instr
        )

        iv_results = iv_model.fit(cov_type='robust')

        print("\n" + str(iv_results))

        # Extract elasticity
        if 'log_price' in iv_results.params.index:
            elasticity = iv_results.params['log_price']
            se = iv_results.std_errors['log_price']
            print(f"\n{'='*60}")
            print(f"IV PRICE ELASTICITY")
            print(f"{'='*60}")
            print(f"Coefficient: {elasticity:.3f}")
            print(f"Std. Error: {se:.3f}")
            print(f"95% CI: [{elasticity - 1.96*se:.3f}, {elasticity + 1.96*se:.3f}]")

        return iv_results

    def hausman_test(self, ols_result, iv_result):
        """Conduct Hausman test for endogeneity."""
        print("\n" + "="*80)
        print("HAUSMAN TEST FOR ENDOGENEITY")
        print("="*80)

        # Compare OLS and IV coefficients
        if 'log_price' in ols_result.params.index and 'log_price' in iv_result.params.index:
            ols_beta = ols_result.params['log_price']
            iv_beta = iv_result.params['log_price']

            ols_se = ols_result.bse['log_price'] if hasattr(ols_result, 'bse') else ols_result.std_errors['log_price']
            iv_se = iv_result.std_errors['log_price']

            diff = iv_beta - ols_beta
            se_diff = np.sqrt(iv_se**2 - ols_se**2) if iv_se**2 > ols_se**2 else np.sqrt(iv_se**2 + ols_se**2)

            t_stat = diff / se_diff
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))  # Approximate df

            print(f"\nOLS coefficient: {ols_beta:.3f} (SE: {ols_se:.3f})")
            print(f"IV coefficient:  {iv_beta:.3f} (SE: {iv_se:.3f})")
            print(f"Difference:      {diff:.3f}")
            print(f"\nTest statistic: {t_stat:.3f}")
            print(f"P-value:        {p_value:.4f}")

            if p_value < 0.05:
                print("\n✓ REJECT H₀: Price is ENDOGENOUS (use IV)")
            else:
                print("\n✗ FAIL TO REJECT H₀: Price may be exogenous (OLS OK)")

    def overidentification_test(self, iv_results):
        """Test overidentifying restrictions (if applicable)."""
        print("\n" + "="*80)
        print("OVERIDENTIFICATION TEST")
        print("="*80)

        # This requires more than one instrument
        # linearmodels provides J-statistic automatically
        if hasattr(iv_results, 'sargan'):
            j_stat = iv_results.sargan.stat
            p_value = iv_results.sargan.pval

            print(f"\nSargan J-statistic: {j_stat:.3f}")
            print(f"P-value:           {p_value:.4f}")

            if p_value > 0.05:
                print("\n✓ FAIL TO REJECT H₀: Instruments are valid")
            else:
                print("\n⚠ REJECT H₀: Instruments may be invalid")
        else:
            print("\n⚠ Test not available (need multiple instruments)")

    def compare_ols_iv(self, ols_result, iv_result):
        """Create comparison table of OLS vs IV."""
        print("\n" + "="*80)
        print("OLS vs IV COMPARISON")
        print("="*80)

        comparison_data = {
            'Variable': ['log_price', 'log_gdp_pc', 'internet_users_pct'],
            'OLS_Coef': [],
            'OLS_SE': [],
            'IV_Coef': [],
            'IV_SE': []
        }

        for var in comparison_data['Variable']:
            # OLS
            if var in ols_result.params.index:
                comparison_data['OLS_Coef'].append(f"{ols_result.params[var]:.4f}")
                se = ols_result.bse[var] if hasattr(ols_result, 'bse') else ols_result.std_errors[var]
                comparison_data['OLS_SE'].append(f"({se:.4f})")
            else:
                comparison_data['OLS_Coef'].append('N/A')
                comparison_data['OLS_SE'].append('')

            # IV
            if var in iv_result.params.index:
                comparison_data['IV_Coef'].append(f"{iv_result.params[var]:.4f}")
                comparison_data['IV_SE'].append(f"({iv_result.std_errors[var]:.4f})")
            else:
                comparison_data['IV_Coef'].append('N/A')
                comparison_data['IV_SE'].append('')

        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))

        # Save comparison
        comparison_file = self.tables_dir / 'ols_vs_iv_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n✓ Saved: {comparison_file}")

        return comparison_df

    def save_results(self, results, filename):
        """Save results to file."""
        output_file = self.output_dir / filename

        with open(output_file, 'w') as f:
            f.write(str(results))

        print(f"✓ Saved: {output_file}")


def main():
    """Main execution function."""
    print("="*80)
    print("IV/2SLS ESTIMATION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize estimator
        iv = IVEstimation()

        # Load data
        df = iv.load_data()
        if df is None:
            return

        # Check instruments
        available_instruments = iv.check_instruments(df)

        if len(available_instruments) < 2:  # Need at least endogenous var + 1 instrument
            print("\n⚠ Insufficient instruments for IV estimation")
            return

        # Remove endogenous variable from instrument list
        instruments = [x for x in available_instruments if x != 'log_price']

        if len(instruments) == 0:
            print("\n⚠ No valid instruments found")
            return

        print(f"\nUsing instruments: {instruments}")

        # First stage
        first_stage, model_data = iv.first_stage_regression(df, instruments)
        iv.save_results(first_stage, 'iv_first_stage.txt')

        # IV/2SLS estimation
        iv_results = iv.estimate_iv2sls(df, instruments)
        iv.save_results(iv_results, 'iv_2sls_results.txt')

        # For comparison, estimate OLS
        print("\n" + "="*80)
        print("OLS FOR COMPARISON")
        print("="*80)

        exog_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct'] 
                     if col in df.columns]
        model_data_ols = df[['log_bandwidth_use'] + exog_vars].dropna()

        X_ols = sm.add_constant(model_data_ols[exog_vars])
        y_ols = model_data_ols['log_bandwidth_use']
        ols_results = sm.OLS(y_ols, X_ols).fit(cov_type='HC1')

        print(ols_results.summary())
        iv.save_results(ols_results, 'ols_for_comparison.txt')

        # Hausman test
        iv.hausman_test(ols_results, iv_results)

        # Overidentification test
        if len(instruments) > 1:
            iv.overidentification_test(iv_results)

        # Compare OLS vs IV
        iv.compare_ols_iv(ols_results, iv_results)

        print("\n" + "="*80)
        print("IV/2SLS ESTIMATION COMPLETE ✓")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  {RESULTS_REGRESSION}")
        print(f"  {RESULTS_TABLES}")
        print("\nNext step: Robustness checks and results compilation")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
