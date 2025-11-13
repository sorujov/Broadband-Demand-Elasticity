# code/analysis/08_robustness_checks.py
"""
================================================================================
Robustness Checks Script
================================================================================
Purpose: Test the robustness of baseline results through alternative specifications
Author: Samir Orujov
Date: November 13, 2025

Robustness checks:
1. Alternative dependent variables (subscriptions vs bandwidth)
2. Different time periods (exclude crisis years)
3. Alternative control variables
4. Subsample analysis (EU only, EaP only)
5. Outlier treatment
6. Alternative functional forms
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Econometrics packages
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import (
        DATA_PROCESSED, RESULTS_ROBUSTNESS, RESULTS_TABLES
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_PROCESSED, RESULTS_ROBUSTNESS, RESULTS_TABLES
    )

class RobustnessChecks:
    """Conduct robustness checks on main results."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_ROBUSTNESS
        self.tables_dir = RESULTS_TABLES

    def load_data(self):
        """Load clean dataset."""
        print("="*80)
        print("LOADING DATA FOR ROBUSTNESS CHECKS")
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

    def check1_alternative_dependent_vars(self, df):
        """Test with alternative dependent variables."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 1: ALTERNATIVE DEPENDENT VARIABLES")
        print("="*80)

        # Define dependent variables
        dep_vars = {
            'log_bandwidth_use': 'Bandwidth Usage (baseline)',
            'log_bb_subs': 'Broadband Subscriptions',
            'bb_subs_per100': 'Subscriptions per 100 (levels)'
        }

        # Filter available variables
        dep_vars = {k: v for k, v in dep_vars.items() if k in df.columns}

        if len(dep_vars) < 2:
            print("\n⚠ Not enough alternative dependent variables")
            return None

        print(f"\nTesting {len(dep_vars)} dependent variables:")
        for var, desc in dep_vars.items():
            print(f"  - {var}: {desc}")

        # Estimate model for each dependent variable
        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct']
                  if col in df.columns]

        results = {}

        for dep_var, desc in dep_vars.items():
            print(f"\nEstimating with {dep_var}...")

            try:
                model_data = df[[dep_var] + X_vars].dropna()

                mod = PanelOLS(
                    model_data[dep_var],
                    model_data[X_vars],
                    entity_effects=True,
                    time_effects=True
                )

                result = mod.fit(cov_type='clustered', cluster_entity=True)
                results[desc] = result

                if 'log_price' in result.params.index:
                    elasticity = result.params['log_price']
                    se = result.std_errors['log_price']
                    print(f"  Price coefficient: {elasticity:.4f} (SE: {se:.4f})")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Create comparison table
        if len(results) > 1:
            self._save_comparison_table(results, 'robustness_alternative_depvars.txt')

        return results

    def check2_time_periods(self, df):
        """Test different time periods."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 2: ALTERNATIVE TIME PERIODS")
        print("="*80)

        # Reset index to access year
        df_reset = df.reset_index()

        # Define time periods
        periods = {
            'Full sample': (2010, 2023),
            'Pre-COVID': (2010, 2019),
            'Post-crisis': (2012, 2023),
            'Recent years': (2015, 2023)
        }

        print("\nTesting time periods:")
        for name, (start, end) in periods.items():
            print(f"  - {name}: {start}-{end}")

        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct']
                  if col in df_reset.columns]
        dep_var = 'log_bandwidth_use' if 'log_bandwidth_use' in df_reset.columns else 'log_bb_subs'

        results = {}

        for period_name, (start_year, end_year) in periods.items():
            print(f"\nEstimating {period_name}...")

            try:
                # Filter data
                df_period = df_reset[
                    (df_reset['year'] >= start_year) & 
                    (df_reset['year'] <= end_year)
                ].copy()

                df_period = df_period.set_index(['country', 'year'])
                model_data = df_period[[dep_var] + X_vars].dropna()

                print(f"  Observations: {len(model_data)}")

                mod = PanelOLS(
                    model_data[dep_var],
                    model_data[X_vars],
                    entity_effects=True,
                    time_effects=True
                )

                result = mod.fit(cov_type='clustered', cluster_entity=True)
                results[period_name] = result

                if 'log_price' in result.params.index:
                    elasticity = result.params['log_price']
                    se = result.std_errors['log_price']
                    print(f"  Price coefficient: {elasticity:.4f} (SE: {se:.4f})")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Create comparison table
        if len(results) > 1:
            self._save_comparison_table(results, 'robustness_time_periods.txt')

        return results

    def check3_subsample_analysis(self, df):
        """Separate analysis for EU and EaP countries."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 3: SUBSAMPLE ANALYSIS")
        print("="*80)

        # Reset index to access region
        df_reset = df.reset_index()

        if 'region' not in df_reset.columns:
            print("\n⚠ Region variable not found")
            return None

        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct']
                  if col in df_reset.columns]
        dep_var = 'log_bandwidth_use' if 'log_bandwidth_use' in df_reset.columns else 'log_bb_subs'

        results = {}

        for region in ['EU', 'EaP']:
            print(f"\nEstimating {region} countries only...")

            try:
                # Filter data
                df_region = df_reset[df_reset['region'] == region].copy()
                df_region = df_region.set_index(['country', 'year'])

                model_data = df_region[[dep_var] + X_vars].dropna()

                print(f"  Countries: {model_data.index.get_level_values(0).nunique()}")
                print(f"  Observations: {len(model_data)}")

                mod = PanelOLS(
                    model_data[dep_var],
                    model_data[X_vars],
                    entity_effects=True,
                    time_effects=True
                )

                result = mod.fit(cov_type='clustered', cluster_entity=True)
                results[f'{region} only'] = result

                if 'log_price' in result.params.index:
                    elasticity = result.params['log_price']
                    se = result.std_errors['log_price']
                    print(f"  Price elasticity: {elasticity:.4f} (SE: {se:.4f})")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Create comparison table
        if len(results) > 1:
            self._save_comparison_table(results, 'robustness_subsamples.txt')

        return results

    def check4_outlier_treatment(self, df):
        """Test robustness to outliers."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 4: OUTLIER TREATMENT")
        print("="*80)

        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct']
                  if col in df.columns]
        dep_var = 'log_bandwidth_use' if 'log_bandwidth_use' in df.columns else 'log_bb_subs'

        print("\nTesting outlier treatments:")
        print("  - Baseline (no treatment)")
        print("  - Winsorize at 1st and 99th percentiles")
        print("  - Trim extreme 1% on each tail")

        results = {}

        # Baseline
        try:
            print("\n1. Baseline (no treatment)...")
            model_data = df[[dep_var] + X_vars].dropna()

            mod = PanelOLS(
                model_data[dep_var],
                model_data[X_vars],
                entity_effects=True,
                time_effects=True
            )
            result = mod.fit(cov_type='clustered', cluster_entity=True)
            results['No treatment'] = result

            if 'log_price' in result.params.index:
                print(f"  Price: {result.params['log_price']:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Winsorize
        try:
            print("\n2. Winsorized data...")
            df_winsor = df.copy()

            for var in [dep_var] + X_vars:
                if var in df_winsor.columns:
                    lower = df_winsor[var].quantile(0.01)
                    upper = df_winsor[var].quantile(0.99)
                    df_winsor[var] = df_winsor[var].clip(lower=lower, upper=upper)

            model_data = df_winsor[[dep_var] + X_vars].dropna()

            mod = PanelOLS(
                model_data[dep_var],
                model_data[X_vars],
                entity_effects=True,
                time_effects=True
            )
            result = mod.fit(cov_type='clustered', cluster_entity=True)
            results['Winsorized 1-99%'] = result

            if 'log_price' in result.params.index:
                print(f"  Price: {result.params['log_price']:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Trimmed
        try:
            print("\n3. Trimmed data...")
            df_trim = df.copy()

            # Identify outliers based on price
            if 'log_price' in df_trim.columns:
                lower = df_trim['log_price'].quantile(0.01)
                upper = df_trim['log_price'].quantile(0.99)
                df_trim = df_trim[
                    (df_trim['log_price'] >= lower) & 
                    (df_trim['log_price'] <= upper)
                ]

            model_data = df_trim[[dep_var] + X_vars].dropna()
            print(f"  Observations after trim: {len(model_data)}")

            mod = PanelOLS(
                model_data[dep_var],
                model_data[X_vars],
                entity_effects=True,
                time_effects=True
            )
            result = mod.fit(cov_type='clustered', cluster_entity=True)
            results['Trimmed 1-99%'] = result

            if 'log_price' in result.params.index:
                print(f"  Price: {result.params['log_price']:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Create comparison table
        if len(results) > 1:
            self._save_comparison_table(results, 'robustness_outliers.txt')

        return results

    def check5_alternative_controls(self, df):
        """Test with different sets of control variables."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECK 5: ALTERNATIVE CONTROL VARIABLES")
        print("="*80)

        dep_var = 'log_bandwidth_use' if 'log_bandwidth_use' in df.columns else 'log_bb_subs'

        # Define control variable sets
        control_sets = {
            'Minimal': ['log_price'],
            'Baseline': ['log_price', 'log_gdp_pc'],
            'Extended': ['log_price', 'log_gdp_pc', 'internet_users_pct'],
            'Full': ['log_price', 'log_gdp_pc', 'internet_users_pct', 
                    'urban_population_pct', 'population_density']
        }

        # Filter to available variables
        control_sets = {
            name: [v for v in vars_list if v in df.columns]
            for name, vars_list in control_sets.items()
        }

        print("\nControl variable sets:")
        for name, vars_list in control_sets.items():
            print(f"  {name}: {vars_list}")

        results = {}

        for set_name, X_vars in control_sets.items():
            if len(X_vars) == 0:
                continue

            print(f"\nEstimating with {set_name} controls...")

            try:
                model_data = df[[dep_var] + X_vars].dropna()

                mod = PanelOLS(
                    model_data[dep_var],
                    model_data[X_vars],
                    entity_effects=True,
                    time_effects=True
                )

                result = mod.fit(cov_type='clustered', cluster_entity=True)
                results[set_name] = result

                if 'log_price' in result.params.index:
                    elasticity = result.params['log_price']
                    se = result.std_errors['log_price']
                    print(f"  Price: {elasticity:.4f} (SE: {se:.4f})")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Create comparison table
        if len(results) > 1:
            self._save_comparison_table(results, 'robustness_controls.txt')

        return results

    def _save_comparison_table(self, results_dict, filename):
        """Save comparison table of results."""
        try:
            output_file = self.output_dir / filename
            
            with open(output_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("ROBUSTNESS CHECK COMPARISON\n")
                f.write("="*80 + "\n\n")
                
                # Extract common parameters
                all_params = set()
                for result in results_dict.values():
                    all_params.update(result.params.index)
                
                # Write header
                f.write(f"{'Variable':<25}")
                for model_name in results_dict.keys():
                    f.write(f"{model_name:<20}")
                f.write("\n" + "-"*80 + "\n")
                
                # Write coefficients and standard errors
                for param in sorted(all_params):
                    f.write(f"{param:<25}")
                    for result in results_dict.values():
                        if param in result.params.index:
                            coef = result.params[param]
                            # Handle both .std_errors and .bse
                            if hasattr(result, 'std_errors'):
                                se = result.std_errors[param]
                            elif hasattr(result, 'bse'):
                                se = result.bse[param]
                            else:
                                se = 0
                            
                            # Significance stars
                            pval = result.pvalues[param] if param in result.pvalues.index else 1
                            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                            
                            f.write(f"{coef:>8.4f}{stars:<3}")
                            f.write(f"({se:>6.4f})     ")
                        else:
                            f.write(" "*20)
                    f.write("\n")
                
                # Write summary statistics
                f.write("\n" + "-"*80 + "\n")
                f.write(f"{'N':<25}")
                for result in results_dict.values():
                    f.write(f"{int(result.nobs):<20}")
                f.write("\n")
                
                f.write(f"{'R-squared':<25}")
                for result in results_dict.values():
                    rsq = result.rsquared if hasattr(result, 'rsquared') else 0
                    f.write(f"{rsq:<20.4f}")
                f.write("\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Note: *** p<0.01, ** p<0.05, * p<0.10\n")

            print(f"\n✓ Saved comparison: {output_file}")

        except Exception as e:
            print(f"\n⚠ Could not create comparison table: {e}")
            import traceback
            traceback.print_exc()

    def create_summary_table(self):
        """Create summary of all robustness checks."""
        print("\n" + "="*80)
        print("ROBUSTNESS CHECKS SUMMARY")
        print("="*80)

        summary_file = self.tables_dir / 'robustness_summary.txt'

        summary_text = """
ROBUSTNESS CHECKS SUMMARY
========================================================================

This document summarizes the robustness checks conducted to validate
the main price elasticity estimates.

CHECKS PERFORMED:
----------------
1. Alternative dependent variables
   - Bandwidth usage (baseline)
   - Broadband subscriptions
   - Subscriptions per 100 (levels)

2. Alternative time periods
   - Full sample (2010-2023)
   - Pre-COVID (2010-2019)
   - Post-crisis (2012-2023)
   - Recent years (2015-2023)

3. Subsample analysis
   - EU countries only
   - EaP countries only

4. Outlier treatment
   - No treatment (baseline)
   - Winsorized at 1st and 99th percentiles
   - Trimmed extreme 1%

5. Alternative control variables
   - Minimal controls (price only)
   - Baseline controls (price + GDP)
   - Extended controls (+ internet users)
   - Full controls (+ urbanization + density)

INTERPRETATION GUIDE:
--------------------
- If results are consistent across specifications: 
  => Main findings are ROBUST

- If results vary significantly:
  => Sensitivity to specification choice
  => Report range of estimates
  => Discuss which specification is preferred

RECOMMENDED REPORTING:
---------------------
In your paper:
- Report baseline results in main table
- Report robustness checks in appendix
- Discuss any meaningful variations
- Emphasize consistency (if present)

See individual output files for detailed results.
"""

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"✓ Summary saved: {summary_file}")


def main():
    """Main execution function."""
    print("="*80)
    print("ROBUSTNESS CHECKS")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize robustness checker
        robust = RobustnessChecks()

        # Load data
        df = robust.load_data()
        if df is None:
            return

        # Run robustness checks
        print("\nRunning 5 robustness checks...")
        print("This may take several minutes...")

        # Check 1: Alternative dependent variables
        robust.check1_alternative_dependent_vars(df)

        # Check 2: Alternative time periods
        robust.check2_time_periods(df)

        # Check 3: Subsample analysis
        robust.check3_subsample_analysis(df)

        # Check 4: Outlier treatment
        robust.check4_outlier_treatment(df)

        # Check 5: Alternative controls
        robust.check5_alternative_controls(df)

        # Create summary
        robust.create_summary_table()

        print("\n" + "="*80)
        print("ROBUSTNESS CHECKS COMPLETE ✓")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  {RESULTS_ROBUSTNESS}")
        print(f"  {RESULTS_TABLES}")
        print("\nNext step: Compile results for publication")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
