"""
================================================================================
Baseline Regression Analysis
================================================================================
Purpose: Estimate baseline price elasticity using panel regression models
Author: Samir Orujov
Date: November 13, 2025

Models:
1. Pooled OLS
2. Fixed Effects (country)
3. Fixed Effects (country + time)
4. Regional heterogeneity (EU vs EaP)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Econometrics packages
from linearmodels.panel import PanelOLS
from linearmodels import PooledOLS
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Add project root to path for imports
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

class BaselineRegression:
    """Estimate baseline price elasticity models."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_REGRESSION
        self.tables_dir = RESULTS_TABLES

    def load_data(self):
        """Load clean dataset and prepare for regression."""
        print("="*80)
        print("LOADING DATA FOR REGRESSION")
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

    def check_variables(self, df):
        """Check that required variables exist."""
        print("\n" + "="*80)
        print("VARIABLE CHECK")
        print("="*80)

        required_vars = {
            'Dependent': ['log_bandwidth_use', 'log_bb_subs'],
            'Price': ['log_price'],
            'Controls': ['log_gdp_pc', 'internet_users_pct'],
            'Regional': ['is_eap', 'log_price_x_eap']
        }

        missing_vars = []
        for category, vars_list in required_vars.items():
            print(f"\n{category} variables:")
            for var in vars_list:
                if var in df.columns:
                    non_missing = df[var].notna().sum()
                    print(f"  ✓ {var} ({non_missing} obs)")
                else:
                    print(f"  ✗ {var} - MISSING")
                    missing_vars.append(var)

        if missing_vars:
            print(f"\n⚠ Warning: {len(missing_vars)} required variables missing")

        return len(missing_vars) == 0

    def estimate_pooled_ols(self, df, dependent_var='log_bandwidth_use'):
        """Estimate pooled OLS model."""
        print("\n" + "="*80)
        print(f"MODEL 1: POOLED OLS")
        print(f"Dependent variable: {dependent_var}")
        print("="*80)

        # Prepare variables
        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct',
                                   'time_trend'] if col in df.columns]

        # Drop missing
        model_data = df[[dependent_var] + X_vars].dropna()

        print(f"\nObservations: {len(model_data)}")
        print(f"Variables: {X_vars}")

        # Estimate
        y = model_data[dependent_var]
        X = sm.add_constant(model_data[X_vars])

        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')  # Robust standard errors

        print("\n" + results.summary().as_text())

        # Extract key coefficient
        if 'log_price' in results.params.index:
            elasticity = results.params['log_price']
            se = results.bse['log_price']
            print(f"\n✓ Price elasticity: {elasticity:.3f} (SE: {se:.3f})")

        return results

    def estimate_fixed_effects(self, df, dependent_var='log_bandwidth_use', 
                               entity_effects=True, time_effects=False):
        """Estimate fixed effects model."""

        model_name = "Fixed Effects"
        if entity_effects and time_effects:
            model_name = "Two-Way Fixed Effects"
        elif entity_effects:
            model_name = "Fixed Effects (Country)"
        elif time_effects:
            model_name = "Fixed Effects (Time)"

        print("\n" + "="*80)
        print(f"MODEL: {model_name}")
        print(f"Dependent variable: {dependent_var}")
        print("="*80)

        # Prepare variables
        X_vars = [col for col in ['log_price', 'log_gdp_pc', 'internet_users_pct']
                  if col in df.columns]

        # Drop missing
        model_data = df[[dependent_var] + X_vars].dropna()

        print(f"\nObservations: {len(model_data)}")
        print(f"Entities: {model_data.index.get_level_values(0).nunique()}")
        print(f"Variables: {X_vars}")

        # Estimate
        mod = PanelOLS(
            model_data[dependent_var],
            model_data[X_vars],
            entity_effects=entity_effects,
            time_effects=time_effects
        )

        results = mod.fit(cov_type='clustered', cluster_entity=True)

        print("\n" + str(results))

        # Extract key coefficient
        if 'log_price' in results.params.index:
            elasticity = results.params['log_price']
            se = results.std_errors['log_price']
            print(f"\n✓ Price elasticity: {elasticity:.3f} (SE: {se:.3f})")

        return results

    def estimate_regional_heterogeneity(self, df, dependent_var='log_bandwidth_use'):
        """Estimate model with regional interaction term."""
        print("\n" + "="*80)
        print(f"MODEL: REGIONAL HETEROGENEITY (EU vs EaP)")
        print(f"Dependent variable: {dependent_var}")
        print("="*80)

        # Prepare variables
        X_vars = [col for col in ['log_price', 'log_price_x_eap', 'log_gdp_pc', 
                                   'internet_users_pct'] if col in df.columns]

        # Drop missing
        model_data = df[[dependent_var] + X_vars].dropna()

        print(f"\nObservations: {len(model_data)}")
        print(f"Entities: {model_data.index.get_level_values(0).nunique()}")
        print(f"Variables: {X_vars}")

        # Estimate with two-way fixed effects
        mod = PanelOLS(
            model_data[dependent_var],
            model_data[X_vars],
            entity_effects=True,
            time_effects=True
        )

        results = mod.fit(cov_type='clustered', cluster_entity=True)

        print("\n" + str(results))

        # Calculate elasticities
        if 'log_price' in results.params.index:
            beta1 = results.params['log_price']
            se1 = results.std_errors['log_price']

            print(f"\n✓ EU elasticity (β₁): {beta1:.3f} (SE: {se1:.3f})")

            if 'log_price_x_eap' in results.params.index:
                beta2 = results.params['log_price_x_eap']
                se2 = results.std_errors['log_price_x_eap']

                eap_elasticity = beta1 + beta2
                print(f"✓ EaP elasticity (β₁+β₂): {eap_elasticity:.3f}")
                print(f"✓ Difference (β₂): {beta2:.3f} (SE: {se2:.3f})")

                # Test if difference is significant
                p_value = results.pvalues['log_price_x_eap']
                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"  Significance: p={p_value:.4f} {sig}")

        return results

    def compare_models(self, results_dict):
        """Create comparison table of all models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)

        # Create manual comparison table (summary_col doesn't work well with mixed model types)
        comparison_data = {}
        
        for name, result in results_dict.items():
            model_info = {}
            
            # Extract coefficients and standard errors
            if hasattr(result, 'params'):
                for param in result.params.index:
                    coef = result.params[param]
                    # Get standard error (different attribute names for different model types)
                    if hasattr(result, 'std_errors'):
                        se = result.std_errors[param]
                    elif hasattr(result, 'bse'):
                        se = result.bse[param]
                    else:
                        se = np.nan
                    
                    # Format with stars for significance
                    if hasattr(result, 'pvalues'):
                        pval = result.pvalues[param]
                        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                        model_info[param] = f"{coef:.4f}{stars}\n({se:.4f})"
                    else:
                        model_info[param] = f"{coef:.4f}\n({se:.4f})"
            
            # Add model statistics
            model_info['N'] = int(result.nobs)
            if hasattr(result, 'rsquared'):
                model_info['R²'] = f"{result.rsquared:.3f}"
            elif hasattr(result, 'rsquared_overall'):
                model_info['R²'] = f"{result.rsquared_overall:.3f}"
            
            comparison_data[name] = model_info
        
        # Convert to DataFrame for nice display
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + str(comparison_df))
        
        # Save to file
        table_file = self.tables_dir / 'baseline_regression_comparison.csv'
        comparison_df.to_csv(table_file)
        
        # Also save as text
        table_txt = self.tables_dir / 'baseline_regression_comparison.txt'
        with open(table_txt, 'w') as f:
            f.write(str(comparison_df))
        
        print(f"\n✓ Saved: {table_file}")
        print(f"✓ Saved: {table_txt}")

        return comparison_df

    def save_results(self, results, filename):
        """Save regression results to file."""
        output_file = self.output_dir / filename

        with open(output_file, 'w') as f:
            f.write(str(results))

        print(f"✓ Saved detailed results: {output_file}")


def main():
    """Main execution function."""
    print("="*80)
    print("BASELINE REGRESSION ANALYSIS")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize analyzer
        reg = BaselineRegression()

        # Load data
        df = reg.load_data()
        if df is None:
            return

        # Check variables
        if not reg.check_variables(df):
            print("\n⚠ Warning: Some required variables missing")
            print("Proceeding with available variables...")

        # Store results
        results_dict = {}

        # Model 1: Pooled OLS
        try:
            results_dict['Pooled_OLS'] = reg.estimate_pooled_ols(df)
            reg.save_results(results_dict['Pooled_OLS'], 'model1_pooled_ols.txt')
        except Exception as e:
            print(f"\n⚠ Pooled OLS failed: {e}")

        # Model 2: Fixed Effects (Country)
        try:
            results_dict['FE_Country'] = reg.estimate_fixed_effects(
                df, entity_effects=True, time_effects=False
            )
            reg.save_results(results_dict['FE_Country'], 'model2_fe_country.txt')
        except Exception as e:
            print(f"\n⚠ Country FE failed: {e}")

        # Model 3: Two-Way Fixed Effects
        try:
            results_dict['FE_TwoWay'] = reg.estimate_fixed_effects(
                df, entity_effects=True, time_effects=True
            )
            reg.save_results(results_dict['FE_TwoWay'], 'model3_fe_twoway.txt')
        except Exception as e:
            print(f"\n⚠ Two-way FE failed: {e}")

        # Model 4: Regional Heterogeneity
        try:
            results_dict['Regional'] = reg.estimate_regional_heterogeneity(df)
            reg.save_results(results_dict['Regional'], 'model4_regional.txt')
        except Exception as e:
            print(f"\n⚠ Regional model failed: {e}")

        # Compare models
        if len(results_dict) > 1:
            reg.compare_models(results_dict)

        print("\n" + "="*80)
        print("BASELINE REGRESSION COMPLETE ✓")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Detailed: {RESULTS_REGRESSION}")
        print(f"  Tables: {RESULTS_TABLES}")
        print("\nNext step: IV/2SLS estimation to address endogeneity")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
