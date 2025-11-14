# code/analysis/08_subscription_elasticity.py
"""
================================================================================
Broadband Subscription (Penetration) Price Elasticity Estimation
================================================================================
Purpose: Estimate price elasticity of broadband subscription rates (extensive margin)
Author: Samir Orujov
Date: November 14, 2025

This complements 07_iv_estimation.py which estimates bandwidth usage elasticity.

Dependent Variable: log(bb_subs_per100) - log of subscriptions per 100 people
Endogenous Variable: log(fixed_broad_price) - fixed broadband price
Instruments: 
  - mobile_broad_price_i271mb_ts_GNI (substitute price)
  - research_development_expenditure (technology cost shifter)

Interpretation:
- Bandwidth elasticity (intensive margin): How much existing users adjust usage
- Subscription elasticity (extensive margin): How many people subscribe/unsubscribe
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings

# Suppress specific statsmodels warning about covariance matrix rank
warnings.filterwarnings('ignore', message='covariance of constraints does not have full rank')

# Econometrics packages
from linearmodels.iv import IV2SLS
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


class SubscriptionElasticityEstimator:
    """Price elasticity estimation for broadband subscription rates."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_REGRESSION
        self.tables_dir = RESULTS_TABLES

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load analysis-ready dataset."""
        print("="*80)
        print("BROADBAND SUBSCRIPTION PRICE ELASTICITY ESTIMATION")
        print("="*80)
        print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nDependent variable: Broadband subscriptions per 100 people (penetration)")
        print("This estimates the EXTENSIVE MARGIN elasticity")

        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        if not self.data_file.exists():
            print(f"\n✗ File not found: {self.data_file}")
            return None

        df = pd.read_csv(self.data_file)
        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Set panel structure
        if 'country' in df.columns and 'year' in df.columns:
            df = df.set_index(['country', 'year'])
            print("✓ Panel structure set: country × year")

        # Create log of subscriptions
        if 'bb_subs_per100' in df.columns:
            df['log_subs'] = np.log(df['bb_subs_per100'])
            print(f"✓ Created log_subs from bb_subs_per100")
            print(f"  Non-missing: {df['log_subs'].notna().sum()} observations")
        else:
            print("✗ bb_subs_per100 not found in data")
            return None

        return df

    def first_stage_regression(self, df, instruments):
        """Estimate first-stage regression and test instrument strength."""
        print("\n" + "="*80)
        print("FIRST-STAGE REGRESSION")
        print("="*80)
        print("Dependent variable: log_price (endogenous)")
        print(f"Instruments: {instruments}")

        y_first = df['log_price']

        # Enhanced control variables
        exog_vars = [
            'log_gdp_pc',
            'internet_users_pct',
            'mobile_subs_per100',
            'population_density',
            'urban_population_pct',
            'electric_power_consumption',
        ]
        exog_vars = [v for v in exog_vars if v in df.columns and df[v].notna().sum() > 100]

        # Build first-stage design matrix
        X_first = pd.DataFrame(index=df.index)
        for var in instruments:
            X_first[var] = df[var]
        for var in exog_vars:
            X_first[var] = df[var]

        # Drop missing
        valid = y_first.notna() & X_first.notna().all(axis=1)
        y_first = y_first[valid]
        X_first = X_first[valid]

        print(f"\nObservations: {len(y_first)}")
        print("Note: First-stage uses cross-sectional + time variation (entity FE in 2nd stage)")

        # Estimate first stage with constant
        first_stage = sm.OLS(y_first, sm.add_constant(X_first)).fit(cov_type='HC1')

        print("\n" + str(first_stage.summary()))

        # Save first stage results
        first_stage_file = self.output_dir / 'subscription_iv_first_stage.txt'
        with open(first_stage_file, 'w') as f:
            f.write(str(first_stage.summary()))

        # Test instrument strength
        print("\n" + "="*60)
        print("WEAK INSTRUMENT TEST")
        print("="*60)

        # F-statistic for instruments
        inst_params = first_stage.params[instruments]
        inst_cov = first_stage.cov_params().loc[instruments, instruments]
        F_stat = (inst_params.T @ np.linalg.inv(inst_cov) @ inst_params) / len(instruments)

        print(f"F-statistic for instruments: {F_stat:.2f}")

        if F_stat < 10:
            print("⚠ WARNING: Instruments may be WEAK (F < 10)")
            if F_stat < 5:
                print("⚠⚠ SEVERELY WEAK instruments (F < 5)")
        else:
            print("✓ Instruments appear adequate (F > 10)")

        print(f"✓ Saved: {first_stage_file}")

        return first_stage, F_stat

    def estimate_subscription_iv(self, df, instruments):
        """Estimate IV model for subscription elasticity with entity fixed effects."""
        print("\n" + "="*80)
        print("PANEL IV ESTIMATION - SUBSCRIPTION ELASTICITY (Entity Fixed Effects)")
        print("="*80)
        print("Dependent variable: log_subs (log subscriptions per 100)")
        print("Endogenous: log_price")
        print(f"Instruments: {instruments}")

        # Prepare variables
        y = df['log_subs']
        endog = df[['log_price']]

        # Enhanced control variables
        exog_vars = [
            'log_gdp_pc',
            'internet_users_pct',
            'mobile_subs_per100',
            'gdp_growth',
            'inflation_gdp_deflator',
            'population_density',
        ]
        exog_vars = [v for v in exog_vars if v in df.columns]
        exog = df[exog_vars]
        
        print(f"Control variables ({len(exog_vars)}): {exog_vars}")

        # Instruments
        instr = df[instruments]

        # Drop missing
        valid = (y.notna() & endog.notna().all(axis=1) & 
                exog.notna().all(axis=1) & instr.notna().all(axis=1))

        y = y[valid]
        endog = endog[valid]
        exog = exog[valid]
        instr = instr[valid]

        print(f"\nObservations: {len(y)}")
        print(f"Entities: {len(y.index.get_level_values(0).unique())}")
        print(f"Time periods: {len(y.index.get_level_values(1).unique())}")

        # Apply within transformation for entity fixed effects
        print("\nApplying within transformation (entity demeaning) for Fixed Effects...")
        
        data_combined = pd.concat([y, endog, exog, instr], axis=1)
        
        # Demean all variables
        data_demeaned = data_combined.copy()
        for col in data_combined.columns:
            entity_mean = data_combined.groupby(level=0)[col].transform('mean')
            data_demeaned[col] = data_combined[col] - entity_mean
        
        # Extract demeaned variables
        y_dm = data_demeaned[y.name]
        endog_dm = data_demeaned[endog.columns]
        exog_dm = data_demeaned[exog_vars]
        instr_dm = data_demeaned[instruments]
        
        print("✓ Within transformation complete (entity FE removed)")
        
        # Estimate IV on demeaned data
        iv_model = IV2SLS(
            dependent=y_dm,
            exog=exog_dm,
            endog=endog_dm,
            instruments=instr_dm
        )

        iv_results = iv_model.fit(cov_type='robust')

        print("\n" + str(iv_results))

        # Save IV results
        iv_file = self.output_dir / 'subscription_iv_results.txt'
        with open(iv_file, 'w') as f:
            f.write(str(iv_results))

        # Extract price elasticity
        print("\n" + "="*60)
        print("SUBSCRIPTION PRICE ELASTICITY (EXTENSIVE MARGIN)")
        print("="*60)

        price_coef = iv_results.params['log_price']
        price_se = iv_results.std_errors['log_price']
        price_ci = iv_results.conf_int().loc['log_price']

        print(f"Coefficient: {price_coef:.3f}")
        print(f"Std. Error: {price_se:.3f}")
        print(f"95% CI: [{price_ci.iloc[0]:.3f}, {price_ci.iloc[1]:.3f}]")
        
        print("\nInterpretation:")
        print(f"A 1% increase in price → {price_coef:.2f}% change in subscription rate")
        
        if abs(price_coef) > 1:
            print("✓ ELASTIC demand (|elasticity| > 1): Price sensitive")
        elif abs(price_coef) > 0:
            print("✓ INELASTIC demand (|elasticity| < 1): Price insensitive")

        print(f"\n✓ Saved: {iv_file}")

        return iv_results

    def estimate_ols_comparison(self, df):
        """Estimate OLS for comparison with IV."""
        print("\n" + "="*80)
        print("OLS FOR COMPARISON (Entity Demeaned)")
        print("="*80)

        y = df['log_subs']

        X_vars = [
            'log_price',
            'log_gdp_pc',
            'internet_users_pct',
            'mobile_subs_per100',
            'gdp_growth',
            'inflation_gdp_deflator',
            'population_density',
        ]
        X_vars = [v for v in X_vars if v in df.columns]
        X = df[X_vars]

        # Drop missing
        valid = y.notna() & X.notna().all(axis=1)
        y = y[valid]
        X = X[valid]

        # Apply within transformation
        print("Applying within transformation for entity fixed effects...")
        data_combined = pd.concat([y, X], axis=1)
        
        data_demeaned = data_combined.copy()
        for col in data_combined.columns:
            entity_mean = data_combined.groupby(level=0)[col].transform('mean')
            data_demeaned[col] = data_combined[col] - entity_mean
        
        y_dm = data_demeaned[y.name]
        X_dm = data_demeaned[X_vars]
        
        print("✓ Within transformation complete (entity FE removed)")
        print(f"Observations: {len(y_dm)}")

        # Estimate OLS on demeaned data
        ols_model = sm.OLS(y_dm, X_dm).fit(cov_type='HC1')

        print("\n" + str(ols_model.summary()))

        ols_file = self.output_dir / 'subscription_ols_comparison.txt'
        with open(ols_file, 'w') as f:
            f.write(str(ols_model.summary()))

        print(f"✓ Saved: {ols_file}")

        return ols_model

    def hausman_test(self, ols_results, iv_results):
        """Perform Hausman test for endogeneity."""
        print("\n" + "="*80)
        print("HAUSMAN TEST FOR ENDOGENEITY")
        print("="*80)

        b_ols = ols_results.params.get('log_price', np.nan)
        b_iv = iv_results.params.get('log_price', np.nan)
        se_ols = ols_results.bse.get('log_price', np.nan)
        se_iv = iv_results.std_errors.get('log_price', np.nan)

        if np.isnan(b_ols) or np.isnan(b_iv):
            print("\n✗ Cannot perform Hausman test")
            return None

        diff = b_iv - b_ols
        var_diff = se_iv**2 - se_ols**2

        if var_diff <= 0:
            var_diff = abs(var_diff)

        h_stat = diff / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(h_stat)))

        print(f"\nOLS coefficient: {b_ols:.3f} (SE: {se_ols:.3f})")
        print(f"IV coefficient:  {b_iv:.3f} (SE: {se_iv:.3f})")
        print(f"Difference:      {diff:.3f}")
        print(f"\nTest statistic: {h_stat:.3f}")
        print(f"P-value:        {p_value:.4f}")

        if p_value < 0.10:
            print("\n✓ REJECT H₀: Price is endogenous (IV preferred)")
        else:
            print("\n✗ FAIL TO REJECT H₀: Price may be exogenous (OLS OK)")

        hausman_file = self.output_dir / 'subscription_hausman_test.txt'
        with open(hausman_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HAUSMAN TEST FOR ENDOGENEITY - SUBSCRIPTION MODEL\n")
            f.write("="*80 + "\n\n")
            f.write(f"OLS coefficient: {b_ols:.3f} (SE: {se_ols:.3f})\n")
            f.write(f"IV coefficient:  {b_iv:.3f} (SE: {se_iv:.3f})\n")
            f.write(f"Difference:      {diff:.3f}\n\n")
            f.write(f"Test statistic: {h_stat:.3f}\n")
            f.write(f"P-value:        {p_value:.4f}\n")

        print(f"\n✓ Saved: {hausman_file}")

        return {'statistic': h_stat, 'p_value': p_value}

    def sargan_hansen_test(self, iv_results, instruments):
        """Test overidentification restrictions."""
        if len(instruments) <= 1:
            print("\n" + "="*80)
            print("OVERIDENTIFICATION TEST")
            print("="*80)
            print("\n⚠ Model is exactly identified (1 instrument)")
            return None

        print("\n" + "="*80)
        print("SARGAN-HANSEN J-TEST (Overidentification)")
        print("="*80)

        try:
            if hasattr(iv_results, 'sargan'):
                j_stat = iv_results.sargan.stat
                j_pval = iv_results.sargan.pval
                df = iv_results.sargan.df
                
                print(f"\nJ-statistic (Sargan): {j_stat:.3f}")
                print(f"P-value:              {j_pval:.4f}")
                print(f"Degrees of freedom:   {df}")

                if j_pval > 0.10:
                    print("\n✓ Cannot reject H₀: Instruments appear valid (p > 0.10)")
                else:
                    print("\n✗ Reject H₀: Instruments may be invalid (p ≤ 0.10)")

                return {'statistic': j_stat, 'p_value': j_pval, 'df': df}
                
        except (AttributeError, KeyError):
            print("\n⚠ Sargan test not available with robust standard errors")
            return None

    def compare_elasticities(self, bandwidth_iv_file, subscription_iv_results):
        """Compare bandwidth and subscription elasticities."""
        print("\n" + "="*80)
        print("COMPARISON: INTENSIVE vs EXTENSIVE MARGIN ELASTICITIES")
        print("="*80)

        # Try to read bandwidth elasticity from previous results
        bandwidth_file = self.output_dir / 'iv_2sls_results.txt'
        
        print("\nElasticity Type Definitions:")
        print("-" * 60)
        print("INTENSIVE MARGIN (Bandwidth):")
        print("  How much do existing subscribers adjust their usage?")
        print("  Dependent variable: log(bandwidth per capita)")
        print("\nEXTENSIVE MARGIN (Subscription):")
        print("  How many people decide to subscribe/unsubscribe?")
        print("  Dependent variable: log(subscriptions per 100)")
        print("-" * 60)

        sub_elasticity = subscription_iv_results.params['log_price']
        sub_se = subscription_iv_results.std_errors['log_price']

        print(f"\nSUBSCRIPTION ELASTICITY: {sub_elasticity:.3f} (SE: {sub_se:.3f})")

        if bandwidth_file.exists():
            # Parse bandwidth elasticity from file
            with open(bandwidth_file, 'r') as f:
                content = f.read()
                # Extract log_price coefficient
                import re
                match = re.search(r'log_price\s+([-\d.]+)\s+([\d.]+)', content)
                if match:
                    bw_elasticity = float(match.group(1))
                    bw_se = float(match.group(2))
                    print(f"BANDWIDTH ELASTICITY:     {bw_elasticity:.3f} (SE: {bw_se:.3f})")
                    print(f"\nTOTAL DEMAND ELASTICITY:  {bw_elasticity + sub_elasticity:.3f}")
                    print("  (Sum of intensive + extensive margins)")

        # Save comparison
        comparison_file = self.tables_dir / 'elasticity_comparison.txt'
        with open(comparison_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BROADBAND DEMAND ELASTICITY DECOMPOSITION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Subscription (Extensive):  {sub_elasticity:.3f} (SE: {sub_se:.3f})\n")
            if bandwidth_file.exists() and match:
                f.write(f"Bandwidth (Intensive):     {bw_elasticity:.3f} (SE: {bw_se:.3f})\n")
                f.write(f"Total Demand:              {bw_elasticity + sub_elasticity:.3f}\n")

        print(f"\n✓ Saved: {comparison_file}")


def main():
    """Main execution function."""

    try:
        estimator = SubscriptionElasticityEstimator()

        # Load data
        df = estimator.load_data()
        if df is None:
            return

        # Define instruments (enhanced with lags)
        instruments = [
            'mobile_broad_price_i271mb_ts_GNI',
            'research_development_expenditure',
            'fixed_broad_price_lag1',
            'mobile_broad_price_lag1'
        ]

        # Check instruments availability and filter
        available_instruments = [inst for inst in instruments if inst in df.columns and df[inst].notna().sum() > 100]
        
        if len(available_instruments) == 0:
            print("\n✗ No instruments available")
            return

        print(f"\nUsing {len(available_instruments)} instruments: {available_instruments}")
        instruments = available_instruments

        # First-stage regression
        first_stage, f_stat = estimator.first_stage_regression(df, instruments)

        # IV estimation for subscription
        iv_results = estimator.estimate_subscription_iv(df, instruments)

        # OLS for comparison
        ols_results = estimator.estimate_ols_comparison(df)

        # Hausman test
        estimator.hausman_test(ols_results, iv_results)

        # Overidentification test
        if len(instruments) > 1:
            estimator.sargan_hansen_test(iv_results, instruments)

        # Compare with bandwidth elasticity
        estimator.compare_elasticities(None, iv_results)

        print("\n" + "="*80)
        print("SUBSCRIPTION ELASTICITY ESTIMATION COMPLETE ✓")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  {estimator.output_dir}")
        print(f"  {estimator.tables_dir}")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
