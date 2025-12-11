# code/analysis/06_regression_with_multiple_imputation.py

"""
================================================================================
Regression Analysis with Multiple Imputation - Rubin's Rules
================================================================================
Purpose: Run regressions on m imputed datasets and pool results
Author: Samir Orujov
Date: December 11, 2025

Implements Rubin's Rules for combining estimates from multiply imputed data:
- Within-imputation variance (W)
- Between-imputation variance (B)
- Total variance: T = W + (1 + 1/m)B
- Degrees of freedom adjustment

References:
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys
- Schafer, J.L. (1997). Analysis of Incomplete Multivariate Data
- Barnard & Rubin (1999). Small-sample degrees of freedom
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from statsmodels.iolib.summary2 import summary_col
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import DATA_PROCESSED
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import DATA_PROCESSED


class MultipleImputationRegression:
    """
    Run regressions on multiply imputed datasets and pool results.
    """
    
    def __init__(self, m=5):
        """
        Initialize with number of imputations.
        
        Parameters:
        - m: Number of imputed datasets (default: 5)
        """
        self.m = m
        self.imputation_dir = DATA_PROCESSED / 'missing_data_analysis'
        self.output_dir = DATA_PROCESSED / 'mi_regression_results'
        self.output_dir.mkdir(exist_ok=True)
        
        self.imputed_datasets = []
        self.results = []
    
    def load_imputed_datasets(self):
        """Load all m imputed datasets."""
        print("="*80)
        print(f"LOADING {self.m} IMPUTED DATASETS")
        print("="*80)
        
        for i in range(1, self.m + 1):
            file_path = self.imputation_dir / f'imputed_data_m{i}.xlsx'
            
            if file_path.exists():
                df = pd.read_excel(file_path, engine='openpyxl')
                self.imputed_datasets.append(df)
                print(f"[{i}/{self.m}] Loaded: {file_path.name}")
            else:
                print(f"âš  Warning: {file_path.name} not found")
        
        print(f"\n[OK] Loaded {len(self.imputed_datasets)} datasets")
        
        return self
    
    def prepare_panel_data(self, df):
        """Prepare data for panel regression."""
        # Create log transformations
        df['log_subs'] = np.log(df['fixed_broadband_subs_i4213tfbb'] + 1)
        df['log_price'] = np.log(df['fixed_broad_price_i154_FBB_ts_GNI'] + 1)
        df['log_gdp'] = np.log(df['gdp_per_capita'] + 1)
        df['log_pop'] = np.log(df['population'] + 1)
        
        # Set panel index
        df = df.set_index(['country', 'year'])
        
        return df
    
    def run_single_regression(self, df, spec='baseline'):
        """
        Run regression on a single imputed dataset.
        
        Specifications:
        - baseline: OLS with country fixed effects
        - twoway: OLS with country + year fixed effects
        - iv: IV/2SLS with instrumental variables
        """
        df_panel = self.prepare_panel_data(df)
        
        if spec == 'baseline':
            # Baseline: Country fixed effects
            formula = 'log_subs ~ log_price + log_gdp + log_pop + EntityEffects'
            
            model = PanelOLS.from_formula(
                formula,
                data=df_panel,
                drop_absorbed=True
            )
            
            result = model.fit(cov_type='clustered', cluster_entity=True)
        
        elif spec == 'twoway':
            # Two-way fixed effects
            formula = 'log_subs ~ log_price + log_gdp + log_pop + EntityEffects + TimeEffects'
            
            model = PanelOLS.from_formula(
                formula,
                data=df_panel,
                drop_absorbed=True
            )
            
            result = model.fit(cov_type='clustered', cluster_entity=True)
        
        elif spec == 'iv':
            # IV/2SLS estimation
            # Endogenous: log_price
            # Instruments: regulatory_quality, mobile_broad_price
            
            # Prepare instruments
            if 'regulatory_quality_estimate' in df.columns:
                df_panel['reg_quality'] = df_panel['regulatory_quality_estimate']
            
            if 'mobile_broad_price_i271mb_ts_GNI' in df.columns:
                df_panel['log_mobile_price'] = np.log(df_panel['mobile_broad_price_i271mb_ts_GNI'] + 1)
            
            # IV formula
            formula = 'log_subs ~ 1 + log_gdp + log_pop + [log_price ~ reg_quality + log_mobile_price]'
            
            model = IV2SLS.from_formula(
                formula,
                data=df_panel
            )
            
            result = model.fit(cov_type='robust')
        
        return result
    
    def pool_results(self, results, var_name='log_price'):
        """
        Pool results using Rubin's rules.
        
        Parameters:
        - results: List of regression results (one per imputation)
        - var_name: Variable name to extract coefficient for
        
        Returns:
        - Dictionary with pooled estimates
        """
        m = len(results)
        
        # Extract coefficients and standard errors
        coefs = []
        ses = []
        
        for result in results:
            try:
                coef = result.params[var_name]
                se = result.std_errors[var_name]
                
                coefs.append(coef)
                ses.append(se)
            except KeyError:
                print(f"âš  Warning: {var_name} not found in results")
                continue
        
        if len(coefs) == 0:
            return None
        
        # Rubin's rules
        # 1. Pooled coefficient (Q_bar)
        Q_bar = np.mean(coefs)
        
        # 2. Within-imputation variance (U_bar)
        U_bar = np.mean([se**2 for se in ses])
        
        # 3. Between-imputation variance (B)
        B = np.var(coefs, ddof=1)
        
        # 4. Total variance (T)
        T = U_bar + (1 + 1/m) * B
        
        # 5. Pooled standard error
        SE_pooled = np.sqrt(T)
        
        # 6. Degrees of freedom (Barnard & Rubin, 1999)
        # Assuming large sample, use normal approximation
        df = (m - 1) * (1 + U_bar / ((1 + 1/m) * B))**2
        
        # 7. t-statistic
        t_stat = Q_bar / SE_pooled
        
        # 8. p-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # 9. 95% confidence interval
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = Q_bar - t_crit * SE_pooled
        ci_upper = Q_bar + t_crit * SE_pooled
        
        # 10. Fraction of missing information (lambda)
        r = (1 + 1/m) * B / T  # Relative increase in variance
        lambda_mi = (r + 2/(df+3)) / (r + 1)  # Fraction of missing info
        
        pooled = {
            'coefficient': Q_bar,
            'se': SE_pooled,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'within_var': U_bar,
            'between_var': B,
            'total_var': T,
            'df': df,
            'fmi': lambda_mi,  # Fraction of missing information
            'riv': r,  # Relative increase in variance
            'm': m
        }
        
        return pooled
    
    def run_mi_analysis(self, spec='baseline'):
        """
        Run regression on all imputed datasets and pool results.
        
        Parameters:
        - spec: Regression specification ('baseline', 'twoway', 'iv')
        """
        print(f"\n{'='*80}")
        print(f"RUNNING {spec.upper()} REGRESSION ON {self.m} IMPUTED DATASETS")
        print("="*80)
        
        # Run regression on each imputed dataset
        results = []
        
        for i, df in enumerate(self.imputed_datasets, 1):
            print(f"\n[{i}/{self.m}] Estimating on imputation {i}...")
            
            try:
                result = self.run_single_regression(df, spec=spec)
                results.append(result)
                
                # Show price coefficient
                if 'log_price' in result.params.index:
                    coef = result.params['log_price']
                    se = result.std_errors['log_price']
                    print(f"  Price elasticity: {coef:.4f} (SE: {se:.4f})")
                
            except Exception as e:
                print(f"  âš  Error: {str(e)}")
                continue
        
        if len(results) == 0:
            print("\nâš  No successful regressions")
            return None
        
        # Pool results for price coefficient
        print("\n" + "="*80)
        print("POOLING RESULTS (RUBIN'S RULES)")
        print("="*80)
        
        pooled = self.pool_results(results, var_name='log_price')
        
        if pooled is not None:
            print(f"\nPooled Price Elasticity Estimate:")
            print(f"  Coefficient: {pooled['coefficient']:.4f}")
            print(f"  Standard Error: {pooled['se']:.4f}")
            print(f"  t-statistic: {pooled['t_stat']:.2f}")
            print(f"  p-value: {pooled['p_value']:.4f}")
            print(f"  95% CI: [{pooled['ci_lower']:.4f}, {pooled['ci_upper']:.4f}]")
            print(f"\nDiagnostics:")
            print(f"  Degrees of freedom: {pooled['df']:.1f}")
            print(f"  Fraction of missing information: {pooled['fmi']:.3f}")
            print(f"  Relative increase in variance: {pooled['riv']:.3f}")
        
        # Save results
        self.save_pooled_results(pooled, spec)
        
        return pooled, results
    
    def save_pooled_results(self, pooled, spec):
        """Save pooled results to Excel."""
        if pooled is None:
            return
        
        # Create summary table
        summary_df = pd.DataFrame({
            'Specification': [spec],
            'Price Elasticity': [pooled['coefficient']],
            'Std. Error': [pooled['se']],
            't-statistic': [pooled['t_stat']],
            'p-value': [pooled['p_value']],
            'CI Lower': [pooled['ci_lower']],
            'CI Upper': [pooled['ci_upper']],
            'DF': [pooled['df']],
            'FMI': [pooled['fmi']],
            'RIV': [pooled['riv']],
            'm': [pooled['m']]
        })
        
        output_file = self.output_dir / f'pooled_results_{spec}.xlsx'
        summary_df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"\n[OK] Saved: {output_file}")
    
    def compare_with_listwise_deletion(self):
        """Compare MI results with listwise deletion (complete case analysis)."""
        print("\n" + "="*80)
        print("ROBUSTNESS: COMPARING WITH LISTWISE DELETION")
        print("="*80)
        
        # Use first imputed dataset and drop missing
        df_complete = self.imputed_datasets[0].copy()
        
        # Identify complete cases for key variables
        key_vars = [
            'fixed_broadband_subs_i4213tfbb',
            'fixed_broad_price_i154_FBB_ts_GNI',
            'gdp_per_capita',
            'population'
        ]
        
        available_vars = [v for v in key_vars if v in df_complete.columns]
        df_complete = df_complete.dropna(subset=available_vars)
        
        print(f"\nComplete cases: {len(df_complete)} observations")
        
        # Run regression
        try:
            result_listwise = self.run_single_regression(df_complete, spec='baseline')
            
            if 'log_price' in result_listwise.params.index:
                coef_lw = result_listwise.params['log_price']
                se_lw = result_listwise.std_errors['log_price']
                
                print(f"\nListwise Deletion Results:")
                print(f"  Price elasticity: {coef_lw:.4f}")
                print(f"  Standard error: {se_lw:.4f}")
                
                return result_listwise
        
        except Exception as e:
            print(f"âš  Error in listwise deletion: {str(e)}")
            return None
    
    def run_complete_analysis(self):
        """Run complete MI regression analysis."""
        print("="*80)
        print("MULTIPLE IMPUTATION REGRESSION ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_imputed_datasets()
        
        # Run baseline specification
        pooled_baseline, results_baseline = self.run_mi_analysis(spec='baseline')
        
        # Run two-way FE specification
        pooled_twoway, results_twoway = self.run_mi_analysis(spec='twoway')
        
        # Robustness: Listwise deletion
        result_listwise = self.compare_with_listwise_deletion()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")
        
        print("\nFor your paper:")
        print("  1. Report pooled MI estimates as main results")
        print("  2. Compare with listwise deletion in robustness section")
        print("  3. Mention FMI (Fraction of Missing Information) in text")
        print("  4. Include comparison table in appendix")
        
        return self


# Main execution
if __name__ == "__main__":
    mi_reg = MultipleImputationRegression(m=5)
    mi_reg.run_complete_analysis()
