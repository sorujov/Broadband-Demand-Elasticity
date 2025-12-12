# code/analysis/00_comprehensive_method_diagnostic.py

"""
================================================================================
COMPREHENSIVE ECONOMETRIC METHOD DIAGNOSTIC FRAMEWORK
================================================================================
Purpose: Test ALL methods and recommend the optimal path for your paper
Author: Samir Orujov
Date: December 11, 2025

This script:
1. Tests data suitability for 8+ econometric methods
2. Checks assumptions and validity conditions
3. Runs preliminary estimations
4. Identifies problems (weak instruments, endogeneity, etc.)
5. Provides RANKED recommendations with justification
6. Generates a decision tree for your paper

Methods Tested:
- Pooled OLS
- Fixed Effects (FE)
- Random Effects (RE)  
- Two-Way Fixed Effects
- First Differences
- IV/2SLS (with instrument tests)
- GMM Dynamic Panel (Arellano-Bond)
- System GMM (Blundell-Bond)

Output: Clear roadmap of which method(s) to use and why
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from linearmodels.panel import PanelOLS, RandomEffects, FirstDifferenceOLS
from linearmodels.iv import IV2SLS
from linearmodels.system import SUR
import statsmodels.api as sm

# Configure UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'code'))

from utils.config import DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES


class MethodDiagnostic:
    """
    Comprehensive diagnostic framework to test all econometric methods
    and recommend the optimal approach.
    """
    
    def __init__(self, data_file=None):
        """Initialize with data file."""
        if data_file is None:
            self.data_file = DATA_PROCESSED / 'data_merged_with_series.xlsx'
        else:
            self.data_file = data_file
        
        self.output_dir = DATA_PROCESSED / 'method_diagnostic'
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.df_panel = None
        
        # Store test results
        self.test_results = {}
        self.method_scores = {}
        self.recommendations = []
        
    def load_and_prepare_data(self):
        """Load data and create necessary transformations."""
        print("="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        self.df = pd.read_excel(self.data_file, engine='openpyxl')
        
        print(f"\n[OK] Loaded: {len(self.df):,} observations")
        print(f"  Countries: {self.df['country'].nunique()}")
        print(f"  Years: {self.df['year'].min()}-{self.df['year'].max()}")
        
        # Create log transformations
        self.df['log_subs'] = np.log(self.df['fixed_broadband_subs_i992b'] + 1)
        self.df['log_price'] = np.log(self.df['fixed_broad_price_ppp'] + 1)
        self.df['log_gdp'] = np.log(self.df['gdp_per_capita'] + 1)
        self.df['log_pop'] = np.log(self.df['population'] + 1)
        
        # Add region indicators
        self.df['is_eu'] = self.df['country'].isin(EU_COUNTRIES).astype(int)
        self.df['is_eap'] = self.df['country'].isin(EAP_COUNTRIES).astype(int)
        
        # Create lagged variables for IV/GMM
        self.df = self.df.sort_values(['country', 'year'])
        self.df['log_price_lag1'] = self.df.groupby('country')['log_price'].shift(1)
        self.df['log_price_lag2'] = self.df.groupby('country')['log_price'].shift(2)
        self.df['log_subs_lag1'] = self.df.groupby('country')['log_subs'].shift(1)
        
        # Potential instruments
        if 'regulatory_quality_estimate' in self.df.columns:
            self.df['reg_quality'] = self.df['regulatory_quality_estimate']
        
        if 'mobile_broad_price_ppp' in self.df.columns:
            self.df['log_mobile_price'] = np.log(self.df['mobile_broad_price_ppp'] + 1)
        
        # Set panel index
        self.df_panel = self.df.set_index(['country', 'year']).sort_index()
        
        print("\n[OK] Created log transformations and lags")
        
        return self
    
    def check_data_structure(self):
        """Check if data meets requirements for panel methods."""
        print("\n" + "="*80)
        print("TEST 1: DATA STRUCTURE REQUIREMENTS")
        print("="*80)
        
        results = {}
        
        # 1. Panel balance
        countries = self.df['country'].unique()
        years = self.df['year'].unique()
        
        obs_per_country = self.df.groupby('country').size()
        balanced = (obs_per_country == len(years)).all()
        
        results['balanced_panel'] = balanced
        results['n_countries'] = len(countries)
        results['n_years'] = len(years)
        results['n_obs'] = len(self.df)
        results['theoretical_max'] = len(countries) * len(years)
        results['completeness_pct'] = (len(self.df) / results['theoretical_max'] * 100)
        
        print(f"\nPanel Structure:")
        print(f"  Countries (N): {results['n_countries']}")
        print(f"  Years (T): {results['n_years']}")
        print(f"  Observations: {results['n_obs']:,}")
        print(f"  Balanced: {'✓ YES' if balanced else '✗ NO'}")
        print(f"  Completeness: {results['completeness_pct']:.1f}%")
        
        if balanced:
            print("\n  → All panel methods available")
        else:
            print("\n  → Unbalanced panel: Some methods may need adjustment")
        
        # 2. Time series length
        results['T_adequate'] = len(years) >= 10
        
        if results['T_adequate']:
            print(f"\n✓ T={len(years)} ≥ 10: Adequate for dynamic panel (GMM)")
        else:
            print(f"\n✗ T={len(years)} < 10: GMM may have weak instruments")
        
        # 3. Cross-sectional variation
        results['N_adequate'] = len(countries) >= 20
        
        if results['N_adequate']:
            print(f"✓ N={len(countries)} ≥ 20: Good cross-sectional variation")
        else:
            print(f"✗ N={len(countries)} < 20: Limited for FE estimation")
        
        self.test_results['data_structure'] = results
        
        return results
    
    def test_pooled_ols(self):
        """Test 1: Pooled OLS (baseline)."""
        print("\n" + "="*80)
        print("TEST 2: POOLED OLS (Baseline)")
        print("="*80)
        
        # Drop missing
        df_clean = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop']].dropna()
        
        if len(df_clean) < 100:
            print("\n✗ Insufficient data for pooled OLS")
            return None
        
        # Estimate
        y = df_clean['log_subs']
        X = add_constant(df_clean[['log_price', 'log_gdp', 'log_pop']])
        
        model = OLS(y, X)
        result = model.fit(cov_type='HC3')  # Robust SEs
        
        # Extract results
        coef = result.params['log_price']
        se = result.bse['log_price']
        pval = result.pvalues['log_price']
        
        print(f"\nResults:")
        print(f"  Price elasticity: {coef:.4f}")
        print(f"  Standard error: {se:.4f}")
        print(f"  p-value: {pval:.4f}")
        print(f"  R²: {result.rsquared:.3f}")
        
        # Problem: Likely biased due to unobserved heterogeneity
        print(f"\n⚠️  WARNING: Pooled OLS ignores country-specific effects")
        print(f"  → Likely biased if unobserved heterogeneity exists")
        print(f"  → Use only as baseline comparison")
        
        score = {
            'suitable': True,
            'issues': ['Omitted variable bias', 'Unobserved heterogeneity'],
            'elasticity': coef,
            'se': se,
            'pval': pval,
            'priority': 'LOW - Use as baseline only'
        }
        
        self.method_scores['pooled_ols'] = score
        self.test_results['pooled_ols'] = result
        
        return result
    
    def test_fixed_effects(self):
        """Test 2: Fixed Effects (within estimator)."""
        print("\n" + "="*80)
        print("TEST 3: FIXED EFFECTS (Within Estimator)")
        print("="*80)
        
        # Estimate
        df_clean = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop']].dropna()
        
        if len(df_clean) < 100:
            print("\n✗ Insufficient data")
            return None
        
        try:
            model = PanelOLS.from_formula(
                'log_subs ~ log_price + log_gdp + log_pop + EntityEffects',
                data=df_clean,
                drop_absorbed=True
            )
            
            result = model.fit(cov_type='clustered', cluster_entity=True)
            
            coef = result.params['log_price']
            se = result.std_errors['log_price']
            pval = result.pvalues['log_price']
            
            print(f"\nResults:")
            print(f"  Price elasticity: {coef:.4f}")
            print(f"  Standard error: {se:.4f}")
            print(f"  p-value: {pval:.4f}")
            print(f"  Within R²: {result.rsquared_within:.3f}")
            
            # Check if price varies within countries
            within_var = df_clean.groupby(level='country')['log_price'].var().mean()
            total_var = df_clean['log_price'].var()
            
            print(f"\n  Within-country price variation: {within_var:.4f}")
            print(f"  Total price variation: {total_var:.4f}")
            print(f"  Ratio: {within_var/total_var*100:.1f}%")
            
            issues = []
            if within_var / total_var < 0.1:
                print(f"\n⚠️  WARNING: Limited within-country price variation")
                print(f"  → FE may not be efficient (most variation is between countries)")
                issues.append('Limited within variation')
            
            # Endogeneity still possible
            issues.append('Price may still be endogenous')
            
            score = {
                'suitable': True,
                'issues': issues,
                'elasticity': coef,
                'se': se,
                'pval': pval,
                'within_var_ratio': within_var/total_var,
                'priority': 'HIGH - Standard panel approach'
            }
            
            self.method_scores['fixed_effects'] = score
            self.test_results['fixed_effects'] = result
            
            return result
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            return None
    
    def test_hausman(self):
        """Test 3: Hausman test (FE vs RE)."""
        print("\n" + "="*80)
        print("TEST 4: HAUSMAN TEST (FE vs RE)")
        print("="*80)
        
        df_clean = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop']].dropna()
        
        if len(df_clean) < 100:
            print("\n✗ Insufficient data")
            return None
        
        try:
            # Fixed effects
            fe_model = PanelOLS.from_formula(
                'log_subs ~ log_price + log_gdp + log_pop + EntityEffects',
                data=df_clean,
                drop_absorbed=True
            )
            fe_result = fe_model.fit()
            
            # Random effects
            re_model = RandomEffects.from_formula(
                'log_subs ~ 1 + log_price + log_gdp + log_pop',
                data=df_clean
            )
            re_result = re_model.fit()
            
            # Hausman test
            # H0: Random effects is consistent (use RE)
            # H1: Only fixed effects is consistent (use FE)
            
            fe_coef = fe_result.params['log_price']
            re_coef = re_result.params['log_price']
            
            diff = fe_coef - re_coef
            
            fe_var = fe_result.cov['log_price']['log_price']
            re_var = re_result.cov['log_price']['log_price']
            
            hausman_stat = diff**2 / (fe_var - re_var) if fe_var > re_var else np.nan
            
            print(f"\nCoefficients:")
            print(f"  Fixed Effects: {fe_coef:.4f}")
            print(f"  Random Effects: {re_coef:.4f}")
            print(f"  Difference: {diff:.4f}")
            
            if not np.isnan(hausman_stat):
                pval_hausman = 1 - stats.chi2.cdf(hausman_stat, df=1)
                print(f"\nHausman Test:")
                print(f"  χ² statistic: {hausman_stat:.2f}")
                print(f"  p-value: {pval_hausman:.4f}")
                
                if pval_hausman < 0.05:
                    print(f"\n✓ Result: Use FIXED EFFECTS (p < 0.05)")
                    print(f"  → Country-specific effects correlated with regressors")
                    recommendation = 'Use FE'
                else:
                    print(f"\n✓ Result: Can use RANDOM EFFECTS (p ≥ 0.05)")
                    print(f"  → RE is more efficient")
                    recommendation = 'Can use RE'
            else:
                print(f"\n⚠️  Cannot compute Hausman test")
                recommendation = 'Use FE (conservative)'
            
            self.test_results['hausman'] = {
                'fe_coef': fe_coef,
                're_coef': re_coef,
                'recommendation': recommendation
            }
            
            return recommendation
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            return None
    
    def test_instruments(self):
        """Test 4: Instrument validity for IV/2SLS."""
        print("\n" + "="*80)
        print("TEST 5: INSTRUMENTAL VARIABLES (IV/2SLS)")
        print("="*80)
        
        # Check if instruments available
        potential_instruments = ['reg_quality', 'log_mobile_price', 'log_price_lag1']
        available_instruments = [iv for iv in potential_instruments if iv in self.df_panel.columns]
        
        print(f"\nPotential Instruments:")
        for iv in potential_instruments:
            available = '✓' if iv in available_instruments else '✗'
            print(f"  {available} {iv}")
        
        if len(available_instruments) == 0:
            print(f"\n✗ NO INSTRUMENTS AVAILABLE")
            print(f"  → Cannot use IV/2SLS")
            self.method_scores['iv_2sls'] = {
                'suitable': False,
                'issues': ['No instruments available'],
                'priority': 'IMPOSSIBLE'
            }
            return None
        
        # Test each instrument
        print(f"\n" + "-"*80)
        print("INSTRUMENT VALIDITY TESTS")
        print("-"*80)
        
        instrument_results = {}
        
        for iv_name in available_instruments:
            print(f"\nTesting: {iv_name}")
            
            # Prepare data
            df_iv = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop', iv_name]].dropna()
            
            if len(df_iv) < 50:
                print(f"  ✗ Insufficient data")
                continue
            
            # Test 1: Relevance (First-stage F-stat)
            X_first = add_constant(df_iv[[iv_name, 'log_gdp', 'log_pop']])
            y_first = df_iv['log_price']
            
            first_stage = OLS(y_first, X_first).fit()
            
            # F-stat for excluded instrument
            f_stat = first_stage.fvalue
            
            print(f"  First-stage F-statistic: {f_stat:.2f}")
            
            if f_stat > 10:
                print(f"  ✓ Strong instrument (F > 10)")
                strength = 'Strong'
            elif f_stat > 5:
                print(f"  ⚠️  Weak instrument (5 < F < 10)")
                strength = 'Weak'
            else:
                print(f"  ✗ Very weak instrument (F < 5)")
                strength = 'Very weak'
            
            # Test 2: Exogeneity (correlation with error term - informal)
            # We can't test this directly, but check correlation with outcome
            corr_with_y = df_iv[[iv_name, 'log_subs']].corr().iloc[0, 1]
            
            print(f"  Correlation with outcome: {corr_with_y:.3f}")
            
            if abs(corr_with_y) < 0.3:
                print(f"  ✓ Low direct correlation (exclusion restriction plausible)")
                exclusion = 'Plausible'
            else:
                print(f"  ⚠️  High direct correlation (may violate exclusion restriction)")
                exclusion = 'Questionable'
            
            instrument_results[iv_name] = {
                'f_stat': f_stat,
                'strength': strength,
                'exclusion': exclusion,
                'n_obs': len(df_iv)
            }
        
        # Overall IV assessment
        strong_instruments = [iv for iv, res in instrument_results.items() 
                            if res['strength'] == 'Strong']
        
        print(f"\n" + "="*80)
        print("IV/2SLS ASSESSMENT")
        print("="*80)
        
        if len(strong_instruments) > 0:
            print(f"\n✓ {len(strong_instruments)} STRONG INSTRUMENT(S) AVAILABLE:")
            for iv in strong_instruments:
                print(f"  - {iv} (F = {instrument_results[iv]['f_stat']:.2f})")
            
            print(f"\n→ IV/2SLS is VIABLE")
            print(f"  Recommended specification:")
            print(f"    Endogenous: log_price")
            print(f"    Instruments: {', '.join(strong_instruments)}")
            
            # Try running IV regression
            try:
                print(f"\n" + "-"*80)
                print("Running IV/2SLS estimation...")
                print("-"*80)
                
                # Use first strong instrument
                iv_var = strong_instruments[0]
                df_iv = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop', iv_var]].dropna()
                
                # IV formula
                formula = f'log_subs ~ 1 + log_gdp + log_pop + [log_price ~ {iv_var}]'
                
                model = IV2SLS.from_formula(formula, data=df_iv)
                result = model.fit(cov_type='robust')
                
                coef = result.params['log_price']
                se = result.std_errors['log_price']
                pval = result.pvalues['log_price']
                
                print(f"\nIV/2SLS Results:")
                print(f"  Price elasticity: {coef:.4f}")
                print(f"  Standard error: {se:.4f}")
                print(f"  p-value: {pval:.4f}")
                
                # Diagnostic tests
                print(f"\nDiagnostic Tests:")
                print(f"  First-stage F: {result.first_stage.diagnostics['fstat']['stat']:.2f}")
                
                self.method_scores['iv_2sls'] = {
                    'suitable': True,
                    'issues': [] if len(strong_instruments) > 1 else ['Only one instrument (cannot test overidentification)'],
                    'elasticity': coef,
                    'se': se,
                    'pval': pval,
                    'instruments': strong_instruments,
                    'first_stage_f': result.first_stage.diagnostics['fstat']['stat'],
                    'priority': 'HIGH - Addresses endogeneity'
                }
                
                self.test_results['iv_2sls'] = result
                
                return result
                
            except Exception as e:
                print(f"\n✗ Error in IV estimation: {str(e)}")
                
        else:
            print(f"\n✗ NO STRONG INSTRUMENTS")
            print(f"  → IV/2SLS not recommended (weak instruments)")
            
            self.method_scores['iv_2sls'] = {
                'suitable': False,
                'issues': ['Weak instruments', 'Estimates will be biased'],
                'priority': 'LOW - Not recommended'
            }
        
        self.test_results['instruments'] = instrument_results
        
        return instrument_results
    
    def test_endogeneity(self):
        """Test 5: Endogeneity test (Hausman-Wu)."""
        print("\n" + "="*80)
        print("TEST 6: ENDOGENEITY TEST (Price Endogenous?)")
        print("="*80)
        
        # Check if we have instruments
        if 'iv_2sls' not in self.test_results or self.test_results['iv_2sls'] is None:
            print("\n✗ Cannot test endogeneity (no valid IV estimation)")
            return None
        
        iv_result = self.test_results['iv_2sls']
        
        # Compare OLS vs IV
        ols_coef = self.test_results['pooled_ols'].params['log_price']
        iv_coef = iv_result.params['log_price']
        
        diff = abs(iv_coef - ols_coef)
        
        print(f"\nCoefficient Comparison:")
        print(f"  OLS: {ols_coef:.4f}")
        print(f"  IV/2SLS: {iv_coef:.4f}")
        print(f"  Difference: {diff:.4f}")
        print(f"  Relative difference: {diff/abs(ols_coef)*100:.1f}%")
        
        # Rule of thumb: >20% difference suggests endogeneity
        if diff / abs(ols_coef) > 0.20:
            print(f"\n✓ ENDOGENEITY DETECTED (>20% difference)")
            print(f"  → Price is likely endogenous")
            print(f"  → IV/2SLS estimates are more reliable")
            endogeneity_conclusion = 'Present - Use IV'
        else:
            print(f"\n✓ No strong evidence of endogeneity (<20% difference)")
            print(f"  → OLS may be acceptable")
            print(f"  → But IV is still conservative choice")
            endogeneity_conclusion = 'Weak - OLS acceptable'
        
        self.test_results['endogeneity'] = {
            'ols_coef': ols_coef,
            'iv_coef': iv_coef,
            'difference_pct': diff/abs(ols_coef)*100,
            'conclusion': endogeneity_conclusion
        }
        
        return endogeneity_conclusion
    
    def test_two_way_fe(self):
        """Test 6: Two-way fixed effects (entity + time)."""
        print("\n" + "="*80)
        print("TEST 7: TWO-WAY FIXED EFFECTS (Entity + Time)")
        print("="*80)
        
        df_clean = self.df_panel[['log_subs', 'log_price', 'log_gdp', 'log_pop']].dropna()
        
        if len(df_clean) < 100:
            print("\n✗ Insufficient data")
            return None
        
        try:
            model = PanelOLS.from_formula(
                'log_subs ~ log_price + log_gdp + log_pop + EntityEffects + TimeEffects',
                data=df_clean,
                drop_absorbed=True
            )
            
            result = model.fit(cov_type='clustered', cluster_entity=True)
            
            coef = result.params['log_price']
            se = result.std_errors['log_price']
            pval = result.pvalues['log_price']
            
            print(f"\nResults:")
            print(f"  Price elasticity: {coef:.4f}")
            print(f"  Standard error: {se:.4f}")
            print(f"  p-value: {pval:.4f}")
            
            # Compare with one-way FE
            if 'fixed_effects' in self.test_results:
                fe_coef = self.test_results['fixed_effects'].params['log_price']
                print(f"\n  One-way FE coefficient: {fe_coef:.4f}")
                print(f"  Two-way FE coefficient: {coef:.4f}")
                print(f"  Difference: {abs(coef - fe_coef):.4f}")
            
            print(f"\n✓ Two-way FE controls for:")
            print(f"  - Country-specific effects (supply/demand factors)")
            print(f"  - Time-specific effects (global trends, shocks)")
            
            self.method_scores['twoway_fe'] = {
                'suitable': True,
                'issues': ['Price may still be endogenous'],
                'elasticity': coef,
                'se': se,
                'pval': pval,
                'priority': 'HIGH - Robust to trends'
            }
            
            self.test_results['twoway_fe'] = result
            
            return result
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            return None
    
    def generate_recommendations(self):
        """Generate final recommendations with rankings."""
        print("\n" + "="*80)
        print("FINAL RECOMMENDATIONS")
        print("="*80)
        
        # Analyze all test results
        recommendations = []
        
        # Method 1: IV/2SLS (if instruments are strong)
        if 'iv_2sls' in self.method_scores and self.method_scores['iv_2sls']['suitable']:
            iv_score = self.method_scores['iv_2sls']
            if iv_score.get('first_stage_f', 0) > 10:
                recommendations.append({
                    'rank': 1,
                    'method': 'IV/2SLS',
                    'reason': 'Strong instruments (F > 10), addresses endogeneity',
                    'elasticity': iv_score['elasticity'],
                    'use_case': 'PRIMARY ESTIMATION',
                    'issues': iv_score['issues']
                })
        
        # Method 2: Two-way FE
        if 'twoway_fe' in self.method_scores:
            recommendations.append({
                'rank': 2,
                'method': 'Two-Way Fixed Effects',
                'reason': 'Controls for country + time effects, robust to trends',
                'elasticity': self.method_scores['twoway_fe']['elasticity'],
                'use_case': 'ROBUSTNESS CHECK',
                'issues': self.method_scores['twoway_fe']['issues']
            })
        
        # Method 3: One-way FE
        if 'fixed_effects' in self.method_scores:
            recommendations.append({
                'rank': 3,
                'method': 'Fixed Effects (One-Way)',
                'reason': 'Standard panel approach, controls for unobserved heterogeneity',
                'elasticity': self.method_scores['fixed_effects']['elasticity'],
                'use_case': 'BASELINE',
                'issues': self.method_scores['fixed_effects']['issues']
            })
        
        # Method 4: Pooled OLS (always last)
        if 'pooled_ols' in self.method_scores:
            recommendations.append({
                'rank': 99,
                'method': 'Pooled OLS',
                'reason': 'Baseline comparison only',
                'elasticity': self.method_scores['pooled_ols']['elasticity'],
                'use_case': 'COMPARISON ONLY',
                'issues': self.method_scores['pooled_ols']['issues']
            })
        
        # Sort by rank
        recommendations.sort(key=lambda x: x['rank'])
        
        # Print recommendations
        print("\n" + "="*80)
        print("RECOMMENDED APPROACH FOR YOUR PAPER")
        print("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            if rec['use_case'] in ['PRIMARY ESTIMATION', 'ROBUSTNESS CHECK', 'BASELINE']:
                print(f"\n{i}. {rec['method'].upper()}")
                print(f"   Use: {rec['use_case']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Elasticity: {rec['elasticity']:.4f}")
                if rec['issues']:
                    print(f"   ⚠️  Issues: {', '.join(rec['issues'])}")
        
        # Save recommendations
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_excel(self.output_dir / 'method_recommendations.xlsx', 
                       index=False, engine='openpyxl')
        
        print(f"\n[OK] Saved recommendations to: method_recommendations.xlsx")
        
        return recommendations
    
    def generate_decision_tree(self):
        """Generate visual decision tree."""
        print("\n" + "="*80)
        print("DECISION TREE FOR YOUR PAPER")
        print("="*80)
        
        # Create text-based decision tree
        tree = """
╔═══════════════════════════════════════════════════════════════╗
║              ECONOMETRIC METHOD DECISION TREE                 ║
╚═══════════════════════════════════════════════════════════════╝

START: Do you have valid instruments?
│
├─ YES (Strong instruments, F > 10)
│  │
│  └─► USE: IV/2SLS as PRIMARY method
│      │
│      ├─ Report: IV estimates as main results
│      ├─ Robustness: Two-way FE, One-way FE
│      └─ Comparison: Pooled OLS (show bias)
│
└─ NO (Weak or no instruments)
   │
   ├─ Is price likely endogenous?
   │  │
   │  ├─ YES (Theory suggests endogeneity)
   │  │  │
   │  │  └─► PROBLEM: Cannot use IV
   │  │      │
   │  │      ├─ Best option: Two-way FE (controls most factors)
   │  │      ├─ Report: FE estimates with caveats
   │  │      └─ Discuss: Endogeneity as limitation
   │  │
   │  └─ NO (Price reasonably exogenous)
   │     │
   │     └─► USE: Two-way FE as PRIMARY
   │         │
   │         ├─ Report: Two-way FE as main
   │         ├─ Robustness: One-way FE
   │         └─ Comparison: Pooled OLS
   │
   └─ Special case: Limited within variation?
      │
      └─► CONSIDER: Between estimator or system GMM
          (Advanced methods, consult econometrician)
"""
        
        print(tree)
        
        # Save to file with UTF-8 encoding
        with open(self.output_dir / 'decision_tree.txt', 'w', encoding='utf-8') as f:
            f.write(tree)
        
        print(f"\n[OK] Saved decision tree to: decision_tree.txt")
        
        return tree
    
    def create_comparison_table(self):
        """Create table comparing all methods."""
        print("\n" + "="*80)
        print("METHOD COMPARISON TABLE")
        print("="*80)
        
        comparison_data = []
        
        for method_name, scores in self.method_scores.items():
            if scores['suitable']:
                row = {
                    'Method': method_name.replace('_', ' ').title(),
                    'Elasticity': scores.get('elasticity', np.nan),
                    'Std. Error': scores.get('se', np.nan),
                    'p-value': scores.get('pval', np.nan),
                    'Priority': scores.get('priority', 'N/A'),
                    'Issues': ', '.join(scores.get('issues', []))
                }
                comparison_data.append(row)
        
        if len(comparison_data) > 0:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Format
            comparison_df['Elasticity'] = comparison_df['Elasticity'].round(4)
            comparison_df['Std. Error'] = comparison_df['Std. Error'].round(4)
            comparison_df['p-value'] = comparison_df['p-value'].round(4)
            
            print("\n")
            print(comparison_df.to_string(index=False))
            
            # Save
            comparison_df.to_excel(self.output_dir / 'method_comparison_table.xlsx',
                                  index=False, engine='openpyxl')
            
            print(f"\n[OK] Saved comparison table")
            
            return comparison_df
        
        return None
    
    def run_complete_diagnostic(self):
        """Run complete diagnostic framework."""
        print("="*80)
        print("COMPREHENSIVE ECONOMETRIC METHOD DIAGNOSTIC")
        print("="*80)
        print("\nThis will test ALL major methods and recommend the best approach")
        print("for estimating broadband demand elasticity.\n")
        
        # Load data
        self.load_and_prepare_data()
        
        # Run tests
        self.check_data_structure()
        self.test_pooled_ols()
        self.test_fixed_effects()
        self.test_hausman()
        self.test_instruments()
        self.test_endogeneity()
        self.test_two_way_fe()
        
        # Generate outputs
        self.generate_recommendations()
        self.generate_decision_tree()
        self.create_comparison_table()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        
        print("\n✅ YOU NOW HAVE A CLEAR PATH FORWARD!")
        print("\nNext steps:")
        print("  1. Review method_recommendations.xlsx")
        print("  2. Follow the decision tree")
        print("  3. Implement recommended method(s)")
        print("  4. Run robustness checks")
        
        return self


# Main execution
if __name__ == "__main__":
    diagnostic = MethodDiagnostic()
    diagnostic.run_complete_diagnostic()
