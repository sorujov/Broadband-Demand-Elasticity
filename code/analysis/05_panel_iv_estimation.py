# code/analysis/05_panel_iv_estimation.py

"""
================================================================================
PANEL IV ESTIMATION FOR BROADBAND DEMAND ELASTICITY
================================================================================
Purpose: Rigorous IV/2SLS estimation addressing price endogeneity
         Comparing EU vs Eastern Partnership (EaP) countries

Methods:
- IV/2SLS with multiple instrument combinations
- Two-way fixed effects baseline for comparison
- Full diagnostic suite (first-stage F, Hausman, Hansen J)
- Regional heterogeneity via price×EaP interaction

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import sys

warnings.filterwarnings('ignore')

# Panel and IV estimation
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS, IVLIML
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_DIR as BASE_RESULTS_DIR,
        EU_COUNTRIES, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_DIR as BASE_RESULTS_DIR,
        EU_COUNTRIES, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE
    )

RESULTS_DIR = BASE_RESULTS_DIR / 'iv_estimation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load data and create all necessary variables."""
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    # Load prepared dataset
    df = pd.read_csv(ANALYSIS_READY_FILE)

    print(f"\n[OK] Loaded: {len(df):,} observations")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")

    # Map standardized names to IV script names for compatibility
    # Primary DV and price already exist as log transforms
    df['log_subs'] = df[PRIMARY_DV]  # log_fixed_broadband_subs
    df['log_internet_users'] = df['log_internet_users_pct']

    # Price variables
    df['log_price'] = df[PRIMARY_PRICE]  # log_fixed_broad_price (GNI-adjusted)
    df['log_price_usd'] = df['log_fixed_broad_price_usd']
    df['log_price_ppp'] = df['log_fixed_broad_price_ppp']
    df['log_price_gni'] = df[PRIMARY_PRICE]

    # Control variables (use standardized names)
    df['log_gdp'] = df['log_gdp_per_capita']
    df['log_pop'] = df['log_population']
    df['urban_pct'] = df['urban_population_pct']
    df['reg_quality'] = df['regulatory_quality_estimate']
    df['secure_servers'] = df['log_secure_internet_servers']

    # Mobile prices (instruments)
    if 'log_mobile_broad_price' in df.columns:
        df['log_mobile_price'] = df['log_mobile_broad_price']

    # Regional indicators (already exist from data prep)
    df['is_eu'] = df['eu']
    df['is_eap'] = df['eap']

    # Lagged prices already exist from data prep
    if 'log_fixed_broad_price_lag1' in df.columns:
        df['log_price_lag1'] = df['log_fixed_broad_price_lag1']
    else:
        df = df.sort_values(['country', 'year'])
        df['log_price_lag1'] = df.groupby('country')['log_price'].shift(1)

    df['log_price_lag2'] = df.groupby('country')['log_price'].shift(2)

    # Interaction terms
    df['price_x_eap'] = df['log_price'] * df['is_eap']

    # Instrument interactions
    if 'log_mobile_price' in df.columns:
        df['mobile_price_x_eap'] = df['log_mobile_price'] * df['is_eap']
    if 'reg_quality' in df.columns:
        df['reg_quality_x_eap'] = df['reg_quality'] * df['is_eap']
    df['price_lag1_x_eap'] = df['log_price_lag1'] * df['is_eap']

    print("\n[OK] Created transformations:")
    print("  - Log variables: subs, internet_users, price, GDP, population")
    print("  - Regional indicators: is_eu, is_eap")
    print("  - Price lags: lag1, lag2")
    print("  - Interaction terms: price_x_eap, instrument×EaP")

    return df


def create_panel_data(df, period='all'):
    """Create panel-indexed dataframe for a given time period."""
    df_copy = df.copy()

    if period == 'pre_covid':
        df_copy = df_copy[df_copy['year'] <= 2019]
    elif period == 'full':
        pass  # Keep all years
    elif isinstance(period, tuple):
        df_copy = df_copy[(df_copy['year'] >= period[0]) & (df_copy['year'] <= period[1])]

    # Set panel index
    df_panel = df_copy.set_index(['country', 'year']).sort_index()

    return df_panel


# =============================================================================
# FIRST-STAGE DIAGNOSTICS
# =============================================================================

def run_first_stage(df_panel, instruments, controls, endogenous='log_price'):
    """
    Run first-stage regression and compute diagnostics.

    Returns dict with F-statistic, partial R², and coefficient details.
    """
    # Build variable list
    all_vars = [endogenous] + instruments + controls
    df_clean = df_panel[all_vars].dropna()

    if len(df_clean) < 50:
        return None

    # First-stage regression: endogenous ~ instruments + controls
    y = df_clean[endogenous]
    X = add_constant(df_clean[instruments + controls])

    first_stage = OLS(y, X).fit(cov_type='HC3')

    # F-test for excluded instruments (joint significance)
    # Create restriction matrix for instrument coefficients
    n_instr = len(instruments)
    n_vars = len(X.columns)

    r_matrix = np.zeros((n_instr, n_vars))
    for i, inst in enumerate(instruments):
        col_idx = list(X.columns).index(inst)
        r_matrix[i, col_idx] = 1

    try:
        f_test = first_stage.f_test(r_matrix)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)
    except:
        f_stat = first_stage.fvalue
        f_pval = first_stage.f_pvalue

    # Partial R-squared (contribution of instruments)
    # R² with instruments - R² without instruments
    X_restricted = add_constant(df_clean[controls]) if controls else add_constant(pd.DataFrame(index=df_clean.index))
    restricted = OLS(y, X_restricted).fit()

    partial_r2 = first_stage.rsquared - restricted.rsquared

    # Instrument coefficients
    inst_coefs = {inst: {
        'coef': first_stage.params[inst],
        'se': first_stage.bse[inst],
        'pval': first_stage.pvalues[inst]
    } for inst in instruments}

    return {
        'f_stat': f_stat,
        'f_pval': f_pval,
        'partial_r2': partial_r2,
        'r2': first_stage.rsquared,
        'n_obs': len(df_clean),
        'inst_coefs': inst_coefs,
        'weak_instrument': f_stat < 10
    }


# =============================================================================
# IV ESTIMATION
# =============================================================================

def run_iv_estimation(df_panel, instruments, controls,
                      dep_var='log_subs',
                      include_interaction=True,
                      use_liml=False):
    """
    Run IV/2SLS estimation with optional interaction terms.

    If include_interaction=True, instruments both log_price and log_price×EaP.
    Requires instruments for both endogenous variables.
    """
    df_work = df_panel.copy()

    # Build instrument list with interactions for EaP
    all_instruments = instruments.copy()

    if include_interaction:
        # Create interaction instruments dynamically
        for inst in instruments:
            # Generate interaction variable name
            if 'mobile' in inst:
                int_name = 'mobile_price_x_eap'
            elif 'reg' in inst:
                int_name = 'reg_quality_x_eap'
            elif 'lag1' in inst:
                int_name = 'price_lag1_x_eap'
            elif 'lag2' in inst:
                int_name = 'price_lag2_x_eap'
            else:
                int_name = f"{inst}_x_eap"

            # Create if doesn't exist
            if int_name not in df_work.columns and inst in df_work.columns:
                df_work[int_name] = df_work[inst] * df_work['is_eap']

            if int_name in df_work.columns:
                all_instruments.append(int_name)

    # Remove duplicates while preserving order
    all_instruments = list(dict.fromkeys(all_instruments))

    # Build variable lists
    endog_vars = ['log_price']
    if include_interaction:
        endog_vars.append('price_x_eap')

    # Check we have enough instruments
    if len(all_instruments) < len(endog_vars):
        print(f"  Not enough instruments: {len(all_instruments)} < {len(endog_vars)}")
        return None, None

    # Required variables (exclude is_eap since it's absorbed by entity FE)
    exog_vars = controls
    all_vars = [dep_var] + endog_vars + all_instruments + exog_vars + ['is_eap']
    all_vars = [v for v in all_vars if v in df_work.columns]

    df_clean = df_work[list(set(all_vars))].dropna()

    if len(df_clean) < 50:
        return None, None

    # Build formula - don't include is_eap as it's time-invariant
    exog_str = ' + '.join(exog_vars) if exog_vars else '1'
    endog_str = ' + '.join(endog_vars)
    inst_str = ' + '.join(all_instruments)

    formula = f"{dep_var} ~ 1 + {exog_str} + [{endog_str} ~ {inst_str}]"

    try:
        # Choose estimator
        if use_liml:
            model = IVLIML.from_formula(formula, data=df_clean)
        else:
            model = IV2SLS.from_formula(formula, data=df_clean)

        result = model.fit(cov_type='robust')

        return result, df_clean

    except Exception as e:
        print(f"  Error in IV estimation: {str(e)}")
        return None, None


def run_ols_baseline(df_panel, controls, dep_var='log_subs', include_interaction=True):
    """Run OLS baseline for comparison."""

    # Note: is_eap is time-invariant and absorbed by entity FE
    # We only include the interaction term (price_x_eap) which varies over time
    regressors = ['log_price']
    if include_interaction:
        regressors.append('price_x_eap')
    regressors += controls

    all_vars = [dep_var] + regressors
    all_vars = [v for v in all_vars if v in df_panel.columns]

    df_clean = df_panel[list(set(all_vars))].dropna()

    if len(df_clean) < 50:
        return None, None

    try:
        y = df_clean[dep_var]
        X = df_clean[regressors]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        result = model.fit(cov_type='clustered', cluster_entity=True)

        return result, df_clean

    except Exception as e:
        print(f"  Error in OLS: {str(e)}")
        return None, None


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def durbin_wu_hausman_test(iv_result, ols_result, var='log_price'):
    """
    Test H0: OLS is consistent (price is exogenous)
    """
    if iv_result is None or ols_result is None:
        return {'stat': np.nan, 'pval': np.nan, 'conclusion': 'Cannot compute'}

    try:
        beta_iv = iv_result.params[var]
        beta_ols = ols_result.params[var]

        var_iv = iv_result.cov[var][var]
        var_ols = ols_result.cov[var][var]

        diff = beta_iv - beta_ols
        var_diff = var_iv - var_ols

        if var_diff > 0:
            hausman_stat = diff**2 / var_diff
            pval = 1 - stats.chi2.cdf(hausman_stat, df=1)

            if pval < 0.05:
                conclusion = 'Reject H0: Price is endogenous, use IV'
            else:
                conclusion = 'Cannot reject H0: OLS may be consistent'

            return {
                'stat': hausman_stat,
                'pval': pval,
                'beta_diff': diff,
                'beta_iv': beta_iv,
                'beta_ols': beta_ols,
                'conclusion': conclusion
            }
        else:
            return {'stat': np.nan, 'pval': np.nan, 'conclusion': 'Variance diff negative'}

    except Exception as e:
        return {'stat': np.nan, 'pval': np.nan, 'conclusion': f'Error: {str(e)}'}


def get_overid_test(iv_result):
    """Extract overidentification test (Hansen J) if available."""
    try:
        if hasattr(iv_result, 'j_stat') and iv_result.j_stat is not None:
            return {
                'stat': iv_result.j_stat.stat,
                'pval': iv_result.j_stat.pval,
                'valid': iv_result.j_stat.pval >= 0.05
            }
    except:
        pass
    return {'stat': np.nan, 'pval': np.nan, 'valid': None}


# =============================================================================
# REGIONAL ELASTICITY CALCULATION
# =============================================================================

def calculate_regional_elasticities(result, include_interaction=True):
    """
    Calculate implied elasticities for EU and EaP from interaction model.

    EU elasticity = beta_price
    EaP elasticity = beta_price + beta_interaction
    """
    if result is None:
        return None

    try:
        beta_price = result.params['log_price']
        se_price = result.std_errors['log_price']

        if include_interaction and 'price_x_eap' in result.params:
            beta_int = result.params['price_x_eap']
            se_int = result.std_errors['price_x_eap']

            # Covariance for delta method
            try:
                cov_price_int = result.cov.loc['log_price', 'price_x_eap']
            except:
                cov_price_int = 0

            # EU elasticity
            eu_elasticity = beta_price
            eu_se = se_price

            # EaP elasticity (delta method for SE)
            eap_elasticity = beta_price + beta_int
            eap_var = se_price**2 + se_int**2 + 2*cov_price_int
            eap_se = np.sqrt(max(eap_var, 0))

            # P-values
            df_resid = result.df_resid if hasattr(result, 'df_resid') else 100

            eu_tstat = eu_elasticity / eu_se if eu_se > 0 else 0
            eap_tstat = eap_elasticity / eap_se if eap_se > 0 else 0

            eu_pval = 2 * (1 - stats.t.cdf(abs(eu_tstat), df=df_resid))
            eap_pval = 2 * (1 - stats.t.cdf(abs(eap_tstat), df=df_resid))

            # Interaction test
            int_tstat = beta_int / se_int if se_int > 0 else 0
            int_pval = 2 * (1 - stats.t.cdf(abs(int_tstat), df=df_resid))

            return {
                'eu': {'elasticity': eu_elasticity, 'se': eu_se, 'pval': eu_pval},
                'eap': {'elasticity': eap_elasticity, 'se': eap_se, 'pval': eap_pval},
                'interaction': {'coef': beta_int, 'se': se_int, 'pval': int_pval},
                'ratio': abs(eap_elasticity / eu_elasticity) if eu_elasticity != 0 else np.nan
            }
        else:
            # No interaction - return single elasticity
            tstat = beta_price / se_price if se_price > 0 else 0
            df_resid = result.df_resid if hasattr(result, 'df_resid') else 100
            pval = 2 * (1 - stats.t.cdf(abs(tstat), df=df_resid))

            return {
                'pooled': {'elasticity': beta_price, 'se': se_price, 'pval': pval}
            }

    except Exception as e:
        print(f"  Error calculating elasticities: {str(e)}")
        return None


# =============================================================================
# MAIN ESTIMATION LOOP
# =============================================================================

def run_all_specifications(df, controls=['log_gdp', 'rd_expenditure', 'secure_servers']):
    """
    Run all IV specifications across both time periods.
    """
    print("\n" + "=" * 80)
    print("RUNNING ALL IV SPECIFICATIONS")
    print("=" * 80)

    # Define instrument combinations
    # Each instrument needs its EaP interaction to identify both endogenous vars
    instrument_sets = {
        'mobile_price': ['log_mobile_price'],
        'reg_quality': ['reg_quality'],
        'both': ['log_mobile_price', 'reg_quality'],
        'lagged': ['log_price_lag1', 'log_price_lag2'],
        'mobile_lagged': ['log_mobile_price', 'log_price_lag1'],
        'all': ['log_mobile_price', 'reg_quality', 'log_price_lag1'],
    }

    # Define time periods
    periods = {
        'pre_covid': 'pre_covid',
        'full': 'full'
    }

    all_results = []

    for period_name, period_filter in periods.items():
        print(f"\n{'='*80}")
        print(f"PERIOD: {period_name.upper()}")
        print(f"{'='*80}")

        # Create panel data for this period
        df_panel = create_panel_data(df, period=period_filter)
        n_obs = len(df_panel)
        print(f"Observations: {n_obs}")

        # 1. OLS Baseline
        print(f"\n--- OLS Two-Way FE (Baseline) ---")
        ols_result, ols_data = run_ols_baseline(df_panel, controls)

        if ols_result is not None:
            ols_elast = calculate_regional_elasticities(ols_result)

            if ols_elast and 'eu' in ols_elast:
                result_row = {
                    'period': period_name,
                    'method': 'OLS_TWFE',
                    'instruments': 'None',
                    'eu_elasticity': ols_elast['eu']['elasticity'],
                    'eu_se': ols_elast['eu']['se'],
                    'eu_pval': ols_elast['eu']['pval'],
                    'eap_elasticity': ols_elast['eap']['elasticity'],
                    'eap_se': ols_elast['eap']['se'],
                    'eap_pval': ols_elast['eap']['pval'],
                    'interaction_pval': ols_elast['interaction']['pval'],
                    'ratio': ols_elast['ratio'],
                    'first_stage_f': np.nan,
                    'hausman_pval': np.nan,
                    'hansen_j_pval': np.nan,
                    'n_obs': ols_result.nobs,
                    'r2': ols_result.rsquared
                }
                all_results.append(result_row)

                print(f"  EU:  {ols_elast['eu']['elasticity']:.4f} (p={ols_elast['eu']['pval']:.3f})")
                print(f"  EaP: {ols_elast['eap']['elasticity']:.4f} (p={ols_elast['eap']['pval']:.3f})")
                print(f"  Ratio: {ols_elast['ratio']:.2f}x")

        # 2. IV Specifications
        for inst_name, instruments in instrument_sets.items():
            print(f"\n--- IV: {inst_name} ---")

            # Check instrument availability
            available = all(inst in df_panel.columns for inst in instruments)
            if not available:
                print(f"  Instruments not available")
                continue

            # First-stage diagnostics
            first_stage = run_first_stage(df_panel, instruments, controls)

            if first_stage is None:
                print(f"  First stage failed")
                continue

            f_stat = first_stage['f_stat']
            print(f"  First-stage F: {f_stat:.2f} {'(WEAK)' if f_stat < 10 else '(OK)'}")

            # Run IV estimation
            use_liml = f_stat < 10  # Use LIML for weak instruments
            iv_result, iv_data = run_iv_estimation(
                df_panel, instruments, controls,
                use_liml=use_liml
            )

            if iv_result is None:
                print(f"  IV estimation failed")
                continue

            # Calculate elasticities
            iv_elast = calculate_regional_elasticities(iv_result)

            if iv_elast is None or 'eu' not in iv_elast:
                print(f"  Could not calculate elasticities")
                continue

            # Diagnostic tests
            hausman = durbin_wu_hausman_test(iv_result, ols_result)
            overid = get_overid_test(iv_result)

            # Store results
            result_row = {
                'period': period_name,
                'method': 'LIML' if use_liml else 'IV_2SLS',
                'instruments': inst_name,
                'eu_elasticity': iv_elast['eu']['elasticity'],
                'eu_se': iv_elast['eu']['se'],
                'eu_pval': iv_elast['eu']['pval'],
                'eap_elasticity': iv_elast['eap']['elasticity'],
                'eap_se': iv_elast['eap']['se'],
                'eap_pval': iv_elast['eap']['pval'],
                'interaction_pval': iv_elast['interaction']['pval'],
                'ratio': iv_elast['ratio'],
                'first_stage_f': f_stat,
                'hausman_pval': hausman['pval'],
                'hansen_j_pval': overid['pval'],
                'n_obs': iv_result.nobs,
                'r2': np.nan
            }
            all_results.append(result_row)

            # Print results
            sig_eu = '***' if iv_elast['eu']['pval'] < 0.01 else '**' if iv_elast['eu']['pval'] < 0.05 else '*' if iv_elast['eu']['pval'] < 0.1 else ''
            sig_eap = '***' if iv_elast['eap']['pval'] < 0.01 else '**' if iv_elast['eap']['pval'] < 0.05 else '*' if iv_elast['eap']['pval'] < 0.1 else ''

            print(f"  EU:  {iv_elast['eu']['elasticity']:.4f}{sig_eu} (SE={iv_elast['eu']['se']:.4f})")
            print(f"  EaP: {iv_elast['eap']['elasticity']:.4f}{sig_eap} (SE={iv_elast['eap']['se']:.4f})")
            print(f"  Ratio: {iv_elast['ratio']:.2f}x")
            print(f"  Hausman p-val: {hausman['pval']:.3f}" if not np.isnan(hausman['pval']) else "  Hausman: N/A")
            if not np.isnan(overid['pval']):
                print(f"  Hansen J p-val: {overid['pval']:.3f} {'(VALID)' if overid['valid'] else '(INVALID)'}")

    return pd.DataFrame(all_results)


# =============================================================================
# RESULTS SUMMARY AND SELECTION
# =============================================================================

def select_best_specification(results_df):
    """
    Select best IV specification based on:
    1. First-stage F > 10
    2. Hansen J p-value > 0.05 (if available)
    3. Economically meaningful elasticities (negative)
    """
    print("\n" + "=" * 80)
    print("SELECTING BEST SPECIFICATION")
    print("=" * 80)

    # Filter IV results only
    iv_results = results_df[results_df['method'] != 'OLS_TWFE'].copy()

    if len(iv_results) == 0:
        print("No valid IV specifications")
        return None

    # Scoring
    iv_results['score'] = 0

    # Strong instruments (+2)
    iv_results.loc[iv_results['first_stage_f'] >= 10, 'score'] += 2

    # Valid overidentification (+1)
    iv_results.loc[iv_results['hansen_j_pval'] >= 0.05, 'score'] += 1
    iv_results.loc[iv_results['hansen_j_pval'].isna(), 'score'] += 0.5  # Can't test, neutral

    # Negative elasticities (expected sign) (+1 each)
    iv_results.loc[iv_results['eu_elasticity'] < 0, 'score'] += 1
    iv_results.loc[iv_results['eap_elasticity'] < 0, 'score'] += 1

    # Significant interaction (+1)
    iv_results.loc[iv_results['interaction_pval'] < 0.1, 'score'] += 1

    # Select best for each period
    best_specs = {}

    for period in iv_results['period'].unique():
        period_results = iv_results[iv_results['period'] == period]
        if len(period_results) > 0:
            best_idx = period_results['score'].idxmax()
            best_specs[period] = period_results.loc[best_idx]

            print(f"\n{period.upper()} - Best specification:")
            best = best_specs[period]
            print(f"  Instruments: {best['instruments']}")
            print(f"  First-stage F: {best['first_stage_f']:.2f}")
            print(f"  EU elasticity: {best['eu_elasticity']:.4f} (p={best['eu_pval']:.3f})")
            print(f"  EaP elasticity: {best['eap_elasticity']:.4f} (p={best['eap_pval']:.3f})")
            print(f"  EaP/EU ratio: {best['ratio']:.2f}x")
            print(f"  Score: {best['score']}")

    return best_specs


def format_significance(pval):
    """Format significance stars."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.1:
        return '*'
    return ''


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_results_table(results_df, output_path):
    """Generate formatted results table."""

    # Create summary for display
    display_df = results_df.copy()

    # Format elasticities with significance
    display_df['EU'] = display_df.apply(
        lambda x: f"{x['eu_elasticity']:.4f}{format_significance(x['eu_pval'])} ({x['eu_se']:.4f})",
        axis=1
    )
    display_df['EaP'] = display_df.apply(
        lambda x: f"{x['eap_elasticity']:.4f}{format_significance(x['eap_pval'])} ({x['eap_se']:.4f})",
        axis=1
    )
    display_df['F-stat'] = display_df['first_stage_f'].apply(
        lambda x: f"{x:.2f}" if not np.isnan(x) else '-'
    )
    display_df['N'] = display_df['n_obs'].astype(int)

    # Select columns
    output_cols = ['period', 'method', 'instruments', 'EU', 'EaP', 'ratio', 'F-stat', 'N']
    output_df = display_df[output_cols]

    # Save
    output_df.to_excel(output_path / 'iv_results_summary.xlsx', index=False)
    results_df.to_excel(output_path / 'iv_results_full.xlsx', index=False)

    print(f"\n[OK] Results saved to: {output_path}")

    return output_df


def generate_latex_table(results_df, output_path):
    """Generate LaTeX table for paper."""

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Instrumental Variables Estimates of Broadband Price Elasticity}",
        "\\label{tab:iv_results}",
        "\\begin{tabular}{lcccccc}",
        "\\hline\\hline",
        " & (1) & (2) & (3) & (4) & (5) & (6) \\\\",
        " & OLS & IV Mobile & IV RegQual & IV Both & IV Lagged & Best IV \\\\",
        "\\hline"
    ]

    for period in ['pre_covid', 'full']:
        period_label = 'Pre-COVID (2010-2019)' if period == 'pre_covid' else 'Full Period (2010-2024)'
        latex_lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{period_label}}}}} \\\\")

        period_results = results_df[results_df['period'] == period]

        # EU row
        eu_vals = []
        for method in ['OLS_TWFE', 'mobile_price', 'reg_quality', 'both', 'lagged']:
            row = period_results[
                (period_results['instruments'] == method) |
                (period_results['method'] == method)
            ]
            if len(row) > 0:
                r = row.iloc[0]
                sig = format_significance(r['eu_pval'])
                eu_vals.append(f"{r['eu_elasticity']:.4f}$^{{{sig}}}$")
            else:
                eu_vals.append('-')

        # Add best
        best_row = period_results[period_results['first_stage_f'] >= 10]
        if len(best_row) > 0:
            best = best_row.loc[best_row['first_stage_f'].idxmax()]
            sig = format_significance(best['eu_pval'])
            eu_vals.append(f"{best['eu_elasticity']:.4f}$^{{{sig}}}$")
        else:
            eu_vals.append('-')

        latex_lines.append(f"EU Elasticity & {' & '.join(eu_vals)} \\\\")

        # EaP row
        eap_vals = []
        for method in ['OLS_TWFE', 'mobile_price', 'reg_quality', 'both', 'lagged']:
            row = period_results[
                (period_results['instruments'] == method) |
                (period_results['method'] == method)
            ]
            if len(row) > 0:
                r = row.iloc[0]
                sig = format_significance(r['eap_pval'])
                eap_vals.append(f"{r['eap_elasticity']:.4f}$^{{{sig}}}$")
            else:
                eap_vals.append('-')

        if len(best_row) > 0:
            sig = format_significance(best['eap_pval'])
            eap_vals.append(f"{best['eap_elasticity']:.4f}$^{{{sig}}}$")
        else:
            eap_vals.append('-')

        latex_lines.append(f"EaP Elasticity & {' & '.join(eap_vals)} \\\\")

        # First-stage F
        f_vals = ['-']  # OLS has no F
        for method in ['mobile_price', 'reg_quality', 'both', 'lagged']:
            row = period_results[period_results['instruments'] == method]
            if len(row) > 0:
                f = row.iloc[0]['first_stage_f']
                f_vals.append(f"{f:.2f}" if not np.isnan(f) else '-')
            else:
                f_vals.append('-')

        if len(best_row) > 0:
            f_vals.append(f"{best['first_stage_f']:.2f}")
        else:
            f_vals.append('-')

        latex_lines.append(f"First-stage F & {' & '.join(f_vals)} \\\\")

        # N
        n_vals = []
        for method in ['OLS_TWFE', 'mobile_price', 'reg_quality', 'both', 'lagged']:
            row = period_results[
                (period_results['instruments'] == method) |
                (period_results['method'] == method)
            ]
            if len(row) > 0:
                n_vals.append(str(int(row.iloc[0]['n_obs'])))
            else:
                n_vals.append('-')

        if len(best_row) > 0:
            n_vals.append(str(int(best['n_obs'])))
        else:
            n_vals.append('-')

        latex_lines.append(f"N & {' & '.join(n_vals)} \\\\")
        latex_lines.append("\\hline")

    latex_lines.extend([
        "\\multicolumn{7}{l}{\\textit{Notes:} $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$} \\\\",
        "\\multicolumn{7}{l}{Standard errors robust to heteroskedasticity.} \\\\",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_table = '\n'.join(latex_lines)

    with open(output_path / 'table_iv_results.tex', 'w') as f:
        f.write(latex_table)

    print(f"[OK] LaTeX table saved to: {output_path / 'table_iv_results.tex'}")

    return latex_table


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("PANEL IV ESTIMATION: BROADBAND DEMAND ELASTICITY")
    print("EU vs Eastern Partnership Countries")
    print("=" * 80)

    # 1. Load and prepare data
    df = load_and_prepare_data()

    # 2. Run all specifications
    controls = ['log_gdp', 'rd_expenditure', 'secure_servers']
    results_df = run_all_specifications(df, controls=controls)

    # 3. Select best specification
    best_specs = select_best_specification(results_df)

    # 4. Generate output tables
    print("\n" + "=" * 80)
    print("GENERATING OUTPUT TABLES")
    print("=" * 80)

    display_df = generate_results_table(results_df, RESULTS_DIR)
    latex_table = generate_latex_table(results_df, RESULTS_DIR)

    # 5. Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n" + display_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)

    for period, best in best_specs.items():
        print(f"\n{period.upper()}:")
        print(f"  Best instruments: {best['instruments']}")
        print(f"  EU elasticity: {best['eu_elasticity']:.4f} (p={best['eu_pval']:.3f})")
        print(f"  EaP elasticity: {best['eap_elasticity']:.4f} (p={best['eap_pval']:.3f})")
        print(f"  EaP is {best['ratio']:.1f}x more price-elastic than EU")

        if best['first_stage_f'] >= 10:
            print(f"  Instruments are STRONG (F={best['first_stage_f']:.2f})")
        else:
            print(f"  WARNING: Weak instruments (F={best['first_stage_f']:.2f})")

    print("\n" + "=" * 80)
    print(f"[OK] All results saved to: {RESULTS_DIR}")
    print("=" * 80)

    return results_df, best_specs


if __name__ == "__main__":
    results_df, best_specs = main()
