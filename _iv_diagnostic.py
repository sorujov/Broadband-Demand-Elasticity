# -*- coding: utf-8 -*-
"""Diagnostic: test all instrument candidates + Arellano-Bond GMM."""

import pandas as pd
import numpy as np
import warnings
import sys
import io

from linearmodels.iv import IV2SLS
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---- Load data ----
df = pd.read_csv('data/processed/analysis_ready_data.csv')
df['year_num'] = df['year'].astype(int)
df['eap_dummy'] = df['eap'].astype(float)
df['covid_dummy'] = (df['year_num'] >= 2020).astype(float)
df = df.sort_values(['country', 'year_num']).reset_index(drop=True)

CONTROLS = [
    'log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
    'regulatory_quality_estimate', 'log_secure_internet_servers',
    'research_development_expenditure', 'log_population_density',
    'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator',
]
DV = 'log_fixed_broadband_subs'
PRICE = 'log_fixed_broad_price'


def run_2sls(df_in, instrs_list, label, covid_inter=False):
    """Run 2SLS with within-transformation, return result dict."""
    d = df_in.copy()
    d['price_x_eap'] = d[PRICE] * d['eap_dummy']
    endog = [PRICE, 'price_x_eap']
    iv_names = []
    for ins in instrs_list:
        d[ins + '_x_eap'] = d[ins] * d['eap_dummy']
        iv_names += [ins, ins + '_x_eap']

    if covid_inter:
        d['price_x_cov'] = d[PRICE] * d['covid_dummy']
        d['price_x_eap_x_cov'] = d['price_x_eap'] * d['covid_dummy']
        endog += ['price_x_cov', 'price_x_eap_x_cov']
        for ins in instrs_list:
            d[ins + '_x_cov'] = d[ins] * d['covid_dummy']
            d[ins + '_x_eap_x_cov'] = d[ins + '_x_eap'] * d['covid_dummy']
            iv_names += [ins + '_x_cov', ins + '_x_eap_x_cov']

    avail = [c for c in CONTROLS if c in d.columns]
    needed = list(dict.fromkeys([DV] + endog + iv_names + avail + ['eap_dummy', 'country', 'year_num']))
    if covid_inter:
        needed.append('covid_dummy')
    d = d[[c for c in needed if c in d.columns]].dropna()
    if len(d) < 50:
        print(f'{label}: INSUFFICIENT OBS ({len(d)})')
        return None

    # ---- First-stage partial F for primary price equation ----
    X_fs = sm.add_constant(d[iv_names + avail].values)
    y_fs = d[PRICE].values
    b_fs = np.linalg.lstsq(X_fs, y_fs, rcond=None)[0]
    yhat = X_fs @ b_fs
    res = y_fs - yhat
    n_fs = len(y_fs)
    k_fs = X_fs.shape[1]
    ssres = np.sum(res ** 2)
    X_r = sm.add_constant(d[avail].values) if avail else np.ones((n_fs, 1))
    b_r = np.linalg.lstsq(X_r, y_fs, rcond=None)[0]
    ssres_r = np.sum((y_fs - X_r @ b_r) ** 2)
    F_p = ((ssres_r - ssres) / len(iv_names)) / (ssres / (n_fs - k_fs))

    # ---- Two-way within demeaning ----
    all_cols = list(dict.fromkeys([DV] + endog + iv_names + avail))
    dm = d[all_cols + ['country', 'year_num']].copy()
    for col in all_cols:
        em = dm.groupby('country')[col].transform('mean')
        tm = dm.groupby('year_num')[col].transform('mean')
        gm = dm[col].mean()
        dm[col] = dm[col] - em - tm + gm

    dep = dm[DV]
    endog_dm = dm[endog]
    iv_dm = dm[iv_names]
    exog_dm = dm[avail] if avail else None

    try:
        model = IV2SLS(dep, exog=exog_dm, endog=endog_dm, instruments=iv_dm)
        res2 = model.fit(cov_type='robust')

        b_eu = res2.params[PRICE]
        se_eu = res2.std_errors[PRICE]
        p_eu = res2.pvalues[PRICE]
        b_int = res2.params['price_x_eap']
        se_int = res2.std_errors['price_x_eap']
        cov_eu_int = res2.cov.loc[PRICE, 'price_x_eap']
        eap_b = b_eu + b_int
        eap_se = np.sqrt(se_eu ** 2 + se_int ** 2 + 2 * cov_eu_int)
        eap_p = 2 * (1 - stats.norm.cdf(abs(eap_b / eap_se)))

        sg = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f'  {label:<52s}  F={F_p:7.1f}  N={len(d)}')
        print(f'      EU ={b_eu:+.4f}{sg(p_eu):3}  SE={se_eu:.4f}  p={p_eu:.3f}')
        print(f'      EaP={eap_b:+.4f}{sg(eap_p):3}  SE={eap_se:.4f}  p={eap_p:.3f}')

        return dict(label=label, F=F_p, n=len(d),
                    b_eu=b_eu, se_eu=se_eu, p_eu=p_eu,
                    b_eap=eap_b, se_eap=eap_se, p_eap=eap_p)
    except Exception as e:
        print(f'  {label}: 2SLS ERROR: {e}')
        return None


# Filters
pre_covid = df[df.year_num <= 2019].copy()
full_sample = df.copy()

print('=' * 70)
print('PRE-COVID (2010-2019)')
print('=' * 70)
run_2sls(pre_covid, ['log_mobile_broad_price'],                         '[A] Mobile price (current instrument)')
run_2sls(pre_covid, ['log_fixed_broad_price_lag1'],                     '[B] Lagged own fixed price')
run_2sls(pre_covid, ['log_int_bandwidth'],                              '[C] Internet bandwidth (supply shifter)')
run_2sls(pre_covid, ['log_fixed_broad_price_lag1', 'log_int_bandwidth'], '[D] Lagged price + bandwidth (overid)')

print()
print('=' * 70)
print('FULL SAMPLE (2010-2024) with COVID interactions')
print('=' * 70)
run_2sls(full_sample, ['log_mobile_broad_price'],                         '[A] Mobile price (current)', True)
run_2sls(full_sample, ['log_fixed_broad_price_lag1'],                     '[B] Lagged own fixed price', True)
run_2sls(full_sample, ['log_int_bandwidth'],                              '[C] Internet bandwidth',     True)
run_2sls(full_sample, ['log_fixed_broad_price_lag1', 'log_int_bandwidth'], '[D] Lagged price + bandwidth (overid)', True)

# ============================================================================
# ARELLANO-BOND / BLUNDELL-BOND GMM
# ============================================================================
print()
print('=' * 70)
print('ARELLANO-BOND DIFFERENCE GMM (pydynpd)')
print('=' * 70)

try:
    from pydynpd import regression as pdreg

    # pydynpd requires: flat df with individual ID (string/int) and time (int)
    # No multi-index. Vars must be present as columns.
    df_ab = df[['country', 'year_num', DV, PRICE,
                'price_x_eap' if 'price_x_eap' in df.columns else 'eap_dummy',
                'eap_dummy'] + CONTROLS].copy().dropna(subset=[DV, PRICE])

    # Create interaction manually
    df_ab['price_x_eap'] = df_ab[PRICE] * df_ab['eap_dummy']

    # pydynpd command string syntax:
    # "depvar L(1:1).depvar indepvar1 indepvar2 | gmm(varname, min_lag, max_lag) iv(exog) | options"
    # steps(2) = two-step; nolevel = difference GMM only; level = system GMM

    avail_ab = [c for c in CONTROLS if c in df_ab.columns]
    ctrl_str = ' '.join(avail_ab)

    # Spec 1: Difference GMM, lagged DV as GMM instrument, price + price_x_eap as endogenous
    cmd1 = (f'{DV} L1.{DV} {PRICE} price_x_eap {ctrl_str} | '
            f'gmm({DV}, 2, 4) gmm({PRICE}, 2, 4) gmm(price_x_eap, 2, 4) iv({ctrl_str}) | steps(2)')
    print(f'\nAB Spec 1 (DiGMM, lags 2-4 as instruments):')
    print(f'  CMD: {cmd1[:80]}...')
    try:
        m1 = pdreg.abond(cmd1, df_ab, ['country', 'year_num'])
        print('  AB Spec 1: OK')
        if hasattr(m1, 'models') and m1.models:
            mod = m1.models[0]
            rt = mod.regression_table
            print(rt[['variable', 'coefficient', 'std_err', 'p_value']].to_string(index=False))
            print(f'  Hansen J p={mod.hansen.p_value:.3f}  AR(1) p={mod.AR_list[0].P_value:.3f}  AR(2) p={mod.AR_list[1].P_value:.3f}')
    except Exception as e:
        print(f'  AB Spec 1 ERROR: {e}')

    # Spec 2: System GMM (Blundell-Bond), adds level equations
    cmd2 = (f'{DV} L1.{DV} {PRICE} price_x_eap {ctrl_str} | '
            f'gmm({DV}, 2, 4) gmm({PRICE}, 2, 4) gmm(price_x_eap, 2, 4) iv({ctrl_str}) | steps(2) level')
    print(f'\nBB Spec 2 (SysGMM level, lags 2-4):')
    try:
        m2 = pdreg.abond(cmd2, df_ab, ['country', 'year_num'])
        print('  SysGMM Spec 2: OK')
        if hasattr(m2, 'models') and m2.models:
            mod2 = m2.models[0]
            rt2 = mod2.regression_table
            print(rt2[['variable', 'coefficient', 'std_err', 'p_value']].to_string(index=False))
            print(f'  Hansen J p={mod2.hansen.p_value:.3f}  AR(1) p={mod2.AR_list[0].P_value:.3f}  AR(2) p={mod2.AR_list[1].P_value:.3f}')
    except Exception as e:
        print(f'  BB Spec 2 ERROR: {e}')

    # Pre-COVID only
    df_ab_pre = df_ab[df_ab.year_num <= 2019].copy()
    cmd3 = (f'{DV} L1.{DV} {PRICE} price_x_eap {ctrl_str} | '
            f'gmm({DV}, 2, 4) gmm({PRICE}, 2, 4) gmm(price_x_eap, 2, 4) iv({ctrl_str}) | steps(2)')
    print(f'\nAB Spec 3 (DiGMM pre-COVID only):')
    try:
        m3 = pdreg.abond(cmd3, df_ab_pre, ['country', 'year_num'])
        if hasattr(m3, 'models') and m3.models:
            mod3 = m3.models[0]
            rt3 = mod3.regression_table
            print(rt3[['variable', 'coefficient', 'std_err', 'p_value']].to_string(index=False))
            print(f'  Hansen J p={mod3.hansen.p_value:.3f}  AR(1) p={mod3.AR_list[0].P_value:.3f}  AR(2) p={mod3.AR_list[1].P_value:.3f}')
    except Exception as e:
        print(f'  AB Spec 3 ERROR: {e}')

except ImportError:
    print('pydynpd not available')
except Exception as e:
    print(f'AB/GMM section error: {e}')
    import traceback; traceback.print_exc()
