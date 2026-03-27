# -*- coding: utf-8 -*-
"""
IV Estimation: 2SLS Robustness with Lagged Own Price Instrument
===============================================================
Instruments fixed broadband price with its own one-year lag
(log_fixed_broad_price_lag1).

Lagged own price rationale:
  - National price baskets are set through annual regulatory/procurement
    cycles, generating high serial persistence (within-demeaned F=65
    pre-COVID; F=99 full-sample; Stock-Yogo 10% threshold = 16.38).
  - Exclusion restriction: past prices affect current subscriptions only
    through current prices; broadband contracts are monthly/annual and
    consumers cannot anticipate next year's regulated tariff at adoption.
  - Mobile broadband price (prior instrument) has within-demeaned F=0.23
    pre-COVID and is therefore discarded.
  - International bandwidth was also tested as an alternative; an
    over-identified spec (lagged price + bandwidth) strongly rejected
    Hansen J (chi2=262.89, p<0.001), indicating bandwidth correlates with
    connection quality in the demand equation. Lagged price is the sole
    preferred just-identified instrument.

Estimates:
  1. OLS baseline (TWFE + Driscoll-Kraay SE) -- pre-COVID reference
  2. Pre-COVID 2SLS, lagged price  (just-identified; Panel A)
  3. Full-sample 2SLS, lagged price + COVID interactions (Panel B)

Outputs:
  results/regression_output/robustness/iv_results.xlsx
  manuscript/tables/table6_iv_robustness.tex
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from pathlib import Path
from scipy import stats
import warnings
import sys
import io

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR
    )

ROBUSTNESS_DIR = RESULTS_REGRESSION / 'robustness'
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONTROLS = [
    'log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
    'regulatory_quality_estimate', 'log_secure_internet_servers',
    'research_development_expenditure', 'log_population_density',
    'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator',
]

# Instrument: one-year lag of own price (within-demeaned F=65 pre-COVID, 99 full-sample)
INSTR_LAG = 'log_fixed_broad_price_lag1'

print("=" * 80)
print("IV ESTIMATION: 2SLS ROBUSTNESS (lagged own price instrument)")
print("=" * 80)

df_full = pd.read_csv(ANALYSIS_READY_FILE)
df_full['year_num'] = df_full['year'].astype(int)
df_full['eap_dummy'] = df_full['eap'].astype(float)
df_full['covid_dummy'] = (df_full['year_num'] >= 2020).astype(float)
df_full = df_full.sort_values(['iso3', 'year_num'] if 'iso3' in df_full.columns else ['country', 'year_num'])
df_full['year_dt'] = pd.to_datetime(df_full['year_num'], format='%Y')
df_full = df_full.set_index(['country', 'year_dt'])

results = []


def run_iv(df_panel, label, instruments_list, covid_interaction=False):
    """
    Run 2SLS with two-way within-transformation (entity + time demeaning).

    Parameters
    ----------
    instruments_list : list of str
        Column names of the excluded instruments (base names, without
        interactions; interactions are created automatically).
    """
    import statsmodels.api as sm

    df_est = df_panel.copy()
    df_est['price_x_eap'] = df_est[PRIMARY_PRICE] * df_est['eap_dummy']

    endog_vars = [PRIMARY_PRICE, 'price_x_eap']
    instr_vars = []
    for ins in instruments_list:
        col_x_eap = ins + '_x_eap'
        df_est[col_x_eap] = df_est[ins] * df_est['eap_dummy']
        instr_vars += [ins, col_x_eap]

    if covid_interaction:
        df_est['price_x_covid']       = df_est[PRIMARY_PRICE]   * df_est['covid_dummy']
        df_est['price_x_eap_x_covid'] = df_est['price_x_eap']   * df_est['covid_dummy']
        endog_vars += ['price_x_covid', 'price_x_eap_x_covid']
        for ins in instruments_list:
            col_x_cov     = ins + '_x_cov'
            col_x_eap_cov = ins + '_x_eap_x_cov'
            df_est[col_x_cov]     = df_est[ins] * df_est['covid_dummy']
            df_est[col_x_eap_cov] = df_est[ins + '_x_eap'] * df_est['covid_dummy']
            instr_vars += [col_x_cov, col_x_eap_cov]

    avail_controls = [c for c in CONTROLS if c in df_est.columns]
    required = list(dict.fromkeys(
        [PRIMARY_DV] + endog_vars + instr_vars + avail_controls + ['eap_dummy']
    ))
    if covid_interaction:
        required.append('covid_dummy')
    df_est = df_est[[c for c in required if c in df_est.columns]].dropna()

    if len(df_est) < 80:
        print(f"  [{label}] Insufficient obs: {len(df_est)}")
        return None

    # ---- Two-way within demeaning ----
    all_vars = list(dict.fromkeys([PRIMARY_DV] + endog_vars + instr_vars + avail_controls))

    def _demean2way(df_in):
        d = df_in.astype(float).copy()
        entity_means = d.groupby(level=0).transform('mean')
        time_means   = d.groupby(level=1).transform('mean')
        grand_mean   = d.mean()
        return d - entity_means - time_means + grand_mean

    df_dm = _demean2way(df_est[all_vars])

    # ---- First-stage partial F for primary price equation ----
    X_fs  = sm.add_constant(df_dm[instr_vars + avail_controls].values)
    y_fs  = df_dm[PRIMARY_PRICE].values
    b_fs  = np.linalg.lstsq(X_fs, y_fs, rcond=None)[0]
    yhat  = X_fs @ b_fs
    res   = y_fs - yhat
    n_obs = len(y_fs)
    k_tot = X_fs.shape[1]
    ssres = np.sum(res ** 2)
    X_res = sm.add_constant(df_dm[avail_controls].values) if avail_controls else np.ones((n_obs, 1))
    b_res = np.linalg.lstsq(X_res, y_fs, rcond=None)[0]
    ssres_r = np.sum((y_fs - X_res @ b_res) ** 2)
    k_instr = len(instr_vars)
    f_stat = ((ssres_r - ssres) / k_instr) / (ssres / (n_obs - k_tot))

    r2_fs = 1 - ssres / np.sum((y_fs - y_fs.mean()) ** 2)
    print(f"\n  [{label}]  First-stage F (within-demeaned, {k_instr} instr.): {f_stat:.1f}"
          f"  (RÂ²={r2_fs:.3f})  N={n_obs}")

    # ---- 2SLS ----
    dep_dm  = df_dm[PRIMARY_DV]
    exog_dm = df_dm[avail_controls] if avail_controls else None
    endog_dm = df_dm[endog_vars]
    instr_dm = df_dm[instr_vars]

    try:
        iv_model = IV2SLS(dep_dm, exog=exog_dm, endog=endog_dm,
                          instruments=instr_dm)
        iv_res = iv_model.fit(cov_type='robust')
    except Exception as e:
        print(f"  [{label}] 2SLS error: {e}")
        return None

    b_eu   = iv_res.params[PRIMARY_PRICE]
    se_eu  = iv_res.std_errors[PRIMARY_PRICE]
    p_eu   = iv_res.pvalues[PRIMARY_PRICE]
    b_int  = iv_res.params['price_x_eap']
    se_int = iv_res.std_errors['price_x_eap']
    cov_eu_int = iv_res.cov.loc[PRIMARY_PRICE, 'price_x_eap']
    eap_b  = b_eu + b_int
    eap_se = np.sqrt(se_eu ** 2 + se_int ** 2 + 2 * cov_eu_int)
    eap_p  = 2 * (1 - stats.norm.cdf(abs(eap_b / eap_se)))

    sg = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    print(f"    EU  (2SLS): {b_eu:+.4f}{sg(p_eu):3s}  SE={se_eu:.4f}  p={p_eu:.3f}")
    print(f"    EaP (2SLS): {eap_b:+.4f}{sg(eap_p):3s}  SE={eap_se:.4f}  p={eap_p:.3f}")

    # ---- Hansen J for over-identified specs ----
    hansen_j = np.nan
    hansen_p = np.nan
    n_endog  = len(endog_vars)
    df_j     = k_instr - n_endog  # over-identification degree
    if df_j > 0:
        # Obtain 2SLS residuals, regress on all instruments + controls
        try:
            yhat_2sls = (iv_res.fitted_values.values
                         if hasattr(iv_res, 'fitted_values')
                         else dep_dm.values - iv_res.resids.values)
            iv_resid = dep_dm.values - yhat_2sls
            Z_all = sm.add_constant(
                np.column_stack([df_dm[instr_vars].values,
                                  df_dm[avail_controls].values])
            )
            b_j = np.linalg.lstsq(Z_all, iv_resid, rcond=None)[0]
            ssr_j = np.sum((iv_resid - Z_all @ b_j) ** 2)
            ssr_e = np.sum(iv_resid ** 2)
            hansen_j = len(iv_resid) * (1 - ssr_j / ssr_e)
            hansen_p = 1 - stats.chi2.cdf(hansen_j, df=df_j)
            print(f"    Hansen J: chiÂ²({df_j})={hansen_j:.2f}  p={hansen_p:.3f}")
        except Exception:
            pass

    row = {
        'specification':       label,
        'covid_interaction':   covid_interaction,
        'instruments':         '+'.join(instruments_list),
        'eu_elasticity':       b_eu,
        'eu_se':               se_eu,
        'eu_pval':             p_eu,
        'eap_elasticity':      eap_b,
        'eap_se':              eap_se,
        'eap_pval':            eap_p,
        'first_stage_f':       f_stat,
        'first_stage_r2':      r2_fs,
        'hansen_j':            hansen_j,
        'hansen_p':            hansen_p,
        'n_obs':               n_obs,
    }

    if covid_interaction:
        for param in ['price_x_covid', 'price_x_eap_x_covid']:
            if param in iv_res.params.index:
                row[f'{param}_coef'] = iv_res.params[param]
                row[f'{param}_pval'] = iv_res.pvalues[param]

    return row


# ============================================================================
# 0. OLS baseline (pre-COVID, TWFE + Driscoll-Kraay) for comparison table row
# ============================================================================
print("\n" + "=" * 60)
print("OLS BASELINE (pre-COVID, TWFE + DK SE)")
print("=" * 60)

df_pre_ols = df_full[df_full.index.get_level_values('year_dt').year <= 2019].copy()
df_pre_ols['price_x_eap'] = df_pre_ols[PRIMARY_PRICE] * df_pre_ols['eap_dummy']
avail_ctrl = [c for c in CONTROLS if c in df_pre_ols.columns]
y_ols = df_pre_ols[PRIMARY_DV]
X_ols = df_pre_ols[[PRIMARY_PRICE, 'price_x_eap'] + avail_ctrl]
ols_res = PanelOLS(y_ols, X_ols, entity_effects=True, time_effects=True).fit(
    cov_type='kernel', kernel='bartlett', bandwidth=3)
b_eu_ols   = ols_res.params[PRIMARY_PRICE]
se_eu_ols  = ols_res.std_errors[PRIMARY_PRICE]
p_eu_ols   = ols_res.pvalues[PRIMARY_PRICE]
b_int_ols  = ols_res.params['price_x_eap']
se_int_ols = ols_res.std_errors['price_x_eap']
eap_b_ols  = b_eu_ols + b_int_ols
eap_se_ols = np.sqrt(se_eu_ols**2 + se_int_ols**2
                     + 2 * ols_res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
eap_p_ols  = 2 * (1 - stats.t.cdf(abs(eap_b_ols / eap_se_ols),
                                    df=ols_res.df_resid))
sg = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
print(f"\n  [OLS Baseline] EU={b_eu_ols:+.4f}{sg(p_eu_ols):3s}"
      f"  EaP={eap_b_ols:+.4f}{sg(eap_p_ols):3s}")

results.insert(0, {
    'specification':     'OLS Baseline (TWFE, DK SE)',
    'covid_interaction': False,
    'instruments':       '---',
    'eu_elasticity':     b_eu_ols,
    'eu_se':             se_eu_ols,
    'eu_pval':           p_eu_ols,
    'eap_elasticity':    eap_b_ols,
    'eap_se':            eap_se_ols,
    'eap_pval':          eap_p_ols,
    'first_stage_f':     np.nan,
    'first_stage_r2':    np.nan,
    'hansen_j':          np.nan,
    'hansen_p':          np.nan,
    'n_obs':             int(ols_res.nobs),
})

# ============================================================================
# 1â€“3: Pre-COVID 2SLS (three IV specs)
# ============================================================================
print("\n" + "=" * 60)
print("PRE-COVID (2010-2019) â€” 2SLS SPECIFICATIONS")
print("=" * 60)

mask_pre = df_full.index.get_level_values('year_dt').year <= 2019

row = run_iv(df_full[mask_pre], 'Pre-COVID: Lagged price',
             [INSTR_LAG], covid_interaction=False)
if row: results.append(row)


# ============================================================================
# 4: Full-sample 2SLS with COVID interactions
# ============================================================================
print("\n" + "=" * 60)
print("FULL SAMPLE (2010-2024) â€” 2SLS WITH COVID INTERACTIONS")
print("=" * 60)

row = run_iv(df_full, 'Full sample: Lagged price',
             [INSTR_LAG], covid_interaction=True)
if row: results.append(row)

# Save to Excel
results_df = pd.DataFrame(results)
out_xlsx = ROBUSTNESS_DIR / 'iv_results.xlsx'
results_df.to_excel(out_xlsx, index=False)
print(f"\n[OK] Excel results saved â†’ {out_xlsx}")


# ============================================================================
# GENERATE LATEX TABLE 6 (Appendix)
# ============================================================================

def fmt(coef, se, pval):
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    coef_str = f'${coef:.3f}^{{{stars}}}$' if stars else f'${coef:.3f}$'
    return coef_str, f'({se:.3f})'


ols_row   = next((r for r in results if r['specification'] == 'OLS Baseline (TWFE, DK SE)'), None)
pre_rows  = [r for r in results if not r.get('covid_interaction') and
             r['specification'] != 'OLS Baseline (TWFE, DK SE)']
full_rows = [r for r in results if r.get('covid_interaction')]

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{IV Robustness: 2SLS Estimates Using Lagged Own Price as Instrument}',
    r'\label{tab:iv_robustness}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\begin{tabular}{lccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{EU Elasticity} & \multicolumn{2}{c}{EaP Elasticity} & \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Specification & Coef. & SE & Coef. & SE & First-Stage $F$ \\',
    r'\midrule',
    r'\multicolumn{6}{l}{\textit{A. Pre-COVID (2010--2019)}} \\[2pt]',
]

if ols_row:
    c_eu, s_eu   = fmt(ols_row['eu_elasticity'],  ols_row['eu_se'],  ols_row['eu_pval'])
    c_eap, s_eap = fmt(ols_row['eap_elasticity'], ols_row['eap_se'], ols_row['eap_pval'])
    lines.append(f"\\phantom{{x}}OLS (TWFE + DK SE) & {c_eu} & {s_eu} & {c_eap} & {s_eap} & --- \\\\[2pt]")

for row in pre_rows:
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    f_str = f"{row['first_stage_f']:.0f}" if pd.notna(row.get('first_stage_f')) else '---'
    label = row['specification'].replace('Pre-COVID: ', '').lower()
    lines.append(f"\\phantom{{x}}2SLS, {label} & {c_eu} & {s_eu} & {c_eap} & {s_eap}"
                 f" & {f_str} \\\\[2pt]")

lines.append(r'\midrule')
lines.append(r'\multicolumn{6}{l}{\textit{B. Full sample (2010--2024), COVID interactions}} \\[2pt]')
for row in full_rows:
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    f_str = f"{row['first_stage_f']:.0f}" if pd.notna(row.get('first_stage_f')) else '---'
    label = row['specification'].replace('Full sample: ', '').lower()
    lines.append(f"\\phantom{{x}}2SLS, {label} & {c_eu} & {s_eu} & {c_eap} & {s_eap}"
                 f" & {f_str} \\\\[2pt]")

lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\par\vspace{4pt}',
    r'\scriptsize',
    r'\textit{Notes:} Instrument: $\log$ fixed broadband price (GNI\%) lagged one year.',
    r'All specifications include country and year fixed effects absorbed via two-way',
    r'within-transformation. First-stage $F$ is a partial $F$-statistic for the excluded',
    r'instrument in the within-transformed first stage; $F > 10$ indicates instrument',
    r'relevance \citep{stock2002testing}.',
    r'EaP elasticity $=$ EU coefficient $+$ EU$\times$EaP interaction; SE via delta method.',
    r'OLS: Driscoll--Kraay SE (kernel bandwidth$=$3). 2SLS: heteroskedasticity-robust SE.',
    r'$^{*}$ $p < 0.10$, $^{**}$ $p < 0.05$, $^{***}$ $p < 0.01$.',
    r'\end{minipage}',
    r'\end{table}',
]

out_tex = MANUSCRIPT_TABLES_DIR / 'table6_iv_robustness.tex'
with open(out_tex, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f"[OK] Table 6 written -> {out_tex}")

print("\n" + "=" * 80)
print("IV ESTIMATION COMPLETE")
print("=" * 80)
