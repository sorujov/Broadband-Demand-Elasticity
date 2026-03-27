# -*- coding: utf-8 -*-
"""
IV Estimation: Instrumental Variables Robustness Check
======================================================
Uses mobile broadband price (log_mobile_broad_price) as an instrument for
fixed broadband price, exploiting the fact that mobile and fixed broadband
share common cost drivers (spectrum, network infrastructure) while mobile
price is plausibly exogenous to unobserved fixed broadband demand shocks.

Estimates:
  1. Pre-COVID baseline (2010-2019) with IV
  2. Full sample (2010-2024) with COVID interaction terms and IV

Reports first-stage F-statistics and 2SLS elasticities for EU and EaP.

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

INSTRUMENT = 'log_mobile_broad_price'  # mobile broadband price as cost shifter

print("=" * 80)
print("IV ESTIMATION: 2SLS ROBUSTNESS")
print("=" * 80)

df_full = pd.read_csv(ANALYSIS_READY_FILE)
df_full['year_num'] = df_full['year'].astype(int)
df_full['eap_dummy'] = df_full['country'].isin(EAP_COUNTRIES).astype(float)
df_full['covid_dummy'] = (df_full['year_num'] >= 2020).astype(float)
df_full['year_dt'] = pd.to_datetime(df_full['year_num'], format='%Y')
df_full = df_full.set_index(['country', 'year_dt'])

results = []


def run_iv(df_panel, label, covid_interaction=False):
    """
    Run 2SLS using linearmodels IV2SLS with entity/time dummies absorbed
    via within-transformation (demeaning).

    Strategy: demean all variables (country + year within transformation),
    then run IV2SLS on demeaned data using mobile price as instrument.
    """
    df_est = df_panel.copy()
    df_est['price_x_eap'] = df_est[PRIMARY_PRICE] * df_est['eap_dummy']
    instr_eap = 'instr_x_eap'
    df_est[instr_eap] = df_est[INSTRUMENT] * df_est['eap_dummy']

    endog_vars = [PRIMARY_PRICE, 'price_x_eap']
    instr_vars = [INSTRUMENT, instr_eap]

    if covid_interaction:
        df_est['price_x_covid']       = df_est[PRIMARY_PRICE]   * df_est['covid_dummy']
        df_est['price_x_eap_x_covid'] = df_est['price_x_eap']   * df_est['covid_dummy']
        df_est['instr_x_covid']       = df_est[INSTRUMENT]       * df_est['covid_dummy']
        df_est['instr_x_eap_x_covid'] = df_est[instr_eap]        * df_est['covid_dummy']
        endog_vars += ['price_x_covid', 'price_x_eap_x_covid']
        instr_vars += ['instr_x_covid', 'instr_x_eap_x_covid']

    avail_controls = [c for c in CONTROLS if c in df_est.columns]
    required = ([PRIMARY_DV] + endog_vars + instr_vars +
                avail_controls + ['eap_dummy'])
    if covid_interaction:
        required += ['covid_dummy']
    df_est = df_est[[c for c in required if c in df_est.columns]].dropna()

    if len(df_est) < 80:
        print(f"  [{label}] Insufficient obs: {len(df_est)}")
        return None

    # Within-transformation (demean by entity and time)
    def within_demean(df, idx_names):
        """Two-way demeaning: subtract entity mean + time mean - grand mean."""
        result = df.copy().astype(float)
        entity_means = df.groupby(level=0).transform('mean')
        time_means   = df.groupby(level=1).transform('mean')
        grand_mean   = df.mean()
        return result - entity_means - time_means + grand_mean

    all_vars = [PRIMARY_DV] + endog_vars + avail_controls + instr_vars
    if covid_interaction:
        all_vars += ['covid_dummy']
    all_vars = list(dict.fromkeys(all_vars))  # deduplicate

    df_dm = within_demean(df_est[all_vars], df_est.index.names)

    dep     = df_dm[PRIMARY_DV]
    exog    = df_dm[avail_controls] if avail_controls else None
    endog   = df_dm[endog_vars]
    instruments = df_dm[instr_vars]

    # First stage: regress each endogenous var on instruments + exog controls
    # Check instrument relevance via OLS F-stat on primary price equation
    from linearmodels.iv import IV2SLS as _IV2SLS
    import statsmodels.api as sm

    first_stage_vars = instr_vars + avail_controls
    X_fs = sm.add_constant(df_dm[first_stage_vars].values)
    y_fs = df_dm[PRIMARY_PRICE].values
    ols_fs = np.linalg.lstsq(X_fs, y_fs, rcond=None)
    y_hat = X_fs @ ols_fs[0]
    resid = y_fs - y_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_fs - y_fs.mean())**2)
    r2_fs  = 1 - ss_res / ss_tot
    n_obs  = len(y_fs)
    k_instr = len(instr_vars)
    k_total = X_fs.shape[1]
    # Partial F-stat for instruments
    # Restricted: no instruments; unrestricted: with instruments
    X_res = sm.add_constant(df_dm[avail_controls].values) if avail_controls else np.ones((n_obs, 1))
    ols_res = np.linalg.lstsq(X_res, y_fs, rcond=None)
    y_hat_r = X_res @ ols_res[0]
    ss_res_r = np.sum((y_fs - y_hat_r)**2)
    f_stat = ((ss_res_r - ss_res) / k_instr) / (ss_res / (n_obs - k_total))

    print(f"\n  [{label}]  First-stage F (instrument relevance): {f_stat:.2f}"
          f"  (R²={r2_fs:.3f})")

    # 2SLS
    try:
        if exog is not None:
            iv_model = _IV2SLS(dep, exog=df_dm[avail_controls], endog=endog,
                               instruments=instruments)
        else:
            iv_model = _IV2SLS(dep, exog=None, endog=endog,
                               instruments=instruments)
        iv_res = iv_model.fit(cov_type='robust')
    except Exception as e:
        print(f"  [{label}] 2SLS error: {e}")
        return None

    b_eu   = iv_res.params[PRIMARY_PRICE]
    se_eu  = iv_res.std_errors[PRIMARY_PRICE]
    p_eu   = iv_res.pvalues[PRIMARY_PRICE]
    b_int  = iv_res.params['price_x_eap']
    se_int = iv_res.std_errors['price_x_eap']

    eap_b   = b_eu + b_int
    eap_se  = np.sqrt(se_eu**2 + se_int**2 + 2 * iv_res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
    eap_pval = 2 * (1 - stats.norm.cdf(abs(eap_b / eap_se)))

    sig_eu  = '***' if p_eu    < 0.01 else '**' if p_eu    < 0.05 else '*' if p_eu    < 0.10 else ''
    sig_eap = '***' if eap_pval < 0.01 else '**' if eap_pval < 0.05 else '*' if eap_pval < 0.10 else ''

    print(f"    EU (2SLS):  {b_eu:7.4f}{sig_eu:3s}  (SE={se_eu:.4f}, p={p_eu:.3f})")
    print(f"    EaP (2SLS): {eap_b:7.4f}{sig_eap:3s}  (SE={eap_se:.4f}, p={eap_pval:.3f})")

    row = {
        'specification':   label,
        'covid_interaction': covid_interaction,
        'eu_elasticity':   b_eu,
        'eu_se':           se_eu,
        'eu_pval':         p_eu,
        'eap_elasticity':  eap_b,
        'eap_se':          eap_se,
        'eap_pval':        eap_pval,
        'first_stage_f':   f_stat,
        'first_stage_r2':  r2_fs,
        'n_obs':           n_obs,
    }

    if covid_interaction:
        for param in ['price_x_covid', 'price_x_eap_x_covid']:
            if param in iv_res.params.index:
                row[f'{param}_coef'] = iv_res.params[param]
                row[f'{param}_pval'] = iv_res.pvalues[param]

    return row


# 1. Pre-COVID IV
print("\n" + "=" * 60)
print("PRE-COVID (2010-2019) — IV BASELINE")
print("=" * 60)
mask_pre = df_full.index.get_level_values('year_dt').year <= 2019
row = run_iv(df_full[mask_pre], 'Pre-COVID 2SLS', covid_interaction=False)
if row:
    results.append(row)

# 2. Full sample IV with COVID interactions
print("\n" + "=" * 60)
print("FULL SAMPLE (2010-2024) — IV WITH COVID INTERACTIONS")
print("=" * 60)
row = run_iv(df_full, 'Full Sample 2SLS (COVID interactions)', covid_interaction=True)
if row:
    results.append(row)

# 3. OLS baseline for comparison (pre-COVID)
print("\n" + "=" * 60)
print("OLS BASELINE (for comparison)")
print("=" * 60)
df_pre = df_full[df_full.index.get_level_values('year_dt').year <= 2019].copy()
df_pre['price_x_eap'] = df_pre[PRIMARY_PRICE] * df_pre['eap_dummy']
avail_ctrl = [c for c in CONTROLS if c in df_pre.columns]
df_ols = df_pre[[PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + avail_ctrl].dropna()
y_ols = df_ols[PRIMARY_DV]
X_ols = df_ols[[PRIMARY_PRICE, 'price_x_eap'] + avail_ctrl]
ols_res = PanelOLS(y_ols, X_ols, entity_effects=True, time_effects=True).fit(
    cov_type='kernel', kernel='bartlett', bandwidth=3)
b_eu_ols  = ols_res.params[PRIMARY_PRICE]
se_eu_ols = ols_res.std_errors[PRIMARY_PRICE]
p_eu_ols  = ols_res.pvalues[PRIMARY_PRICE]
b_int_ols = ols_res.params['price_x_eap']
se_int_ols = ols_res.std_errors['price_x_eap']
eap_b_ols  = b_eu_ols + b_int_ols
eap_se_ols = np.sqrt(se_eu_ols**2 + se_int_ols**2 + 2 * ols_res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
eap_p_ols  = 2 * (1 - stats.t.cdf(abs(eap_b_ols / eap_se_ols), df=ols_res.df_resid))
sig_eu_ols  = '***' if p_eu_ols  < 0.01 else '**' if p_eu_ols  < 0.05 else '*' if p_eu_ols  < 0.10 else ''
sig_eap_ols = '***' if eap_p_ols < 0.01 else '**' if eap_p_ols < 0.05 else '*' if eap_p_ols < 0.10 else ''
print(f"\n  [OLS Baseline (TWFE + DK SE)]")
print(f"    EU:  {b_eu_ols:7.4f}{sig_eu_ols:3s}  (p={p_eu_ols:.3f})")
print(f"    EaP: {eap_b_ols:7.4f}{sig_eap_ols:3s}  (p={eap_p_ols:.3f})")
results.insert(0, {
    'specification':   'OLS Baseline (TWFE, DK SE)',
    'covid_interaction': False,
    'eu_elasticity':   b_eu_ols,
    'eu_se':           se_eu_ols,
    'eu_pval':         p_eu_ols,
    'eap_elasticity':  eap_b_ols,
    'eap_se':          eap_se_ols,
    'eap_pval':        eap_p_ols,
    'first_stage_f':   np.nan,
    'first_stage_r2':  np.nan,
    'n_obs':           int(ols_res.nobs),
})

# Save to Excel
results_df = pd.DataFrame(results)
out_xlsx = ROBUSTNESS_DIR / 'iv_results.xlsx'
results_df.to_excel(out_xlsx, index=False)
print(f"\n[OK] Excel results saved → {out_xlsx}")


# ============================================================================
# GENERATE LATEX TABLE 6 (Appendix)
# ============================================================================

def fmt(coef, se, pval):
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    if stars:
        return f'${coef:.3f}^{{{stars}}}$', f'({se:.3f})'
    return f'${coef:.3f}$', f'({se:.3f})'


lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{IV Robustness: 2SLS Estimates with Mobile Broadband Price as Instrument}',
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
]

for row in results:
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    f_str = f"{row['first_stage_f']:.1f}" if pd.notna(row.get('first_stage_f')) else '---'
    lines.append(f"{row['specification']} & {c_eu} & {s_eu} & {c_eap} & {s_eap} & {f_str} \\\\[2pt]")

lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\par\vspace{4pt}',
    r'\scriptsize',
    r'\textit{Notes:} Instrument: log mobile broadband price (GNI\%), which captures',
    r'common supply-side cost shifters while being plausibly exogenous to fixed broadband',
    r'demand shocks. 2SLS uses within-transformed (entity + time demeaned) variables.',
    r'First-stage $F$ statistic for primary price equation; $F > 10$ indicates instrument',
    r'relevance. OLS row: Driscoll--Kraay standard errors (kernel bandwidth$=$3);',
    r'2SLS rows: heteroskedasticity-robust standard errors. All in parentheses.',
    r'$^{*}$ p $<$ 0.10, $^{**}$ p $<$ 0.05, $^{***}$ p $<$ 0.01.',
    r'\end{minipage}',
    r'\end{table}',
]

out_tex = MANUSCRIPT_TABLES_DIR / 'table6_iv_robustness.tex'
with open(out_tex, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"[OK] Table 6 written → {out_tex}")

print("\n" + "=" * 80)
print("✓ IV ESTIMATION COMPLETE")
print("=" * 80)
