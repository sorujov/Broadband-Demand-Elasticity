# -*- coding: utf-8 -*-
"""
Period-Split Analysis: Pre-COVID / Acute COVID / Post-Acute COVID
=================================================================
Splits the COVID period (2020-2024) into:
  - Acute COVID:    2020-2021
  - Post-acute:     2022-2024

Uses a single full-sample model with period interaction dummies to avoid
the T=2 problem that arises from running subsamples for the acute period.

Specification:
  ln(Subs) = β1*ln(P) + β2*(ln(P)×EaP)
           + β3*(ln(P)×Acute) + β4*(ln(P)×EaP×Acute)
           + β5*(ln(P)×PostAcute) + β6*(ln(P)×EaP×PostAcute)
           + controls + αi + δt + ε

Implied elasticities:
  EU,  Pre:        β1
  EaP, Pre:        β1 + β2
  EU,  Acute:      β1 + β3
  EaP, Acute:      β1 + β2 + β3 + β4
  EU,  PostAcute:  β1 + β5
  EaP, PostAcute:  β1 + β2 + β5 + β6

Also reports Ukraine/Belarus-excluded robustness for the COVID periods.

Outputs:
  results/regression_output/full_sample_covid_analysis/period_split_results.xlsx
  manuscript/tables/table5_period_split.tex
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
from scipy import stats
import warnings
import sys
import io

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR
    )

FULL_SAMPLE_DIR = RESULTS_REGRESSION / 'full_sample_covid_analysis'
FULL_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONTROLS = [
    'log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
    'regulatory_quality_estimate', 'log_secure_internet_servers',
    'research_development_expenditure', 'log_population_density',
    'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator',
]

CONFLICT_COUNTRIES = ['UKR', 'BLR']

print("=" * 80)
print("PERIOD-SPLIT ANALYSIS (Interaction Approach)")
print("=" * 80)

df_full = pd.read_csv(ANALYSIS_READY_FILE)
df_full['year_num'] = df_full['year'].astype(int)
df_full['eap_dummy']      = df_full['country'].isin(EAP_COUNTRIES).astype(float)
df_full['acute_dummy']    = ((df_full['year_num'] >= 2020) & (df_full['year_num'] <= 2021)).astype(float)
df_full['postacute_dummy'] = (df_full['year_num'] >= 2022).astype(float)
df_full['year_dt'] = pd.to_datetime(df_full['year_num'], format='%Y')
df_full = df_full.set_index(['country', 'year_dt'])


def run_period_split(df_panel, label):
    """
    Estimate three-period model using interaction dummies.
    Returns dict with all period × region elasticities.
    """
    df_est = df_panel.copy()
    # Interaction terms: Price × period × region
    df_est['price_x_eap']           = df_est[PRIMARY_PRICE] * df_est['eap_dummy']
    df_est['price_x_acute']          = df_est[PRIMARY_PRICE] * df_est['acute_dummy']
    df_est['price_x_eap_x_acute']    = df_est[PRIMARY_PRICE] * df_est['eap_dummy'] * df_est['acute_dummy']
    df_est['price_x_post']           = df_est[PRIMARY_PRICE] * df_est['postacute_dummy']
    df_est['price_x_eap_x_post']     = df_est[PRIMARY_PRICE] * df_est['eap_dummy'] * df_est['postacute_dummy']

    avail = [c for c in CONTROLS if c in df_est.columns]
    regressors = [PRIMARY_PRICE, 'price_x_eap',
                  'price_x_acute', 'price_x_eap_x_acute',
                  'price_x_post',  'price_x_eap_x_post'] + avail
    required = [PRIMARY_DV] + regressors
    df_est = df_est[required].dropna()

    if len(df_est) < 100:
        print(f"  [{label}] Insufficient obs: {len(df_est)}")
        return None

    y = df_est[PRIMARY_DV]
    X = df_est[regressors]

    res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=3)

    # Extract coefficients
    b = res.params; s = res.std_errors; p = res.pvalues
    cov = res.cov

    def get_se(*keys):
        """SE of sum of coefficients using delta method (correlated terms)."""
        total = sum(b[k] for k in keys)
        variance = sum(cov.loc[ki, kj] for ki in keys for kj in keys)
        se = np.sqrt(variance)
        pval = 2 * (1 - stats.t.cdf(abs(total / se), df=res.df_resid))
        return total, se, pval

    eu_pre_b,   eu_pre_se,   eu_pre_pv   = b[PRIMARY_PRICE], s[PRIMARY_PRICE], p[PRIMARY_PRICE]
    eap_pre_b,  eap_pre_se,  eap_pre_pv  = get_se(PRIMARY_PRICE, 'price_x_eap')
    eu_ac_b,    eu_ac_se,    eu_ac_pv    = get_se(PRIMARY_PRICE, 'price_x_acute')
    eap_ac_b,   eap_ac_se,   eap_ac_pv  = get_se(PRIMARY_PRICE, 'price_x_eap',
                                                   'price_x_acute', 'price_x_eap_x_acute')
    eu_pa_b,    eu_pa_se,    eu_pa_pv    = get_se(PRIMARY_PRICE, 'price_x_post')
    eap_pa_b,   eap_pa_se,   eap_pa_pv  = get_se(PRIMARY_PRICE, 'price_x_eap',
                                                   'price_x_post', 'price_x_eap_x_post')

    def sig(pv):
        return '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.10 else ''

    n_countries = df_est.index.get_level_values('country').nunique()
    print(f"\n  [{label}]  N={res.nobs}, Countries={n_countries}, R²={res.rsquared:.3f}")
    print(f"    Pre-COVID:  EU={eu_pre_b:.4f}{sig(eu_pre_pv)}  EaP={eap_pre_b:.4f}{sig(eap_pre_pv)}")
    print(f"    Acute:      EU={eu_ac_b:.4f}{sig(eu_ac_pv)}   EaP={eap_ac_b:.4f}{sig(eap_ac_pv)}")
    print(f"    Post-acute: EU={eu_pa_b:.4f}{sig(eu_pa_pv)}  EaP={eap_pa_b:.4f}{sig(eap_pa_pv)}")

    return {
        'label':         label,
        # Pre-COVID
        'eu_pre':       eu_pre_b,  'eu_pre_se':  eu_pre_se,  'eu_pre_pv':  eu_pre_pv,
        'eap_pre':      eap_pre_b, 'eap_pre_se': eap_pre_se, 'eap_pre_pv': eap_pre_pv,
        # Acute COVID
        'eu_ac':        eu_ac_b,   'eu_ac_se':   eu_ac_se,   'eu_ac_pv':   eu_ac_pv,
        'eap_ac':       eap_ac_b,  'eap_ac_se':  eap_ac_se,  'eap_ac_pv':  eap_ac_pv,
        # Post-acute
        'eu_pa':        eu_pa_b,   'eu_pa_se':   eu_pa_se,   'eu_pa_pv':   eu_pa_pv,
        'eap_pa':       eap_pa_b,  'eap_pa_se':  eap_pa_se,  'eap_pa_pv':  eap_pa_pv,
        # Interaction terms
        'b_acute_int':  b['price_x_acute'],  'p_acute_int':  p['price_x_acute'],
        'b_post_int':   b['price_x_post'],   'p_post_int':   p['price_x_post'],
        'b_triple_ac':  b['price_x_eap_x_acute'], 'p_triple_ac': p['price_x_eap_x_acute'],
        'b_triple_pa':  b['price_x_eap_x_post'],  'p_triple_pa': p['price_x_eap_x_post'],
        'n_obs':        int(res.nobs),
        'n_countries':  n_countries,
        'r_squared':    res.rsquared,
    }


# Full sample (all 33 countries)
print("\nFULL SAMPLE:")
row_full = run_period_split(df_full, 'Full sample (33 countries)')

# Excluding Ukraine and Belarus from COVID periods
df_excl = df_full.copy()
yr_arr = df_excl.index.get_level_values('year_dt').year
covid_conflict_mask = (
    (yr_arr >= 2020) &
    df_excl.index.get_level_values('country').isin(CONFLICT_COUNTRIES)
)
df_excl = df_excl[~covid_conflict_mask]
# Zero out conflict countries' COVID dummies in remaining observations
# (only needed if they have pre-2020 obs, which is fine - those are kept)
print("\nEXCLUDING UKR/BLR FROM COVID PERIODS:")
row_excl = run_period_split(df_excl, 'Excl. UKR/BLR from COVID periods')

results = [r for r in [row_full, row_excl] if r is not None]

# Save to Excel
out_xlsx = FULL_SAMPLE_DIR / 'period_split_results.xlsx'
results_df = pd.DataFrame([{k: v for k, v in r.items()
                             if not k.startswith('b_') and not k.startswith('p_triple')}
                            for r in results])
pd.DataFrame(results).to_excel(out_xlsx, index=False)
print(f"\n[OK] Excel results saved → {out_xlsx}")


# ============================================================================
# GENERATE LATEX TABLE 5
# ============================================================================

def fmt(coef, se, pval):
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    if stars:
        return f'${coef:.3f}^{{{stars}}}$', f'({se:.3f})'
    return f'${coef:.3f}$', f'({se:.3f})'


def build_row(row, prefix_eu, prefix_eap):
    """Build 6-cell row (3 periods × EU/EaP) from result dict."""
    b_eu_pre,  s_eu_pre  = fmt(row['eu_pre'],  row['eu_pre_se'],  row['eu_pre_pv'])
    b_eap_pre, s_eap_pre = fmt(row['eap_pre'], row['eap_pre_se'], row['eap_pre_pv'])
    b_eu_ac,   s_eu_ac   = fmt(row['eu_ac'],   row['eu_ac_se'],   row['eu_ac_pv'])
    b_eap_ac,  s_eap_ac  = fmt(row['eap_ac'],  row['eap_ac_se'],  row['eap_ac_pv'])
    b_eu_pa,   s_eu_pa   = fmt(row['eu_pa'],   row['eu_pa_se'],   row['eu_pa_pv'])
    b_eap_pa,  s_eap_pa  = fmt(row['eap_pa'],  row['eap_pa_se'],  row['eap_pa_pv'])
    coef_row = f'{b_eu_pre} & {b_eap_pre} & {b_eu_ac} & {b_eap_ac} & {b_eu_pa} & {b_eap_pa}'
    se_row   = f'{s_eu_pre} & {s_eap_pre} & {s_eu_ac} & {s_eap_ac} & {s_eu_pa} & {s_eap_pa}'
    return coef_row, se_row


row_a = row_full if row_full else results[0]
row_b = row_excl if row_excl else None

coef_a, se_a = build_row(row_a, 'eu', 'eap') if row_a else ('--', '--')
coef_b, se_b = build_row(row_b, 'eu', 'eap') if row_b else ('--', '--')

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Period-Split Elasticity Estimates: Pre-COVID, Acute COVID, and Post-Acute}',
    r'\label{tab:period_split}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\small',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{Pre-COVID} & \multicolumn{2}{c}{Acute COVID} & \multicolumn{2}{c}{Post-Acute} \\',
    r'& \multicolumn{2}{c}{2010--2019} & \multicolumn{2}{c}{2020--2021} & \multicolumn{2}{c}{2022--2024} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
    r'& EU & EaP & EU & EaP & EU & EaP \\',
    r'\midrule',
    r'\textit{Panel A: Full Sample (33 countries)} \\',
    f'Elasticity & {coef_a} \\\\',
    f'& {se_a} \\\\',
    r'\\',
    f'Observations & \\multicolumn{{6}}{{c}}{{{row_a["n_obs"]}}} \\\\' if row_a else '',
    f'R-squared   & \\multicolumn{{6}}{{c}}{{{row_a["r_squared"]:.2f}}} \\\\' if row_a else '',
]

if row_b:
    lines += [
        r'\midrule',
        r'\textit{Panel B: Excluding Ukraine \& Belarus from COVID Periods} \\',
        f'Elasticity & {coef_b} \\\\',
        f'& {se_b} \\\\',
        r'\\',
        f'Observations & \\multicolumn{{6}}{{c}}{{{row_b["n_obs"]}}} \\\\',
        f'R-squared   & \\multicolumn{{6}}{{c}}{{{row_b["r_squared"]:.2f}}} \\\\',
    ]

lines += [
    r'\midrule',
    r'Country FE & \multicolumn{6}{c}{Yes} \\',
    r'Year FE    & \multicolumn{6}{c}{Yes} \\',
    r'Price measure & \multicolumn{6}{c}{GNI\% (income-relative)} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\par\vspace{4pt}',
    r'\small',
    r'\textit{Notes:} Estimates come from a single regression on the full 2010--2024 panel',
    r'using price$\times$period interaction dummies for Acute COVID (2020--2021) and Post-Acute',
    r'(2022--2024). Pre-COVID elasticities are the base-period coefficients. This interaction',
    r'approach avoids the short-panel problem of estimating 2-year subsamples.',
    r'Full controls. Driscoll--Kraay standard errors (bandwidth = 3) in parentheses.',
    r'Panel~B excludes Ukraine and Belarus from the COVID periods only (their pre-2020',
    r'observations are retained). $^{*}$ p $<$ 0.10, $^{**}$ p $<$ 0.05, $^{***}$ p $<$ 0.01.',
    r'\end{minipage}',
    r'\end{table}',
]

# Remove any empty strings from lines
lines = [l for l in lines if l]

out_tex = MANUSCRIPT_TABLES_DIR / 'table5_period_split.tex'
with open(out_tex, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"[OK] Table 5 written → {out_tex}")

print("\n" + "=" * 80)
print("✓ PERIOD-SPLIT ANALYSIS COMPLETE")
print("=" * 80)
