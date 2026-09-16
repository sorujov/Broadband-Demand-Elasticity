# -*- coding: utf-8 -*-
"""
EaP Jackknife Robustness: Leave-One-Country-Out
================================================
Loops over the 6 EaP countries, dropping one at a time, and re-estimates
the main pre-COVID baseline (Full Controls, GNI% price).

Reports:
  - EaP elasticity for each leave-one-out sample
  - Range and any high-leverage country (where exclusion shifts results materially)

Outputs:
  results/regression_output/robustness/eap_jackknife.xlsx
  manuscript/tables/table7_jackknife.tex   (if variation is notable)
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

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR, COUNTRY_NAMES
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.config import (
        ANALYSIS_READY_FILE, RESULTS_REGRESSION, EAP_COUNTRIES,
        PRIMARY_DV, PRIMARY_PRICE, MANUSCRIPT_TABLES_DIR, COUNTRY_NAMES
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

print("=" * 80)
print("EaP JACKKNIFE ROBUSTNESS: LEAVE-ONE-COUNTRY-OUT")
print("=" * 80)

df_full = pd.read_csv(ANALYSIS_READY_FILE)
df_full['year_num'] = df_full['year'].astype(int)
df_pre = df_full[df_full['year_num'] <= 2019].copy()
df_pre['eap_dummy'] = df_pre['country'].isin(EAP_COUNTRIES).astype(float)
df_pre['year_dt'] = pd.to_datetime(df_pre['year_num'], format='%Y')
df_pre = df_pre.set_index(['country', 'year_dt'])


def run_baseline(df_panel, label):
    df_est = df_panel.copy()
    df_est['price_x_eap'] = df_est[PRIMARY_PRICE] * df_est['eap_dummy']
    avail = [c for c in CONTROLS if c in df_est.columns]
    required = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + avail
    df_est = df_est[required].dropna()

    if len(df_est) < 50:
        print(f"  [{label}] Insufficient obs: {len(df_est)}")
        return None

    y = df_est[PRIMARY_DV]
    X = df_est[[PRIMARY_PRICE, 'price_x_eap'] + avail]
    res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=3)

    b_eu   = res.params[PRIMARY_PRICE]
    b_int  = res.params['price_x_eap']
    se_eu  = res.std_errors[PRIMARY_PRICE]
    se_int = res.std_errors['price_x_eap']

    eap_b   = b_eu + b_int
    eap_se  = np.sqrt(se_eu**2 + se_int**2 + 2 * res.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
    eu_pval  = res.pvalues[PRIMARY_PRICE]
    eap_pval = 2 * (1 - stats.t.cdf(abs(eap_b / eap_se), df=res.df_resid))

    sig_eu  = '***' if eu_pval  < 0.01 else '**' if eu_pval  < 0.05 else '*' if eu_pval  < 0.10 else ''
    sig_eap = '***' if eap_pval < 0.01 else '**' if eap_pval < 0.05 else '*' if eap_pval < 0.10 else ''
    n_eap = df_est[df_est.index.get_level_values('country').isin(EAP_COUNTRIES)].shape[0]

    print(f"\n  [{label}]")
    print(f"    EU:  {b_eu:7.4f}{sig_eu:3s}  (p={eu_pval:.3f})")
    print(f"    EaP: {eap_b:7.4f}{sig_eap:3s}  (p={eap_pval:.3f})")
    print(f"    N={res.nobs}, N_EaP obs={n_eap}, R²={res.rsquared:.3f}")

    return {
        'label':          label,
        'dropped_country': label.replace('Drop ', '').replace(' (full EaP)', ''),
        'eu_elasticity':  b_eu,
        'eu_se':          se_eu,
        'eu_pval':        eu_pval,
        'eap_elasticity': eap_b,
        'eap_se':         eap_se,
        'eap_pval':       eap_pval,
        'n_obs':          int(res.nobs),
        'r_squared':      res.rsquared,
    }


# Full EaP baseline (all 6 countries)
print("\nFULL EaP SAMPLE (all 6 countries):")
full_row = run_baseline(df_pre, 'Full EaP (baseline)')
results = [full_row] if full_row else []

# Jackknife: drop one EaP country at a time
print("\nJACKKNIFE RESULTS:")
for country_code in EAP_COUNTRIES:
    country_name = COUNTRY_NAMES.get(country_code, country_code)
    df_jack = df_pre[
        ~((df_pre.index.get_level_values('country') == country_code) &
          (df_pre['eap_dummy'] == 1))
    ].copy()
    # Re-assign eap_dummy with the dropped country removed
    # (the country is removed entirely, not just reclassified)
    remaining_eap = [c for c in EAP_COUNTRIES if c != country_code]
    df_jack = df_pre[
        ~(df_pre.index.get_level_values('country') == country_code)
    ].copy()
    df_jack['eap_dummy'] = df_jack.index.get_level_values('country').isin(
        remaining_eap).astype(float)

    row = run_baseline(df_jack, f'Drop {country_name}')
    if row:
        results.append(row)

# Save to Excel
results_df = pd.DataFrame(results)
out_xlsx = ROBUSTNESS_DIR / 'eap_jackknife.xlsx'
results_df.to_excel(out_xlsx, index=False)
print(f"\n[OK] Excel results saved → {out_xlsx}")

# Summary statistics
eap_range = results_df['eap_elasticity']
print(f"\n{'='*60}")
print("JACKKNIFE SUMMARY")
print(f"{'='*60}")
print(f"  EaP elasticity range: [{eap_range.min():.4f}, {eap_range.max():.4f}]")
print(f"  Mean:  {eap_range.mean():.4f}")
print(f"  SD:    {eap_range.std():.4f}")
if full_row:
    baseline_eap = full_row['eap_elasticity']
    for _, row in results_df.iterrows():
        if row['label'] == 'Full EaP (baseline)':
            continue
        diff = abs(row['eap_elasticity'] - baseline_eap)
        flag = '  <-- HIGH LEVERAGE' if diff > 0.10 else ''
        print(f"  Drop {row['dropped_country']}: {row['eap_elasticity']:.4f}  "
              f"(diff={diff:.4f}){flag}")


# ============================================================================
# GENERATE LATEX TABLE 7
# ============================================================================

def fmt(coef, se, pval):
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    if stars:
        return f'${coef:.3f}^{{{stars}}}$', f'({se:.3f})'
    return f'${coef:.3f}$', f'({se:.3f})'


lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{EaP Jackknife Robustness: Leave-One-Country-Out (Pre-COVID Baseline)}',
    r'\label{tab:jackknife}',
    r'\begin{minipage}{\textwidth}',
    r'\begin{adjustbox}{width=\textwidth}',
    r'\scriptsize',
    r'\begin{tabular}{lcccccc}',
    r'\toprule',
    r'& \multicolumn{2}{c}{EU Elasticity} & \multicolumn{2}{c}{EaP Elasticity} & & \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Sample & Coef. & SE & Coef. & SE & N & R$^2$ \\',
    r'\midrule',
]

for _, row in results_df.iterrows():
    c_eu, s_eu   = fmt(row['eu_elasticity'],  row['eu_se'],  row['eu_pval'])
    c_eap, s_eap = fmt(row['eap_elasticity'], row['eap_se'], row['eap_pval'])
    label = row['label']
    # Bold the full-sample baseline row
    if 'baseline' in label.lower():
        label = r'\textbf{' + label + r'}'
    lines.append(f'{label} & {c_eu} & {s_eu} & {c_eap} & {s_eap} & '
                 f"{int(row['n_obs'])} & {row['r_squared']:.2f} \\\\")

lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{adjustbox}',
    r'\par\vspace{4pt}',
    r'\scriptsize',
    r'\textit{Notes:} Each row drops one EaP country from the sample and',
    r're-estimates the baseline model (Full Controls, GNI\% price, pre-COVID 2010--2019).',
    r'Driscoll--Kraay standard errors (bandwidth = 3) in parentheses.',
    r'$^{*}$ p $<$ 0.10, $^{**}$ p $<$ 0.05, $^{***}$ p $<$ 0.01.',
    r'\end{minipage}',
    r'\end{table}',
]

out_tex = MANUSCRIPT_TABLES_DIR / 'table7_jackknife.tex'
with open(out_tex, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"[OK] Table 7 written → {out_tex}")

print("\n" + "=" * 80)
print("✓ EaP JACKKNIFE COMPLETE")
print("=" * 80)
