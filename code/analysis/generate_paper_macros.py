"""
Generate LaTeX \\newcommand macros from regression output xlsx files.

Reads all result xlsx files and writes manuscript/paper_macros.tex with
\\newcommand definitions for every number cited in the paper body and abstract.

Usage:
    python code/analysis/generate_paper_macros.py

Re-run whenever regression outputs are regenerated.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

try:
    from code.utils.config import RESULTS_REGRESSION, MANUSCRIPT_DIR
except (ImportError, ModuleNotFoundError):
    from utils.config import RESULTS_REGRESSION, MANUSCRIPT_DIR

# ── helpers ──────────────────────────────────────────────────────────────────

def count_sig(df, col_pval, threshold=0.05):
    """Return count of rows where col_pval < threshold."""
    return int((df[col_pval] < threshold).sum())

def pct_sig(df, col_pval, threshold=0.10):
    """Return integer percentage of rows where col_pval < threshold."""
    n = len(df)
    if n == 0:
        return float('nan')
    return int(round(100 * (df[col_pval] < threshold).sum() / n))

def stars(pval):
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.10: return '*'
    return ''

def fmt(val, decimals=3):
    return f'{val:.{decimals}f}'

def fmt_large(n_float):
    """Round to nearest 100,000 and format with LaTeX comma separator."""
    n = round(n_float / 100000) * 100000
    s = f'{int(n):,}'.replace(',', '{,}')
    return s

# ── load data ─────────────────────────────────────────────────────────────────

pre_ext   = pd.read_excel(RESULTS_REGRESSION / 'pre_covid_analysis' /
                          'extended_control_specifications.xlsx')
pre_price = pd.read_excel(RESULTS_REGRESSION / 'pre_covid_analysis' /
                          'price_robustness_matrix.xlsx')
full_ext  = pd.read_excel(RESULTS_REGRESSION / 'full_sample_covid_analysis' /
                          'extended_control_specifications.xlsx')
full_price = pd.read_excel(RESULTS_REGRESSION / 'full_sample_covid_analysis' /
                           'price_robustness_matrix.xlsx')
period    = pd.read_excel(RESULTS_REGRESSION / 'full_sample_covid_analysis' /
                          'period_split_results.xlsx')
yby       = pd.read_excel(RESULTS_REGRESSION / 'full_sample_covid_analysis' /
                          'year_by_year_elasticities.xlsx')
placebo   = pd.read_excel(RESULTS_REGRESSION / 'full_sample_covid_analysis' /
                          'placebo_test_results.xlsx')
jackknife = pd.read_excel(RESULTS_REGRESSION / 'robustness' / 'eap_jackknife.xlsx')
iv_res    = pd.read_excel(RESULTS_REGRESSION / 'robustness' / 'iv_results.xlsx')
samp_res  = pd.read_excel(RESULTS_REGRESSION / 'robustness' / 'sample_restrictions.xlsx')

# ── baseline (Full Controls, GNI%, pre-COVID) ─────────────────────────────────

fc_mask  = pre_ext['specification'].str.contains('Full Controls', case=False, na=False)
fc_row   = pre_ext[fc_mask].iloc[-1]   # last in case of duplicates

eu_base_b    = fc_row['eu_elasticity']
eu_base_se   = fc_row['eu_se']
eu_base_pval = fc_row['eu_pval']
eap_base_b   = fc_row['eap_elasticity']
eap_base_se  = fc_row['eap_se']
eap_base_pval = fc_row['eap_pval']
int_base_b   = fc_row['interaction_coef']
int_base_pval = fc_row['interaction_pval']
gdp_base_b   = fc_row['gdp_coef']   if 'gdp_coef'  in fc_row.index else float('nan')
gdp_base_pval= fc_row['gdp_pval']   if 'gdp_pval'  in fc_row.index else float('nan')
baseline_n   = int(fc_row['n_obs'])
baseline_r2  = fc_row['r_squared']

# EaP / EU stability across all control specs (GNI%, pre-COVID, in pre_ext)
eu_spec_min   = pre_ext['eu_elasticity'].min()
eu_spec_max   = pre_ext['eu_elasticity'].max()
eap_spec_min  = pre_ext['eap_elasticity'].min()
eap_spec_max  = pre_ext['eap_elasticity'].max()
eap_spec_mean = pre_ext['eap_elasticity'].mean()
eap_spec_std  = pre_ext['eap_elasticity'].std()

# ── price robustness (pre-COVID, 24 specs = 3 price × 8 control) ──────────────

gni_rows = pre_price[pre_price['price_measure'] == 'GNI%']
ppp_rows = pre_price[pre_price['price_measure'] == 'PPP']
usd_rows = pre_price[pre_price['price_measure'] == 'USD']

# EaP significance counts per measure (at p<0.05 and p<0.10)
pct_gni_sig_eap_10 = pct_sig(gni_rows, 'eap_pval', 0.10)
pct_gni_sig_eap_05 = pct_sig(gni_rows, 'eap_pval', 0.05)
pct_ppp_sig_eap_10 = pct_sig(ppp_rows, 'eap_pval', 0.10)
pct_ppp_sig_eap_05 = pct_sig(ppp_rows, 'eap_pval', 0.05)

cnt_gni_sig_eap_05 = count_sig(gni_rows, 'eap_pval', 0.05)
cnt_ppp_sig_eap_05 = count_sig(ppp_rows, 'eap_pval', 0.05)
cnt_usd_sig_eap_05 = count_sig(usd_rows, 'eap_pval', 0.05)

# EU significance counts per measure
cnt_gni_sig_eu_05  = count_sig(gni_rows, 'eu_pval', 0.05)
cnt_ppp_sig_eu_05  = count_sig(ppp_rows, 'eu_pval', 0.05)
cnt_usd_sig_eu_05  = count_sig(usd_rows, 'eu_pval', 0.05)
n_per_measure      = len(gni_rows)   # typically 8

# Robustness matrix across ALL 24 specs (for total count claims)
rob_n_specs      = len(pre_price)

# Robustness matrix stats using GNI% specs only (8 specs, primary price measure)
# The paper's range/mean/SD claims in the text refer to the primary GNI% measure
rob_eap_min      = gni_rows['eap_elasticity'].min()
rob_eap_max      = gni_rows['eap_elasticity'].max()
rob_eap_mean     = gni_rows['eap_elasticity'].mean()
rob_eap_sd       = gni_rows['eap_elasticity'].std()
rob_eu_min       = gni_rows['eu_elasticity'].min()
rob_eu_max       = gni_rows['eu_elasticity'].max()
rob_eu_mean      = gni_rows['eu_elasticity'].mean()
rob_eu_sd        = gni_rows['eu_elasticity'].std()
rob_eap_sig_all  = count_sig(gni_rows, 'eap_pval', 0.01)   # all 8 at 1%
rob_eu_sig_05    = count_sig(gni_rows, 'eu_pval',  0.05)   # at 5%
rob_eu_sig_10    = count_sig(gni_rows, 'eu_pval',  0.10)   # at 10%
rob_gni_n_specs  = len(gni_rows)   # number of GNI% specs (typically 8)

# EaP elasticity ranges by price measure (pre-COVID, Full Controls row)
fc_gni = gni_rows[gni_rows['control_spec'].str.contains('Full Controls', case=False, na=False)]
fc_ppp = ppp_rows[ppp_rows['control_spec'].str.contains('Full Controls', case=False, na=False)]
fc_usd = usd_rows[usd_rows['control_spec'].str.contains('Full Controls', case=False, na=False)]

eap_gni_fc_b  = fc_gni['eap_elasticity'].iloc[0] if len(fc_gni) else float('nan')
eap_ppp_fc_b  = fc_ppp['eap_elasticity'].iloc[0] if len(fc_ppp) else float('nan')
eap_usd_fc_b  = fc_usd['eap_elasticity'].iloc[0] if len(fc_usd) else float('nan')

eu_gni_fc_b   = fc_gni['eu_elasticity'].iloc[0]  if len(fc_gni) else float('nan')
eu_ppp_fc_b   = fc_ppp['eu_elasticity'].iloc[0]  if len(fc_ppp) else float('nan')
eu_usd_fc_b   = fc_usd['eu_elasticity'].iloc[0]  if len(fc_usd) else float('nan')

# ── COVID full-sample (Full Controls, GNI%) ───────────────────────────────────

fc_full_mask = full_ext['specification'].str.contains('Full Controls', case=False, na=False)
# full_ext also has a price_measure column
if 'price_measure' in full_ext.columns:
    fc_full_mask &= (full_ext['price_measure'] == 'GNI%')
fc_full_row  = full_ext[fc_full_mask].iloc[-1]

# Pre-COVID baselines sourced from period_split_results.xlsx Panel A
# so macros correctly match the values reported in Table~\ref{tab:period_split}
_ps_panel_a   = period.iloc[0]
eu_pre_full_b     = _ps_panel_a['eu_pre']
eu_pre_full_pval  = _ps_panel_a['eu_pre_pv']
eap_pre_full_b    = _ps_panel_a['eap_pre']
eap_pre_full_pval = _ps_panel_a['eap_pre_pv']
eu_covid_b        = fc_full_row['eu_covid_elasticity']
eu_covid_pval     = fc_full_row['eu_covid_pval']
eap_covid_b       = fc_full_row['eap_covid_elasticity']
eap_covid_pval    = fc_full_row['eap_covid_pval']
full_n            = int(fc_full_row['n_obs'])

# Pre-COVID baselines from the interaction model (price_robustness_matrix)
# These match Figure 4 exactly (same data source)
_fp_mask = (full_price['control_spec'].str.contains('Full Controls', case=False, na=False) &
            (full_price['price_measure'] == 'GNI%'))
_fp_row  = full_price[_fp_mask].iloc[0]
eu_pre_int_b      = _fp_row['eu_pre_elasticity']
eu_pre_int_pval   = _fp_row['eu_pre_pval']
eap_pre_int_b     = _fp_row['eap_pre_elasticity']
eap_pre_int_pval  = _fp_row['eap_pre_pval']
eu_covid_int_b    = _fp_row['eu_covid_elasticity']
eu_covid_int_pval = _fp_row['eu_covid_pval']
eap_covid_int_b   = _fp_row['eap_covid_elasticity']
eap_covid_int_pval = _fp_row['eap_covid_pval']

# COVID interaction terms (price×COVID and triple)
eu_cov_int_b    = fc_full_row['price_x_covid_coef']
eu_cov_int_pval = fc_full_row['price_x_covid_pval']
eap_triple_b    = fc_full_row['triple_interaction_coef']
eap_triple_pval = fc_full_row['triple_interaction_pval']

# ── year-by-year (2015 is the reference year cited in the paper) ──────────────

row_2015         = yby[yby['year'] == 2015]
eu_yby_2015_b    = float(row_2015['eu_elasticity'].iloc[0]) if len(row_2015) else float('nan')
eu_yby_2015_pval = float(row_2015['eu_pval'].iloc[0])       if len(row_2015) else float('nan')

# ── placebo test ──────────────────────────────────────────────────────────────

def _placebo_row(keyword):
    mask = placebo['test_type'].str.contains(keyword, case=False, na=False)
    return placebo[mask].iloc[0]

eu_placebo_row   = _placebo_row('EU Effect')
eap_placebo_row  = _placebo_row('EaP')
eu_placebo_b     = float(eu_placebo_row['coefficient'])
eu_placebo_pval  = float(eu_placebo_row['pvalue'])
eap_placebo_b    = float(eap_placebo_row['coefficient'])
eap_placebo_pval = float(eap_placebo_row['pvalue'])

# ── jackknife ─────────────────────────────────────────────────────────────────

jk_baseline = jackknife[jackknife['dropped_country'].isna() |
                        (jackknife['dropped_country'] == '')]
jk_dropped  = jackknife[jackknife['dropped_country'].notna() &
                        (jackknife['dropped_country'] != '')]

eap_jk_min = jk_dropped['eap_elasticity'].min()
eap_jk_max = jk_dropped['eap_elasticity'].max()
jk_min_country = jk_dropped.loc[jk_dropped['eap_elasticity'].idxmin(), 'dropped_country']
jk_max_country = jk_dropped.loc[jk_dropped['eap_elasticity'].idxmax(), 'dropped_country']
# All EaP jackknife estimates significant at 1%?
all_jk_sig = (jk_dropped['eap_pval'] < 0.01).all()

# ── IV ────────────────────────────────────────────────────────────────────────

# Select pre-COVID 2SLS row (skip OLS row)
iv_2sls_mask = ~iv_res['specification'].str.contains('OLS', case=False, na=False)
iv_pre_mask  = iv_2sls_mask & ~iv_res['covid_interaction'].astype(bool)
iv_pre_row   = iv_res[iv_pre_mask].iloc[0]

iv_eap_b    = iv_pre_row['eap_elasticity']
iv_eap_pval = iv_pre_row['eap_pval']
iv_first_f  = iv_pre_row['first_stage_f']

# Full-sample IV row
iv_full_mask = iv_2sls_mask & iv_res['covid_interaction'].astype(bool)
iv_full_row  = iv_res[iv_full_mask].iloc[0] if iv_full_mask.any() else None

iv_full_f = iv_full_row['first_stage_f'] if iv_full_row is not None else float('nan')

# Full-sample IV elasticities (credible spec, F > 10)
iv_full_eu_b   = iv_full_row['eu_elasticity']  if iv_full_row is not None else float('nan')
iv_full_eu_se  = iv_full_row['eu_se']           if iv_full_row is not None else float('nan')
iv_full_eap_b  = iv_full_row['eap_elasticity']  if iv_full_row is not None else float('nan')
iv_full_eap_se = iv_full_row['eap_se']          if iv_full_row is not None else float('nan')

# ── sample restrictions ───────────────────────────────────────────────────────

def get_samp_row(keyword):
    mask = samp_res['restriction'].str.lower().str.contains(keyword, na=False)
    rows = samp_res[mask]
    return rows.iloc[0] if len(rows) else None

samp_bal  = get_samp_row('balanced')
samp_out  = get_samp_row('outlier')
samp_hi   = get_samp_row('high')
samp_mid  = get_samp_row('middle|lower')

# ── period split (Full Controls, GNI%, Panel A) ───────────────────────────────

# label = 'Full Controls (GNI%)' or similar for Panel A
ps_row = period.iloc[0]  # Panel A = first row

ps_eu_pre  = ps_row['eu_pre']
ps_eap_pre = ps_row['eap_pre']
ps_eu_ac   = ps_row['eu_ac']
ps_eap_ac  = ps_row['eap_ac']
ps_eu_pa   = ps_row['eu_pa']
ps_eap_pa  = ps_row['eap_pa']

# ── economic interpretation (derived values) ──────────────────────────────────

EAP_MEAN_SUBS  = 20.0   # mean EaP subscriptions per 100 (external data point)
EU_MEAN_SUBS   = 35.0   # mean EU subscriptions per 100 (external data point)
EAP_POPULATION = 75.0   # total EaP-6 population in millions
PRICE_CHANGE   = 0.10   # 10% price change used in the text

eap_pct_change = abs(eap_base_b) * PRICE_CHANGE * 100          # e.g. 6.1
eap_add_subs   = abs(eap_base_b) * PRICE_CHANGE * EAP_MEAN_SUBS  # e.g. 1.2
eap_add_total  = eap_add_subs / 100.0 * EAP_POPULATION * 1e6   # e.g. 900000
eu_add_subs    = abs(eu_base_b)  * PRICE_CHANGE * EU_MEAN_SUBS   # e.g. 0.35

# ── build macro dict ──────────────────────────────────────────────────────────

macros = {}

def add(name, value):
    macros[name] = str(value)

# --- Baseline (pre-COVID, Full Controls, GNI%) ---
add('EUBaseB',     fmt(eu_base_b))
add('EUBaseBr',    fmt(eu_base_b, 2))        # 2dp for inline text
add('EUBaseSE',    fmt(eu_base_se))
add('EUBasePval',  fmt(eu_base_pval))
add('EUBaseStars', stars(eu_base_pval))
add('EaPBaseB',    fmt(eap_base_b))
add('EaPBaseBr',   fmt(eap_base_b, 2))       # 2dp for inline text
add('EaPBaseBAbs', fmt(abs(eap_base_b), 2))  # absolute value for revenue discussion
add('EaPBaseSE',   fmt(eap_base_se))
add('EaPBasePval', fmt(eap_base_pval))
add('EaPBaseStars',stars(eap_base_pval))
add('IntCoef',     fmt(int_base_b, 2))
add('IntPval',     fmt(int_base_pval))
add('IntStars',    stars(int_base_pval))

# GDP coefficient (needed as \gdpcoef in results.tex)
add('GDPCoef',  fmt(gdp_base_b, 2) if not np.isnan(gdp_base_b) else r'\text{n/a}')
add('GDPPval',  fmt(gdp_base_pval) if not np.isnan(gdp_base_pval) else r'\text{n/a}')
add('GDPStars', stars(gdp_base_pval) if not np.isnan(gdp_base_pval) else '')
add('gdpcoef',  fmt(gdp_base_b, 2) if not np.isnan(gdp_base_b) else r'\text{n/a}')   # alias
add('BaselineN',   str(baseline_n))
add('BaselineRsq', fmt(baseline_r2))
eap_eu_ratio = abs(eap_base_b) / abs(eu_base_b) if abs(eu_base_b) > 0 else float('nan')
add('EaPEURatio',  fmt(eap_eu_ratio, 1))

# --- Control-spec stability (pre-COVID, GNI%) ---
add('EaPSpecMin',  fmt(eap_spec_min, 2))
add('EaPSpecMax',  fmt(eap_spec_max, 2))
add('EaPSpecMean', fmt(eap_spec_mean, 2))
add('EaPSpecSD',   fmt(eap_spec_std, 2))
add('EUSpecMin',   fmt(eu_spec_min, 2))
add('EUSpecMax',   fmt(eu_spec_max, 2))

# --- Price robustness per-measure significance counts ---
add('PctGNISigEaP',    str(pct_gni_sig_eap_05))
add('PctPPPSigEaP',    str(pct_ppp_sig_eap_05))
add('PctGNISigEaPten', str(pct_gni_sig_eap_10))
add('PctPPPSigEaPten', str(pct_ppp_sig_eap_10))
add('CntGNISigEaP',    str(cnt_gni_sig_eap_05))
add('CntPPPSigEaP',    str(cnt_ppp_sig_eap_05))
add('CntUSDSigEaP',    str(cnt_usd_sig_eap_05))
add('CntGNISigEU',     str(cnt_gni_sig_eu_05))
add('CntPPPSigEU',     str(cnt_ppp_sig_eu_05))
add('CntUSDSigEU',     str(cnt_usd_sig_eu_05))
add('NPerMeasure',     str(n_per_measure))

add('EaPGNIFCB',   fmt(eap_gni_fc_b) if not np.isnan(eap_gni_fc_b) else r'\text{n/a}')
add('EaPPPPFCB',   fmt(eap_ppp_fc_b) if not np.isnan(eap_ppp_fc_b) else r'\text{n/a}')
add('EaPUSDBCB',   fmt(eap_usd_fc_b) if not np.isnan(eap_usd_fc_b) else r'\text{n/a}')

# --- Robustness matrix summary (GNI% specs = primary price measure, 8 specs) ---
add('RobEaPMin',    fmt(rob_eap_min, 2))
add('RobEaPMax',    fmt(rob_eap_max, 2))
add('RobEaPMean',   fmt(rob_eap_mean, 2))
add('RobEaPSD',     fmt(rob_eap_sd, 2))
add('RobEUMin',     fmt(rob_eu_min, 2))
add('RobEUMax',     fmt(rob_eu_max, 2))
add('RobEUMean',    fmt(rob_eu_mean, 2))
add('RobEUSD',      fmt(rob_eu_sd, 2))
add('RobNSpecs',    str(rob_n_specs))          # total specs across all price measures
add('RobGNINSpecs', str(rob_gni_n_specs))      # GNI%-only spec count
add('RobEaPSigAll', str(rob_eap_sig_all))
add('RobEUSigFive', str(rob_eu_sig_05))
add('RobEUSigTen',  str(rob_eu_sig_10))

# --- COVID full-sample implied elasticities and interactions ---
add('EUPreFullB',     fmt(eu_pre_full_b, 2))
add('EUPreFullPval',  fmt(eu_pre_full_pval))
add('EUPreFullStars', stars(eu_pre_full_pval))
add('EaPPreFullB',    fmt(eap_pre_full_b, 2))
add('EaPPreFullPval', fmt(eap_pre_full_pval))
add('EaPPreFullStars',stars(eap_pre_full_pval))
add('EUCovidB',       fmt(eu_covid_b, 2))
add('EUCovidPval',    fmt(eu_covid_pval))
add('EUCovidStars',   stars(eu_covid_pval))
add('EaPCovidB',      fmt(eap_covid_b, 2))
add('EaPCovidPval',   fmt(eap_covid_pval))
add('EaPCovidStars',  stars(eap_covid_pval))
add('FullSampleN',    str(full_n))
# COVID interaction term (price × COVID)
add('EUCovInt',       fmt(eu_cov_int_b, 2))
add('EUCovIntPval',   fmt(eu_cov_int_pval))
add('EUCovIntStars',  stars(eu_cov_int_pval))
# Triple interaction (price × EaP × COVID)
add('EaPTripleInt',   fmt(eap_triple_b, 2))
add('EaPTriplePval',  fmt(eap_triple_pval))
add('EaPTripleStars', stars(eap_triple_pval))

# --- Interaction model pre-COVID baselines (Figure 4 data source) ---
add('EUPreIntB',      fmt(eu_pre_int_b, 2))
add('EUPreIntPval',   fmt(eu_pre_int_pval))
add('EUPreIntStars',  stars(eu_pre_int_pval))
add('EaPPreIntB',     fmt(eap_pre_int_b, 2))
add('EaPPreIntPval',  fmt(eap_pre_int_pval))
add('EaPPreIntStars', stars(eap_pre_int_pval))

# --- Year-by-year: 2015 EU elasticity cited in the text ---
if not np.isnan(eu_yby_2015_b):
    add('EUYbyYRefElast', fmt(eu_yby_2015_b, 2))
    add('EUYbyYRefPval',  fmt(eu_yby_2015_pval))

# --- Placebo test ---
add('PlaceboEaP',      fmt(eap_placebo_b, 2))
add('PlaceboEaPPval',  fmt(eap_placebo_pval))
add('PlaceboEaPStars', stars(eap_placebo_pval))
add('PlaceboEU',       fmt(eu_placebo_b, 2))
add('PlaceboEUPval',   fmt(eu_placebo_pval))
add('PlaceboEUStars',  stars(eu_placebo_pval))

# --- Jackknife ---
add('EaPJKMin',        fmt(eap_jk_min, 2))
add('EaPJKMax',        fmt(eap_jk_max, 2))
add('EaPJKMinCountry', str(jk_min_country))
add('EaPJKMaxCountry', str(jk_max_country))
add('EaPJKAllSig',     'true' if all_jk_sig else 'false')

# --- IV (pre-COVID: just-identified lagged price) ---
iv_pre_eu_b    = iv_pre_row['eu_elasticity']
iv_pre_eu_se   = iv_pre_row['eu_se']
iv_pre_eu_pval = iv_pre_row['eu_pval']
iv_pre_eap_b   = iv_pre_row['eap_elasticity']
iv_pre_eap_se  = iv_pre_row['eap_se']
iv_pre_eap_pval= iv_pre_row['eap_pval']
iv_pre_n       = int(iv_pre_row['n_obs'])

add('IVFirstF',     fmt(iv_first_f, 0))   # integer display, e.g. "65"
add('IVPreEUB',     fmt(iv_pre_eu_b, 3))
add('IVPreEUSE',    fmt(iv_pre_eu_se, 3))
add('IVPreEUPval',  fmt(iv_pre_eu_pval, 3))
add('IVPreEUStars', stars(iv_pre_eu_pval))
add('IVPreEaPB',    fmt(iv_pre_eap_b, 3))
add('IVPreEaPSE',   fmt(iv_pre_eap_se, 3))
add('IVPreEaPPval', fmt(iv_pre_eap_pval, 3))
add('IVPreEaPStars',stars(iv_pre_eap_pval))
add('IVPreN',       str(iv_pre_n))

# --- IV (full-sample: lagged price + COVID interactions) ---
add('IVFullF',   fmt(iv_full_f, 0) if not np.isnan(iv_full_f) else r'\text{n/a}')
if not np.isnan(iv_full_eu_b):
    iv_full_eu_pval  = iv_full_row['eu_pval']
    iv_full_eap_pval = iv_full_row['eap_pval']
    iv_full_n        = int(iv_full_row['n_obs'])
    add('IVFullEUB',     fmt(iv_full_eu_b, 3))
    add('IVFullEUSE',    fmt(iv_full_eu_se, 3))
    add('IVFullEUPval',  fmt(iv_full_eu_pval, 3))
    add('IVFullEUStars', stars(iv_full_eu_pval))
    add('IVFullEaPB',    fmt(iv_full_eap_b, 3))
    add('IVFullEaPSE',   fmt(iv_full_eap_se, 3))
    add('IVFullEaPPval', fmt(iv_full_eap_pval, 3))
    add('IVFullEaPStars',stars(iv_full_eap_pval))
    add('IVFullN',       str(iv_full_n))

# Legacy aliases kept for backward compatibility
add('IVEaPB',    fmt(iv_pre_eap_b, 2))
add('IVEaPPval', fmt(iv_pre_eap_pval, 3))
add('IVEaPStars',stars(iv_pre_eap_pval))

# --- Sample restrictions ---
if samp_bal is not None:
    add('SampBalN', str(int(samp_bal['n_obs'])))
if samp_hi  is not None:
    add('SampHiEUB',    fmt(samp_hi['eu_elasticity']))
    add('SampHiEUSE',   fmt(samp_hi['eu_se']))
    add('SampHiEUPval', fmt(samp_hi['eu_pval']))
    add('SampHiN',      str(int(samp_hi['n_obs'])))
if samp_mid is not None:
    add('SampMidEaPB',    fmt(samp_mid['eap_elasticity']))
    add('SampMidEaPSE',   fmt(samp_mid['eap_se']))
    add('SampMidEaPPval', fmt(samp_mid['eap_pval']))
    add('SampMidN',       str(int(samp_mid['n_obs'])))
if samp_out is not None:
    add('SampOutEaPN',  str(int(samp_out['n_obs'])))

# --- Economic interpretation (derived values) ---
add('EaPTenPctChg',      f'{eap_pct_change:.1f}')     # e.g. "6.1"
add('EaPAddSubsHundred', f'{eap_add_subs:.1f}')        # e.g. "1.2"
add('EaPAddSubsTotal',   fmt_large(eap_add_total))     # e.g. "900{,}000"
add('EUAddSubsHundred',  f'{eu_add_subs:.2f}')         # e.g. "0.35"

# ── descriptive statistics (from analysis_ready_data.csv) ─────────────────────

from code.utils.config import DATA_PROCESSED

desc_df = pd.read_csv(DATA_PROCESSED / 'analysis_ready_data.csv')
desc_eu  = desc_df[desc_df['region'] == 'EU']
desc_eap = desc_df[desc_df['region'] == 'EaP']

def dstat(series, decimals=1):
    """Return (mean, sd) formatted to given decimals."""
    return (f'{series.mean():.{decimals}f}', f'{series.std():.{decimals}f}')

# Variables for the descriptive stats table (label, column, decimals, thousands_divisor)
DESC_VARS = [
    (r'\textit{Dependent Variable}', None, None, None),
    ('Fixed broadband subs (per 100)', 'fixed_broadband_subs_alt', 1, None),
    ('SECTION', r'\textit{Price Variables}', None, None),
    (r'Price (\% of GNI per capita)', 'fixed_broad_price', 2, None),
    (r'Price (PPP US\$)', 'fixed_broad_price_ppp', 1, None),
    (r'Price (Nominal US\$)', 'fixed_broad_price_usd', 1, None),
    ('SECTION', r'\textit{Economic Variables}', None, None),
    (r'GDP per capita (US\$ 1000s)', 'gdp_per_capita', 1, 1000),
    (r'GDP growth (\%)', 'gdp_growth', 2, None),
    (r'Inflation (\%)', 'inflation_gdp_deflator', 1, None),
    ('SECTION', r'\textit{Socioeconomic Variables}', None, None),
    ('Urban population (\\%)', 'urban_population_pct', 1, None),
    ('Tertiary enrollment (\\%)', 'education_tertiary_pct', 1, None),
    ('Regulatory quality (index)', 'regulatory_quality_estimate', 2, None),
    ('SECTION', r'\textit{Infrastructure Variables}', None, None),
    ("Int'l bandwidth (Gbit/s)", 'int_bandwidth', 0, 1000),
    ('Secure servers (per million)', 'secure_internet_servers', 0, None),
    (r'R\&D expenditure (\% GDP)', 'research_development_expenditure', 2, None),
]

def fmt_int_comma(val):
    """Format integer with LaTeX comma grouping: 25624 -> 25{,}624."""
    s = f'{int(round(val)):,}'.replace(',', '{,}')
    return s

table_rows = []
for entry in DESC_VARS:
    if entry[0] == 'SECTION':
        # Section header with addlinespace
        table_rows.append(r'\addlinespace')
        table_rows.append(entry[1] + ' \\\\')
        continue
    label, col, dec, divisor = entry
    if col is None:
        # First section header (no addlinespace before it)
        table_rows.append(label + ' \\\\')
        continue
    vals = []
    for subset in [desc_df, desc_eu, desc_eap]:
        s = subset[col].dropna()
        if divisor:
            s = s / divisor
        if dec == 0 and col == 'secure_internet_servers':
            vals.extend([fmt_int_comma(s.mean()), fmt_int_comma(s.std())])
        else:
            m = f'{s.mean():.{dec}f}'
            sd = f'{s.std():.{dec}f}'
            vals.extend([m, sd])
    # Handle negative regulatory quality for EaP
    row = f'{label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\'
    # Format negative sign properly
    row = row.replace('& -', '& $-$')
    table_rows.append(row)

n_full = len(desc_df)
n_eu   = len(desc_eu)
n_eap  = len(desc_eap)
n_countries_full = desc_df['country'].nunique()
n_countries_eu   = desc_eu['country'].nunique()
n_countries_eap  = desc_eap['country'].nunique()
n_years = desc_df['year'].nunique()

# Generate the table .tex file
desc_table_tex = r"""\begin{table}[htbp]
\centering
\caption{Descriptive Statistics by Region (2010--2024)}
\label{tab:descriptives}
\begin{threeparttable}
\begin{adjustbox}{max width=\textwidth}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Full Sample} & \multicolumn{2}{c}{EU (""" + str(n_countries_eu) + r""")} & \multicolumn{2}{c}{EaP (""" + str(n_countries_eap) + r""")} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Variable & Mean & SD & Mean & SD & Mean & SD \\
\midrule
""" + '\n'.join(table_rows) + r"""
\midrule
Observations & \multicolumn{2}{c}{""" + str(n_full) + r"""} & \multicolumn{2}{c}{""" + str(n_eu) + r"""} & \multicolumn{2}{c}{""" + str(n_eap) + r"""} \\
Countries & \multicolumn{2}{c}{""" + str(n_countries_full) + r"""} & \multicolumn{2}{c}{""" + str(n_countries_eu) + r"""} & \multicolumn{2}{c}{""" + str(n_countries_eap) + r"""} \\
Years & \multicolumn{2}{c}{""" + str(n_years) + r"""} & \multicolumn{2}{c}{""" + str(n_years) + r"""} & \multicolumn{2}{c}{""" + str(n_years) + r"""} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\begin{tablenotes}[flushleft]
\scriptsize
\item \textit{Notes:} Summary statistics for balanced panel of """ + str(n_countries_full) + r""" countries over 2010--2024. EU includes """ + str(n_countries_eu) + r""" member states; EaP includes Armenia, Azerbaijan, Belarus, Georgia, Moldova, and Ukraine. Price and GDP figures in current US\$. Data sources: ITU (telecommunications), World Bank WDI (economic variables), World Bank WGI (governance).
\end{tablenotes}
\end{threeparttable}
\end{table}"""

desc_table_path = MANUSCRIPT_DIR / 'tables' / 'table_descriptives.tex'
desc_table_path.parent.mkdir(parents=True, exist_ok=True)
desc_table_path.write_text(desc_table_tex, encoding='utf-8')
print(f'Wrote descriptive stats table to {desc_table_path.relative_to(BASE_DIR)}')

# --- Descriptive stats inline macros for data.tex / discussion.tex text ---
add('DescSubsEU',   f'{desc_eu["fixed_broadband_subs_alt"].mean():.1f}')
add('DescSubsEaP',  f'{desc_eap["fixed_broadband_subs_alt"].mean():.1f}')
add('DescSubsFull', f'{desc_df["fixed_broadband_subs_alt"].mean():.1f}')
add('DescGDPEU',    f'{desc_eu["gdp_per_capita"].mean():,.0f}')
add('DescGDPEaP',   f'{desc_eap["gdp_per_capita"].mean():,.0f}')
add('DescPriceGNIEaP', f'{desc_eap["fixed_broad_price"].mean():.1f}')
add('DescPriceGNIEU',  f'{desc_eu["fixed_broad_price"].mean():.1f}')

# ── write paper_macros.tex ────────────────────────────────────────────────────

out_path = MANUSCRIPT_DIR / 'paper_macros.tex'
lines = [
    '% Auto-generated by code/analysis/generate_paper_macros.py',
    '% DO NOT EDIT MANUALLY -- re-run script to regenerate.',
    '% Add  \\input{paper_macros}  in paper.tex preamble after \\usepackage lines.',
    '',
]
for name, val in macros.items():
    lines.append(f'\\newcommand{{\\{name}}}{{{val}}}')
lines.append('')  # trailing newline

out_path.write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote {len(macros)} macros to {out_path.relative_to(BASE_DIR)}')
for k, v in macros.items():
    print(f'  \\{k:<26} = {v}')
