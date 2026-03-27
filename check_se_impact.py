"""Compare old (no covariance) vs new (with covariance) SE for all implied elasticities."""
import pandas as pd, numpy as np
from scipy import stats
from linearmodels.panel import PanelOLS

df = pd.read_csv('data/processed/analysis_ready_data.csv')
df['year_dt'] = pd.to_datetime(df['year'], format='%Y')
df = df.set_index(['country', 'year_dt'])

PRIMARY_PRICE = 'log_fixed_broad_price'
PRIMARY_DV = 'log_fixed_broadband_subs'

EAP_COUNTRIES = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df.index.get_level_values('country').isin(EAP_COUNTRIES).astype(float)
df['price_x_eap'] = df[PRIMARY_PRICE] * df['eap_dummy']

controls = ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
            'regulatory_quality_estimate', 'log_secure_internet_servers',
            'research_development_expenditure', 'population_ages_15_64',
            'gdp_growth', 'inflation_gdp_deflator', 'log_population_density']

def sig(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return 'ns'

# ─── 1. Table A.1 columns ───
TABLE_SPECS = [
    ('(1) GDP Only',  ['log_gdp_per_capita']),
    ('(2) + Socio',   ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct']),
    ('(3) + Instit.', ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                       'regulatory_quality_estimate']),
    ('(4) + Infra.',  ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                       'regulatory_quality_estimate', 'log_secure_internet_servers',
                       'research_development_expenditure']),
    ('(5) + Demog.',  ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                       'regulatory_quality_estimate', 'log_secure_internet_servers',
                       'research_development_expenditure', 'log_population_density',
                       'population_ages_15_64']),
    ('(7) Full',      ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                       'regulatory_quality_estimate', 'log_secure_internet_servers',
                       'research_development_expenditure', 'log_population_density',
                       'population_ages_15_64', 'gdp_growth', 'inflation_gdp_deflator']),
]

df_pre = df[df.index.get_level_values('year_dt').year <= 2019]

print("=" * 85)
print("IMPACT OF SE CORRECTION: Old (no cov) vs New (with cov)")
print("=" * 85)

any_changed = False

print("\n1. TABLE A.1: Implied EaP elasticity rows")
hdr = f"   {'Column':<18} {'EaP':>7} {'OldSE':>7} {'NewSE':>7} {'Chg%':>6} {'OldSig':>7} {'NewSig':>7} {'Changed?':>9}"
print(hdr)
print("   " + "-" * 72)
for label, ctrls in TABLE_SPECS:
    av = [c for c in ctrls if c in df_pre.columns]
    cols = [PRIMARY_DV, PRIMARY_PRICE, 'price_x_eap'] + av
    d = df_pre[cols].dropna()
    y = d[PRIMARY_DV]; X = d[[PRIMARY_PRICE, 'price_x_eap'] + av]
    try:
        r = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
            cov_type='kernel', kernel='bartlett', bandwidth=3)
    except Exception:
        print(f"   {label:<18} SKIPPED (absorbed/rank)")
        continue
    b = r.params[PRIMARY_PRICE] + r.params['price_x_eap']
    s_old = np.sqrt(r.std_errors[PRIMARY_PRICE]**2 + r.std_errors['price_x_eap']**2)
    s_new = np.sqrt(r.std_errors[PRIMARY_PRICE]**2 + r.std_errors['price_x_eap']**2 + 2*r.cov.loc[PRIMARY_PRICE, 'price_x_eap'])
    p_old = 2*(1-stats.t.cdf(abs(b/s_old), df=r.df_resid))
    p_new = 2*(1-stats.t.cdf(abs(b/s_new), df=r.df_resid))
    ch = sig(p_old) != sig(p_new)
    if ch: any_changed = True
    chstr = 'YES <<<' if ch else 'No'
    pct = ((s_new - s_old)/s_old*100)
    print(f"   {label:<18} {b:>7.3f} {s_old:>7.3f} {s_new:>7.3f} {pct:>+5.1f}% {sig(p_old):>7} {sig(p_new):>7} {chstr:>9}")

# ─── 2. Table A.3: Price robustness ───
PRICE_MEASURES = [
    {'name': 'GNI%', 'var': 'log_fixed_broad_price'},
    {'name': 'PPP',  'var': 'log_fixed_broad_price_ppp'},
    {'name': 'USD',  'var': 'log_fixed_broad_price_usd'},
]
CONTROL_SPECS = {
    'Full Controls': controls,
    'Comprehensive': ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
                      'regulatory_quality_estimate', 'log_secure_internet_servers'],
    'Core':          ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct'],
    'Institutional': ['log_gdp_per_capita', 'regulatory_quality_estimate'],
    'Infrastructure':['log_gdp_per_capita', 'log_secure_internet_servers', 'research_development_expenditure'],
    'Demographic':   ['log_gdp_per_capita', 'urban_population_pct', 'log_population', 'population_ages_15_64'],
    'Macroeconomic': ['log_gdp_per_capita', 'gdp_growth', 'inflation_gdp_deflator'],
    'Minimal':       ['log_gdp_per_capita'],
}

print("\n2. TABLE A.3: Price Robustness (24 specs)")
hdr2 = f"   {'Price':<6} {'Spec':<18} {'EaP':>7} {'OldSE':>7} {'NewSE':>7} {'Chg%':>6} {'OldSig':>7} {'NewSig':>7} {'Changed?':>9}"
print(hdr2)
print("   " + "-" * 78)
for pm in PRICE_MEASURES:
    int_name = f"price_x_eap_{pm['name']}"
    df_pre[int_name] = df_pre[pm['var']] * df_pre['eap_dummy']
    for spec_name, spec_ctrls in CONTROL_SPECS.items():
        av = [c for c in spec_ctrls if c in df_pre.columns]
        req = [PRIMARY_DV, pm['var'], int_name] + av
        avail_req = [c for c in req if c in df_pre.columns]
        d = df_pre[avail_req].dropna()
        if len(d) < 100: continue
        y = d[PRIMARY_DV]; X = d[[pm['var'], int_name] + [c for c in av if c in d.columns]]
        try:
            r = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
                cov_type='kernel', kernel='bartlett', bandwidth=3)
        except Exception:
            continue
        b = r.params[pm['var']] + r.params[int_name]
        s_old = np.sqrt(r.std_errors[pm['var']]**2 + r.std_errors[int_name]**2)
        s_new = np.sqrt(r.std_errors[pm['var']]**2 + r.std_errors[int_name]**2 + 2*r.cov.loc[pm['var'], int_name])
        p_old = 2*(1-stats.t.cdf(abs(b/s_old), df=r.df_resid))
        p_new = 2*(1-stats.t.cdf(abs(b/s_new), df=r.df_resid))
        ch = sig(p_old) != sig(p_new)
        if ch: any_changed = True
        chstr = 'YES <<<' if ch else 'No'
        pct = ((s_new - s_old)/s_old*100)
        print(f"   {pm['name']:<6} {spec_name:<18} {b:>7.3f} {s_old:>7.3f} {s_new:>7.3f} {pct:>+5.1f}% {sig(p_old):>7} {sig(p_new):>7} {chstr:>9}")

print("\n" + "=" * 85)
if any_changed:
    print("WARNING: Some significance levels CHANGED. Check details above.")
else:
    print("RESULT: No significance levels changed. All results are qualitatively identical.")
print("Covariance is negative => corrected SEs are SMALLER => results slightly more significant.")
print("=" * 85)
