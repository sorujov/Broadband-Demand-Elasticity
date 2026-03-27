"""Verify that res.cov uses the same DK kernel/bw=3 as res.std_errors."""
import pandas as pd, numpy as np
from linearmodels.panel import PanelOLS

df = pd.read_csv('data/processed/analysis_ready_data.csv')
df['year_dt'] = pd.to_datetime(df['year'], format='%Y')
df = df.set_index(['country', 'year_dt'])
df = df[df.index.get_level_values('year_dt').year <= 2019]

PP = 'log_fixed_broad_price'
DV = 'log_fixed_broadband_subs'
EAP = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['eap_dummy'] = df.index.get_level_values('country').isin(EAP).astype(float)
df['price_x_eap'] = df[PP] * df['eap_dummy']

ctrls = ['log_gdp_per_capita', 'urban_population_pct', 'education_tertiary_pct',
         'regulatory_quality_estimate', 'log_secure_internet_servers',
         'research_development_expenditure', 'population_ages_15_64',
         'gdp_growth', 'inflation_gdp_deflator', 'log_population_density']
av = [c for c in ctrls if c in df.columns]
cols = [DV, PP, 'price_x_eap'] + av
d = df[cols].dropna()
y = d[DV]; X = d[[PP, 'price_x_eap'] + av]

# Fit with DK bw=3
r3 = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
    cov_type='kernel', kernel='bartlett', bandwidth=3)
# Fit with DK bw=1
r1 = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
    cov_type='kernel', kernel='bartlett', bandwidth=1)
# Fit with clustered
rc = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)

print("=" * 75)
print("VERIFICATION: res.cov uses same kernel/bandwidth as res.std_errors")
print("=" * 75)

print("\n1. SE from cov diagonal vs reported SE (DK bw=3)")
vars2 = [PP, 'price_x_eap']
se_from_cov = np.sqrt(np.diag(r3.cov.loc[vars2, vars2].values))
se_reported = r3.std_errors[vars2].values
print(f"   SE(price)     from cov diagonal: {se_from_cov[0]:.10f}")
print(f"   SE(price)     from std_errors:   {se_reported[0]:.10f}")
print(f"   SE(interact.) from cov diagonal: {se_from_cov[1]:.10f}")
print(f"   SE(interact.) from std_errors:   {se_reported[1]:.10f}")
print(f"   Perfect match: {np.allclose(se_from_cov, se_reported, atol=1e-15)}")

print("\n2. Cov matrix changes with bandwidth (proves it's bandwidth-specific)")
cov3 = r3.cov.loc[PP, 'price_x_eap']
cov1 = r1.cov.loc[PP, 'price_x_eap']
covc = rc.cov.loc[PP, 'price_x_eap']
print(f"   Cov(price, interact) DK bw=3:   {cov3:.10f}")
print(f"   Cov(price, interact) DK bw=1:   {cov1:.10f}")
print(f"   Cov(price, interact) Clustered: {covc:.10f}")
print(f"   bw=3 != bw=1: {not np.isclose(cov3, cov1)}")
print(f"   bw=3 != clust: {not np.isclose(cov3, covc)}")

print("\n3. SE(price) also changes with bandwidth (same source)")
print(f"   SE(price) DK bw=3:   {r3.std_errors[PP]:.10f}")
print(f"   SE(price) DK bw=1:   {r1.std_errors[PP]:.10f}")
print(f"   SE(price) Clustered: {rc.std_errors[PP]:.10f}")

print("\n4. Full 2x2 VCV submatrix for DK bw=3:")
sub = r3.cov.loc[vars2, vars2]
print(sub.to_string())
corr = cov3 / (r3.std_errors[PP] * r3.std_errors['price_x_eap'])
print(f"\n   Correlation(price, interact): {corr:.4f}")

print("\n5. Implied EaP SE comparison")
eap_b = r3.params[PP] + r3.params['price_x_eap']
print(f"   EaP elasticity = {eap_b:.4f}")
for label, r in [('DK bw=3', r3), ('DK bw=1', r1), ('Clustered', rc)]:
    s1 = r.std_errors[PP]
    s2 = r.std_errors['price_x_eap']
    cov = r.cov.loc[PP, 'price_x_eap']
    se_nocov = np.sqrt(s1**2 + s2**2)
    se_cov   = np.sqrt(s1**2 + s2**2 + 2*cov)
    print(f"   {label:<12}  Var(p)={s1**2:.6f}  Var(i)={s2**2:.6f}  2*Cov={2*cov:+.6f}  "
          f"SE_old={se_nocov:.4f}  SE_new={se_cov:.4f}")

print("\n6. linearmodels internals check")
print(f"   r3._cov_type:   {r3._cov_type}")
print(f"   r3._bandwidth:  {getattr(r3, '_bandwidth', 'N/A')}")
print(f"   r3._kernel:     {getattr(r3, '_kernel', 'N/A')}")
# Check if cov_config is stored
if hasattr(r3, '_cov_config'):
    print(f"   r3._cov_config: {r3._cov_config}")

print("\n" + "=" * 75)
print("CONCLUSION: res.cov is the FULL variance-covariance matrix from the SAME")
print("DK kernel estimation that produces res.std_errors. Both use bandwidth=3.")
print("=" * 75)
