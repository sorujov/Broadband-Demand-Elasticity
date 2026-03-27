import pandas as pd, scipy.stats as st

df = pd.read_excel('results/regression_output/full_sample_covid_analysis/price_robustness_matrix.xlsx')
print("Columns:", df.columns.tolist())
print()

# Re-derive df_resid from stored t-stat and p-value
# t = b/se, p = 2*(1-T.cdf(|t|, df))  =>  T.ppf(1-p/2, df) = |t|
# We can back out df by trying values
row = df[(df['control_spec'] == 'Full Controls (Baseline)') & (df['price_measure'] == 'GNI%')].iloc[0]
b, se, p_stored = row.eu_pre_elasticity, row.eu_pre_se, row.eu_pre_pval
t_obs = abs(b / se)
print(f"Observed |t| = {t_obs:.6f}")
print(f"Stored p     = {p_stored:.6f}")
print()
for df_try in [400, 420, 440, 450, 460, 480, 495]:
    p_try = 2 * st.t.sf(t_obs, df=df_try)
    print(f"  df={df_try}: p={p_try:.6f}  diff={abs(p_try - p_stored):.8f}")

# Also check the pre-covid baseline regression
print()
df2 = pd.read_excel('results/regression_output/pre_covid_analysis/price_robustness_matrix.xlsx')
print("Pre-COVID columns:", df2.columns.tolist())
row2 = df2[(df2['control_spec'] == 'Full Controls (Baseline)') & (df2['price_measure'] == 'GNI%')].iloc[0]
b2, se2, p2 = row2['eu_elasticity'], row2['eu_se'], row2['eu_pval']
t2 = abs(b2/se2)
print(f"Pre-COVID EU |t|={t2:.6f}  p_stored={p2:.6f}")
for df_try in [240, 250, 260, 270, 280, 290, 300]:
    p_try = 2 * st.t.sf(t2, df=df_try)
    print(f"  df={df_try}: p={p_try:.6f}  diff={abs(p_try - p2):.8f}")
