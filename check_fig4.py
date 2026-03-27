import pandas as pd
import scipy.stats as st

df = pd.read_excel('results/regression_output/full_sample_covid_analysis/price_robustness_matrix.xlsx')
b = df[(df['control_spec']=='Full Controls (Baseline)') & (df['price_measure']=='GNI%')].iloc[0]
t90 = st.t.ppf(0.95, df=b['df_resid'])
print(f"df_resid={b['df_resid']:.0f}, t90={t90:.4f}")
print()
print("=== PRE-COVID ===")
eu_pre = b['eu_pre_elasticity']; eu_pre_se = b['eu_pre_se']; eu_pre_p = b['eu_pre_pval']
eap_pre = b['eap_pre_elasticity']; eap_pre_se = b['eap_pre_se']; eap_pre_p = b['eap_pre_pval']
print(f"EU:  elast={eu_pre:.4f}, SE={eu_pre_se:.4f}, p={eu_pre_p:.4f}")
print(f"     90%CI: [{eu_pre-t90*eu_pre_se:.4f}, {eu_pre+t90*eu_pre_se:.4f}]  crosses_zero={eu_pre-t90*eu_pre_se<0<eu_pre+t90*eu_pre_se}")
print(f"EaP: elast={eap_pre:.4f}, SE={eap_pre_se:.4f}, p={eap_pre_p:.4f}")
print(f"     90%CI: [{eap_pre-t90*eap_pre_se:.4f}, {eap_pre+t90*eap_pre_se:.4f}]  crosses_zero={eap_pre-t90*eap_pre_se<0<eap_pre+t90*eap_pre_se}")
print()
print("=== COVID ===")
eu_c = b['eu_covid_elasticity']; eu_c_se = b['eu_covid_se']; eu_c_p = b['eu_covid_pval']
eap_c = b['eap_covid_elasticity']; eap_c_se = b['eap_covid_se']; eap_c_p = b['eap_covid_pval']
print(f"EU:  elast={eu_c:.4f}, SE={eu_c_se:.4f}, p={eu_c_p:.4f}")
print(f"     90%CI: [{eu_c-t90*eu_c_se:.4f}, {eu_c+t90*eu_c_se:.4f}]  crosses_zero={eu_c-t90*eu_c_se<0<eu_c+t90*eu_c_se}")
print(f"EaP: elast={eap_c:.4f}, SE={eap_c_se:.4f}, p={eap_c_p:.4f}")
print(f"     90%CI: [{eap_c-t90*eap_c_se:.4f}, {eap_c+t90*eap_c_se:.4f}]  crosses_zero={eap_c-t90*eap_c_se<0<eap_c+t90*eap_c_se}")
print()
print("=== STARS ===")
def stars(p):
    if p<0.01: return '***'
    if p<0.05: return '**'
    if p<0.10: return '*'
    return '(none)'
print(f"EU pre: {stars(eu_pre_p)}, EaP pre: {stars(eap_pre_p)}")
print(f"EU covid: {stars(eu_c_p)}, EaP covid: {stars(eap_c_p)}")
print()
print("=== YLIM COMPUTATION ===")
all_vals = [eu_pre - t90*eu_pre_se, eap_pre - t90*eap_pre_se,
            eu_c + t90*eu_c_se, eap_c + t90*eap_c_se]
print(f"all_vals used for ylim: {[round(v,4) for v in all_vals]}")
print(f"ylim: ({min(all_vals)-0.12:.4f}, {max(all_vals)+0.12:.4f})")
print(f"NOTE: COVID lower bounds NOT in all_vals:")
print(f"  EU covid lower: {eu_c-t90*eu_c_se:.4f}")
print(f"  EaP covid lower: {eap_c-t90*eap_c_se:.4f}")
