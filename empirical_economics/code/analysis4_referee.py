"""Checks added after internal referee review. Run after analysis.py/analysis2.py."""
import pandas as pd, numpy as np, json, warnings, common as c, infer as I
from scipy import stats
warnings.filterwarnings('ignore')
R=json.load(open('results/results.json')); B={}
d=pd.read_pickle('panel_final.pkl')
CORE=['log_gdp_per_capita','urban_population_pct','education_tertiary_pct','regulatory_quality_estimate','population_ages_15_64','gdp_growth','inflation_gdp_deflator','log_population_density']
DC=['D'+x for x in CORE]
def row(r,n): return {k:dict(b=float(r.params[k]),se=float(r.std_errors[k]),p=float(r.pvalues[k])) for k in n}
pre=d[d.year<=2019].copy(); pre['Dpe']=pre.Dp*pre.eap
s=pre.dropna(subset=['Dy','Dp','Dpe']+DC)
# 1. FD with country effects (growth-rate regression with country intercepts)
r=c.twfe(s,'Dy',['Dp','Dpe']+DC,'cl',ent=True); e=c.lincomb(r,{'Dp':1,'Dpe':1})
r2=c.twfe(s,'Dy',['Dp']+DC,'cl',ent=True)
B['fd_fe']=dict(pooled=row(r2,['Dp'])['Dp'],eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),int=row(r,['Dpe'])['Dpe'],n=int(r.nobs))
old=I.design
I.design=lambda a,b_,cc,fe=('country','year'):old(a,b_,cc,fe=('country','year'))
B['fd_fe']['wcr_int']=I.wcr(s.reset_index(drop=True),'Dy',['Dp','Dpe']+DC,{'Dpe':1},B=4999)[3]
I.design=old
# 2. EaP-specific year effects: identification of EaP slope from within-EaP cross-section
s2=s.copy(); s2['gy']=s2.eap.astype(int).astype(str)+'_'+s2.year.astype(str)
I.design=lambda a,b_,cc,fe=('gy',):old(a,b_,cc,fe=('gy',))
from linearmodels.panel import PanelOLS
dd=s2.set_index(['country','year'])
m=PanelOLS(dd.Dy,dd[['Dp','Dpe']+DC],other_effects=dd[['gy']]).fit(cov_type='clustered',cluster_entity=True)
e=c.lincomb(m,{'Dp':1,'Dpe':1})
B['fd_groupyear']=dict(eu=row(m,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),int=row(m,['Dpe'])['Dpe'],n=int(m.nobs),
                        wcr_int=I.wcr(s2.reset_index(drop=True),'Dy',['Dp','Dpe']+DC,{'Dpe':1},B=4999)[3])
I.design=old
# 3. unweighted mean group
sl=pd.read_csv('results/country_fd_slopes_pre.csv')
B['mg_unweighted']={g:dict(b=float(x.b.mean()),se=float(x.b.std(ddof=1)/np.sqrt(len(x)))) for g,x in [('eu',sl[sl.eap==0]),('eap',sl[sl.eap==1])]}
for g in B['mg_unweighted'].values(): g['p']=float(2*stats.t.sf(abs(g['b']/g['se']),5 if g is B['mg_unweighted'].get('eap') else 26))
B['mg_perm_two_sided']=R['T3']['mg']['perm_p']
B['ranksum_two_sided']=float(stats.mannwhitneyu(sl[sl.eap==1].b,sl[sl.eap==0].b,alternative='two-sided').pvalue)
# weight share of each EaP country in inverse-variance mean
ea=sl[sl.eap==1]; w=1/ea.se**2; B['ivw_weights_eap']=dict(zip(ea.country,(w/w.sum()).round(3)))
# 4. EaP-EU difference by year (FD)
yrs=[y for y in range(2011,2020) if y!=2018]
dd=d[d.year.isin(yrs)].copy(); X=['Dp']; 
dd['Dpe']=dd.Dp*dd.eap
for y in yrs:
    dd[f'e{y}']=dd.Dpe*(dd.year==y); X.append(f'e{y}')
X=[x for x in X if x!='Dpe']
r=c.twfe(dd.dropna(subset=['Dy','Dp']+DC),'Dy',X+DC,'cl',ent=False)
B['eap_by_year']={y:dict(b=float(r.params[f'e{y}']+r.params['Dp']),se=float(c.lincomb(r,{'Dp':1,f'e{y}':1})[1]),n_eap=int(dd[(dd.year==y)&(dd.eap==1)].Dp.notna().sum())) for y in yrs}
# pooled OLS weights by year: share of sum of squared residualised Dp*EaP
# 5. EU part of decomposition GDP coefficient etc. and post-2019 trend reconciliation are in results.json already
# 6. headroom with K=70
h=d.copy(); h['hl70']=1-h.groupby('country').s100.shift(1)/70; h['hc']=h.hl70-h.hl70.mean(); h['Dpxh']=h.Dp*h.hc
hp=h[h.year<=2019].copy(); hp['Dpxe']=hp.Dp*hp.eap
r=c.twfe(hp.dropna(subset=['Dy','Dp','Dpxh']+DC),'Dy',['Dp','Dpxh']+DC,'cl',ent=False); B['headroom70_pre']=row(r,['Dp','Dpxh'])
r=c.twfe(h.dropna(subset=['Dy','Dp','Dpxh','Dpxpost']+DC),'Dy',['Dp','Dpxh','Dpxpost']+DC,'cl',ent=False); B['headroom70_full']=row(r,['Dp','Dpxh','Dpxpost'])
r=c.twfe(hp.dropna(subset=['Dy','Dp','Dpxh','Dpxe']+DC),'Dy',['Dp','Dpxe','Dpxh']+DC,'cl',ent=False); B['headroom70_eap']=row(r,['Dp','Dpxe','Dpxh'])
# 7. GDP coefficients in decomposition
R['B']=B
json.dump(R,open('results/results.json','w'),indent=1,default=float)
print(json.dumps(B,indent=1,default=float)[:4000])
# 8. additional sub-period windows quoted in the text
R=json.load(open('results/results.json'))
W={}
for a_,b_ in [(2011,2016),(2014,2016)]:
    s=d[d.year.between(a_,b_)].copy(); s['Dpe']=s.Dp*s.eap
    s=s.dropna(subset=['Dy','Dp','Dpe']+DC)
    r=c.twfe(s,'Dy',['Dp','Dpe']+DC,'cl',ent=False); e=c.lincomb(r,{'Dp':1,'Dpe':1})
    W[f'{a_}_{b_}']=dict(eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs))
# bloc averages
g=d[d.eap==1].groupby('year')[['Dy','Dp']].mean()
R['B']['subperiod_extra']=W; R['B']['eap_bloc_means']={int(k):v for k,v in g.round(4).to_dict('index').items()}
json.dump(R,open('results/results.json','w'),indent=1,default=float)
print(W)
