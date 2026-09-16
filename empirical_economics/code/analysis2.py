import pandas as pd, numpy as np, json, warnings, common as c, infer as I
from scipy import stats
warnings.filterwarnings('ignore')
RES=json.load(open('results/results.json')); A={}
d=pd.read_pickle('panel_final.pkl').sort_values(['country','year']).reset_index(drop=True); d0=pd.read_pickle('panel.pkl').sort_values(['country','year']).reset_index(drop=True)
assert (d.country.values==d0.country.values).all() and (d.year.values==d0.year.values).all()
CORE=['log_gdp_per_capita','urban_population_pct','education_tertiary_pct','regulatory_quality_estimate','population_ages_15_64','gdp_growth','inflation_gdp_deflator','log_population_density']
DC=['D'+x for x in CORE]
def row(r,n): return {k:dict(b=float(r.params[k]),se=float(r.std_errors[k]),p=float(r.pvalues[k])) for k in n}
def cl(df,y,X,ent=False):
    s=df.dropna(subset=[y]+X); return c.twfe(s,y,X,'cl',ent=ent), s
# 1. long differences within basket regimes
LD={}
base=d0.copy()
base['p']=d['p'].values  # carried-forward set missing
for k in [1,3,5,7]:
    rows=[]
    for (a,b) in [(2010,2017),(2018,2024)]:
        s=base[(base.year>=a)&(base.year<=b)].copy().sort_values(['country','year'])
        g=s.groupby('country')
        for v in ['y','p']+CORE: s['K'+v]=s[v]-g[v].shift(k)
        s['reg']=a; rows.append(s)
    s=pd.concat(rows).dropna(subset=['Ky','Kp'])
    s['yr_reg']=s.year.astype(str)
    s['Kpe']=s.Kp*s.eap
    for lab,sub in [('old',s[s.reg==2010]),('new',s[s.reg==2018])]:
        if len(sub)<40: continue
        r,ss=cl(sub,'Ky',['Kp','Kpe']+['K'+x for x in CORE])
        e=c.lincomb(r,{'Kp':1,'Kpe':1})
        LD[f'{lab}_k{k}']=dict(eu=row(r,['Kp'])['Kp'],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs))
A['longdiff']=LD
# 2. decompose ratio: implied annual GNI pc
d['lgni']=np.log(d0.fixed_broad_price_usd/(d0.fixed_broad_price/100)*12)
d.loc[d.p.isna(),'lgni']=np.nan
g=d.sort_values(['country','year']).groupby('country')
d['Dlgni']=g.lgni.diff(); d.loc[d.year==2018,'Dlgni']=np.nan
d['Dp_usd2']=d.Dp_usd
pre=d[d.year<=2019].copy()
for v in ['Dp_usd2','Dlgni']: pre[v+'e']=pre[v]*pre.eap
DEC={}
for lab,ctrl in [('with_gdp',DC),('no_gdp',[x for x in DC if x!='Dlog_gdp_per_capita'])]:
    X=['Dp_usd2','Dp_usd2e','Dlgni','Dlgnie']+ctrl
    r,s=cl(pre,'Dy',X)
    out=row(r,X[:4])
    out['eap_usd']=dict(zip(['b','se','p'],c.lincomb(r,{'Dp_usd2':1,'Dp_usd2e':1})))
    out['eap_gni']=dict(zip(['b','se','p'],c.lincomb(r,{'Dlgni':1,'Dlgnie':1})))
    # affordability restriction for EaP: usd + gni = 0
    out['eap_restr']=dict(zip(['b','se','p'],c.lincomb(r,{'Dp_usd2':1,'Dp_usd2e':1,'Dlgni':1,'Dlgnie':1})))
    out['eu_restr']=dict(zip(['b','se','p'],c.lincomb(r,{'Dp_usd2':1,'Dlgni':1})))
    DEC[lab]=out
A['decomp']=DEC
# crisis exclusions
crisis={('AZE',2015),('AZE',2016),('UKR',2014),('UKR',2015),('BLR',2011),('BLR',2015),('MDA',2015),('GEO',2015),('ARM',2015)}
def fdpair(df):
    df=df.copy(); df['Dpe_']=df.Dp*df.eap
    r,_=cl(df,'Dy',['Dp','Dpe_']+DC); e=c.lincomb(r,{'Dp':1,'Dpe_':1})
    return dict(eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs))
A['no_crisis']=fdpair(pre[[ (cc,y) not in crisis for cc,y in zip(pre.country,pre.year)]])
A['no_aze']=fdpair(pre[pre.country!='AZE'])
A['no_aze_geo']=fdpair(pre[~pre.country.isin(['AZE','GEO'])])
# 3. lead placebo
s=d.sort_values(['country','year']).copy(); s['FDp']=s.groupby('country').Dp.shift(-1)
s=s[s.year<=2018]  # lead defined within pre sample (2019 lead missing for most)
s['FDpe']=s.FDp*s.eap; s['Dpe']=s.Dp*s.eap
r,_=cl(s,'Dy',['Dp','Dpe','FDp','FDpe']+DC)
A['lead']=row(r,['Dp','Dpe','FDp','FDpe'])|{'eap_lead':dict(zip(['b','se','p'],c.lincomb(r,{'FDp':1,'FDpe':1}))),'eap_cur':dict(zip(['b','se','p'],c.lincomb(r,{'Dp':1,'Dpe':1}))),'n':int(r.nobs)}
# 4. equivalence (TOST) EU FD vs |0.10|; 90% CI
eu=RES['T3']['fd']['Dp']
lo,hi=eu['b']-1.645*eu['se'],eu['b']+1.645*eu['se']
A['tost_eu']=dict(ci90=[lo,hi],p_tost=float(max(stats.norm.sf((eu['b']+0.10)/eu['se']),stats.norm.sf((0.10-eu['b'])/eu['se']))))
nu=RES['T5']['tw_ld']['pxld']; A['nuts_ci95']=[nu['b']-1.96*nu['se'],nu['b']+1.96*nu['se']]
nu2=RES['T5']['tw_ld']['p']; A['nuts_p_ci95']=[nu2['b']-1.96*nu2['se'],nu2['b']+1.96*nu2['se']]
# 5. FD sample composition
fdS=d[(d.year<=2024)].dropna(subset=['Dy','Dp']+DC)
A['fd_obs_by_year']=fdS.groupby('year').size().to_dict()
# 6. robust MG
sl=pd.read_csv('results/country_fd_slopes_pre.csv')
A['mg_robust']={g:dict(median=float(x.b.median()),trim_mean=float(stats.trim_mean(x.b,0.2)),
    n_neg=int((x.b<0).sum()),n=len(x), inv_var_mean=float(np.sum(x.b/x.se**2)/np.sum(1/x.se**2)),
    inv_var_se=float(np.sqrt(1/np.sum(1/x.se**2)))) for g,x in [('eu',sl[sl.eap==0]),('eap',sl[sl.eap==1])]}
# sign test EaP vs EU median
A['mg_ranksum_p']=float(stats.mannwhitneyu(sl[sl.eap==1].b,sl[sl.eap==0].b,alternative='less').pvalue)
# 7. residual persistence of static TWFE (pre, balanced)
bal=d0[d0.year<=2019]
r=c.twfe(bal,'y',['p']+CORE,'cl')
e=r.resids.reset_index(); e.columns=['country','year','e']; e=e.sort_values(['country','year'])
e['le']=e.groupby('country').e.shift(1); ee=e.dropna()
A['resid_ar1']=float(np.corrcoef(ee['e'].values,ee['le'].values)[0,1])
fdr=I  # noop
# satellite night lights as statistics-independent activity control
s=d.copy(); s['lntl']=d0.log_ntl_pc.values
s['Dlntl']=s.groupby('country').lntl.diff(); s.loc[s.year==2014,'Dlntl']=np.nan
p2=s[s.year<=2019].copy(); p2['Dpe_']=p2.Dp*p2.eap
ctrl_ntl=[x for x in DC if x!='Dlog_gdp_per_capita']+['Dlntl']
r,_=cl(p2,'Dy',['Dp','Dpe_']+ctrl_ntl); e=c.lincomb(r,{'Dp':1,'Dpe_':1})
A['ntl']=dict(eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),ntl=row(r,['Dlntl'])['Dlntl'],n=int(r.nobs))
r,_=cl(p2,'Dy',['Dp','Dpe_']+DC+['Dlntl']); e=c.lincomb(r,{'Dp':1,'Dpe_':1})
A['ntl_plus_gdp']=dict(eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs))
# growth of lights vs GDP growth correlation (plausibility)
A['ntl_gdp_corr_fd']=float(s[['Dlntl','Dlog_gdp_per_capita']].dropna().corr().iloc[0,1])
# sub-period FD, EU vs EaP
SP={}
def mk_fd(df,lab):
    e=df.country.isin(lab)*1.0; df['Dpe']=df.Dp*e; return df
old=I.design
for lab,(a_,b_) in {'1113':(2011,2013),'1417':(2014,2017),'1924':(2019,2024)}.items():
    s=d[d.year.between(a_,b_)].copy(); s['Dpe']=s.Dp*s.eap; s['eapf']=s.eap
    X=['Dp','Dpe']+DC
    s=s.dropna(subset=['Dy']+X).reset_index(drop=True)
    r=c.twfe(s,'Dy',X,'cl',ent=False); e=c.lincomb(r,{'Dp':1,'Dpe':1})
    I.design=lambda df_,y_,X_,fe=('year',):old(df_,y_,X_,fe=('year',))
    wc=I.wcr(s,'Dy',X,{'Dpe':1},B=4999)[3]; ri=I.ri_label(s,'Dy',X,'eapf',mk_fd,'Dpe',B=1999)[2]
    I.design=old
    SP[lab]=dict(eu=row(r,['Dp'])['Dp'],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs),wcr_int=wc,ri_int=ri)
A['subperiod']=SP
RES['A']=A
json.dump(RES,open('results/results.json','w'),indent=1,default=float)
def pr(o,p=''):
    if isinstance(o,dict):
        if 'b' in o and 'se' in o: print(p,f"{o['b']:.3f} ({o['se']:.3f}) p={o['p']:.3f}"); return
        for k,v in o.items(): pr(v,p+'.'+str(k))
    else: print(p,o)
pr(A,'A')
