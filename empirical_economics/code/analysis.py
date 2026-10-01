"""Revision analysis for Applied Economics submission.
Produces results/results.json, tables/*.tex (main + supplement), figures/*.pdf
"""
import pandas as pd, numpy as np, json, warnings, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
import common as c, infer as I, dyn
from pathlib import Path
from scipy import stats
OUT=Path('results'); OUT.mkdir(exist_ok=True)
RES={}
B=9999; BRI=4999

CORE=['log_gdp_per_capita','urban_population_pct','education_tertiary_pct','regulatory_quality_estimate',
      'population_ages_15_64','gdp_growth','inflation_gdp_deflator','log_population_density']
d0=pd.read_pickle('panel.pkl')
d0['post']=(d0.year>=2020)*1.0
# ---- data flags
d0['carry']=d0['carry'].astype(bool)
RES['n_carry2019']=int(d0.carry.sum())
d=d0.copy()
for v in ['p','p_ppp','p_usd']:
    d.loc[d.carry,v]=np.nan
d['pxe']=d.p*d.eap; d['pxpost']=d.p*d.post; d['pxepost']=d.pxe*d.post
d['tr']=(d.year-2017)/10
T={}
for cc in sorted(d.country.unique()):
    T['tr_'+cc]=(d.country==cc)*d.tr
d=pd.concat([d,pd.DataFrame(T)],axis=1)
TR=sorted([x for x in d if x.startswith('tr_')])[1:]
d=d.sort_values(['country','year']).reset_index(drop=True)
g=d.groupby('country')
for v in ['y','p','p_ppp','p_usd','pxe','pxpost','pxepost','log_internet_users_pct','log_mob_price']+CORE+['log_secure_internet_servers','research_development_expenditure']:
    d['D'+v]=g[v].diff()
# basket break: drop the 2017->2018 change
for v in ['Dp','Dp_ppp','Dp_usd','Dpxe','Dpxpost','Dpxepost']:
    d.loc[d.year==2018,v]=np.nan
DC=['D'+x for x in CORE]
d['hl']=1-d.groupby('country').s100.shift(1)/60
d.to_pickle('panel_final.pkl')

def cl(df,y,X,ent=True,time=True):
    s=df.dropna(subset=[y]+X)
    return c.twfe(s,y,X,'cl',ent=ent,time=time), s
def row(r,names):
    return {n:dict(b=float(r.params[n]),se=float(r.std_errors[n]),p=float(r.pvalues[n])) for n in names}
def fd_design(df,y,X):
    return I.design(df,y,X,fe=('year',))
def wcr_fd(df,y,X,w,**k):
    old=I.design; I.design=lambda a,b,cc,fe=('year',):old(a,b,cc,fe=('year',))
    try: return I.wcr(df,y,X,w,**k)
    finally: I.design=old
def wcr_tw(df,y,X,w,**k):
    return I.wcr(df,y,X,w,**k)

pre=d[d.year<=2019]
# =====================================================================
# TABLE 2: pooled price coefficient, static -> diffusion-robust (2010-2019 and 2010-2024)
T2={}
for win,sub in [('pre',pre),('full',d)]:
    r,s=cl(sub,'y',['p']+CORE); T2[f'static_{win}']=row(r,['p'])|{'n':int(r.nobs),'wcr':I.wcr(s,'y',['p']+CORE,{'p':1},B=B)[3]}
    r,s=cl(sub,'y',['p']+CORE+TR); T2[f'trend_{win}']=row(r,['p'])|{'n':int(r.nobs),'wcr':I.wcr(s,'y',['p']+CORE+TR,{'p':1},B=B)[3]}
    r,s=cl(sub,'Dy',['Dp']+DC,ent=False); T2[f'fd_{win}']=row(r,['Dp'])|{'n':int(r.nobs),'wcr':wcr_fd(s,'Dy',['Dp']+DC,{'Dp':1},B=B)[3]}
    # distributed lag FD (cumulative 2-year)
    s=sub.copy(); s['LDp']=s.groupby('country').Dp.shift(1)
    r,s2=cl(s,'Dy',['Dp','LDp']+DC,ent=False)
    e=c.lincomb(r,{'Dp':1,'LDp':1}); T2[f'fdlag_{win}']={'cum':dict(b=e[0],se=e[1],p=e[2]),'n':int(r.nobs)}
    # dynamic LSDV (price assumed predetermined) on balanced original prices
    bal=d0[d0.year<=(2019 if win=='pre' else 2024)].copy(); bal.loc[bal.carry,'p']=np.nan; bal['y_lag']=bal.groupby('country').y.shift(1)
    r,_=cl(bal,'y',['y_lag','p']+CORE)
    lr=r.params.p/(1-r.params.y_lag)
    T2[f'lsdv_{win}']=row(r,['y_lag','p'])|{'n':int(r.nobs),'lr':float(lr)}
    # difference GMM, collapsed, lags 2-3, price endogenous
    gm=dyn.ab_gmm(bal,'y',['p'],['log_gdp_per_capita'],lags=(2,3))
    t=gm['tab']
    T2[f'gmm_{win}']=dict(rho=dict(b=t.b1.Ly,se=t.se1.Ly,p=t.p1.Ly),p=dict(b=t.b1.p,se=t.se1.p,p=t.p1.p),
        lr=list(dyn.longrun(gm,['p'])),J=gm['pJ'],m1=gm['pm1'],m2=gm['pm2'],ninst=gm['n_inst'],n=gm['nobs'])
RES['T2']=T2; print('T2 done')

# =====================================================================
# TABLE 3: EU vs EaP (2010-2019)
T3={}
def mk_lv(df,lab):
    e=df.country.isin(lab)*1.0; df['pxe']=df.p*e; return df
def mk_fd(df,lab):
    e=df.country.isin(lab)*1.0; df['Dpxe']=df.Dp*e; return df
pre=d[d.year<=2019].copy(); pre['eapf']=pre.eap
for lab,y,X,tw in [('static','y',['p','pxe']+CORE,True),('trend','y',['p','pxe']+CORE+TR,True),('fd','Dy',['Dp','Dpxe']+DC,False)]:
    r,s=cl(pre,y,X,ent=tw)
    b,i=X[0],X[1]
    eap=c.lincomb(r,{b:1,i:1})
    f = I.wcr if tw else wcr_fd
    out=row(r,[b,i])|{'eap':dict(b=eap[0],se=eap[1],p=eap[2]),'n':int(r.nobs)}
    out['wcr_eu']=f(s,y,X,{b:1},B=B)[3]; out['wcr_int']=f(s,y,X,{i:1},B=B)[3]; out['wcr_eap']=f(s,y,X,{b:1,i:1},B=B)[3]
    if tw:
        out['ri_int']=I.ri_label(s,y,X,'eapf',mk_lv if lab!='fd' else mk_fd,i,B=BRI)[2]
    else:
        old=I.design; I.design=lambda a,bb,cc,fe=('year',):old(a,bb,cc,fe=('year',))
        out['ri_int']=I.ri_label(s,y,X,'eapf',mk_fd,i,B=BRI)[2]; I.design=old
    # leave-one-EaP-out
    loo=[]
    for cc in c.EAP:
        s2=s[s.country!=cc]; X2=[x for x in X if s2[x].abs().sum()>0 and x!='tr_'+sorted(s2.country.unique())[0]]; r2=c.twfe(s2,y,X2,'cl',ent=tw); loo.append(c.lincomb(r2,{b:1,i:1})[0])
    out['loo']=[float(min(loo)),float(max(loo))]
    T3[lab]=out
# mean-group FD slopes (country-specific; heterogeneity-robust)
slopes=[]
for cc,s in pre.dropna(subset=['Dy','Dp']).groupby('country'):
    f=smf.ols('Dy~Dp+Dlog_gdp_per_capita',s).fit()
    slopes.append(dict(country=cc,eap=int(s.eap.iloc[0]),b=f.params.Dp,se=f.bse.Dp,n=len(s)))
sl=pd.DataFrame(slopes); sl.to_csv(OUT/'country_fd_slopes_pre.csv',index=False)
mg={}
for gname,gg in [('eu',sl[sl.eap==0]),('eap',sl[sl.eap==1])]:
    # trimmed of the 5% most extreme? report plain and median
    mg[gname]=dict(mean=float(gg.b.mean()),se=float(gg.b.std(ddof=1)/np.sqrt(len(gg))),median=float(gg.b.median()),n=len(gg))
mg['diff_t_p']=float(stats.ttest_ind(sl[sl.eap==1].b,sl[sl.eap==0].b,equal_var=False).pvalue)
mg['perm_p']=float(np.mean([abs(np.mean(x[:6])-np.mean(x[6:]))>=abs(mg['eap']['mean']-mg['eu']['mean']) for x in [np.random.default_rng(i).permutation(sl.b.values) for i in range(BRI)]]))
T3['mg']=mg
# TWFE implicit weights on country slopes (static, pre): share of residualised price variance
s=pre.dropna(subset=['y','p']+CORE)
w=c.demean2(d0[(d0.year<=2019)],['p','y']+CORE)
import numpy.linalg as la
Xc=w[CORE].values; pr=w.p.values - Xc@la.lstsq(Xc,w.p.values,rcond=None)[0]
wt=pd.Series(pr**2,index=w.country).groupby(level=0).sum(); wt/=wt.sum()
T3['twfe_weight_eap']=float(wt[c.EAP].sum()); wt.to_csv(OUT/'twfe_weights_pre.csv')
RES['T3']=T3; print('T3 done')

# =====================================================================
# TABLE 4: change after 2019 and basket regimes
T4={}
full=d.copy(); full['eapf']=full.eap
for lab,y,X,tw in [('static','y',['p','pxpost']+CORE,True),('trend','y',['p','pxpost']+CORE+TR,True),('fd','Dy',['Dp','Dpxpost']+DC,False)]:
    r,s=cl(full,y,X,ent=tw); f=I.wcr if tw else wcr_fd
    post=c.lincomb(r,{X[0]:1,X[1]:1})
    T4[lab]=row(r,X[:2])|{'post':dict(b=post[0],se=post[1],p=post[2]),'wcr_shift':f(s,y,X,{X[1]:1},B=B)[3],'n':int(r.nobs)}
for lab,y,X,tw in [('static_g','y',['p','pxe','pxpost','pxepost']+CORE,True),('trend_g','y',['p','pxe','pxpost','pxepost']+CORE+TR,True),('fd_g','Dy',['Dp','Dpxe','Dpxpost','Dpxepost']+DC,False)]:
    r,s=cl(full,y,X,ent=tw); f=I.wcr if tw else wcr_fd
    T4[lab]=row(r,X[:4])|{'eap_shift':dict(zip(['b','se','p'],c.lincomb(r,{X[2]:1,X[3]:1}))),'n':int(r.nobs),
        'wcr_eu_shift':f(s,y,X,{X[2]:1},B=B)[3],'wcr_eap_shift':f(s,y,X,{X[2]:1,X[3]:1},B=B)[3]}
# basket regimes
for lab,a,b_ in [('r1',2010,2017),('r2',2018,2024)]:
    s=d[(d.year>=a)&(d.year<=b_)]
    r,_=cl(s,'y',['p']+CORE); T4[f'static_{lab}']=row(r,['p'])|{'n':int(r.nobs)}
    s=s[s.year>a]
    r,_=cl(s,'Dy',['Dp']+DC,ent=False); T4[f'fd_{lab}']=row(r,['Dp'])|{'n':int(r.nobs)}
RES['T4']=T4; print('T4 done')

# year-specific slopes: static vs FD (Figure 1)
ys={}
X=['p']; dd=d.copy()
for yv in range(2011,2025): dd[f'px{yv}']=dd.p*(dd.year==yv); X.append(f'px{yv}')
r,_=cl(dd,'y',X+CORE)
ys['static']=[dict(zip(['year','b','se','p'],[yv,*c.lincomb(r,{'p':1,**({f'px{yv}':1} if yv>2010 else {})})])) for yv in range(2010,2025) if yv!=2019 or True]
dd=d.copy(); X=['Dp']; yrs=[yv for yv in range(2011,2025) if yv not in (2018,)]
base=yrs[0]
for yv in yrs[1:]: dd[f'Dpx{yv}']=dd.Dp*(dd.year==yv); X.append(f'Dpx{yv}')
r,_=cl(dd,'Dy',X+DC,ent=False)
ys['fd']=[dict(zip(['year','b','se','p'],[yv,*c.lincomb(r,{'Dp':1,**({f'Dpx{yv}':1} if yv>base else {})})])) for yv in yrs]
RES['yearly']=ys
# 5-year pooled FD by period for plotting robustness
print('yearly done')

# =====================================================================
# TABLE 5: NUTS-2
R=pd.read_csv('eurostat/regional_panel.csv'); R=R[~R.geo.str.startswith('FRY')]
R=R.merge(d0[['country','year','p','log_gdp_per_capita']],on=['country','year'],how='left')
R.loc[R.merge(d0[['country','year','carry']],on=['country','year'],how='left').carry.values==True,'p']=np.nan
R['y']=np.log(R.bb_hh); R['lg']=np.log(R.gdp_pps)
first=R.sort_values('year').groupby('geo').first()
R['ld']=R.geo.map(first.gdp_pps<75).astype(float)
R['pxld']=R.p*R.ld; R['post']=(R.year>=2020)*1.0; R['pxldpost']=R.pxld*R.post; R['pxpost']=R.p*R.post
R['cy']=R.country+R.year.astype(str)
R=R.dropna(subset=['y','p','lg'])
from linearmodels.panel import PanelOLS
def rfit(X,fe,sub):
    dd=sub.set_index(['geo','year'])
    m=PanelOLS(dd.y,dd[X],entity_effects=True,other_effects=dd[['cy']]) if fe=='cy' else PanelOLS(dd.y,dd[X],entity_effects=True,time_effects=True)
    return m.fit(cov_type='clustered',clusters=dd[['country']].astype('category').apply(lambda s:s.cat.codes))
T5={}
R2=R.rename(columns={'geo':'country_','country':'nat'}).assign(country=lambda x:x.nat)  # for WCR clustering by nation
for lab,X,fe,sub in [('tw_all',['p','lg','log_gdp_per_capita'],'tw',R),('tw_ld',['p','pxld','lg','log_gdp_per_capita'],'tw',R),
                     ('cy_ld',['pxld','lg'],'cy',R),('cy_ld_post',['pxld','pxldpost','lg'],'cy',R),
                     ('tw_ld_1017',['p','pxld','lg','log_gdp_per_capita'],'tw',R[R.year<=2017])]:
    r=rfit(X,fe,sub); T5[lab]=row(r,[x for x in X if x.startswith('p')])|{'n':int(r.nobs),'regions':int(sub.geo.nunique()),'countries':int(sub.country.nunique())}
    # WCR by country with region FE + year (or country-year) FE
    s=sub.copy(); fes=('geo','year') if fe=='tw' else ('geo','cy')
    old=I.design; I.design=lambda a,bb,cc,fe=fes:old(a,bb,cc,fe=fes)
    key='pxld' if 'pxld' in X else 'p'
    T5[lab]['wcr_'+key]=I.wcr(s,'y',X,{key:1},B=1999)[3]; I.design=old
T5['n_ld_regions']=int(R[R.ld==1].geo.nunique())
RES['T5']=T5; print('T5 done')

# =====================================================================
# SUPPLEMENT: robustness grid for FD pooled & EaP (pre) and static
S={}
def fd_pair(df,price='Dp',ctrl=DC,y='Dy'):
    df=df.copy(); df['Dpe_']=df[price]*df.eap
    r,_=cl(df,y,[price,'Dpe_']+ctrl,ent=False); e=c.lincomb(r,{price:1,'Dpe_':1})
    r2,_=cl(df,y,[price]+ctrl,ent=False)
    return dict(pooled=row(r2,[price])[price],eu=row(r,[price])[price],eap=dict(b=e[0],se=e[1],p=e[2]),n=int(r.nobs))
pre=d[d.year<=2019]
S['base']=fd_pair(pre)
S['ppp']=fd_pair(pre,'Dp_ppp'); S['usd']=fd_pair(pre,'Dp_usd')
S['fullctrl']=fd_pair(pre,ctrl=DC+['Dlog_secure_internet_servers','Dresearch_development_expenditure'])
S['gdponly']=fd_pair(pre,ctrl=['Dlog_gdp_per_capita'])
S['mobprice']=fd_pair(pre,ctrl=DC+['Dlog_mob_price'])
S['users']=fd_pair(pre,y='Dlog_internet_users_pct')
dc=d0.copy().sort_values(['country','year'])
for v in ['y','p','y_count']+CORE: dc['D'+v]=dc.groupby('country')[v].diff()
S['with_carry']=fd_pair(dc[dc.year<=2019])
S['count_dv']=fd_pair(dc[dc.year<=2019],y='Dy_count')
S['with_2018']=fd_pair(dc[dc.year<=2019])  # same as with_carry; includes 2018 change
S['end2017']=fd_pair(pre[pre.year<=2017])
# DK SEs
r=c.twfe(pre.dropna(subset=['Dy','Dp','Dpxe']+DC),'Dy',['Dp','Dpxe']+DC,'dk',ent=False)
S['dk']=dict(eu=row(r,['Dp'])['Dp'],eap=dict(zip(['b','se','p'],c.lincomb(r,{'Dp':1,'Dpxe':1}))))
# full sample exclusions
fs=d.copy()
fs2=fs[~((fs.country=='UKR')&(fs.year>=2022))&~((fs.country=='BLR')&(fs.year>=2020))]
for lab,sub in [('full_base',fs),('full_nowar',fs2),('full_end2021',fs[fs.year<=2021])]:
    r,_=cl(sub,'Dy',['Dp','Dpxpost']+DC,ent=False); S[lab]=row(r,['Dp','Dpxpost'])|{'n':int(r.nobs)}
    r,_=cl(sub,'y',['p','pxpost']+CORE,ent=True); S[lab+'_static']=row(r,['p','pxpost'])
# functional forms (static, pre): level-level and log-level, elasticities at means by group
for lab,form in [('linlin','s100 ~ fixed_broad_price'),('loglin','y ~ fixed_broad_price'),('linlog','s100 ~ p')]:
    pass
ff={}
pp=pre.dropna(subset=['p']).copy()
pp['pr']=pp.fixed_broad_price; pp['prxe']=pp.pr*pp.eap; pp['Ds']=pp.groupby('country').s100.diff(); pp['Dpr']=pp.groupby('country').pr.diff()
pp.loc[pp.year==2018,'Dpr']=np.nan; pp['Dprxe']=pp.Dpr*pp.eap; pp['Dpxe2']=pp.Dp*pp.eap; pp['Dy2']=pp.Dy
for lab,y,X in [('linlin','Ds',['Dpr','Dprxe']),('loglin','Dy2',['Dpr','Dprxe']),('linlog','Ds',['Dp','Dpxe2'])]:
    r,s=cl(pp,y,X+DC,ent=False)
    m_eu=s[s.eap==0]; m_ea=s[s.eap==1]
    def el(bb,m):
        P=m.pr.mean(); Q=m.s100.mean()
        return {'linlin':bb*P/Q,'loglin':bb*P,'linlog':bb/Q}[lab]
    beu=r.params[X[0]]; bea=beu+r.params[X[1]]
    ff[lab]=dict(eu=float(el(beu,m_eu)),eap=float(el(bea,m_ea)),p_eu=float(r.pvalues[X[0]]),p_eap=float(c.lincomb(r,{X[0]:1,X[1]:1})[2]))
S['funcform']=ff
# headroom interaction in FD (saturation)
h=d.copy(); h['hc']=h.hl-h.hl.mean(); h['Dpxh']=h.Dp*h.hc
r,_=cl(h,'Dy',['Dp','Dpxh','Dpxpost']+DC,ent=False); S['fd_headroom']=row(r,['Dp','Dpxh','Dpxpost'])
r,_=cl(h[h.year<=2019],'Dy',['Dp','Dpxh']+DC,ent=False); S['fd_headroom_pre']=row(r,['Dp','Dpxh'])
h['Dpxhe']=h.Dpxh*0; 
r,_=cl(h[h.year<=2019].assign(Dpxe=lambda x:x.Dp*x.eap),'Dy',['Dp','Dpxe','Dpxh']+DC,ent=False); S['fd_headroom_eap_pre']=row(r,['Dp','Dpxe','Dpxh'])
# logistic diffusion (log-odds) in FD, K grid
for K in [55,70,100]:
    h['lo']=np.log(h.s100/(K-h.s100)); h['Dlo']=h.groupby('country').lo.diff()
    hp=h[h.year<=2019]
    S[f'logodds_K{K}']=fd_pair(hp,y='Dlo')
# Hausman-type IV (static, pre, pooled) first-stage and AR set
from linearmodels.iv import IV2SLS
bal=d0[d0.year<=2019].copy()
w=c.demean2(bal,['y','p','z_sub','z_nn']+CORE).reset_index(drop=True)
clu=w.country.astype('category').cat.codes
ivres={}
for z in ['z_sub','z_nn']:
    r=IV2SLS(w.y,w[CORE],w[['p']],w[[z]]).fit(cov_type='clustered',clusters=clu)
    # Anderson-Rubin grid
    grid=np.linspace(-6,4,501); acc=[]
    for b0 in grid:
        yy=w.y-b0*w.p
        import statsmodels.api as sm
        f=sm.OLS(yy,sm.add_constant(pd.concat([w[[z]],w[CORE]],axis=1))).fit(cov_type='cluster',cov_kwds={'groups':clu})
        if f.pvalues[z]>0.05: acc.append(b0)
    ivres[z]=dict(b=float(r.params.p),se=float(r.std_errors.p),F=float(r.first_stage.diagnostics['f.stat'].iloc[0]),
                  ar_lo=float(min(acc)) if acc else None, ar_hi=float(max(acc)) if acc else None,
                  ar_bounded=bool(acc and min(acc)>grid[0] and max(acc)<grid[-1]), ar_empty=not acc)
S['iv']=ivres
S['twfe_weight_eap']=RES['T3']['twfe_weight_eap']
RES['S']=S
json.dump(RES,open(OUT/'results.json','w'),indent=1,default=float)
print('ALL DONE')
