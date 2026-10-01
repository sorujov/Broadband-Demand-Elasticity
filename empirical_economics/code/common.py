import pandas as pd, numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

EAP = ['ARM','AZE','BLR','GEO','MDA','UKR']
SUBREG = {'DNK':'N','FIN':'N','SWE':'N','EST':'B','LVA':'B','LTU':'B',
 'AUT':'W','BEL':'W','DEU':'W','FRA':'W','IRL':'W','LUX':'W','NLD':'W',
 'CYP':'S','ESP':'S','GRC':'S','ITA':'S','MLT':'S','PRT':'S',
 'CZE':'C','HUN':'C','POL':'C','SVK':'C','BGR':'SE','ROU':'SE','HRV':'SE','SVN':'SE',
 'BLR':'EE','MDA':'EE','UKR':'EE','ARM':'SC','AZE':'SC','GEO':'SC'}
CAP = {'AUT':(48.21,16.37),'BEL':(50.85,4.35),'BGR':(42.70,23.32),'HRV':(45.81,15.98),'CYP':(35.17,33.36),
 'CZE':(50.08,14.44),'DNK':(55.68,12.57),'EST':(59.44,24.75),'FIN':(60.17,24.94),'FRA':(48.86,2.35),
 'DEU':(52.52,13.40),'GRC':(37.98,23.73),'HUN':(47.50,19.04),'IRL':(53.35,-6.26),'ITA':(41.90,12.50),
 'LVA':(56.95,24.11),'LTU':(54.69,25.28),'LUX':(49.61,6.13),'MLT':(35.90,14.51),'NLD':(52.37,4.90),
 'POL':(52.23,21.01),'PRT':(38.72,-9.14),'ROU':(44.43,26.10),'SVK':(48.15,17.11),'SVN':(46.06,14.51),
 'ESP':(40.42,-3.70),'SWE':(59.33,18.07),'ARM':(40.18,44.51),'AZE':(40.41,49.87),'BLR':(53.90,27.57),
 'GEO':(41.72,44.78),'MDA':(47.01,28.86),'UKR':(50.45,30.52)}
CTRL = ['log_gdp_per_capita','urban_population_pct','education_tertiary_pct','regulatory_quality_estimate',
        'log_secure_internet_servers','research_development_expenditure','population_ages_15_64',
        'gdp_growth','inflation_gdp_deflator','log_population_density']
CTRL_LABELS = {'log_gdp_per_capita':'Log GDP per capita','urban_population_pct':'Urban population (%)',
 'education_tertiary_pct':'Tertiary enrolment (%)','regulatory_quality_estimate':'Regulatory quality (WGI)',
 'log_secure_internet_servers':'Log secure servers per million','research_development_expenditure':'R\\&D (% GDP)',
 'population_ages_15_64':'Working-age share (%)','gdp_growth':'GDP growth (%)','inflation_gdp_deflator':'Inflation (%)',
 'log_population_density':'Log population density'}

def hav(a,b):
    R=6371; la1,lo1=np.radians(a); la2,lo2=np.radians(b)
    h=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(h))

def load():
    d = pd.read_csv('repo/data/processed/analysis_ready_data.csv')
    d['eap'] = d.country.isin(EAP).astype(float)
    d['p'] = d.log_fixed_broad_price
    d['p_usd'] = d.log_fixed_broad_price_usd
    d['p_ppp'] = d.log_fixed_broad_price_ppp
    d['s100'] = d.fixed_broadband_subs_alt             # ITU per-100 indicator
    d['y'] = np.log(d.s100)                            # primary DV (per 100 inhabitants)
    d['y_count'] = d.log_fixed_broadband_subs          # original submission DV (log count)
    d['covid'] = (d.year>=2020).astype(float)
    d['post18'] = (d.year>=2018).astype(float)
    d['subreg'] = d.country.map(SUBREG)
    d['log_mob_price'] = d.log_mobile_broad_price
    # Hausman-type instruments
    d = d.sort_values(['country','year']).reset_index(drop=True)
    z1=[]; z2=[]
    for i,r in d.iterrows():
        same = d[(d.year==r.year)&(d.subreg==r.subreg)&(d.country!=r.country)]
        z1.append(same.p.mean())
        oth = d[(d.year==r.year)&(d.country!=r.country)].copy()
        oth['dist'] = [hav(CAP[r.country],CAP[c]) for c in oth.country]
        nn = oth.nsmallest(3,'dist')
        w = 1/nn.dist; z2.append((nn.p*w).sum()/w.sum())
    d['z_sub']=z1; d['z_nn']=z2
    d['p_lag']=d.groupby('country').p.shift(1)
    d['y_lag']=d.groupby('country').y.shift(1)
    d['s_lag']=d.groupby('country').s100.shift(1)
    return d

def twfe(df, y, X, cov='dk', ent=True, time=True):
    dd = df.set_index(['country','year'])
    m = PanelOLS(dd[y], dd[X], entity_effects=ent, time_effects=time)
    if cov=='dk': return m.fit(cov_type='kernel', kernel='bartlett', bandwidth=3)
    if cov=='cl': return m.fit(cov_type='clustered', cluster_entity=True)
    if cov=='rob': return m.fit(cov_type='robust')
    raise ValueError

def lincomb(res, w):
    """w: dict name->weight. returns est, se, p (normal)"""
    from scipy import stats
    names = list(w); a = np.array([w[n] for n in names])
    b = res.params[names].values; V = res.cov.loc[names,names].values
    est = a@b; se = np.sqrt(a@V@a)
    df = getattr(res,'df_resid',None)
    p = 2*(1-stats.norm.cdf(abs(est/se)))
    return est, se, p

def demean2(df, cols, tol=1e-10, maxit=500):
    """two-way within transform by alternating projections (valid for unbalanced panels).
    Rows with any missing value in cols are dropped first."""
    df = df.dropna(subset=cols).copy()
    out = df[['country','year']].copy()
    for c in cols:
        x = df[c].astype(float).copy()
        for _ in range(maxit):
            x0 = x.copy()
            x = x - x.groupby(df['country']).transform('mean')
            x = x - x.groupby(df['year']).transform('mean')
            if np.max(np.abs(x-x0)) < tol: break
        out[c] = x
    return out

def stars(p):
    return '^{***}' if p<0.01 else '^{**}' if p<0.05 else '^{*}' if p<0.10 else ''
def fmt(b,se=None,p=None,dec=2):
    s = f"${b:.{dec}f}{stars(p) if p is not None else ''}$"
    return s
def fse(se,dec=2): return f"({se:.{dec}f})"
