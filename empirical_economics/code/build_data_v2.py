"""Rebuild the analysis panel from primary sources (ITU price-basket workbook 2008-2025,
World Bank WDI/WGI API vintage July 2026, harmonised DMSP/VIIRS night lights).
No forward filling of prices or subscriptions. Interior gaps in controls are linearly
interpolated within country; edge gaps take the nearest observed value (flagged)."""
import pandas as pd, numpy as np, requests, common as c
EU="AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE".split()
ISO=EU+c.EAP; YRS=list(range(2010,2025))
x=pd.read_excel('raw/itu_prices.xlsx','economies_2008-2025')
x=x[x.IsoCode.isin(ISO)]
def chain(pieces):
    out=[]
    for code,(a,b) in pieces:
        for unit,suf in [('GNI','_GNI'),('PPP','_PPP'),('USD','$')]:
            s=x[x.Code==code+suf]
            for _,r in s.iterrows():
                for y in range(a,b+1):
                    out.append(dict(country=r.IsoCode,year=y,unit=unit,v=r[y]))
    o=pd.DataFrame(out).pivot_table(index=['country','year'],columns='unit',values='v').reset_index()
    return o
fb=chain([('i154_FBB',(2010,2017)),('i154_FBB5',(2018,2024))]).rename(columns={'GNI':'fixed_broad_price','PPP':'fixed_broad_price_ppp','USD':'fixed_broad_price_usd'})
mb=chain([('i271md_pd_B1GB',(2013,2017)),('i271mb_1GB5',(2018,2020)),('i271mb_2GB',(2021,2024))]).rename(columns={'GNI':'mobile_broad_price','PPP':'mobile_broad_price_ppp','USD':'mobile_broad_price_usd'})
P=pd.MultiIndex.from_product([ISO,YRS],names=['country','year']).to_frame(index=False)
d=P.merge(fb,how='left').merge(mb,how='left')
# ITU repeats 2018 values in 2019 for part of the sample: flag
d=d.sort_values(['country','year'])
prev=d.groupby('country')[['fixed_broad_price','fixed_broad_price_ppp','fixed_broad_price_usd']].shift(1)
d['carry']=(d.year==2019)&(d[['fixed_broad_price','fixed_broad_price_ppp','fixed_broad_price_usd']].values==prev.values).all(axis=1)
# World Bank
C=';'.join(ISO)
import os, time, json as _j
os.makedirs('raw/wbcache',exist_ok=True)
def wb(ind,src=''):
    fn=f'raw/wbcache/{ind}.json'
    if os.path.exists(fn):
        r=_j.load(open(fn))
        return pd.DataFrame([dict(country=e['countryiso3code'],year=int(e['date']),v=e['value']) for e in r[1]])
    for k in range(6):
        try:
            r=requests.get(f"https://api.worldbank.org/v2/country/{C}/indicator/{ind}?format=json&date=2010:2024&per_page=2000{src}",timeout=90).json(); break
        except Exception as e:
            time.sleep(3*(k+1))
    _j.dump(r,open(fn,'w'))
    return pd.DataFrame([dict(country=e['countryiso3code'],year=int(e['date']),v=e['value']) for e in r[1]])
W={'IT.NET.BBND.P2':'s100','IT.NET.BBND':'fixed_broadband_subs','IT.NET.USER.ZS':'internet_users_pct','SP.POP.TOTL':'population',
   'NY.GDP.PCAP.CD':'gdp_per_capita','SP.URB.TOTL.IN.ZS':'urban_population_pct','SE.TER.ENRR':'education_tertiary_pct',
   'SP.POP.1564.TO.ZS':'population_ages_15_64','NY.GDP.MKTP.KD.ZG':'gdp_growth','NY.GDP.DEFL.KD.ZG':'inflation_gdp_deflator',
   'EN.POP.DNST':'population_density','IT.NET.SECR.P6':'secure_internet_servers','GB.XPD.RSDV.GD.ZS':'research_development_expenditure'}
for ind,name in W.items():
    d=d.merge(wb(ind).rename(columns={'v':name}),how='left')
d=d.merge(wb('GOV_WGI_RQ.EST','&source=3').rename(columns={'v':'regulatory_quality_estimate'}),how='left')
# night lights
n=pd.read_csv('raw/ntl_country.csv')[['country','year','ntl_sum']]
d=d.merge(n,how='left')
d['population_density']=d['population_density'].astype(float)
# density = population / land area: extend with population where missing (area constant)
area=(d.population/d.population_density).groupby(d.country).transform('median')
d['population_density']=d.population_density.fillna(d.population/area)
CTRLS=['gdp_per_capita','urban_population_pct','education_tertiary_pct','population_ages_15_64','gdp_growth',
       'inflation_gdp_deflator','regulatory_quality_estimate','secure_internet_servers','research_development_expenditure']
imp={}
for v in CTRLS:
    miss=d[v].isna()
    d[v]=d.groupby('country')[v].transform(lambda s: s.interpolate(limit_area='inside'))
    inside=miss & d[v].notna()
    d[v]=d.groupby('country')[v].transform(lambda s: s.ffill().bfill())
    imp[v]=dict(interior=int(inside.sum()),edge=int((miss&~inside).sum()))
d['imputed_controls']=0
print('imputation counts',imp)
# derived
d['eap']=d.country.isin(c.EAP).astype(float); d['region']=np.where(d.eap==1,'EaP','EU')
d['y']=np.log(d.s100); d['y_count']=np.log(d.fixed_broadband_subs)
d['p']=np.log(d.fixed_broad_price); d['p_usd']=np.log(d.fixed_broad_price_usd); d['p_ppp']=np.log(d.fixed_broad_price_ppp)
d['log_fixed_broad_price']=d.p
d['log_mob_price']=np.log(d.mobile_broad_price); d['log_mobile_broad_price']=d.log_mob_price
d['log_gdp_per_capita']=np.log(d.gdp_per_capita); d['log_population_density']=np.log(d.population_density)
d['log_secure_internet_servers']=np.log(d.secure_internet_servers)
d['log_internet_users_pct']=np.log(d.internet_users_pct)
d['log_ntl_pc']=np.log(d.ntl_sum/d.population)
d['subreg']=d.country.map(c.SUBREG)
z1=[];z2=[]
for i,r in d.iterrows():
    same=d[(d.year==r.year)&(d.subreg==r.subreg)&(d.country!=r.country)]
    z1.append(same.p.mean())
    oth=d[(d.year==r.year)&(d.country!=r.country)&d.p.notna()].copy()
    oth['dist']=[c.hav(c.CAP[r.country],c.CAP[k]) for k in oth.country]
    nn=oth.nsmallest(3,'dist'); w_=1/nn.dist; z2.append((nn.p*w_).sum()/w_.sum())
d['z_sub']=z1; d['z_nn']=z2
g=d.groupby('country')
d['p_lag']=g.p.shift(1); d['y_lag']=g.y.shift(1); d['s_lag']=g.s100.shift(1)
d=d.reset_index(drop=True)
d.to_pickle('panel.pkl'); d.to_csv('results/analysis_panel_v2.csv',index=False)
print(d.shape, 'carry2019:',int(d.carry.sum()), 'missing s100:',int(d.s100.isna().sum()), 'missing p:',int(d.p.isna().sum()))
print(d[['s100','fixed_broad_price','mobile_broad_price','gdp_per_capita','regulatory_quality_estimate','ntl_sum']].describe().T[['count','mean','min','max']])
import json; json.dump(imp,open('results/imputation_counts.json','w'))
