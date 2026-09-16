import pandas as pd, numpy as np
bb=pd.read_csv('eurostat/broad_h.csv'); gd=pd.read_csv('eurostat/gdp_pps.csv')
bb['cc']=bb.geo.str[:2]; bb['lvl']=bb.geo.str.len()-2
ISO={'AT':'AUT','BE':'BEL','BG':'BGR','CY':'CYP','CZ':'CZE','DE':'DEU','DK':'DNK','EE':'EST','EL':'GRC','ES':'ESP','FI':'FIN','FR':'FRA','HR':'HRV','HU':'HUN','IE':'IRL','IT':'ITA','LT':'LTU','LU':'LUX','LV':'LVA','MT':'MLT','NL':'NLD','PL':'POL','PT':'PRT','RO':'ROU','SE':'SWE','SI':'SVN','SK':'SVK'}
bb=bb[bb.cc.isin(ISO)]
# choose finest level with >=2 regions and decent coverage per country
rows=[]
for cc,g in bb.groupby('cc'):
    for lvl in [2,1,0]:
        gl=g[g.lvl==lvl]
        if gl.geo.nunique()>=2 and len(gl)>=gl.geo.nunique()*6: rows.append(gl); break
    else:
        pass
R=pd.concat(rows)
# countries with only national data: skip (no within-country variation)
R=R.rename(columns={'value':'bb_hh','time':'year'})
gd=gd.rename(columns={'value':'gdp_pps','time':'year'})
R=R.merge(gd[['geo','year','gdp_pps']],on=['geo','year'],how='left')
R['country']=R.cc.map(ISO)
print(R.groupby('cc').agg(lvl=('lvl','first'),regions=('geo','nunique'),n=('bb_hh','size'),gdp_na=('gdp_pps',lambda s:s.isna().sum())))
R=R.dropna(subset=['gdp_pps','bb_hh'])
R.to_csv('eurostat/regional_panel.csv',index=False)
print(len(R), R.geo.nunique(), R.country.nunique(), R.year.min(), R.year.max())
