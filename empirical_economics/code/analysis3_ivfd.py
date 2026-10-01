"""First-difference Hausman-type IV (Online Resource). Run after analysis.py and analysis2.py."""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from linearmodels.iv import IV2SLS
d=pd.read_pickle('panel_final.pkl')
CORE=['log_gdp_per_capita','urban_population_pct','education_tertiary_pct','regulatory_quality_estimate','population_ages_15_64','gdp_growth','inflation_gdp_deflator','log_population_density']
d['Dz']=d.groupby('country').z_sub.diff(); d['Dzn']=d.groupby('country').z_nn.diff()
d.loc[d.year==2018,['Dz','Dzn']]=np.nan
out={}
for win in [2019,2024]:
  s=d[d.year<=win].dropna(subset=['Dy','Dp','Dz','Dzn']+['D'+x for x in CORE]).copy()
  yd=pd.get_dummies(s.year,drop_first=True,prefix='y').astype(float)
  ex=pd.concat([s[['D'+x for x in CORE]],yd],axis=1); ex['const']=1
  for z in [['Dz'],['Dzn'],['Dz','Dzn']]:
    r=IV2SLS(s.Dy,ex,s[['Dp']],s[z]).fit(cov_type='clustered',clusters=s.country.astype('category').cat.codes)
    out[f'{win}_{"+".join(z)}']=dict(b=float(r.params.Dp),se=float(r.std_errors.Dp),F=float(r.first_stage.diagnostics['f.stat'].iloc[0]),n=int(r.nobs))
R=json.load(open('results/results.json')); R['S']['iv_fd']=out; json.dump(R,open('results/results.json','w'),indent=1,default=float)
print(out)
