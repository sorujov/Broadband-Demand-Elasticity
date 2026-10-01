"""Monte Carlo: year-specific TWFE price slopes under catch-up diffusion with zero price effect."""
import numpy as np, pandas as pd, json
from linearmodels.panel import PanelOLS
rng=np.random.default_rng(7)
def one(N=33,T=15,late=6):
    rows=[]
    mid=np.r_[rng.normal(-4,2,N-late), rng.normal(3,1.5,late)]   # diffusion midpoint relative to 2010 (years)
    K=rng.uniform(35,50,N); speed=rng.uniform(0.25,0.45,N)
    p0=np.r_[rng.normal(0,0.3,N-late), rng.normal(1.3,0.3,late)]  # late adopters start with high affordability ratio
    for i in range(N):
        for t in range(T):
            stage=1/(1+np.exp(-speed[i]*(t-mid[i])))
            s=K[i]*stage*np.exp(rng.normal(0,0.03))
            lp=p0[i]-1.2*(stage-1/(1+np.exp(speed[i]*mid[i])))*(1 if i>=N-late else 0.3)+rng.normal(0,0.15)
            rows.append(dict(c=i,t=t,y=np.log(s),p=lp))
    d=pd.DataFrame(rows)
    X=['p']
    for t in range(1,T):
        d[f'p{t}']=d.p*(d.t==t); X.append(f'p{t}')
    dd=d.set_index(['c','t'])
    r=PanelOLS(dd.y,dd[X],entity_effects=True,time_effects=True).fit()
    lev=[r.params.p]+[r.params.p+r.params[f'p{t}'] for t in range(1,T)]
    d=d.sort_values(['c','t']); d['Dy']=d.groupby('c').y.diff(); d['Dp']=d.groupby('c').p.diff()
    e=d.dropna(); X=['Dp']
    for t in range(2,T):
        e[f'q{t}']=e.Dp*(e.t==t); X.append(f'q{t}')
    ee=e.set_index(['c','t'])
    r2=PanelOLS(ee.Dy,ee[X],time_effects=True).fit()
    fd=[r2.params.Dp]+[r2.params.Dp+r2.params[f'q{t}'] for t in range(2,T)]
    rp=PanelOLS(dd.y,dd[['p']],entity_effects=True,time_effects=True).fit().params.p
    rf=PanelOLS(ee.Dy,ee[['Dp']],time_effects=True).fit().params.Dp
    return lev,fd,rp,rf
L=[];F=[];P=[];Q=[]
for _ in range(200):
    a,b,cc,dd_=one(); L.append(a);F.append(b);P.append(cc);Q.append(dd_)
L=np.array(L);F=np.array(F)
res=dict(levels_mean=L.mean(0).tolist(),fd_mean=F.mean(0).tolist(),pooled_levels=float(np.mean(P)),pooled_fd=float(np.mean(Q)),
         share_increasing=float(np.mean(L[:,-1]>L[:,0])))
json.dump(res,open('results/simulation.json','w'),indent=1)
print({k:(np.round(v,2) if isinstance(v,list) else round(v,3)) for k,v in res.items()})
