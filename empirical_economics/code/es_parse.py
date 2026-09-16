import json, itertools, pandas as pd
def parse(ds):
    j=json.load(open(f'eurostat/{ds}.json'))
    ids=j['id']; size=j['size']
    cats=[sorted(j['dimension'][k]['category']['index'].items(), key=lambda x:x[1]) for k in ids]
    rows=[]
    for k,v in j['value'].items():
        k=int(k); idx=[]
        for s in reversed(size):
            idx.append(k%s); k//=s
        idx=idx[::-1]
        rows.append({ids[i]:cats[i][idx[i]][0] for i in range(len(ids))}|{'value':v})
    return pd.DataFrame(rows)
if __name__=="__main__":
  pass
bb=parse('isoc_r_broad_h'); bb=bb[bb.unit=='PC_HH']
gd=parse('nama_10r_2gdp'); gd=gd[gd.unit=='PPS_EU27_2020_HAB']
bb.to_csv('eurostat/broad_h.csv',index=False); gd.to_csv('eurostat/gdp_pps.csv',index=False)
bb['lvl']=bb.geo.str.len()-2
bb['cc']=bb.geo.str[:2]
print(bb[bb.lvl==2].groupby('cc').agg(n=('value','size'),regions=('geo','nunique'),y0=('time','min'),y1=('time','max')))
print(bb[bb.lvl==1].groupby('cc').agg(n=('value','size'),regions=('geo','nunique')))
print(bb.time.unique())
