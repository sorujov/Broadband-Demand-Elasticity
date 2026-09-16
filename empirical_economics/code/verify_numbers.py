"""Audit: every number quoted in the manuscript prose must be traceable to results.json (within rounding)."""
import json, re
R=json.load(open('results/results.json'))
vals=[]
def walk(o):
    if isinstance(o,dict):
        for v in o.values(): walk(v)
    elif isinstance(o,list):
        for v in o: walk(v)
    elif isinstance(o,(int,float)): vals.append(float(o))
walk(R)
import pandas as pd
d=pd.read_pickle('panel.pkl')
# add descriptive numbers
d['per']=pd.cut(d.year,[2009,2013,2017,2024],labels=['a','b','c'])
for c in ['s100','fixed_broad_price','fixed_broad_price_ppp']:
    vals+=list(d.groupby(['region','per'])[c].mean().values)
for e in R['yearly']['static']+R['yearly']['fd']: vals+= [e['b'],e['se']]
tex=open('paper/manuscript.tex').read()
body=tex[tex.index('\\section{Introduction}'):tex.index('\\backmatter')]
body=re.sub(r'\\input\{[^}]*\}','',body)
nums=re.findall(r'(?<![\w.])(\$?[-+]?\$?\d+\.\d+)',body)
bad=[]
for n in set(nums):
    x=float(n.replace('$',''))
    dec=len(n.split('.')[1])
    ok=any(abs(abs(v)-abs(x))<=0.5*10**(-dec)+1e-9 for v in vals)
    if not ok: bad.append(n)
print('checked',len(set(nums)),'unmatched:',sorted(bad,key=lambda s:float(s.replace('$',''))))
