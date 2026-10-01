import json,sys
R=json.load(open('results/results.json'))
def pr(o,p=''):
    if isinstance(o,dict):
        if 'b' in o and 'se' in o: print(p,f"{o['b']:.3f} ({o['se']:.3f}) p={o['p']:.3f}"); return
        for k,v in o.items(): pr(v,p+'.'+str(k))
    elif isinstance(o,list):
        if all(not isinstance(i,dict) for i in o): print(p,[round(i,3) if isinstance(i,float) else i for i in o])
        else: print(p,'list')
    else: print(p, round(o,4) if isinstance(o,float) else o)
for k in sys.argv[1:]: pr(R[k],k)
