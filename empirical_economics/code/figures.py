import json, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
  'axes.edgecolor':'#555','axes.linewidth':0.6,'xtick.color':'#333','ytick.color':'#333','axes.labelcolor':'#222',
  'grid.color':'#e3e3e3','grid.linewidth':0.5})
BLUE,ORANGE,INK='#2a78d6','#eb6834','#222'
R=json.load(open('results/results.json'))
# Figure 1: year-specific slopes, levels vs first differences
fig,ax=plt.subplots(1,2,figsize=(7.0,2.8),sharey=True)
for a,key,title,col,mk in [(ax[0],'static','(a) Levels, two-way fixed effects',BLUE,'o'),(ax[1],'fd','(b) First differences, year effects',ORANGE,'s')]:
    e=pd.DataFrame(R['yearly'][key])
    a.axhline(0,color='#888',lw=0.7)
    a.axvspan(2017.5,2024.6,color='#f2f2f2',zorder=0,lw=0)
    a.errorbar(e.year,e.b,yerr=1.96*e.se,fmt=mk,color=col,ms=4.5,lw=1.2,capsize=0,elinewidth=1.2,mec='white',mew=0.6)
    a.plot(e.year,e.b,color=col,lw=1,alpha=.6)
    a.set_title(title,fontsize=9,loc='left',color=INK)
    a.set_xticks(range(2010,2025,2)); a.grid(axis='y')
    a.text(2021,-0.8,'5 GB basket (2018+)',ha='center',fontsize=7.5,color='#555')
ax[0].set_ylabel('Coefficient on log price (%GNI)')
ax[0].set_ylim(-0.85,0.5)
fig.tight_layout(); fig.savefig('figs/fig1_yearly_slopes.pdf'); fig.savefig('figs/fig1_yearly_slopes.png',dpi=300)
# Figure 2: country-specific FD slopes 2010-2019
sl=pd.read_csv('results/country_fd_slopes_pre.csv').sort_values('b')
fig,a=plt.subplots(figsize=(3.6,5.6))
y=np.arange(len(sl))
for i,(_,r) in enumerate(sl.iterrows()):
    col=ORANGE if r.eap else BLUE; mk='s' if r.eap else 'o'
    a.plot([r.b-1.96*r.se,r.b+1.96*r.se],[i,i],color=col,lw=1.1,alpha=.8)
    a.plot(r.b,i,mk,color=col,ms=4.5,mec='white',mew=.6)
a.axvline(0,color='#888',lw=.7)
mg=R['A']['mg_robust']
a.axvline(mg['eap']['inv_var_mean'],color=ORANGE,lw=.8,ls='--'); a.axvline(mg['eu']['inv_var_mean'],color=BLUE,lw=.8,ls='--')
a.set_yticks(y); a.set_yticklabels(sl.country,fontsize=7)
for t,(_,r) in zip(a.get_yticklabels(),sl.iterrows()):
    if r.eap: t.set_fontweight('bold')
a.set_xlim(-1.6,1.6); a.set_xlabel('Country-specific first-difference coefficient')
a.plot([],[],'o',color=BLUE,label='EU'); a.plot([],[],'s',color=ORANGE,label='EaP (bold labels)')
a.legend(frameon=False,fontsize=7.5,loc='lower right'); a.grid(axis='x')
fig.tight_layout(); fig.savefig('figs/fig2_country_slopes.pdf'); fig.savefig('figs/fig2_country_slopes.png',dpi=300)
print('ok')
# Figure S1: simulation
sim=json.load(open('results/simulation.json'))
fig,a=plt.subplots(figsize=(4.6,2.8))
yrs=list(range(2010,2025))
a.axhline(0,color='#888',lw=.7)
a.plot(yrs,sim['levels_mean'],'-o',color=BLUE,ms=4,lw=1.2,mec='white',mew=.6,label='Levels, two-way fixed effects')
a.plot(yrs[1:],sim['fd_mean'],'-s',color=ORANGE,ms=4,lw=1.2,mec='white',mew=.6,label='First differences')
a.set_ylabel('Mean estimated coefficient'); a.set_xticks(range(2010,2025,2)); a.grid(axis='y')
a.set_title('True price effect = 0 (200 simulated panels)',fontsize=9,loc='left')
a.legend(frameon=False,fontsize=7.5,loc='lower right')
fig.tight_layout(); fig.savefig('figs/figS1_simulation.pdf'); fig.savefig('figs/figS1_simulation.png',dpi=200)
