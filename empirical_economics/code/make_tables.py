import json, pandas as pd, numpy as np
R=json.load(open('results/results.json')); T='paper/tables/'
def st(p): return '^{***}' if p<0.01 else '^{**}' if p<0.05 else '^{*}' if p<0.10 else ''
def b(x,dec=2): return f"${x['b']:.{dec}f}{st(x['p'])}$"
def se(x,dec=2): return f"({x['se']:.{dec}f})"
def pv(p): return '$<$0.001' if p<0.001 else f"{p:.3f}"
def num(x,dec=2): return f"${x:.{dec}f}$"
def write(name,s): open(T+name,'w').write(s)

# ---------- Table 1: literature ----------
lit=r"""\begin{table}[t]
\caption{Selected estimates of broadband demand responses to price}\label{tab:lit}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{2.6cm}p{2.7cm}p{2.6cm}p{4.2cm}@{}}
\toprule
Study & Setting and period & Data and method & Headline estimate \\
\midrule
\citet{rosston2010household} & United States, 2009--2010 & Household choice experiment & WTP about US\$20 per month for reliability and US\$45--48 for faster service \\
\citet{cardona2009demand} & Austria, 2006 & Household survey, nested logit & DSL own-price elasticity from $-0.97$ to $-2.55$, depending on platform competition \\
\citet{galperin2013price} & Latin America and the Caribbean, 2010 & Country cross-section, IV & Elastic in Latin America (10\% price cut raises penetration by about 22\%); inelastic in the OECD comparison \\
\citet{grzybowski2014market} & Slovakia, 2010s & 6{,}446 households, mixed logit & DSL own-price elasticity about $-3.0$; all fixed broadband about $-2.0$ \\
\citet{dauvin2014estimating} & EU-27, 2006--2010 & NUTS-1 regional panel & Diffusion depends on competition and unbundling prices; no single elasticity \\
\citet{liu2018distinguishing} & United States, 2015--2016 & Discrete-choice survey & WTP concave in speed; about US\$8.7 per month for wired-level latency \\
\citet{mendez2021lifeline} & United States, Lifeline programme & Structural model of subsidised adoption & US\$9.25 subsidy raises low-income adoption by about 6\% \\
\citet{lindlacher2021low} & Germany & Household survey, mixed logit & Low-speed users rarely switch to high-speed plans where available \\
\citet{sinclair2023assessing} & Australia (NBN), 2022 & Choice experiment & WTP for download speed grew by about 9\% per year \\
\citet{wilson2025public} & United States, municipal fibre & Structural entry and investment model & Public entry crowds out private fibre investment \\
This article & EU-27 and EaP-6, 2010--2024 & Country panel; levels, trends, first and long differences & EU: about zero after removing diffusion; EaP: $-0.17$ to $-0.46$, driven by bloc-wide co-movement, not identified within the group \\
\botrule
\end{tabular*}
\footnotetext{WTP: willingness to pay. Estimates as reported by the authors; they refer to different margins (adoption, technology or tier choice) and are not directly comparable.}
\end{table}
"""
write('tab_lit.tex',lit)

# ---------- Table 2: descriptives ----------
d=pd.read_pickle('panel.pkl').sort_values(['country','year'])
d.loc[d.carry.astype(bool),'p']=np.nan
d['per']=pd.cut(d.year,[2009,2013,2017,2024],labels=['2010--13','2014--17','2018--24'])
d['dy']=d.groupby('country').y.diff(); d['dp']=d.groupby('country').p.diff(); d.loc[d.year==2018,'dp']=np.nan
rows=[('Subscriptions per 100','s100',1),('Price, \\% of GNI p.c.','fixed_broad_price',2),
      ('Price, PPP\\$','fixed_broad_price_ppp',1),('GDP p.c., US\\$ thousand','gdpk',1),
      ('Internet users, \\%','internet_users_pct',1),('$\\Delta$ log subscriptions','dy',3),('$\\Delta$ log price','dp',3)]
d['gdpk']=d.gdp_per_capita/1000
s=r"""\begin{table}[t]
\caption{Descriptive statistics: period means by country group}\label{tab:desc}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{EU-27} & \multicolumn{3}{c}{Eastern Partnership} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
& 2010--13 & 2014--17 & 2018--24 & 2010--13 & 2014--17 & 2018--24 \\
\midrule
"""
for lab,v,dec in rows:
    vals=[d[(d.region==g)&(d.per==p)][v].mean() for g in ['EU','EaP'] for p in ['2010--13','2014--17','2018--24']]
    s+=lab+' & '+' & '.join(f"${x:.{dec}f}$" for x in vals)+r' \\'+'\n'
nobs=[int(d[(d.region==g)&(d.per==p)].s100.notna().sum()) for g in ['EU','EaP'] for p in ['2010--13','2014--17','2018--24']]
s+=r'\midrule'+'\nCountry-years & '+' & '.join(map(str,nobs))+r' \\'+'\n'
s+=r"""\botrule
\end{tabular*}
\footnotetext{Sources: ITU ICT Price Baskets 2008--2025; ITU subscriptions and Internet use via World Bank WDI (July 2026 vintage). The price basket is the cheapest plan of the largest operator with at least 1\,GB (2008--2017) or 5\,GB (from 2018) of monthly data at 256\,kbit/s or more. Changes in log price exclude the 2017--2018 change, which mixes price movements with the basket revision, and the repeated 2019 values. EaP: Armenia, Azerbaijan, Belarus, Georgia, Moldova, Ukraine.}
\end{table}
"""
write('tab_desc.tex',s)

# ---------- Table 3: pooled static vs within-differenced ----------
T2=R['T2']
def cell(k,var): return T2[k][var]
s=r"""\begin{table}[t]
\caption{Pooled price coefficient: levels versus within-differenced and dynamic estimators}\label{tab:pooled}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
& (1) & (2) & (3) & (4) & (5) & (6) \\
& \shortstack{TWFE\\levels} & \shortstack{+ country\\trends} & \shortstack{First\\differences} & \shortstack{FD, 2-year\\cumulative} & \shortstack{Dynamic\\LSDV} & \shortstack{Difference\\GMM} \\
\midrule
"""
for win,lab in [('pre','A. 2010--2019'),('full','B. 2010--2024')]:
    s+=r'\multicolumn{7}{@{}l}{\textit{Panel '+lab+r'}}\\'+'\n'
    c1,c2,c3=T2[f'static_{win}']['p'],T2[f'trend_{win}']['p'],T2[f'fd_{win}']['Dp']
    c4=T2[f'fdlag_{win}']['cum']; c5=T2[f'lsdv_{win}']['p']; c6=T2[f'gmm_{win}']['p']
    s+='Log price & '+' & '.join(b(x) for x in [c1,c2,c3,c4,c5,c6])+r'\\'+'\n'
    s+=' & '+' & '.join(se(x) for x in [c1,c2,c3,c4,c5,c6])+r'\\'+'\n'
    s+='WCR bootstrap $p$ & '+' & '.join([pv(T2[f'static_{win}']['wcr']),pv(T2[f'trend_{win}']['wcr']),pv(T2[f'fd_{win}']['wcr']),'--','--','--'])+r'\\'+'\n'
    s+='Lagged dependent variable & -- & -- & -- & -- & '+b(T2[f'lsdv_{win}']['y_lag'])+' & '+b(T2[f'gmm_{win}']['rho'])+r'\\'+'\n'
    g=T2[f'gmm_{win}']
    s+=f"Long-run effect & -- & -- & -- & -- & {num(T2[f'lsdv_{win}']['lr'])} & {num(g['lr'][0])} ({g['lr'][1]:.2f})"+r'\\'+'\n'
    s+=f"Hansen $J$ / AR(2) $p$ & -- & -- & -- & -- & -- & {g['J']:.2f} / {g['m2']:.2f}"+r'\\'+'\n'
    s+='Observations & '+' & '.join(str(T2[k]['n']) for k in [f'static_{win}',f'trend_{win}',f'fd_{win}',f'fdlag_{win}',f'lsdv_{win}'])+f" & {g['n']}"+r'\\'+'\n'
    if win=='pre': s+=r'\midrule'+'\n'
s+=r"""\botrule
\end{tabular*}
\footnotetext{Dependent variable: log fixed-broadband subscriptions per 100 inhabitants (first-differenced in columns 3--4, 6). Price: log entry-level basket as \% of GNI per capita. Columns 1--2 include country and year effects; columns 3--4 include year effects; all include the core controls of Section~\ref{sec:data}. Column 2 adds country-specific linear trends. Column 4 reports the sum of the current and lagged price-change coefficients. Column 5 adds the lagged dependent variable to column 1 (the short-run coefficient is shown; Nickell bias is of order $1/T$). Column 6 is one-step difference GMM with price treated as endogenous, instruments collapsed and limited to lags 2--3 (12 instruments in Panel A, 17 in Panel B), log GDP per capita as the only control and year effects; long-run standard error by the delta method. Carried-forward 2019 prices are treated as missing and the 2017--2018 change is excluded in columns 3--4. Standard errors clustered by country (33 clusters) in parentheses. WCR: wild cluster restricted bootstrap $p$-value, Webb weights, 9{,}999 replications. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
write('tab_pooled.tex',s)

# ---------- Table 4: EU vs EaP ----------
T3=R['T3']; LD=R['A']['longdiff']; Bq=R['B']
MGu=Bq['mg_unweighted']
s=r"""\begin{table}[t]
\caption{Price coefficients for EU and Eastern Partnership countries, 2010--2019}\label{tab:groups}
\scriptsize\setlength{\tabcolsep}{1pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
& (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
& \shortstack{TWFE\\levels} & \shortstack{+ country\\trends} & \shortstack{First\\differences} & \shortstack{FD +\\country FE} & \shortstack{FD + group\\year FE} & \shortstack{3-year\\diff.} & \shortstack{Mean\\group} \\
\midrule
"""
eu=[T3['static']['p'],T3['trend']['p'],T3['fd']['Dp'],Bq['fd_fe']['eu'],Bq['fd_groupyear']['eu'],LD['old_k3']['eu'],MGu['eu']]
ea=[T3['static']['eap'],T3['trend']['eap'],T3['fd']['eap'],Bq['fd_fe']['eap'],Bq['fd_groupyear']['eap'],LD['old_k3']['eap'],MGu['eap']]
s+='EU & '+' & '.join(b(x) for x in eu)+r'\\'+'\n'
s+=' & '+' & '.join(se(x) for x in eu)+r'\\'+'\n'
s+='EaP & '+' & '.join(b(x) for x in ea)+r'\\'+'\n'
s+=' & '+' & '.join(se(x) for x in ea)+r'\\'+'\n'
s+=r'\midrule'+'\n'+r'\multicolumn{8}{@{}l}{\textit{$p$-values for the EaP--EU difference}}\\'+'\n'
ints=[T3['static']['pxe'],T3['trend']['pxe'],T3['fd']['Dpxe'],Bq['fd_fe']['int'],Bq['fd_groupyear']['int']]
s+='Cluster-robust & '+' & '.join(pv(x['p']) for x in ints)+r' & & \\'+'\n'
s+='Wild bootstrap & '+' & '.join(pv(v) for v in [T3['static']['wcr_int'],T3['trend']['wcr_int'],T3['fd']['wcr_int'],Bq['fd_fe']['wcr_int'],Bq['fd_groupyear']['wcr_int']])+r' & & \\'+'\n'
s+='Randomisation & '+' & '.join(pv(T3[k]['ri_int']) for k in ['static','trend','fd'])+r' & & & & '+pv(Bq['mg_perm_two_sided'])+r'\\'+'\n'
s+='Rank-sum & & & & & & & '+pv(Bq['ranksum_two_sided'])+r'\\'+'\n'
loo_txt='; '.join(f"({n}) ${T3[k]['loo'][0]:.2f}$ to ${T3[k]['loo'][1]:.2f}$" for n,k in [(1,'static'),(2,'trend'),(3,'fd')])
s+='Observations & '+' & '.join(str(x) for x in [T3['static']['n'],T3['trend']['n'],T3['fd']['n'],Bq['fd_fe']['n'],Bq['fd_groupyear']['n'],LD['old_k3']['n']])+r' & 33\\'+'\n'
s+=r"""\botrule
\end{tabular*}
\footnotetext{Columns 1--6 are single regressions in which the price coefficient is interacted with an EaP indicator; the EaP row reports the sum of the main and interaction coefficients. Column 4 adds country effects to the first-difference model (country-specific trends in levels). Column 5 replaces the common year effects by year effects specific to each group, so that the EaP coefficient is identified only from differences among the six EaP countries within a year. Column 6 uses overlapping 3-year differences within the 2010--2017 basket regime, with year effects (5-year differences: EU $""" + f"{LD['old_k5']['eu']['b']:.2f}" + r"""$, EaP $""" + f"{LD['old_k5']['eap']['b']:.2f}" + r"""$). Column 7 is the unweighted average of country-specific first-difference slopes \citep{pesaran1995estimating} with the standard error from their dispersion; the inverse-variance-weighted averages are $""" + f"{R['A']['mg_robust']['eu']['inv_var_mean']:.2f}" + r"""$ (EU) and $""" + f"{R['A']['mg_robust']['eap']['inv_var_mean']:.2f}" + r"""$ (EaP), the latter with 48\% of the weight on Belarus; its $p$-values come from permuting group labels across the 33 slopes and from a two-sided Mann--Whitney test. Randomisation inference reassigns the EaP label to 6 of the 33 countries at random (4{,}999 draws) and compares studentised statistics. Wild bootstrap: WCR, Webb weights. Range of the EaP coefficient when one EaP country is left out: """+loo_txt+r""". Other notes as in Table~\ref{tab:pooled}.}
\end{table}
"""
write('tab_groups.tex',s)

# ---------- Table 5: timing ----------
T4=R['T4']; SP=R['A']['subperiod']
s=r"""\begin{table}[t]
\caption{Did the price response change? Basket regimes, sub-periods and the post-2019 period}\label{tab:timing}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
\multicolumn{6}{@{}l}{\textit{Panel A. Pooled coefficient within each ITU basket regime}}\\
& \multicolumn{2}{c}{1\,GB basket, 2010--2017} & \multicolumn{2}{c}{5\,GB basket, 2018--2024} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
& TWFE levels & First differences & TWFE levels & First differences & \\
Log price & """+' & '.join(b(T4[k]['p' if 'static' in k else 'Dp']) for k in ['static_r1','fd_r1','static_r2','fd_r2'])+r""" & \\
& """+' & '.join(se(T4[k]['p' if 'static' in k else 'Dp']) for k in ['static_r1','fd_r1','static_r2','fd_r2'])+r""" & \\
Observations & """+' & '.join(str(T4[k]['n']) for k in ['static_r1','fd_r1','static_r2','fd_r2'])+r""" & \\
\midrule
\multicolumn{6}{@{}l}{\textit{Panel B. First differences by sub-period}}\\
& 2011--2013 & 2014--2017 & 2019--2024 & & \\
EU & """+' & '.join(b(SP[k]['eu']) for k in ['1113','1417','1924'])+r""" & & \\
& """+' & '.join(se(SP[k]['eu']) for k in ['1113','1417','1924'])+r""" & & \\
EaP & """+' & '.join(b(SP[k]['eap']) for k in ['1113','1417','1924'])+r""" & & \\
& """+' & '.join(se(SP[k]['eap']) for k in ['1113','1417','1924'])+r""" & & \\
EaP--EU: WCR / RI $p$ & """+' & '.join(f"{SP[k]['wcr_int']:.2f} / {SP[k]['ri_int']:.2f}" for k in ['1113','1417','1924'])+r""" & & \\
Observations & """+' & '.join(str(SP[k]['n']) for k in ['1113','1417','1924'])+r""" & & \\
\midrule
\multicolumn{6}{@{}l}{\textit{Panel C. Price $\times$ post-2019 (2020--2024), full sample}}\\
& \multicolumn{2}{c}{TWFE levels} & + country trends & First differences & \\
Log price & \multicolumn{2}{c}{"""+b(T4['static']['p'])+'} & '+b(T4['trend']['p'])+' & '+b(T4['fd']['Dp'])+r""" & \\
& \multicolumn{2}{c}{"""+se(T4['static']['p'])+'} & '+se(T4['trend']['p'])+' & '+se(T4['fd']['Dp'])+r""" & \\
Log price $\times$ post-2019 & \multicolumn{2}{c}{"""+b(T4['static']['pxpost'])+'} & '+b(T4['trend']['pxpost'])+' & '+b(T4['fd']['Dpxpost'])+r""" & \\
& \multicolumn{2}{c}{"""+se(T4['static']['pxpost'])+'} & '+se(T4['trend']['pxpost'])+' & '+se(T4['fd']['Dpxpost'])+r""" & \\
WCR $p$, shift & \multicolumn{2}{c}{"""+pv(T4['static']['wcr_shift'])+'} & '+pv(T4['trend']['wcr_shift'])+' & '+pv(T4['fd']['wcr_shift'])+r""" & \\
Post-2019 coefficient & \multicolumn{2}{c}{"""+b(T4['static']['post'])+'} & '+b(T4['trend']['post'])+' & '+b(T4['fd']['post'])+r""" & \\
Shift, EU / EaP & \multicolumn{2}{c}{"""+f"{T4['static_g']['pxpost']['b']:.2f} / {T4['static_g']['eap_shift']['b']:.2f}"+'} & '+f"{T4['trend_g']['pxpost']['b']:.2f} / {T4['trend_g']['eap_shift']['b']:.2f}"+' & '+f"{T4['fd_g']['Dpxpost']['b']:.2f} / {T4['fd_g']['eap_shift']['b']:.2f}"+r""" & \\
WCR $p$, EU / EaP shift & \multicolumn{2}{c}{"""+f"{pv(T4['static_g']['wcr_eu_shift'])} / {pv(T4['static_g']['wcr_eap_shift'])}"+'} & '+f"{pv(T4['trend_g']['wcr_eu_shift'])} / {pv(T4['trend_g']['wcr_eap_shift'])}"+' & '+f"{pv(T4['fd_g']['wcr_eu_shift'])} / {pv(T4['fd_g']['wcr_eap_shift'])}"+r""" & \\
Observations & \multicolumn{2}{c}{"""+str(T4['static']['n'])+'} & '+str(T4['trend']['n'])+' & '+str(T4['fd']['n'])+r""" & \\
\botrule
\end{tabular*}
\footnotetext{Panel A estimates the pooled model separately within the two ITU basket definitions. Panel B interacts the price change with an EaP indicator within each sub-period; WCR and RI $p$-values refer to the EaP--EU difference. Panel C interacts price with an indicator for 2020--2024; the split model interacts it further with the EaP indicator. Other notes as in Table~\ref{tab:pooled}.}
\end{table}
"""
import re
s=re.sub(r'(?<=[\s{/])-(\d)', r'$-$\1', s)
write('tab_timing.tex',s)

# ---------- Table 6: what drives the EaP estimate ----------
A=R['A']; S=R['S']
def eaprow(lab,x,key='eap'): 
    return f"{lab} & {b(x[key])} & {se(x[key])} & {b(x['eu'])} & {se(x['eu'])} & {x.get('n','')}"+r'\\'+'\n'
s=r"""\begin{table}[t]
\caption{First-difference estimates, 2010--2019: what drives the EaP coefficient?}\label{tab:eapchecks}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
& \multicolumn{2}{c}{EaP} & \multicolumn{2}{c}{EU} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Specification & Coef. & SE & Coef. & SE & Obs. \\
\midrule
"""
s+=eaprow('(1) Baseline',S['base'])
s+=r'\multicolumn{6}{@{}l}{\textit{Price measurement}}\\'+'\n'
s+=eaprow('(2) Price in PPP dollars',S['ppp'])+eaprow('(3) Price in current US dollars',S['usd'])
s+=eaprow('(4) Incl.\\ repeated 2019 prices, 2017--18 change',S['with_carry'])
s+=r'\multicolumn{6}{@{}l}{\textit{Decomposing the affordability ratio (with log GDP per capita)}}\\'+'\n'
dc=A['decomp']['with_gdp']
s+=f"(5a) Log tariff, US\\$ & {b(dc['eap_usd'])} & {se(dc['eap_usd'])} & {b(dc['Dp_usd2'])} & {se(dc['Dp_usd2'])} & \\\\\n"
s+=f"(5b) Log GNI p.c.\\ (ITU denominator) & {b(dc['eap_gni'])} & {se(dc['eap_gni'])} & {b(dc['Dlgni'])} & {se(dc['Dlgni'])} & \\\\\n"
s+=f"(5c) Test tariff $+$ GNI $=0$, $p$ & \\multicolumn{{2}}{{c}}{{{dc['eap_restr']['p']:.2f}}} & \\multicolumn{{2}}{{c}}{{{dc['eu_restr']['p']:.2f}}} & \\\\\n"
dn=A['decomp']['no_gdp']
s+=f"(6a) Log tariff, US\\$, no GDP control & {b(dn['eap_usd'])} & {se(dn['eap_usd'])} & {b(dn['Dp_usd2'])} & {se(dn['Dp_usd2'])} & \\\\\n"
s+=f"(6b) Log GNI p.c., no GDP control & {b(dn['eap_gni'])} & {se(dn['eap_gni'])} & {b(dn['Dlgni'])} & {se(dn['Dlgni'])} & \\\\\n"
s+=r'\multicolumn{6}{@{}l}{\textit{Sample and controls}}\\'+'\n'
s+=eaprow('(7) Excl.\\ currency-crisis years',A['no_crisis'])+eaprow('(8) Excluding Azerbaijan',A['no_aze'])
s+=eaprow('(9) Night lights instead of GDP',A['ntl'])+eaprow('(10) Night lights and GDP',A['ntl_plus_gdp'])
s+=eaprow('(11) Log GDP per capita only',S['gdponly'])+eaprow('(12) Adding secure servers and R\\&D',S['fullctrl'])
s+=r'\multicolumn{6}{@{}l}{\textit{Outcome and functional form}}\\'+'\n'
s+=eaprow('(13) Internet users (\\%)',S['users'])+eaprow('(14) Log count of subscriptions',S['count_dv'])
s+=eaprow('(15) Log-odds, ceiling 70 per 100',S['logodds_K70'])
s+=r'\multicolumn{6}{@{}l}{\textit{Timing}}\\'+'\n'
ld=A['lead']
s+=f"(16a) Current price change & {b(ld['eap_cur'])} & {se(ld['eap_cur'])} & {b(ld['Dp'])} & {se(ld['Dp'])} & {ld['n']}\\\\\n"
s+=f"(16b) Next year's price change & {b(ld['eap_lead'])} & {se(ld['eap_lead'])} & {b(ld['FDp'])} & {se(ld['FDp'])} & \\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{All rows are first-difference regressions with year effects and core controls unless stated, 2010--2019, clustered standard errors (33 countries). Row 5 replaces the log affordability ratio by its two components, the US-dollar tariff and the Atlas GNI per capita implied by ITU's ratio; under a pure affordability response their coefficients are equal and opposite. Row 7 drops Azerbaijan 2015--16, Ukraine 2014--15, Belarus 2011 and 2015, and Armenia, Georgia and Moldova 2015. Night lights: harmonised DMSP/VIIRS radiance summed over national territory (\citealp{li2020harmonized}, 2024 release); the 2013--2014 sensor change is excluded. Row 15: $\ln[s/(70-s)]$. Row 16 adds next year's price change; the EaP coefficient on the current change in 16a is the sum of main and interaction terms. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
write('tab_eapchecks.tex',s)

# ---------- Table 7: regional ----------
T5=R['T5']
s=r"""\begin{table}[t]
\caption{Regional evidence: households with broadband in 175 EU regions, 2010--2021}\label{tab:regional}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
& (1) & (2) & (3) & (4) & (5) \\
FE & \shortstack{Region,\\year} & \shortstack{Region,\\year} & \shortstack{Region,\\country-year} & \shortstack{Region,\\country-year} & \shortstack{Region, year\\2010--2017} \\
\midrule
National log price & """+b(T5['tw_all']['p'])+' & '+b(T5['tw_ld']['p'])+' & -- & -- & '+b(T5['tw_ld_1017']['p'])+r"""\\
& """+se(T5['tw_all']['p'])+' & '+se(T5['tw_ld']['p'])+' & & & '+se(T5['tw_ld_1017']['p'])+r"""\\
Price $\times$ less-developed & & """+b(T5['tw_ld']['pxld'])+' & '+b(T5['cy_ld']['pxld'])+' & '+b(T5['cy_ld_post']['pxld'])+' & '+b(T5['tw_ld_1017']['pxld'])+r"""\\
& & """+se(T5['tw_ld']['pxld'])+' & '+se(T5['cy_ld']['pxld'])+' & '+se(T5['cy_ld_post']['pxld'])+' & '+se(T5['tw_ld_1017']['pxld'])+r"""\\
Price $\times$ less-dev. $\times$ 2020--21 & & & & """+b(T5['cy_ld_post']['pxldpost'])+r""" & \\
& & & & """+se(T5['cy_ld_post']['pxldpost'])+r""" & \\
WCR $p$ (first coef.) & """+' & '.join(pv(T5[k].get('wcr_p',T5[k].get('wcr_pxld'))) for k in ['tw_all','tw_ld','cy_ld','cy_ld_post','tw_ld_1017'])+r"""\\
Observations & """+' & '.join(str(T5[k]['n']) for k in ['tw_all','tw_ld','cy_ld','cy_ld_post','tw_ld_1017'])+r"""\\
\botrule
\end{tabular*}
\footnotetext{Dependent variable: log share of households with broadband (Eurostat \texttt{isoc\_r\_broad\_h}), finest regional level available (NUTS-2; NUTS-1 for Germany, Greece and Poland), 19 member states with at least two regions. National price: log ITU basket as \% of GNI per capita. Less-developed region: GDP per head in PPS below 75\% of the EU average in the first observed year (46 regions), the EU cohesion-policy threshold. All columns control for log regional GDP per head in PPS; columns 1, 2 and 5 also for national log GDP per capita. Standard errors clustered by member state (19 clusters); WCR with 1{,}999 replications. The 95\% confidence interval for the interaction in column 2 is $[""" + f"{R['A']['nuts_ci95'][0]:.2f}, {R['A']['nuts_ci95'][1]:.2f}" + r"""]$. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
write('tab_regional.tex',s)
print('tables written')
