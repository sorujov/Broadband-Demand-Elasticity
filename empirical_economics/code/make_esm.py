import json, pandas as pd, numpy as np
R=json.load(open('results/results.json')); S=R['S']; A=R['A']
def st(p): return '^{***}' if p<0.01 else '^{**}' if p<0.05 else '^{*}' if p<0.10 else ''
def b(x): return f"${x['b']:.2f}{st(x['p'])}$"
def se(x): return f"({x['se']:.2f})"
def m(v): return f"${v:.2f}$"
out=[]
# S1 robustness grid FD pre
rows=[('Baseline','base'),('Price in PPP dollars','ppp'),('Price in current US dollars','usd'),('GDP only','gdponly'),
      ('+ servers and R\\&D','fullctrl'),('+ mobile basket (2014--19)','mobprice'),
      ('Internet users outcome','users'),('Incl.\\ repeated 2019, 2017--18','with_carry'),
      ('Log count of subscriptions','count_dv'),('Sample ending in 2017','end2017'),
      ('Log-odds, $K=55$','logodds_K55'),('Log-odds, $K=70$','logodds_K70'),('Log-odds, $K=100$','logodds_K100')]
s=r"""\begin{table}[h]
\caption{First-difference estimates, 2010--2019: pooled, EU and EaP coefficients under alternative choices}\label{tab:S1}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
& \multicolumn{2}{c}{Pooled} & \multicolumn{2}{c}{EU} & \multicolumn{2}{c}{EaP} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Specification & Coef. & SE & Coef. & SE & Coef. & SE & Obs.\\
\midrule
"""
for lab,k in rows:
    x=S[k]; s+=f"{lab} & {b(x['pooled'])} & {se(x['pooled'])} & {b(x['eu'])} & {se(x['eu'])} & {b(x['eap'])} & {se(x['eap'])} & {x['n']}\\\\\n"
dk=S['dk']
s+=f"Driscoll--Kraay SEs & & & {b(dk['eu'])} & {se(dk['eu'])} & {b(dk['eap'])} & {se(dk['eap'])} & 240\\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{First differences with year effects and core controls; clustered standard errors unless stated. Log-odds rows use $\ln[s/(K-s)]$ as the outcome. The mobile-broadband basket is available from 2013, which restricts the differenced sample to 2014--2019. Driscoll--Kraay: Bartlett kernel, bandwidth 3. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
out.append(s)
# S2 functional form & headroom
ff=S['funcform']
s=r"""\begin{table}[h]
\caption{Functional form and headroom, first differences, 2010--2019}\label{tab:S2}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
\multicolumn{5}{@{}l}{\textit{Panel A. Implied elasticity at group means}}\\
Form & EU & $p$ & EaP & $p$ \\
\midrule
"""
for lab,k in [('Linear ($s$ on price)','linlin'),('Log-linear ($\\ln s$ on price)','loglin'),('Linear-log ($s$ on $\\ln$ price)','linlog')]:
    s+=f"{lab} & {m(ff[k]['eu'])} & {ff[k]['p_eu']:.3f} & {m(ff[k]['eap'])} & {ff[k]['p_eap']:.3f}\\\\\n"
s+=r"""\midrule
\multicolumn{5}{@{}l}{\textit{Panel B. Price coefficient varying with headroom $h_{i,t-1}=1-s_{i,t-1}/70$ (centred)}}\\
& \multicolumn{2}{c}{2010--2019} & \multicolumn{2}{c}{2010--2024}\\
"""
Bq=R['B']; hp=Bq['headroom70_pre']; hf=Bq['headroom70_full']; he=Bq['headroom70_eap']
s+=f"$\\Delta\\ln p$ & {b(hp['Dp'])} {se(hp['Dp'])} & & {b(hf['Dp'])} {se(hf['Dp'])} & \\\\\n"
s+=f"$\\Delta\\ln p\\times h$ & {b(hp['Dpxh'])} {se(hp['Dpxh'])} & & {b(hf['Dpxh'])} {se(hf['Dpxh'])} & \\\\\n"
s+=f"$\\Delta\\ln p\\times$ post-2019 & & & {b(hf['Dpxpost'])} {se(hf['Dpxpost'])} & \\\\\n"
s+=r"\multicolumn{5}{@{}l}{\textit{Adding the EaP interaction (2010--2019)}}\\"+"\n"
s+=f"$\\Delta\\ln p$; $\\times$EaP; $\\times h$ & \\multicolumn{{4}}{{l}}{{{b(he['Dp'])} {se(he['Dp'])};\\quad {b(he['Dpxe'])} {se(he['Dpxe'])};\\quad {b(he['Dpxh'])} {se(he['Dpxh'])}}}\\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{Panel A: elasticities evaluated at group mean price and penetration; $p$-values refer to the underlying coefficient (EaP: sum of main and interaction terms). Panel B: clustered standard errors in parentheses. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
out.append(s)
# S3 post-2019 robustness
s=r"""\begin{table}[h]
\caption{Price $\times$ post-2019: sample restrictions}\label{tab:S3}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
& \multicolumn{2}{c}{TWFE levels} & \multicolumn{2}{c}{First differences} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Sample & Price & Price $\times$ post & Price & Price $\times$ post & Obs. (FD)\\
\midrule
"""
for lab,k in [('Full sample','full_base'),('Excl.\\ UKR 2022--24, BLR 2020--24','full_nowar'),('Ending in 2021','full_end2021')]:
    a=S[k+'_static']; f=S[k]
    s+=f"{lab} & {b(a['p'])} & {b(a['pxpost'])} & {b(f['Dp'])} & {b(f['Dpxpost'])} & {f['n']}\\\\\n"
    s+=f" & {se(a['p'])} & {se(a['pxpost'])} & {se(f['Dp'])} & {se(f['Dpxpost'])} & \\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{Core controls; country and year effects (levels) or year effects (first differences); clustered standard errors. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}
"""
out.append(s)
# S4 IV
iv=S['iv']; ivf=S['iv_fd']
s=r"""\begin{table}[h]
\caption{Hausman-type instrumental-variable estimates, pooled}\label{tab:S4}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
Estimator & Instrument & Coef. (SE) & First-stage $F$ & Anderson--Rubin 95\% set\\
\midrule
"""
for z,lab in [('z_sub','sub-regional'),('z_nn','3 nearest capitals')]:
    x=iv[z]; ar=f"[{x['ar_lo']:.1f}, {x['ar_hi']:.1f}]"+('' if x['ar_bounded'] else ' (grid edge)')
    s+=f"Levels, 2010--2019 & {lab} & {x['b']:.2f} ({x['se']:.2f}) & {x['F']:.1f} & {ar}\\\\\n"
for k,lab in [('2019_Dz','sub-regional'),('2019_Dzn','nearest capitals'),('2019_Dz+Dzn','both'),('2024_Dz','sub-regional'),('2024_Dzn','nearest capitals'),('2024_Dz+Dzn','both')]:
    x=ivf[k]; w='2010--2019' if k.startswith('2019') else '2010--2024'
    s+=f"FD, {w} & {lab} & {x['b']:.2f} ({x['se']:.2f}) & {x['F']:.1f} & --\\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{Instruments are averages of log prices of other sample countries in the same year: the leave-one-out mean of the country's sub-region, or the inverse-distance-weighted mean of the three countries with the nearest capitals. Levels regressions are two-way demeaned; FD regressions include year effects. Core controls; standard errors clustered by country. The Anderson--Rubin (AR) set was computed on a grid over $[-6,4]$. Sub-regions: Nordic (DNK, FIN, SWE), Baltic (EST, LVA, LTU), Western (AUT, BEL, DEU, FRA, IRL, LUX, NLD), Southern (CYP, ESP, GRC, ITA, MLT, PRT), Central (CZE, HUN, POL, SVK), South-Eastern (BGR, HRV, ROU, SVN), South Caucasus (ARM, AZE, GEO), Eastern (BLR, MDA, UKR). The first-difference first stages are too weak for the estimates to be informative.}
\end{table}
"""
out.append(s)
# S5 data flags and FD composition, weights, slopes
sl=pd.read_csv('results/country_fd_slopes_pre.csv'); wt=pd.read_csv('results/twfe_weights_pre.csv'); wt.columns=['country','w']
sl=sl.merge(wt,on='country').sort_values(['eap','b'])
s=r"""\begin{table}[h]
\caption{Country-specific first-difference slopes (2010--2019) and weights in the levels regression}\label{tab:S5}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc|lcccc@{}}
\toprule
Country & Slope & SE & Obs. & TWFE weight & Country & Slope & SE & Obs. & TWFE weight\\
\midrule
"""
rows=list(sl.itertuples())
half=(len(rows)+1)//2
BF=chr(92)+'textbf{'
def nm(r): return BF+r.country+'}' if r.eap else r.country
for i in range(half):
    r1=rows[i]; c1=f"{nm(r1)} & {r1.b:.2f} & {r1.se:.2f} & {r1.n} & {r1.w:.3f}"
    if i+half<len(rows):
        r2=rows[i+half]; c2=f"{nm(r2)} & {r2.b:.2f} & {r2.se:.2f} & {r2.n} & {r2.w:.3f}"
    else: c2='& & & &'
    s+=c1+' & '+c2+'\\\\\n'
s+=r"""\botrule
\end{tabular*}
\footnotetext{Slopes: regression of the annual change in log subscriptions per 100 on the change in log price and log GDP per capita with a constant, by country; conventional standard errors. EaP countries in bold. TWFE weight: country's share of the sum of squared residualised prices in the 2010--2019 levels regression (price residualised on country and year effects and core controls); the six EaP countries account for """+f"{R['T3']['twfe_weight_eap']*100:.0f}"+r"""\%.}
\end{table}
"""
out.append(s)
fd=A['fd_obs_by_year']; imp=json.load(open('results/imputation_counts.json'))
carry=sorted(pd.read_pickle('panel.pkl').query('carry').country)
s=r"""\begin{table}[h]
\caption{Data construction: flags, imputations and sample composition}\label{tab:S6}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{4.2cm}p{8.2cm}@{}}
\toprule
Item & Treatment \\
\midrule
ITU basket definition & 1\,GB minimum (2010--2017), 5\,GB minimum (2018--2024); the 2017--2018 change is never used in differenced regressions.\\
2019 prices equal to 2018 in all three units & """+str(len(carry))+" countries ("+', '.join(carry)+r"""); set to missing.\\
Missing in source & Romania 2013 price; Bulgaria 2024 subscriptions; left missing.\\
Mobile-broadband basket & Chain of ITU data-only baskets: postpaid computer-based 1\,GB (2013--2017), 1.5\,GB (2018--2020), 2\,GB (2021--2024).\\
Tertiary enrolment & """+f"{imp['education_tertiary_pct']['interior']} interior values interpolated, {imp['education_tertiary_pct']['edge']} end values set to nearest observation"+r""".\\
R\&D expenditure (robustness only) & """+f"{imp['research_development_expenditure']['edge']} end values set to nearest observation"+r""".\\
Population density & 2024 values computed from population and land area.\\
Night-time lights & Harmonised DMSP (to 2013) and simulated VIIRS (from 2014) radiance summed over Natural Earth national boundaries (European France only); the 2013--2014 change is excluded.\\
First-difference observations by year & """+', '.join(f"{k}: {v}" for k,v in fd.items())+r""".\\
Earlier version of this panel & Used $\ln(1+p)$ instead of $\ln p$, forward-filled missing prices and subscriptions, and averaged different mobile baskets; all corrected here.\\
\botrule
\end{tabular*}
\end{table}
"""
out.append(s)
by=R['B']['eap_by_year']
s=r"""\begin{table}[h]
\caption{EaP first-difference coefficient by year and identification checks, 2010--2019}\label{tab:S7}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
\multicolumn{4}{@{}l}{\textit{Panel A. EaP coefficient by year (common EU slope, year effects, core controls)}}\\
Year & Coef. & SE & EaP countries with data\\
\midrule
"""
for y,v in by.items():
    s+=f"{y} & {m(v['b'])} & ({v['se']:.2f}) & {v['n_eap']}\\\\\n"
fe=R['B']['fd_fe']; gy=R['B']['fd_groupyear']
s+=r"""\midrule
\multicolumn{4}{@{}l}{\textit{Panel B. Alternative fixed effects in first differences}}\\
Specification & EU & EaP & WCR $p$ (difference)\\
"""
s+=f"Year effects (baseline) & {b(R['T3']['fd']['Dp'])} & {b(R['T3']['fd']['eap'])} & {R['T3']['fd']['wcr_int']:.3f}\\\\\n"
s+=f"Year and country effects & {b(fe['eu'])} & {b(fe['eap'])} & {fe['wcr_int']:.3f}\\\\\n"
s+=f"Group-specific year effects & {b(gy['eu'])} & {b(gy['eap'])} & {gy['wcr_int']:.3f}\\\\\n"
s+=r"""\botrule
\end{tabular*}
\footnotetext{Panel A: coefficient on the change in log price for EaP countries in each year, from one regression in which the EaP slope is year-specific and the EU slope common; 2018 is omitted because of the basket revision, and in 2019 only two EaP countries report a new price. Panel B: first-difference models of Table 4 in the article; with group-specific year effects the EaP coefficient is identified only from differences among EaP countries within a year. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{figS1_simulation.pdf}
\caption{Simulated year-specific coefficients when the true price coefficient is zero. 33 countries follow logistic adoption paths (ceilings 35--50 per 100, speeds 0.25--0.45); 27 have diffusion midpoints before the sample (mean $-4$ years), six start late (mean midpoint in year 3) with affordability ratios about 1.3 log points higher that fall with the stage of diffusion. Idiosyncratic noise: 0.03 in log adoption, 0.15 in log price. Lines show means over 200 panels of the year-specific coefficients from the levels regression with country and year effects and from first differences with year effects. The pooled coefficients average $"""+f"{json.load(open('results/simulation.json'))['pooled_levels']:.2f}"+r"""$ (levels) and $"""+f"{json.load(open('results/simulation.json'))['pooled_fd']:.2f}"+r"""$ (first differences)}\label{fig:S1}
\end{figure}
"""
out.append(s)
doc=r"""\documentclass[pdflatex,sn-basic]{sn-jnl}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\graphicspath{{figures/}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\title{Online Resource 1 for ``What can country panels tell us about broadband price responsiveness? Evidence from the European Union and the Eastern Partnership, 2010--2024''}
\abstract{This Online Resource reports additional estimates and the data-construction details referred to in the article. All estimates can be reproduced from public data with the code in the replication package.}
\maketitle
\section*{Additional tables}
"""+'\n'.join(out)+r"""
\end{document}
"""
open('paper/ESM_1.tex','w').write(doc)
print('ok')
