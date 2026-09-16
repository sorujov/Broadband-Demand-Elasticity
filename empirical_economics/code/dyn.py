import numpy as np, pandas as pd
from scipy import stats

def ab_gmm(df, y, xs_endog, xs_exog, lags=(2,3), timedum=True, collapse=True):
    """Difference GMM (Arellano-Bond) with collapsed lag-limited instruments.
    Regressors: L1.y, xs_endog (instrumented with lags of their levels), xs_exog (strictly exog, instrument themselves in FD).
    Returns dict with one-step robust results, two-step coefficients, Hansen J, AR(1), AR(2)."""
    df = df.sort_values(['country','year']).copy()
    g = df.groupby('country')
    df['Ly'] = g[y].shift(1)
    cols = [y,'Ly']+xs_endog+xs_exog
    for v in cols: df['D'+v] = df[v] - df.groupby('country')[v].shift(1)
    years = sorted(df.year.unique())
    # time dummies in FD
    tds=[]
    if timedum:
        for t in years[3:]:
            df[f'td{t}'] = (df.year==t)*1.0
            df[f'Dtd{t}'] = df[f'td{t}'] - df.groupby('country')[f'td{t}'].shift(1)
            tds.append(f'td{t}')
    est = df[df['DLy'].notna() & df['D'+y].notna()].copy()
    # need y_{t-2} available: first usable year = years[2]
    est = est[est.year>=years[2]]
    est = est.dropna(subset=['D'+v for v in [y,'Ly']+xs_endog+xs_exog])
    Xn = ['Ly']+xs_endog+xs_exog+tds
    # instruments: collapsed, for each lag l in lags, column = level of var at t-l (0 if missing)
    Zcols=[]
    for v in [y]+xs_endog:
        for l in range(lags[0], lags[1]+1):
            name=f'Z_{v}_L{l}'
            est[name]=df.groupby('country')[v].shift(l).loc[est.index].fillna(0.0)
            Zcols.append(name)
    for v in xs_exog+tds:
        est['ZD'+v]=est['D'+v]; Zcols.append('ZD'+v)
    ids = est.country.values
    Y = est['D'+y].values; X = est[['D'+x for x in Xn]].values; Z = est[Zcols].values
    groups = [np.where(ids==i)[0] for i in np.unique(ids)]
    # one-step weight with H (MA(1))
    A = np.zeros((Z.shape[1],)*2)
    for ix in groups:
        T=len(ix); H=2*np.eye(T)-np.eye(T,k=1)-np.eye(T,k=-1)
        A += Z[ix].T@H@Z[ix]
    W1 = np.linalg.pinv(A)
    def gmm(W):
        XZ=X.T@Z; M=np.linalg.pinv(XZ@W@XZ.T); b=M@XZ@W@(Z.T@Y); return b,M
    b1,M1 = gmm(W1); e1 = Y-X@b1
    S = np.zeros_like(A)
    for ix in groups:
        s=Z[ix].T@e1[ix]; S+=np.outer(s,s)
    XZ=X.T@Z
    V1 = M1@XZ@W1@S@W1@XZ.T@M1 * len(groups)/(len(groups)-1)
    W2 = np.linalg.pinv(S); b2,M2 = gmm(W2); e2=Y-X@b2
    gbar = Z.T@e2; J = gbar@W2@gbar; dfJ = Z.shape[1]-X.shape[1]
    # AR tests on one-step residuals (simple m-statistics)
    def ar(order):
        num=0; den=0
        for ix in groups:
            e=e1[ix]
            if len(e)>order:
                a=e[order:]*e[:-order]; num+=a.sum(); den+=a.sum()**2
        return num/np.sqrt(den)
    m1, m2 = ar(1), ar(2)
    se1 = np.sqrt(np.diag(V1))
    out = pd.DataFrame({'b1':b1,'se1':se1,'p1':2*(1-stats.norm.cdf(np.abs(b1/se1))),'b2':b2}, index=Xn)
    return dict(tab=out, V=pd.DataFrame(V1,index=Xn,columns=Xn), J=J, dfJ=dfJ, pJ=1-stats.chi2.cdf(J,dfJ),
                m1=m1, pm1=2*(1-stats.norm.cdf(abs(m1))), m2=m2, pm2=2*(1-stats.norm.cdf(abs(m2))),
                n_inst=Z.shape[1], N=len(groups), nobs=len(Y))

def longrun(res, names, rho='Ly'):
    """long-run effect sum(beta)/(1-rho) with delta-method SE"""
    t=res['tab']; V=res['V']
    b=t.b1; s=sum(b[n] for n in names); r=b[rho]; lr=s/(1-r)
    grad=pd.Series(0.0,index=t.index)
    for n in names: grad[n]=1/(1-r)
    grad[rho]=s/(1-r)**2
    se=np.sqrt(grad@V@grad)
    return lr, se, 2*(1-stats.norm.cdf(abs(lr/se)))
