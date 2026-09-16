import numpy as np, pandas as pd
WEBB = np.array([-np.sqrt(1.5),-1,-np.sqrt(.5),np.sqrt(.5),1,np.sqrt(1.5)])

def design(df, y, X, fe=('country','year')):
    D = [df[X].to_numpy(float)]
    for f in fe:
        dm = pd.get_dummies(df[f], drop_first=True).to_numpy(float); D.append(dm)
    Z = np.column_stack(D + [np.ones(len(df))])
    return df[y].to_numpy(float), Z

def ols_cr(yv, Z, g, k_idx, R):
    """returns beta, Rb, se(Rb) CR1 clustered by g"""
    ZtZi = np.linalg.pinv(Z.T@Z); b = ZtZi@Z.T@yv; u = yv - Z@b
    G = np.unique(g); n,k = Z.shape
    meat = np.zeros((k,k))
    for gg in G:
        s = Z[g==gg].T@u[g==gg]; meat += np.outer(s,s)
    V = ZtZi@meat@ZtZi * (len(G)/(len(G)-1))*((n-1)/(n-k))
    r = np.zeros(k); r[k_idx] = R
    return b, r@b, np.sqrt(r@V@r)

def wcr(df, y, X, weights, B=9999, seed=1, cluster='country', null=0.0):
    """Wild cluster restricted bootstrap (Webb weights) for H0: sum w_j beta_j = null.
    weights: dict name->w. Returns (est, se, t, p_boot)."""
    df = df.dropna(subset=[y]+X).reset_index(drop=True)
    yv, Z = design(df, y, X); g = df[cluster].to_numpy()
    idx = [X.index(n) for n in weights]; R = np.array([weights[n] for n in weights])
    b, est, se = ols_cr(yv, Z, g, idx, R); t0 = (est-null)/se
    # restricted estimation: impose R b = null via constrained LS
    k = Z.shape[1]; r = np.zeros(k); r[idx] = R
    ZtZi = np.linalg.pinv(Z.T@Z)
    br = b - ZtZi@r*((r@b-null)/(r@ZtZi@r))
    ur = yv - Z@br
    G, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    P = ZtZi@Z.T
    ts = np.empty(B)
    # precompute cluster pieces
    rows = [np.where(inv==j)[0] for j in range(len(G))]
    n,kk = Z.shape; cfac = (len(G)/(len(G)-1))*((n-1)/(n-kk))
    Zr = [Z[ix] for ix in rows]
    a = r@ZtZi
    for bb in range(B):
        v = rng.choice(WEBB, len(G))
        ys = Z@br + ur*v[inv]
        bs = P@ys; us = ys - Z@bs
        # CR1 variance of r'b: sum_g (a' Z_g' u_g)^2
        sc = np.array([a@(Zr[j].T@us[rows[j]]) for j in range(len(G))])
        ses = np.sqrt(cfac*np.sum(sc**2))
        ts[bb] = (r@bs-null)/ses
    p = np.mean(np.abs(ts) >= abs(t0))
    return est, se, t0, p

def ri_label(df, y, X, label_col, make_cols, target, B=4999, seed=2, n_treat=6):
    """Randomisation inference: reassign the group label (country-level) at random.
    make_cols(df, lab) must rebuild interaction columns from label; target = coefficient name.
    Uses the studentised (CR1) t-statistic."""
    df = df.dropna(subset=[y]+[x for x in X if x in df]).reset_index(drop=True)
    countries = df.country.unique(); rng = np.random.default_rng(seed)
    def tstat(lab):
        dd = make_cols(df.copy(), lab)
        yv, Z = design(dd, y, X); g = dd.country.to_numpy()
        _, est, se = ols_cr(yv, Z, g, [X.index(target)], np.array([1.0]))
        return est, est/se
    true_lab = set(df.loc[df[label_col]==1,'country'].unique())
    e0, t0 = tstat(true_lab)
    ts = []
    for _ in range(B):
        lab = set(rng.choice(countries, n_treat, replace=False))
        ts.append(tstat(lab)[1])
    ts = np.array(ts)
    return e0, t0, np.mean(np.abs(ts) >= abs(t0))
