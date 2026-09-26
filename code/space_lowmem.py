import os, sys
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import in200_probe as IP

CHUNK = 20000


class Space200LowMem(IP.Space200):
    def __init__(self, Q, refs, cfg):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.linear_model import LogisticRegression
        from scipy.linalg import cholesky, solve_triangular
        self.cfg, self.Q, self.refs = cfg, Q, refs
        self.mus = np.vstack([R.mean(0) for R in refs])
        self.n_c = np.array([len(R) for R in refs])
        self.C = len(refs)
        self.mu_g = np.average(self.mus, axis=0, weights=self.n_c)
        N = len(Q)

        Dc = cdist(Q, self.mus)
        self.nc = Dc.argmin(1)
        self.d_cls = Dc[np.arange(N), self.nc]
        self.d_glob = np.linalg.norm(Q - self.mu_g, axis=1)
        self.n_assigned = self.n_c[self.nc]
        del Dc

        self.sigma_t = np.empty(N); self.r = np.empty(N)
        for c in np.unique(self.nc):
            m = self.nc == c
            W = refs[c] - self.mus[c]
            v = Q[m] - self.mus[c]
            rr = np.linalg.norm(v, axis=1)
            u = v / (rr[:, None] + 1e-12)
            self.r[m] = rr
            self.sigma_t[m] = (u @ W.T).std(axis=1)

        u_g = (Q - self.mu_g) / (self.d_glob[:, None] + 1e-12)
        vw = np.zeros(N)
        for c in range(self.C):
            vw += (u_g @ (refs[c] - self.mus[c]).T).var(axis=1)
        self.sigma_w = np.sqrt(vw / self.C)
        self.N_all = int(self.n_c.sum())

        self.knn_std = np.empty(N); self.lid = np.empty(N); self.dk = np.empty(N)
        for c in np.unique(self.nc):
            m = self.nc == c
            R = refs[c]
            kk = min(max(3, int(round(cfg["k_frac"] * len(R)))), len(R))
            d, _ = NearestNeighbors(n_neighbors=kk).fit(R).kneighbors(Q[m])
            self.knn_std[m] = d.std(1)
            lk = min(cfg["k_lid"], kk)
            dd = np.maximum(d[:, :lk], 1e-12)
            self.lid[m] = 1.0 / (np.log(dd[:, -1:] / dd[:, :-1]).mean(1) + 1e-12)
            self.dk[m] = dd[:, -1]

        X = np.vstack(refs)
        self.knn = NearestNeighbors(n_neighbors=cfg["k_knn"]).fit(X).kneighbors(Q)[0][:, -1]

        D = X.shape[1]
        Sw = np.zeros((D, D))
        for R in refs:
            Xc_ = R - R.mean(0)
            Sw += Xc_.T @ Xc_
        Sw /= max(len(X) - self.C, 1)
        a = cfg["maha_shrink"]
        Sw = (1 - a) * Sw + a * (np.trace(Sw) / D) * np.eye(D)
        Wm = solve_triangular(cholesky(Sw, lower=True), np.eye(D), lower=True).T
        self.maha = cdist(Q @ Wm, self.mus @ Wm).min(1)
        del Sw, Wm

        y = np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])
        self.probe = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
        lg = self.probe.decision_function(Q)
        if lg.ndim == 1:
            lg = np.column_stack([-lg, lg])
        mx = lg.max(1, keepdims=True)
        p = np.exp(lg - mx); p /= p.sum(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
        self.energy, self.msp = -lse, -p.max(1)
        self.maxlogit = -lg.max(1)
        self.entropy = -(p * np.log(p + 1e-12)).sum(1)
        lt = lg / cfg["odin_T"]
        mt = lt.max(1, keepdims=True)
        pt = np.exp(lt - mt); pt /= pt.sum(1, keepdims=True)
        self.odin = -pt.max(1)

        cov = np.zeros((D, D)); nrm_sum = 0.0
        for s in range(0, len(X), CHUNK):
            xc = X[s:s + CHUNK] - self.mu_g
            cov += xc.T @ xc
        cov /= max(len(X) - 1, 1)
        w_, V = np.linalg.eigh(cov)
        Vv = V[:, np.argsort(w_)[::-1]][:, :cfg["vim_P"]]
        norms = np.empty(len(X))
        for s in range(0, len(X), CHUNK):
            xc = X[s:s + CHUNK] - self.mu_g
            norms[s:s + CHUNK] = np.linalg.norm(xc - (xc @ Vv) @ Vv.T, axis=1)
        qc = Q - self.mu_g
        res = np.linalg.norm(qc - (qc @ Vv) @ Vv.T, axis=1)
        al = float(lg.max(1).mean() / (norms.mean() + 1e-12))
        self.vim = al * res - lse

        tau_n = np.empty(len(X))
        for s in range(0, len(X), CHUNK):
            tau_n[s:s + CHUNK] = np.linalg.norm(X[s:s + CHUNK] - self.mu_g, axis=1)
        self.tau = float(np.percentile(tau_n, cfg["hinge_tau_q"]))
        self.hinge = self.d_cls + cfg["hinge_lam"] * np.maximum(0.0, self.tau - self.d_glob)


def verify():
    import derma_probe as DP, exponent_probe as EP
    DP.CFG["cache_dir"] = "./feat_cache_dino"
    groups, refs = EP.build_cifar(np.random.default_rng(0), "c100")
    Q = np.vstack([groups[g] for g in groups])
    A = IP.Space200(Q, refs, DP.CFG); B = Space200LowMem(Q, refs, DP.CFG)
    worst = 0.0
    for k in ["d_cls", "sigma_t", "sigma_w", "knn_std", "lid", "knn", "maha", "energy",
              "msp", "maxlogit", "entropy", "odin", "vim", "tau", "hinge"]:
        d = float(np.abs(np.asarray(getattr(A, k), float) - np.asarray(getattr(B, k), float)).max())
        worst = max(worst, d); print(f"    {k:10} max|diff| = {d:.3e}")
    d = float(np.abs(A.theory(True) - B.theory(True)).max()); worst = max(worst, d)
    print(f"    {'theory':10} max|diff| = {d:.3e}")
    print(f"\n  [verify] 최대 편차 {worst:.3e} -> {'통과' if worst < 1e-8 else '** 실패 **'}")


if __name__ == "__main__":
    verify()
