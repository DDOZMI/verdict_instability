import os, sys, json, time, argparse
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derma_probe as DP
import exponent_probe as EP

CACHE_FMT = "./feat_cache_{}_{}"
N_REF     = 400
N_QUERY   = 800
SCORES11  = ["knn_std", "lid", "d_cls", "knn", "maha", "vim",
             "energy", "maxlogit", "odin", "msp", "entropy"]
FLAGS5    = ["knn_std", "lid", "energy", "msp", "maha"]
COVS      = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ALPHAS    = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
LAMBDAS   = [0.0, 1.0, 2.5, 5.0, 10.0]


class Space200(DP.Space):

    def __init__(self, Q, refs, cfg):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.linear_model import LogisticRegression
        from scipy.linalg import cholesky, solve_triangular
        self.cfg, self.Q, self.refs = cfg, Q, refs
        self.mus = np.vstack([R.mean(0) for R in refs])
        self.mu_g = np.vstack(refs).mean(0)
        self.n_c = np.array([len(R) for R in refs])
        self.C = len(refs)
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
            Wc = refs[c] - self.mus[c]
            vw += (u_g @ Wc.T).var(axis=1)
        self.sigma_w = np.sqrt(vw / self.C)
        self.N_all = int(self.n_c.sum())

        self.knn_std = np.empty(N); self.lid = np.empty(N); self.dk = np.empty(N)
        for c in np.unique(self.nc):
            m = self.nc == c
            R = refs[c]
            kk = min(max(3, int(round(cfg["k_frac"] * len(R)))), len(R))
            nn = NearestNeighbors(n_neighbors=kk).fit(R)
            d, _ = nn.kneighbors(Q[m])
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
            Xc = R - R.mean(0)
            Sw += Xc.T @ Xc
        Sw /= max(len(X) - self.C, 1)
        a = cfg["maha_shrink"]
        Sw = (1 - a) * Sw + a * (np.trace(Sw) / D) * np.eye(D)
        Wm = solve_triangular(cholesky(Sw, lower=True), np.eye(D), lower=True).T
        self.maha = cdist(Q @ Wm, self.mus @ Wm).min(1)

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
        Xc = X - self.mu_g
        w_, V = np.linalg.eigh((Xc.T @ Xc) / max(len(Xc) - 1, 1))
        Vv = V[:, np.argsort(w_)[::-1]][:, :cfg["vim_P"]]
        res = np.linalg.norm((Q - self.mu_g) - (Q - self.mu_g) @ Vv @ Vv.T, axis=1)
        al = float(lg.max(1).mean() /
                   (np.linalg.norm(Xc - Xc @ Vv @ Vv.T, axis=1).mean() + 1e-12))
        self.vim = al * res - lse

        self.tau = float(np.percentile(np.linalg.norm(Xc, axis=1), cfg["hinge_tau_q"]))
        self.hinge = self.d_cls + cfg["hinge_lam"] * np.maximum(0.0, self.tau - self.d_glob)

    def truth(self, n_boot, rng):
        B = np.zeros((n_boot, len(self.Q)))
        for b in range(n_boot):
            rb = [R[rng.integers(0, len(R), len(R))] for R in self.refs]
            mb = np.vstack([R.mean(0) for R in rb])
            mg = np.average(mb, axis=0, weights=self.n_c)
            dc = cdist(self.Q, mb).min(1)
            dg = np.linalg.norm(self.Q - mg, axis=1)
            B[b] = dc + self.cfg["hinge_lam"] * np.maximum(0.0, self.tau - dg)
            print(f"    boot {b+1}/{n_boot}", end="\r")
        print()
        return B.std(0)


def build(repr_key, rng, bench="in200"):
    cd = CACHE_FMT.format(bench, repr_key)
    ftr = np.load(f"{cd}/{bench}_feat_train.npy").astype(np.float64)
    ytr = np.load(f"{cd}/{bench}_lab_train.npy")
    fte = np.load(f"{cd}/{bench}_feat_test.npy").astype(np.float64)
    meta = json.load(open(f"{cd}/BENCH_META.json"))
    C = int(ytr.max()) + 1

    refs = []
    for c in range(C):
        X = ftr[ytr == c]
        if len(X) > N_REF:
            X = X[rng.choice(len(X), N_REF, replace=False)]
        refs.append(X)

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    groups = {"in": sub(fte, N_QUERY)}
    near_p = f"{cd}/{bench}_feat_near.npy"
    if os.path.exists(near_p):
        groups["near"] = sub(np.load(near_p).astype(np.float64), N_QUERY)
    for s in meta["far_sets"]:
        groups[s] = sub(np.load(f"{cd}/{bench}_feat_far_{s}.npy").astype(np.float64), N_QUERY)
    return groups, refs


def alloc_eff_dim(S, refs, mask):
    out = []
    eig = {}
    for c in np.unique(S.nc[mask]):
        W = refs[c] - S.mus[c]
        Sc = (W.T @ W) / max(len(W) - 1, 1)
        w, V = np.linalg.eigh(Sc)
        eig[c] = V
    idx = np.where(mask)[0]
    for i in idx:
        c = S.nc[i]
        d = S.Q[i] - S.mus[c]
        pr = (eig[c].T @ d) ** 2
        p = pr / (pr.sum() + 1e-30)
        out.append(1.0 / (p ** 2).sum())
    return float(np.mean(out))


def within(x, G):
    o = np.asarray(x, dtype=float).copy()
    for g in np.unique(G):
        o[G == g] -= o[G == g].mean()
    return o


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def r2(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    return float(LinearRegression().fit(zs(p[m])[:, None], zs(t[m]))
                 .score(zs(p[m])[:, None], zs(t[m])))


def aurc(v, c):
    o = np.argsort(c)
    tz = getattr(np, "trapezoid", None) or np.trapz
    return float(tz(np.asarray(v)[o], np.asarray(c)[o]))


def selfcheck():
    DP.CFG["cache_dir"] = "./feat_cache_dino"
    rng = np.random.default_rng(0)
    groups, refs = EP.build_cifar(rng, "c100")
    Q = np.vstack([groups[g] for g in groups])
    print(f"  CIFAR-100: refs={len(refs)}x{len(refs[0])}  Q={len(Q)}")
    A = DP.Space(Q, refs, DP.CFG)
    B = Space200(Q, refs, DP.CFG)
    worst = 0.0
    for k in ["d_cls", "d_glob", "sigma_t", "r", "sigma_w", "knn_std", "lid", "dk",
              "knn", "maha", "energy", "msp", "maxlogit", "entropy", "odin", "vim",
              "hinge", "nc", "n_assigned"]:
        a, b = np.asarray(getattr(A, k), float), np.asarray(getattr(B, k), float)
        d = np.abs(a - b).max()
        worst = max(worst, d)
        print(f"    {k:12} max|diff| = {d:.3e}")
    ta = A.theory(True); tb = B.theory(True)
    print(f"    {'theory':12} max|diff| = {np.abs(ta-tb).max():.3e}")
    r1 = A.truth(20, np.random.default_rng(7)); r2_ = B.truth(20, np.random.default_rng(7))
    print(f"    {'truth(B=20)':12} max|diff| = {np.abs(r1-r2_).max():.3e}")
    worst = max(worst, np.abs(ta-tb).max(), np.abs(r1-r2_).max())
    print(f"\n  [selfcheck] 최대 편차 {worst:.3e}  ->  {'통과' if worst < 1e-8 else '** 실패 **'}")


def main(repr_key, n_seeds, n_boot, bench):
    out = {"repr": repr_key, "bench": bench, "n_seeds": n_seeds, "n_boot": n_boot,
           "cfg": {k: DP.CFG[k] for k in ["hinge_lam", "hinge_tau_q", "k_knn", "k_lid",
                                          "vim_P", "maha_shrink", "k_frac", "odin_T"]},
           "n_ref": N_REF, "n_query": N_QUERY}
    t00 = time.time()

    rng = np.random.default_rng(0)
    groups, refs = build(repr_key, rng, bench)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    print(f"[build] C={len(refs)} n_c={len(refs[0])} Q={len(Q)} groups={ {g:len(v) for g,v in groups.items()} }")
    out["groups"] = {g: int(len(v)) for g, v in groups.items()}

    S = Space200(Q, refs, DP.CFG)
    print(f"[Space200] probe acc={S.probe.score(np.vstack(refs), np.concatenate([np.full(len(R),i) for i,R in enumerate(refs)])):.3f}  ({time.time()-t00:.0f}s)")
    T = S.truth(n_boot, rng)
    Th = S.theory(True)
    Th_nbar = S.theory(False)
    print(f"[truth] B={n_boot}  ({time.time()-t00:.0f}s)")

    out["fit"] = dict(r2=r2(Th, T), ratio=float(np.median(T / (Th + 1e-12))),
                      r2_within=r2(within(Th, G), within(T, G)),
                      r2_nbar=r2(Th_nbar, T))
    print(f"\n[1] tab:fit   R^2={out['fit']['r2']:.3f}  median T/T̂={out['fit']['ratio']:.3f}"
          f"  (within {out['fit']['r2_within']:.3f}, n̄ 대조 {out['fit']['r2_nbar']:.3f})")

    print("\n[2] tab:groups")
    rows = []
    for g in groups:
        m = G == g
        rho = float(spearmanr(S.r[m], S.sigma_t[m]).statistic)
        aed = alloc_eff_dim(S, refs, m)
        rows.append(dict(group=g, rho_r_sigma=rho, alloc_eff_dim=aed,
                         mean_sigma_t=float(S.sigma_t[m].mean())))
        print(f"    {g:14} rho(r,sigma_t)={rho:+.3f}   alloc.eff.dim={aed:6.1f}   "
              f"mean sigma_t={S.sigma_t[m].mean():.3f}")
    pooled = float(spearmanr(S.r, S.sigma_t).statistic)
    out["groups_table"] = dict(rows=rows, pooled_rho=pooled)
    print(f"    {'pooled':14} rho(r,sigma_t)={pooled:+.3f}")

    print("\n[3] tab:eleven  (group 내 중심화)")
    M = S.metrics()
    el, agree = [], 0
    print(f"    {'score':>10} {'rho(M,s_t)':>11} {'rho(M,T̂)':>10} {'rho(M,T)':>10}  rule")
    for nm in SCORES11:
        v = M[nm]
        a = float(spearmanr(within(v, G), within(S.sigma_t, G)).statistic)
        b = float(spearmanr(within(v, G), within(Th, G)).statistic)
        c = float(spearmanr(within(v, G), within(T, G)).statistic)
        ok = (b > 0) == (c > 0); agree += ok
        el.append(dict(score=nm, rho_sigma_t=a, rho_that=b, rho_T=c, rule=bool(ok)))
        print(f"    {nm:>10} {a:+11.3f} {b:+10.3f} {c:+10.3f}  {'O' if ok else '** X **'}")
    n_neg = sum(1 for e in el if e["rho_T"] < 0)
    out["eleven"] = dict(rows=el, rule_ok=int(agree), n_negative=int(n_neg))
    print(f"    -> eq:rule {agree}/11,  rho(M,T)<0 인 점수 {n_neg}/11")

    print("\n[4] tab:grid  (부호 일치 / 11, 이 설정 단독)")
    sd_D = S.sigma_w / np.sqrt(S.N_all)
    pen_unit = DP.rect_gauss_var(S.tau - S.d_glob, sd_D)
    grid = {}
    for al in ALPHAS:
        row = []
        for lam in LAMBDAS:
            pred = S.sigma_t ** 2 / (S.n_assigned ** al) + (lam ** 2) * pen_unit
            k = sum(((spearmanr(within(M[nm], G), within(pred, G)).statistic > 0) ==
                     (spearmanr(within(M[nm], G), within(T, G)).statistic > 0))
                    for nm in SCORES11)
            row.append(int(k))
        grid[str(al)] = row
        print(f"    a={al:<5}" + "".join(f"{x:>6}" for x in row))
    out["grid"] = dict(alphas=ALPHAS, lambdas=LAMBDAS, table=grid)

    print(f"\n[5] tab:worse  ({n_seeds} seeds, pooled/uncentered)")
    seed_rows = []
    for s in range(n_seeds):
        rs = np.random.default_rng(s)
        gs, rf = build(repr_key, rs, bench)
        Qs = np.vstack([gs[g] for g in gs])
        Ss = Space200(Qs, rf, DP.CFG)
        Ts = Ss.truth(n_boot, rs)
        Ms = Ss.metrics(); Ths = Ss.theory(True)
        rand = float(np.mean([aurc([Ts[u <= np.percentile(u, c*100)].mean() for c in COVS], COVS)
                              for u in (rs.random(len(Qs)) for _ in range(20))]))
        rec = {"seed": s, "meanT": float(Ts.mean()), "rand": rand,
               "level_analytic": float(0.5 * Ts.mean())}
        for nm in FLAGS5:
            f = Ms[nm]
            a = aurc([Ts[f <= np.percentile(f, c*100)].mean() for c in COVS], COVS)
            rec[nm] = dict(aurc=float(a), delta=float(a/rand - 1) * 100,
                           rho_that=float(spearmanr(Ms[nm], Ths).statistic))
        seed_rows.append(rec)
        print(f"    seed {s}: meanT={Ts.mean():.4f} rand={rand:.4f} " +
              " ".join(f"{nm}={rec[nm]['delta']:+.2f}%" for nm in FLAGS5))
    out["worse"] = dict(seeds=seed_rows)

    print(f"\n{'score':>10} {'rho(M,T̂)':>10} {'AURC':>9} {'Delta%':>9} {'95%CI half':>11} {'sign':>6}")
    summ = {}
    for nm in FLAGS5:
        d = np.array([r[nm]["delta"] for r in seed_rows])
        a = np.array([r[nm]["aurc"] for r in seed_rows])
        rh = np.array([r[nm]["rho_that"] for r in seed_rows])
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        sg = f"{int((d > 0).sum() if d.mean() > 0 else (d < 0).sum())}/{len(d)}"
        summ[nm] = dict(rho_that=float(rh.mean()), aurc=float(a.mean()),
                        delta=float(d.mean()), ci_half=float(half), sign=sg,
                        delta_sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0)
        print(f"{nm:>10} {rh.mean():+10.3f} {a.mean():9.4f} {d.mean():+9.2f} {half:11.2f} {sg:>6}")
    rr = np.array([r["rand"] for r in seed_rows])
    summ["random"] = dict(aurc=float(rr.mean()),
                          level_analytic=float(np.mean([r["level_analytic"] for r in seed_rows])))
    out["worse"]["summary"] = summ
    print(f"{'random':>10} {'':>10} {rr.mean():9.4f}   (예측 ½T̄ = {summ['random']['level_analytic']:.4f})")

    p = f"{bench}_probe_{repr_key}.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\n[saved] {p}   총 {(time.time()-t00)/60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", default="dino")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--boot", type=int, default=DP.CFG["n_boot"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--bench", choices=["in200", "in1k"], default="in200")
    a = ap.parse_args()
    selfcheck() if a.selfcheck else main(a.repr, a.seeds, a.boot, a.bench)
