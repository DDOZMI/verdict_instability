import os, sys, json, time, argparse
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derma_probe as DP

N_QUERY_GRP = 800
N_QUERY_ID  = 20000
RATIOS      = [1.0, 6.1, 20.6, 58.7]
N_PERM      = 5
POOL_MAX    = 500


class LightSpace:

    def __init__(self, Q, refs, cfg):
        self.cfg, self.Q, self.refs = cfg, Q, refs
        self.mus = np.vstack([R.mean(0) for R in refs])
        self.n_c = np.array([len(R) for R in refs])
        self.C = len(refs)
        self.N_all = int(self.n_c.sum())
        self.mu_g = np.average(self.mus, axis=0, weights=self.n_c)

        Dc = cdist(Q, self.mus)
        self.nc = Dc.argmin(1)
        self.d_cls = Dc[np.arange(len(Q)), self.nc]
        del Dc
        self.d_glob = np.linalg.norm(Q - self.mu_g, axis=1)
        self.n_assigned = self.n_c[self.nc]

        self.sigma_t = np.empty(len(Q)); self.r = np.empty(len(Q))
        for c in np.unique(self.nc):
            m = self.nc == c
            W = refs[c] - self.mus[c]
            v = Q[m] - self.mus[c]
            rr = np.linalg.norm(v, axis=1)
            u = v / (rr[:, None] + 1e-12)
            self.r[m] = rr
            self.sigma_t[m] = (u @ W.T).std(axis=1)

        u_g = (Q - self.mu_g) / (self.d_glob[:, None] + 1e-12)
        vw = np.zeros(len(Q))
        for c in range(self.C):
            Wc = refs[c] - self.mus[c]
            vw += (u_g @ Wc.T).var(axis=1)
        self.sigma_w = np.sqrt(vw / self.C)

        Xc = np.vstack(refs) - self.mu_g
        self.tau = float(np.percentile(np.linalg.norm(Xc, axis=1), cfg["hinge_tau_q"]))

    def theory(self, use_nc=True, with_pen=True):
        n = self.n_assigned if use_nc else np.full(len(self.Q), self.n_c.mean())
        v = self.sigma_t ** 2 / n
        if with_pen:
            sd = self.sigma_w / np.sqrt(self.N_all)
            v = v + (self.cfg["hinge_lam"] ** 2) * DP.rect_gauss_var(self.tau - self.d_glob, sd)
        return np.sqrt(v)

    def truth(self, n_boot, rng, fixed_argmin=False, with_pen=True, quiet=False):
        B = np.zeros((n_boot, len(self.Q)))
        idx = self.nc
        for b in range(n_boot):
            rb = [R[rng.integers(0, len(R), len(R))] for R in self.refs]
            mb = np.vstack([R.mean(0) for R in rb])
            if fixed_argmin:
                dc = np.linalg.norm(self.Q - mb[idx], axis=1)
            else:
                dc = cdist(self.Q, mb).min(1)
            s = dc
            if with_pen:
                mg = np.average(mb, axis=0, weights=self.n_c)
                dg = np.linalg.norm(self.Q - mg, axis=1)
                s = s + self.cfg["hinge_lam"] * np.maximum(0.0, self.tau - dg)
            B[b] = s
            if not quiet:
                print(f"      boot {b+1}/{n_boot}", end="\r")
        if not quiet:
            print()
        return B.std(0)


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def r2(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    return float(LinearRegression().fit(zs(p[m])[:, None], zs(t[m]))
                 .score(zs(p[m])[:, None], zs(t[m])))


def load(bench, repr_key="dino"):
    cd = f"./feat_cache_{bench}_{repr_key}"
    ftr = np.load(f"{cd}/{bench}_feat_train.npy").astype(np.float64)
    ytr = np.load(f"{cd}/{bench}_lab_train.npy")
    fte = np.load(f"{cd}/{bench}_feat_test.npy").astype(np.float64)
    meta = json.load(open(f"{cd}/BENCH_META.json"))
    C = int(ytr.max()) + 1
    pool = [ftr[ytr == c] for c in range(C)]
    return pool, fte, meta, cd


def draw_counts(C, ratio, rng):
    if ratio <= 1.0:
        return np.full(C, POOL_MAX, dtype=int)
    lo = POOL_MAX / ratio
    n = np.exp(rng.uniform(np.log(lo), np.log(POOL_MAX), size=C))
    return np.clip(np.round(n), 5, POOL_MAX).astype(int)


def make_refs(pool, counts, rng):
    out = []
    for c, n in enumerate(counts):
        X = pool[c]
        out.append(X if len(X) <= n else X[rng.choice(len(X), n, replace=False)])
    return out


def query_groups(fte, meta, cd, bench, rng, n_grp):
    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]
    g = {"in": sub(fte, n_grp)}
    p = f"{cd}/{bench}_feat_near.npy"
    if os.path.exists(p):
        g["near"] = sub(np.load(p).astype(np.float64), n_grp)
    for s in meta["far_sets"]:
        g[s] = sub(np.load(f"{cd}/{bench}_feat_far_{s}.npy").astype(np.float64), n_grp)
    return np.vstack([g[k] for k in g]), {k: len(v) for k, v in g.items()}


def selfcheck():
    import in200_probe as IP
    rng = np.random.default_rng(0)
    groups, refs = IP.build("dino", rng, "in200")
    Q = np.vstack([groups[g] for g in groups])
    print(f"  ImageNet-200: C={len(refs)} n_c={len(refs[0])} Q={len(Q)}")
    A = IP.Space200(Q, refs, DP.CFG)
    B = LightSpace(Q, refs, DP.CFG)
    worst = 0.0
    for k in ["d_cls", "d_glob", "r", "sigma_t", "sigma_w", "nc", "n_assigned", "tau"]:
        a, b = np.asarray(getattr(A, k), float), np.asarray(getattr(B, k), float)
        d = float(np.abs(a - b).max()); worst = max(worst, d)
        print(f"    {k:12} max|diff| = {d:.3e}")
    for tag, ka in [("theory(n_c)", True), ("theory(n̄)", False)]:
        d = float(np.abs(A.theory(ka) - B.theory(ka)).max()); worst = max(worst, d)
        print(f"    {tag:12} max|diff| = {d:.3e}")
    ta = A.truth(20, np.random.default_rng(7))
    tb = B.truth(20, np.random.default_rng(7), quiet=True)
    d = float(np.abs(ta - tb).max()); worst = max(worst, d)
    print(f"    {'truth(B=20)':12} max|diff| = {d:.3e}")
    print(f"\n  [selfcheck] 최대 편차 {worst:.3e}  ->  {'통과' if worst < 1e-8 else '** 실패 **'}")


def main(bench, repr_key, n_boot):
    t0 = time.time()
    pool, fte, meta, cd = load(bench, repr_key)
    C = len(pool)
    OUT = {"bench": bench, "repr": repr_key, "C": C, "n_boot": n_boot,
           "pool_max": POOL_MAX, "n_query_grp": N_QUERY_GRP, "n_query_id": N_QUERY_ID}
    print(f"[load] C={C}  pool={len(pool[0])}/class  ({time.time()-t0:.0f}s)")

    qrng = np.random.default_rng(0)
    Qg, gsizes = query_groups(fte, meta, cd, bench, qrng, N_QUERY_GRP)
    Qid = fte if len(fte) <= N_QUERY_ID else fte[
        np.random.default_rng(1).choice(len(fte), N_QUERY_ID, replace=False)]
    print(f"[query] A/B: {len(Qg)} {gsizes}   C/D: ID {len(Qid)}")
    OUT["groups"] = gsizes

    def fit(counts, Q, seed, tag):
        rng = np.random.default_rng(seed)
        refs = make_refs(pool, counts, rng)
        S = LightSpace(Q, refs, DP.CFG)
        T = S.truth(n_boot, rng, quiet=True)
        print(f"    [{tag}] n_c {counts.min()}~{counts.max()} "
              f"({counts.max()/counts.min():.1f}x)  ({time.time()-t0:.0f}s)")
        return S, T

    print("\n[A] 불균형 스윕 — n_c 무작위 log-uniform 배정")
    A_rows = []
    counts_by_ratio = {}
    for R in RATIOS:
        cnt = draw_counts(C, R, np.random.default_rng(100 + int(R * 10)))
        counts_by_ratio[R] = cnt
        S, T = fit(cnt, Qg, 200 + int(R * 10), f"ratio {R}")
        a, b = r2(S.theory(True), T), r2(S.theory(False), T)
        A_rows.append(dict(target_ratio=R, actual_ratio=float(cnt.max() / cnt.min()),
                           n_min=int(cnt.min()), n_max=int(cnt.max()),
                           n_mean=float(cnt.mean()), r2_nc=a, r2_nbar=b, gain=a - b,
                           ratio_med=float(np.median(T / (S.theory(True) + 1e-12)))))
        print(f"      R²(n_c)={a:.3f}  R²(n̄)={b:.3f}  gain={a-b:+.3f}  "
              f"median T/T̂={A_rows[-1]['ratio_med']:.3f}")
    OUT["A_sweep"] = A_rows

    print(f"\n[B] 치환 불변성 — 58.7x 카운트 다중집합 고정, 클래스 배정만 {N_PERM} 번 치환")
    base = counts_by_ratio[58.7]
    B_rows = []
    for p in range(N_PERM):
        cnt = base[np.random.default_rng(300 + p).permutation(C)]
        S, T = fit(cnt, Qg, 400 + p, f"perm {p}")
        a, b = r2(S.theory(True), T), r2(S.theory(False), T)
        B_rows.append(dict(perm=p, r2_nc=a, r2_nbar=b, gain=a - b))
        print(f"      R²(n_c)={a:.3f}  R²(n̄)={b:.3f}  gain={a-b:+.3f}")
    g = np.array([r["gain"] for r in B_rows])
    OUT["B_perm"] = dict(rows=B_rows, gain_mean=float(g.mean()), gain_sd=float(g.std(ddof=1)),
                         gain_min=float(g.min()), gain_max=float(g.max()),
                         counts_multiset_fixed=True)
    print(f"    -> gain {g.mean():.3f} ± {g.std(ddof=1):.3f}  범위 [{g.min():.3f}, {g.max():.3f}]")

    print("\n[C] 동일 클래스, 두 카운트 — Derma 에서 불가능한 검정")
    C_pairs = [(400, 50), (400, 100), (500, 25)]
    C_rows = []
    for hi, lo in C_pairs:
        Sh, Th = fit(np.full(C, hi), Qid, 500 + hi, f"n_c={hi}")
        Sl, Tl = fit(np.full(C, lo), Qid, 500 + hi, f"n_c={lo}")
        rat = []
        for c in range(C):
            mh, ml = Sh.nc == c, Sl.nc == c
            if mh.sum() >= 5 and ml.sum() >= 5:
                rat.append(Tl[ml].mean() / (Th[mh].mean() + 1e-12))
        rat = np.array(rat)
        pred = np.sqrt(hi / lo)
        C_rows.append(dict(hi=hi, lo=lo, predicted=float(pred), n_classes=int(len(rat)),
                           median=float(np.median(rat)), mean=float(rat.mean()),
                           sd=float(rat.std(ddof=1)),
                           q10=float(np.quantile(rat, .1)), q90=float(np.quantile(rat, .9)),
                           rel_err_median=float(np.median(rat) / pred - 1)))
        print(f"      n_c {hi}->{lo}: 예측 {pred:.3f}  실측 중앙값 {np.median(rat):.3f} "
              f"(±sd {rat.std(ddof=1):.3f}, 10~90% {np.quantile(rat,.1):.3f}~{np.quantile(rat,.9):.3f}, "
              f"{len(rat)} 클래스)  상대오차 {np.median(rat)/pred-1:+.1%}")
    OUT["C_within_class"] = C_rows

    print("\n[D] gamma_c = T̄_c sqrt(n_c) / sigma_t,c  (58.7x 무작위 배정, ID 질의)")
    cnt = counts_by_ratio[58.7]
    rng = np.random.default_rng(900)
    refs = make_refs(pool, cnt, rng)
    S = LightSpace(Qid, refs, DP.CFG)
    Tf = S.truth(n_boot, rng, fixed_argmin=True, with_pen=False, quiet=True)
    Tp = S.truth(n_boot, np.random.default_rng(900), with_pen=False, quiet=True)
    edges = np.array([5, 15, 30, 60, 120, 250, 501])
    D_rows = []
    for i in range(len(edges) - 1):
        cls = np.where((cnt >= edges[i]) & (cnt < edges[i + 1]))[0]
        m = np.isin(S.nc, cls)
        if m.sum() < 20:
            continue
        nn = np.sqrt(S.n_assigned[m])
        D_rows.append(dict(bin=f"{edges[i]}-{edges[i+1]-1}", n_classes=int(len(cls)),
                           n_query=int(m.sum()), mean_nc=float(cnt[cls].mean()),
                           gamma=float((Tp[m] * nn / (S.sigma_t[m] + 1e-12)).mean()),
                           gamma_fixed=float((Tf[m] * nn / (S.sigma_t[m] + 1e-12)).mean())))
        d = D_rows[-1]
        print(f"      n_c {d['bin']:>8}  클래스 {d['n_classes']:>4}  질의 {d['n_query']:>6}  "
              f"gamma={d['gamma']:.3f}   argmin 고정 gamma={d['gamma_fixed']:.3f}")
    OUT["D_gamma"] = D_rows

    p = f"nc_control_{bench}_{repr_key}.json"
    json.dump(OUT, open(p, "w"), indent=2, default=float)
    print(f"\n[saved] {p}   총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["in200", "in1k"], default="in1k")
    ap.add_argument("--repr", default="dino")
    ap.add_argument("--boot", type=int, default=DP.CFG["n_boot"])
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck() if a.selfcheck else main(a.bench, a.repr, a.boot)
