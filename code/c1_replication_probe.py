import os, sys, json, time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derma_probe as DP

COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
FLAGS = ["knn_std", "lid", "energy", "msp", "maha"]
N_SEEDS = int(os.environ.get("C1_SEEDS", 10))
N_RAND = 20
N_BOOT = int(os.environ.get("C1_BOOT", DP.CFG["n_boot"]))
OUT_JSON = os.environ.get("C1_OUT", "c1_replication_probe.json")

_trap = getattr(np, "trapezoid", getattr(np, "trapz", None))


def auc(vals):
    o = np.argsort(COVS)
    return float(_trap(np.asarray(vals)[o], np.asarray(COVS)[o]))


def aurc(f, T):
    return auc([float(T[f <= np.percentile(f, c * 100)].mean()) for c in COVS])


def truth_fast(S, n_boot, rng):
    B = np.zeros((n_boot, len(S.Q)))
    for b in range(n_boot):
        rb = [R[rng.integers(0, len(R), len(R))] for R in S.refs]
        mb = np.vstack([R.mean(0) for R in rb])
        mg = np.average(mb, axis=0, weights=S.n_c)
        dc = cdist(S.Q, mb).min(1)
        dg = np.linalg.norm(S.Q - mg, axis=1)
        B[b] = dc + S.cfg["hinge_lam"] * np.maximum(0.0, S.tau - dg)
    return B.std(0)


def build_cifar100(rng, n_ref=400):
    cd = DP.CFG["cache_dir"]

    def p(k, s):
        return os.path.join(cd, f"dino_{k}_{s}.npy")
    ftr, ytr = np.load(p("feat", "train")).astype(float), np.load(p("lab", "train"))
    fte, yte = np.load(p("feat", "test")).astype(float), np.load(p("lab", "test"))
    held, hout = list(range(50)), list(range(50, 100))
    refs = []
    for c in held:
        X = ftr[ytr == c]
        if len(X) > n_ref:
            X = X[rng.choice(len(X), n_ref, replace=False)]
        refs.append(X)

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    m_in = np.where(np.isin(yte, held))[0]
    if len(m_in) > 800:
        m_in = rng.choice(m_in, 800, replace=False)
    groups = {"in": fte[m_in], "near": sub(fte[np.isin(yte, hout)], 800)}
    for s in ["svhn", "dtd_lr32", "lsun", "isun", "places365_lr32", "cifar10"]:
        f = os.path.join(cd, f"dino_feat_far_{s}.npy")
        if os.path.exists(f):
            groups[s] = sub(np.load(f).astype(float), 800)
    return groups, refs


def build_derma(rng):
    ftr, ytr = DP.get("train")
    fte, yte = DP.get("test")
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}
    held = sorted([c for c, n in cnt.items() if n >= DP.CFG["min_n"]])
    near = sorted([c for c, n in cnt.items() if n < DP.CFG["min_n"]])
    refs = [ftr[ytr == c] for c in held]

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = DP.CFG["n_query"]
    groups = {"in": sub(fte[np.isin(yte, held)], nq),
              "near": sub(fte[np.isin(yte, near)], nq)}
    for s in DP.CFG["ood_far"]:
        F = DP.load_far(s)
        if F is not None:
            groups[s] = sub(F, nq)
    return groups, refs


def one_seed(builder, seed):
    rng = np.random.default_rng(seed)
    groups, refs = builder(rng)
    Q = np.vstack([groups[g] for g in groups])
    S = DP.Space(Q, refs, dict(DP.CFG))
    T = truth_fast(S, N_BOOT, rng)
    That = S.theory(use_nc=True)
    M = dict(S.metrics())

    a = {fn: aurc(M[fn], T) for fn in FLAGS}
    r_hat = {fn: float(spearmanr(M[fn], That).statistic) for fn in FLAGS}
    r_true = {fn: float(spearmanr(M[fn], T).statistic) for fn in FLAGS}
    rand = [aurc(rng.random(len(Q)), T) for _ in range(N_RAND)]
    return dict(seed=seed, n_q=int(len(Q)), C=len(refs), n_boot=N_BOOT,
                meanT=float(T.mean()),
                aurc=a, rho_That=r_hat, rho_T=r_true,
                rho_That_T=float(spearmanr(That, T).statistic),
                aurc_random=float(np.mean(rand)),
                aurc_random_draws=[float(v) for v in rand],
                aurc_random_analytic=float(0.5 * T.mean()))


def summarize(rows, tag):
    print("\n" + "=" * 78)
    print(f"[{tag}]  seeds={len(rows)}  n_q={rows[0]['n_q']}  C={rows[0]['C']}")
    print("=" * 78)

    lvl = np.array([r["aurc_random"] for r in rows])
    ana = np.array([r["aurc_random_analytic"] for r in rows])
    print(f"  random AURC  mean={lvl.mean():.4f}  sd={lvl.std(ddof=1):.4f}  "
          f"CV={100*lvl.std(ddof=1)/lvl.mean():.2f}%   range={lvl.min():.4f}-{lvl.max():.4f}")
    print(f"  0.5*meanT    mean={ana.mean():.4f}  (random baseline 의 해석적 값)")
    print(f"  ** (a) run-to-run LEVEL 산포 = 지적자가 말한 'drift' 의 정체 **")
    print(f"     seed 간 절대 AURC 최대 격차 = {100*(lvl.max()/lvl.min()-1):+.2f}%")

    print(f"\n  ** (b) 대응 대비 (M - random)/random, seed 간 **")
    print(f"  {'score':>9} {'mean%':>8} {'sd%':>7} {'95% CI':>18} {'sign':>7} {'min%':>8} {'max%':>8}")
    res = {}
    for fn in FLAGS:
        g = np.array([100 * (r["aurc"][fn] / r["aurc_random"] - 1) for r in rows])
        m, sd = g.mean(), g.std(ddof=1)
        se = sd / np.sqrt(len(g))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        nsign = int(np.sum(np.sign(g) == np.sign(m)))
        print(f"  {fn:>9} {m:+8.2f} {sd:7.2f}  [{lo:+7.2f},{hi:+7.2f}] "
              f"{nsign:>3}/{len(g)} {g.min():+8.2f} {g.max():+8.2f}")
        res[fn] = dict(mean=float(m), sd=float(sd), ci=[float(lo), float(hi)],
                       sign_agree=nsign, n=len(g), per_seed=[float(v) for v in g])

    print(f"\n  ** rho(M,That) / rho(M,T) — 예측자의 시드 안정성 **")
    print(f"  {'score':>9} {'rho(M,That)':>18} {'sign':>7} {'rho(M,T)':>18} {'sign':>7} {'rule ok':>8}")
    rho = {}
    for fn in FLAGS:
        rh = np.array([r["rho_That"][fn] for r in rows])
        rt = np.array([r["rho_T"][fn] for r in rows])
        sh = int(np.sum(np.sign(rh) == np.sign(rh.mean())))
        st = int(np.sum(np.sign(rt) == np.sign(rt.mean())))
        ok = int(np.sum(np.sign(rh) == np.sign(rt)))
        print(f"  {fn:>9} {rh.mean():+9.3f}+-{rh.std(ddof=1):.3f} {sh:>3}/{len(rh)} "
              f"{rt.mean():+9.3f}+-{rt.std(ddof=1):.3f} {st:>3}/{len(rt)} {ok:>4}/{len(rh)}")
        rho[fn] = dict(That_mean=float(rh.mean()), That_sd=float(rh.std(ddof=1)),
                       That_sign=sh, T_mean=float(rt.mean()), T_sd=float(rt.std(ddof=1)),
                       T_sign=st, rule_agree=ok, n=len(rh),
                       That_per_seed=[float(v) for v in rh],
                       T_per_seed=[float(v) for v in rt])

    print(f"\n  ** LEVEL 산포 대 GAP 산포 **")
    lev_cv = 100 * lvl.std(ddof=1) / lvl.mean()
    for fn in FLAGS:
        print(f"    {fn:>9}: |gap|={abs(res[fn]['mean']):.2f}%  gap_sd={res[fn]['sd']:.2f}%  "
              f"|gap|/gap_sd = {abs(res[fn]['mean'])/max(res[fn]['sd'],1e-9):.1f}")
    print(f"    (level CV = {lev_cv:.2f}% — 모든 행에 공통으로 곱해지므로 gap 에는 들어가지 않음)")
    return dict(tag=tag, level_cv=float(lev_cv),
                level_mean=float(lvl.mean()), level_sd=float(lvl.std(ddof=1)),
                level_analytic=float(ana.mean()),
                gaps=res, rho=rho, rows=rows)


def save(OUT):
    old = {}
    if os.path.exists(OUT_JSON):
        try:
            old = json.load(open(OUT_JSON))
        except (ValueError, OSError):
            old = {}
    old.update(OUT)
    json.dump(old, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\n[saved] {OUT_JSON}  (keys: {', '.join(sorted(old))})")


def main():
    OUT = {}
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"B = {N_BOOT} bootstrap replicates,  seeds = {N_SEEDS}")

    if which in ("both", "cifar"):
        rows = []
        for s in range(N_SEEDS):
            t0 = time.time()
            rows.append(one_seed(build_cifar100, s))
            print(f"  [cifar seed {s}] meanT={rows[-1]['meanT']:.4f} "
                  f"rand={rows[-1]['aurc_random']:.4f} "
                  f"energy={rows[-1]['aurc']['energy']:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        OUT["cifar100"] = summarize(rows, "CIFAR-100  n_ref=400")
        save(OUT)

    if which in ("both", "derma"):
        rows = []
        for s in range(N_SEEDS):
            t0 = time.time()
            rows.append(one_seed(build_derma, s))
            print(f"  [derma seed {s}] meanT={rows[-1]['meanT']:.4f} "
                  f"rand={rows[-1]['aurc_random']:.4f} "
                  f"maha={rows[-1]['aurc']['maha']:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        OUT["derma"] = summarize(rows, "DermaMNIST C=3")

    save(OUT)


if __name__ == "__main__":
    main()
