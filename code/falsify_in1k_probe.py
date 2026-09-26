import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, time
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anisotropy_probe as AP

P_LIST = [1, 5, 10, 20, 50, 100]
N_BINS = 20


def main(setting="in1k", out=None):
    out = out or f"falsify_{setting}.json"
    import derma_probe as DP
    from numpy.random import default_rng
    t0 = time.time()
    if setting == "in1k":
        import in200_probe as IP
        rng = default_rng(DP.CFG["seed"])
        groups, refs = IP.build("dino", rng, bench="in1k")
    else:
        import exponent_probe as EP, odin_real_probe as ORP
        ORDER = ["c100", "c10", "derma400", "derma200", "derma80"]
        ALL = {"c100": ("cifar", "c100"), "c10": ("cifar", "c10"),
               "derma400": ("derma", 400), "derma200": ("derma", 200),
               "derma80": ("derma", 80)}
        rng = default_rng(EP.CFG["seed"])
        for prev in ORDER:
            if prev == setting:
                break
            kind, arg = ALL[prev]
            if kind == "cifar":
                _, _, pr, _ = ORP.build_cifar_idx(rng, arg)
            else:
                _, _, pr = ORP.build_derma_idx(rng, arg)
            ORP.burn_truth(rng, pr, EP.CFG["n_boot"])
            print(f"  [stream] {prev} 소비 (C={len(pr)})", flush=True)
        kind, arg = ALL[setting]
        groups, refs = (EP.build_cifar(rng, arg) if kind == "cifar"
                        else EP.build_derma(rng, arg))
    Q = np.vstack([groups[k] for k in groups])
    mus = np.vstack([R.mean(0) for R in refs])
    nc = cdist(Q, mus).argmin(1)
    N = len(Q)
    print(f"[{setting}] Q={Q.shape} C={len(refs)}  build {time.time()-t0:.0f}s")

    r = np.empty(N); st = np.empty(N)
    alpha = {P: np.empty(N) for P in P_LIST}
    PRs = []
    t1 = time.time()
    for j, c in enumerate(np.unique(nc)):
        m = nc == c
        W = refs[c] - mus[c]
        Cov = (W.T @ W) / max(len(W) - 1, 1)
        lam, V = np.linalg.eigh(Cov)
        o = np.argsort(lam)[::-1]
        lam, V = lam[o], V[:, o]
        PRs.append(AP.participation_ratio(np.maximum(lam, 0)))
        v = Q[m] - mus[c]
        rr = np.linalg.norm(v, axis=1)
        u = v / (rr[:, None] + 1e-12)
        r[m] = rr
        st[m] = np.sqrt(np.maximum((u @ Cov * u).sum(1), 0))
        proj2 = (u @ V) ** 2
        cs = np.cumsum(proj2, axis=1)
        for P in P_LIST:
            alpha[P][m] = cs[:, min(P, cs.shape[1]) - 1]
        if (j + 1) % 200 == 0:
            print(f"    {j+1} classes  {time.time()-t1:.0f}s", flush=True)

    rho0 = float(spearmanr(r, st).statistic)
    res = dict(setting=setting, n_query=N, C=len(refs),
               rho_r_sigma_t=rho0, PR_median=float(np.median(PRs)), alpha={})
    print(f"\n  rho(r, sigma_t) = {rho0:+.4f}   PR median = {np.median(PRs):.1f}")
    print(f"  {'P':>5}{'rho(r,alpha)':>14}{'rho(alpha,st)':>15}{'partial rho':>13}")
    for P in P_LIST:
        a = alpha[P]
        d = dict(rho_r_alpha=float(spearmanr(r, a).statistic),
                 rho_alpha_st=float(spearmanr(a, st).statistic),
                 partial_rho_r_st=float(AP.partial_rho(r, st, a, N_BINS)))
        res["alpha"][P] = d
        print(f"  {P:>5}{d['rho_r_alpha']:>+14.4f}{d['rho_alpha_st']:>+15.4f}"
              f"{d['partial_rho_r_st']:>+13.4f}")
    json.dump(res, open(out, "w"), indent=2)
    print(f"  -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--setting", default="in1k")
    main(ap.parse_args().setting)
