import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, time, argparse
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SETTINGS = {
    "c100":     dict(kind="cifar", which="c100"),
    "c10":      dict(kind="cifar", which="c10"),
    "derma400": dict(kind="derma", min_n=400),
    "derma200": dict(kind="derma", min_n=200),
    "derma80":  dict(kind="derma", min_n=80),
}
ORDER = ["c100", "c10", "derma400", "derma200", "derma80"]


def g_fun(a):
    pb = norm.sf(a)
    ph = norm.pdf(a)
    return a * a * pb - a * ph - (a * pb - ph) ** 2


def var_min(s1, s2, delta):
    th2 = s1 ** 2 + s2 ** 2
    th = np.sqrt(th2)
    a = delta / np.maximum(th, 1e-300)
    return s1 ** 2 * norm.cdf(a) + s2 ** 2 * norm.sf(a) + th2 * g_fun(a), a


def two_nearest(S, refs):
    from scipy.spatial.distance import cdist
    Q, mus = S.Q, S.mus
    Dc = cdist(Q, mus)
    idx = np.argsort(Dc, axis=1)[:, :2]
    c1, c2 = idx[:, 0], idx[:, 1]
    d1 = Dc[np.arange(len(Q)), c1]
    d2 = Dc[np.arange(len(Q)), c2]
    def sig(cs):
        out = np.empty(len(Q))
        for c in np.unique(cs):
            m = cs == c
            W = refs[c] - S.mus[c]
            v = Q[m] - S.mus[c]
            u = v / (np.linalg.norm(v, axis=1)[:, None] + 1e-12)
            out[m] = (u @ W.T).std(axis=1)
        return out
    s1 = sig(c1) / np.sqrt(S.n_c[c1])
    s2 = sig(c2) / np.sqrt(S.n_c[c2])
    return c1, c2, d1, d2, s1, s2


def main(setting, n_boot, out):
    import gpu_logreg; gpu_logreg.install()
    import derma_probe as DP
    import exponent_probe as EP
    import odin_real_probe as ORP
    from numpy.random import default_rng

    if setting == "in1k":
        import in200_probe as IP, space_lowmem as SL
        rng = default_rng(DP.CFG["seed"])
        groups, refs = IP.build("dino", rng, bench="in1k")
        cfg, SpaceCls = DP.CFG, SL.Space200LowMem
        return _run(setting, n_boot, out, groups, refs, cfg, SpaceCls, rng, DP)
    SP = SETTINGS[setting]
    rng = default_rng(EP.CFG["seed"])
    for prev in ORDER:
        if prev == setting:
            break
        P = SETTINGS[prev]
        if P["kind"] == "cifar":
            _, _, pr, _ = ORP.build_cifar_idx(rng, P["which"])
        else:
            _, _, pr = ORP.build_derma_idx(rng, P["min_n"])
        ORP.burn_truth(rng, pr, EP.CFG["n_boot"])
    cfg = dict(DP.CFG)
    if SP["kind"] == "cifar":
        groups, refs = EP.build_cifar(rng, SP["which"])
    else:
        groups, refs = EP.build_derma(rng, SP["min_n"]); cfg["min_n"] = SP["min_n"]
    return _run(setting, n_boot, out, groups, refs, cfg, DP.Space, rng, DP)


def _run(setting, n_boot, out, groups, refs, cfg, SpaceCls, rng, DP):
    import numpy as np
    Q = np.vstack([groups[k] for k in groups])
    print(f"[{setting}] Q={Q.shape} C={len(refs)} n_c={sorted(set(len(R) for R in refs))[:4]}")

    S = SpaceCls(Q, refs, cfg)
    c1, c2, d1, d2, s1, s2 = two_nearest(S, refs)
    v_corr, alpha = var_min(s1, s2, d2 - d1)
    v_plain = s1 ** 2

    from derma_probe import rect_gauss_var
    sd_D = S.sigma_w / np.sqrt(S.N_all)
    v_pen = (cfg["hinge_lam"] ** 2) * rect_gauss_var(S.tau - S.d_glob, sd_D)
    That_plain = np.sqrt(v_plain + v_pen)
    That_corr = np.sqrt(v_corr + v_pen)

    print(f"  부트스트랩 B={n_boot} ...")
    B = np.zeros((n_boot, len(Q))); SW = np.zeros(len(Q))
    from scipy.spatial.distance import cdist
    t0 = time.time()
    for b in range(n_boot):
        rb = [R[rng.integers(0, len(R), len(R))] for R in refs]
        mb = np.vstack([R.mean(0) for R in rb])
        mg = np.vstack(rb).mean(0)
        Db = cdist(Q, mb)
        arg = Db.argmin(1)
        SW += (arg != c1)
        dg = np.linalg.norm(Q - mg, axis=1)
        B[b] = Db.min(1) + cfg["hinge_lam"] * np.maximum(0.0, S.tau - dg)
        if (b + 1) % 50 == 0:
            print(f"    {b+1}/{n_boot}  {time.time()-t0:.0f}s", flush=True)
    T = B.std(0)
    sw_obs = SW / n_boot
    sw_pred = norm.sf(alpha)

    def r2(p, t):
        return float(np.corrcoef(p, t)[0, 1] ** 2)

    res = dict(setting=setting, n_boot=n_boot, C=len(refs), n_query=int(len(Q)),
               switch=dict(obs_mean=float(sw_obs.mean()), pred_mean=float(sw_pred.mean()),
                           corr=float(np.corrcoef(sw_obs, sw_pred)[0, 1])),
               fit=dict(r2_plain=r2(That_plain, T), r2_corr=r2(That_corr, T),
                        med_plain=float(np.median(T / That_plain)),
                        med_corr=float(np.median(T / That_corr))))
    print(f"\n  [전환율] 실측 {sw_obs.mean()*100:.1f}%  예측 {sw_pred.mean()*100:.1f}%  "
          f"질의별 상관 {res['switch']['corr']:+.3f}")
    print(f"  [적합]  R2  {res['fit']['r2_plain']:.4f} -> {res['fit']['r2_corr']:.4f}   "
          f"median T/That  {res['fit']['med_plain']:.4f} -> {res['fit']['med_corr']:.4f}")

    rows = []
    print(f"\n  {'n_c':>6}{'gamma(논문)':>13}{'gamma_corr':>12}{'전환 실측':>11}{'전환 예측':>11}")
    for c in np.unique(c1):
        m = c1 == c
        if m.sum() < 20:
            continue
        gam = float(T[m].mean() * np.sqrt(S.n_c[c]) / (S.sigma_t[m].mean() + 1e-12))
        gam_c = float(T[m].mean() / (That_corr[m].mean() + 1e-12))
        gam_p = float(T[m].mean() / (That_plain[m].mean() + 1e-12))
        rows.append(dict(n_c=int(S.n_c[c]), n_q=int(m.sum()), gamma_paper=gam,
                         gamma_plain=gam_p, gamma_corr=gam_c,
                         sw_obs=float(sw_obs[m].mean()), sw_pred=float(sw_pred[m].mean())))
        print(f"  {S.n_c[c]:>6}{gam:>13.3f}{gam_c:>12.3f}"
              f"{sw_obs[m].mean()*100:>10.1f}%{sw_pred[m].mean()*100:>10.1f}%")
    res["classes"] = rows
    json.dump(res, open(out, "w"), indent=2)
    print(f"  -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.setting, a.boot, a.out or f"min_correction_{a.setting}.json")
