import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, time
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIN_NS = [400, 200, 80]
EXPECT = {400: (0.9742029906364273, 1.0054802048363438),
          200: (0.9731083979959770, 0.9671255204324787),
          80:  (0.9225379849068377, 0.9760028890047293)}
TOL = 5e-9


def build_fixed(min_n, ftr, ytr, fte, yte, cnt, held_ref, near_ref, rng):
    import derma_probe as DP
    held_in = sorted([c for c, n in cnt.items() if n >= min_n])
    refs = [ftr[ytr == c] for c in held_in]

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = DP.CFG["n_query"]
    groups = {"in": sub(fte[np.isin(yte, held_ref)], nq)}
    if len(near_ref) > 0:
        Qn = fte[np.isin(yte, near_ref)]
        if len(Qn) > 0:
            groups["near"] = sub(Qn, nq)
    for s in DP.CFG["ood_far"]:
        F = DP.load_far(s)
        if F is not None:
            groups[s] = sub(F, nq)
    return groups, refs, held_in


def truth_and_switch(S, refs, c1, n_boot, rng):
    import derma_probe as DP
    Q, lam, tau = S.Q, S.cfg["hinge_lam"], S.tau
    B = np.zeros((n_boot, len(Q)))
    SW = np.zeros(len(Q))
    for b in range(n_boot):
        rb = [R[rng.integers(0, len(R), len(R))] for R in refs]
        mb = np.vstack([R.mean(0) for R in rb])
        mg = np.vstack(rb).mean(0)
        Dc = np.linalg.norm(Q[:, None, :] - mb[None, :, :], axis=2)
        SW += (Dc.argmin(1) != c1)
        dg = np.linalg.norm(Q - mg, axis=1)
        B[b] = Dc.min(1) + lam * np.maximum(0.0, tau - dg)
    return B.std(0), SW / n_boot


def main(n_boot, out_json):
    import gpu_logreg; gpu_logreg.install()
    import derma_probe as DP, in200_probe as IP
    import min_correction_probe as MC
    t0 = time.time()

    rng = np.random.default_rng(0)
    ftr, ytr = DP.get("train"); fte, yte = DP.get("test")
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}
    held_ref = sorted([c for c, n in cnt.items() if n >= 400])
    near_ref = sorted([c for c, n in cnt.items() if n < 400])

    res = {}
    for min_n in MIN_NS:
        groups, refs, held_in = build_fixed(min_n, ftr, ytr, fte, yte,
                                            cnt, held_ref, near_ref, rng)
        Q = np.vstack([groups[g] for g in groups])
        cfg = dict(DP.CFG); cfg["min_n"] = min_n
        S = DP.Space(Q, refs, cfg)
        C = len(refs)
        print(f"\n[min_n={min_n}] C={C} Q={Q.shape} n_c={[len(R) for R in refs]}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        c1, c2, d1, d2, s1, s2 = MC.two_nearest(S, refs)
        T, sw_obs = truth_and_switch(S, refs, c1, n_boot, rng)

        th_plain = S.theory(use_nc=True)
        r2_p = IP.r2(th_plain, T)
        med_p = float(np.median(T / (th_plain + 1e-12)))

        e_r2, e_med = EXPECT[min_n]
        ok = abs(r2_p - e_r2) < TOL and abs(med_p - e_med) < TOL
        print(f"  [자체검증] R2 {r2_p:.10f} (기대 {e_r2:.10f})  "
              f"ratio {med_p:.10f} (기대 {e_med:.10f})  -> {'일치' if ok else '** 불일치 **'}",
              flush=True)
        assert ok, f"min_n={min_n} rng 정렬 실패. 재현되지 않으면 진행하지 않는다."

        v_corr, alpha = MC.var_min(s1, s2, d2 - d1)
        sd_D = S.sigma_w / np.sqrt(S.N_all)
        v_pen = (cfg["hinge_lam"] ** 2) * DP.rect_gauss_var(S.tau - S.d_glob, sd_D)
        th_corr = np.sqrt(v_corr + v_pen)
        sw_pred = norm.sf(alpha)

        per_class = []
        for i in np.argsort([len(R) for R in refs]):
            m = c1 == i
            if m.sum() < 20:
                continue
            per_class.append(dict(
                n_c=int(len(refs[i])), n_q=int(m.sum()),
                gamma_plain=float(T[m].mean() / (th_plain[m].mean() + 1e-12)),
                gamma_corr=float(T[m].mean() / (th_corr[m].mean() + 1e-12))))
        gp = [r["gamma_plain"] for r in per_class]; gc_ = [r["gamma_corr"] for r in per_class]
        print("  [클래스별] n_c " + " ".join(f"{r['n_c']}" for r in per_class))
        print("     plain  " + " ".join(f"{x:.3f}" for x in gp) + f"   폭 {max(gp)-min(gp):.3f}")
        print("     corr   " + " ".join(f"{x:.3f}" for x in gc_) + f"   폭 {max(gc_)-min(gc_):.3f}",
              flush=True)

        res[str(min_n)] = dict(
            C=C, n_query=int(len(Q)), n_c=[int(len(R)) for R in refs],
            per_class=per_class,
            spread_plain=float(max(gp) - min(gp)), spread_corr=float(max(gc_) - min(gc_)),
            switch=dict(obs_mean=float(sw_obs.mean()), pred_mean=float(sw_pred.mean()),
                        corr=float(np.corrcoef(sw_obs, sw_pred)[0, 1])),
            fit=dict(r2_plain=r2_p, r2_corr=IP.r2(th_corr, T),
                     med_plain=med_p,
                     med_corr=float(np.median(T / (th_corr + 1e-12)))))
        f = res[str(min_n)]
        print(f"  [전환율] 실측 {f['switch']['obs_mean']*100:.1f}%  "
              f"예측 {f['switch']['pred_mean']*100:.1f}%  상관 {f['switch']['corr']:+.3f}")
        print(f"  [적합]  R2 {f['fit']['r2_plain']:.4f} -> {f['fit']['r2_corr']:.4f}   "
              f"med {f['fit']['med_plain']:.4f} -> {f['fit']['med_corr']:.4f}", flush=True)

    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\n[saved] {out_json}   총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    main(a.boot, "derma_mincorr_paperpath.json")
