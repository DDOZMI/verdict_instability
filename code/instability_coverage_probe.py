import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCORES = ["knn_std", "lid", "energy", "msp", "maha"]
COVS_PAPER = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
COVS_FINE = [round(1.0 - 0.01 * i, 2) for i in range(51)]
SETTINGS = {
    "c100":     dict(kind="cifar", which="c100", odin="CIFAR-100"),
    "c10":      dict(kind="cifar", which="c10",  odin="CIFAR-10"),
    "derma400": dict(kind="derma", min_n=400,    odin="Derma C=3"),
    "derma200": dict(kind="derma", min_n=200,    odin="Derma C=5"),
    "derma80":  dict(kind="derma", min_n=80,     odin="Derma C=7"),
}
ORDER = ["c100", "c10", "derma400", "derma200", "derma80"]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def curve(f, T, covs):
    return [float(T[f <= np.percentile(f, c * 100)].mean()) for c in covs]


def area(v, c):
    o = np.argsort(c)
    return float(_trapz(np.asarray(v)[o], np.asarray(c)[o]))


def main(setting, n_boot, out):
    import gpu_logreg; gpu_logreg.install()
    import derma_probe as DP
    from numpy.random import default_rng

    t0 = time.time()
    if setting == "in1k":
        import in200_probe as IP, space_lowmem as SL
        rng = default_rng(DP.CFG["seed"])
        groups, refs = IP.build("dino", rng, bench="in1k")
        cfg, SpaceCls = DP.CFG, SL.Space200LowMem
    else:
        import exponent_probe as EP, odin_real_probe as ORP
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
        SpaceCls = DP.Space
    Q = np.vstack([groups[k] for k in groups])
    print(f"[{setting}] Q={Q.shape} C={len(refs)}")

    S = SpaceCls(Q, refs, cfg)
    print(f"  Space {time.time()-t0:.0f}s, 부트스트랩 B={n_boot} ...")
    T = S.truth(n_boot, rng)
    M = dict(S.metrics())

    res = dict(setting=setting, n_boot=n_boot, n_query=int(len(Q)), C=len(refs),
               covs_fine=COVS_FINE, covs_paper=COVS_PAPER,
               T_mean=float(T.mean()), curves={}, aurc={}, delta={})

    rnd_fine = np.zeros(len(COVS_FINE)); rnd_paper = np.zeros(len(COVS_PAPER))
    for s in range(20):
        u = default_rng(9000 + s).random(len(Q))
        rnd_fine += np.array(curve(u, T, COVS_FINE))
        rnd_paper += np.array(curve(u, T, COVS_PAPER))
    rnd_fine /= 20; rnd_paper /= 20
    res["curves"]["random"] = rnd_fine.tolist()
    a_rand = area(rnd_paper, COVS_PAPER)
    res["aurc"]["random"] = a_rand
    res["delta"]["random"] = 0.0

    print(f"  {'score':<9}{'AURC':>9}{'Delta %':>10}   (T̄={T.mean():.4f}, baseline={a_rand:.4f})")
    for k in SCORES:
        f = np.asarray(M[k], float)
        res["curves"][k] = curve(f, T, COVS_FINE)
        a = area(curve(f, T, COVS_PAPER), COVS_PAPER)
        d = 100 * (a / a_rand - 1)
        res["aurc"][k], res["delta"][k] = a, d
        print(f"  {k:<9}{a:>9.4f}{d:>+10.2f}")

    json.dump(res, open(out, "w"), indent=2)
    print(f"  -> {out}  ({time.time()-t0:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.setting, a.boot, a.out or f"instability_coverage_{a.setting}.json")
