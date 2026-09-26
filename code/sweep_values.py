import os, sys, json
import numpy as np

PROJECT = "/path/to/your/project"
sys.path.insert(0, PROJECT)
import derma_probe as DP

COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
FLAGS = ["knn_std", "lid", "random", "energy", "msp", "maha"]
N_SWEEP = [25, 50, 100, 200, 400]
rng = np.random.default_rng(DP.CFG["seed"])
cd = DP.CFG["cache_dir"]

_trap = getattr(np, "trapezoid", getattr(np, "trapz", None))


def auc(vals):
    o = np.argsort(COVS)
    return float(_trap(np.asarray(vals)[o], np.asarray(COVS)[o]))


def aurc(f, T):
    return auc([float(T[f <= np.percentile(f, c * 100)].mean()) for c in COVS])


def build(n_ref):
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

    groups = {"in": sub(fte[np.isin(yte, held)], 800),
              "near": sub(fte[np.isin(yte, hout)], 800)}
    for s in ["svhn", "dtd_lr32", "lsun", "isun", "places365_lr32", "cifar10"]:
        f = os.path.join(cd, f"dino_feat_far_{s}.npy")
        if os.path.exists(f):
            groups[s] = sub(np.load(f).astype(float), 800)
    return groups, refs


def main():
    print("=" * 72)
    print(f"sweep_values — §6 과 동일 detector (hinge_lam={DP.CFG['hinge_lam']})")
    print("=" * 72)
    rows = {}
    meanT = {}
    for n_ref in N_SWEEP:
        groups, refs = build(n_ref)
        Q = np.vstack([groups[g] for g in groups])
        S = DP.Space(Q, refs, dict(DP.CFG))
        T = S.truth(DP.CFG["n_boot"], rng)
        M = dict(S.metrics()); M["random"] = rng.random(len(Q))
        meanT[n_ref] = float(T.mean())
        for fn in FLAGS:
            rows.setdefault(fn, {})[n_ref] = aurc(M[fn], T)
        print(f"  n_c={n_ref:>4}  meanT={T.mean():.4f}  " +
              " ".join(f"{fn}={rows[fn][n_ref]:.4f}" for fn in FLAGS))

    print("\n  ** 부호(기준선 대비) 크기별 고정 확인 **")
    for fn in FLAGS:
        if fn == "random":
            continue
        sides = ["+" if rows[fn][n] < rows["random"][n] else "-" for n in N_SWEEP]
        ok = len(set(sides)) == 1
        print(f"    {fn:>9}: {' '.join(sides)}  {'** 고정 **' if ok else '** 뒤집힘 **'}")

    print(f"\n  ** 검증: n_c=400 열을 sec6_values CIFAR-100 과 대조하라. **")
    print(f"     knn_std={rows['knn_std'][400]:.4f}  maha={rows['maha'][400]:.4f}  "
          f"energy={rows['energy'][400]:.4f}")

    def fmt(fn):
        tn = {"knn_std": r"\texttt{knn\_std}", "lid": r"\texttt{lid}",
              "random": r"\emph{random}", "energy": r"\texttt{energy}",
              "msp": r"\texttt{msp}", "maha": r"\texttt{maha}"}[fn]
        vals = [rows[fn][n] for n in N_SWEEP]
        if fn == "random":
            cells = " & ".join(r"\emph{%.4f}" % v for v in vals)
        else:
            cells = " & ".join("%.4f" % v for v in vals)
        return f"{tn} & {cells} \\\\"

    body = "\n".join([fmt("knn_std"), fmt("lid"), r"\midrule", fmt("random"),
                      r"\midrule", fmt("energy"), fmt("msp"), fmt("maha")])
    mt = " & ".join("%.3f" % meanT[n] for n in N_SWEEP)
    print("\n" + "=" * 72)
    print("LaTeX 표 (부록 D 의 tab:sweep 을 이걸로 교체):")
    print("=" * 72)
    print(r"Mean instability & " + mt + r" \\")
    print(r"\midrule")
    print(body)

    with open("sweep_values.json", "w") as f:
        json.dump(dict(rows=rows, meanT=meanT,
                       hinge_lam=DP.CFG["hinge_lam"]), f, indent=2, default=float)
    print("\n[saved] sweep_values.json")


if __name__ == "__main__":
    main()
