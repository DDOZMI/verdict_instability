import os, json, numpy as np

import derma_probe as DP

CFG = dict(
    n_boot=DP.CFG["n_boot"],
    seed=DP.CFG["seed"],
    alphas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    lambdas=[0.0, 1.0, 2.5, 5.0, 10.0],
    min_n_list=[400, 200, 80],
    cifar_ref=400,
    cifar_nq=800,
    out_json="exponent_probe.json",
)

STD = ["knn_std", "lid", "d_cls", "knn", "maha", "vim",
       "energy", "msp", "maxlogit", "entropy", "odin"]

CIFAR = dict(ood=["svhn", "dtd_lr32", "lsun", "isun", "places365_lr32"])


def build_cifar(rng, which):
    cd = DP.CFG["cache_dir"]
    pre = None if which == "c100" else "c10"
    held = list(range(50)) if which == "c100" else [0, 1, 2, 3, 4]
    hout = list(range(50, 100)) if which == "c100" else [5, 6, 7, 8, 9]
    extra = ["cifar10"] if which == "c100" else ["cifar100"]

    def p(k, s):
        t = f"{pre}_{s}" if pre else s
        return os.path.join(cd, f"dino_{k}_{t}.npy")

    for k in ["feat", "lab"]:
        for s in ["train", "test"]:
            if not os.path.exists(p(k, s)):
                print(f"  [skip {which}] 없음: {os.path.basename(p(k, s))}")
                return None
    ftr, ytr = np.load(p("feat", "train")).astype(np.float64), np.load(p("lab", "train"))
    fte, yte = np.load(p("feat", "test")).astype(np.float64), np.load(p("lab", "test"))

    refs = []
    for c in held:
        X = ftr[ytr == c]
        if len(X) > CFG["cifar_ref"]:
            X = X[rng.choice(len(X), CFG["cifar_ref"], replace=False)]
        refs.append(X)

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = CFG["cifar_nq"]
    groups = {"in": sub(fte[np.isin(yte, held)], nq),
              "near": sub(fte[np.isin(yte, hout)], nq)}
    for s in CIFAR["ood"] + extra:
        f = os.path.join(cd, f"dino_feat_far_{s}.npy")
        if os.path.exists(f):
            groups[s] = sub(np.load(f).astype(np.float64), nq)
    return groups, refs


def build_derma(rng, min_n):
    ftr, ytr = DP.get("train")
    fte, yte = DP.get("test")
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}
    hr = sorted([c for c, n in cnt.items() if n >= 400])
    nr = sorted([c for c, n in cnt.items() if n < 400])
    hi = sorted([c for c, n in cnt.items() if n >= min_n])
    refs = [ftr[ytr == c] for c in hi]

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = DP.CFG["n_query"]
    groups = {"in": sub(fte[np.isin(yte, hr)], nq),
              "near": sub(fte[np.isin(yte, nr)], nq)}
    for s in DP.CFG["ood_far"]:
        F = DP.load_far(s)
        if F is not None:
            groups[s] = sub(F, nq)
    return groups, refs


def evaluate(groups, refs, cfg, rng, tag):
    from scipy.stats import spearmanr
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    S = DP.Space(Q, refs, cfg)
    T = S.truth(CFG["n_boot"], rng)
    M = dict(S.metrics())

    sig2 = S.sigma_t ** 2
    nc = S.n_assigned.astype(float)
    sd_D = S.sigma_w / np.sqrt(S.N_all)
    pen_unit = DP.rect_gauss_var(S.tau - S.d_glob, sd_D)

    def w(x):
        o = np.asarray(x, float).copy()
        for k in np.unique(G):
            m = G == k
            o[m] -= o[m].mean()
        return o

    wT = w(T)
    sT = {nm: float(spearmanr(w(M[nm]), wT).statistic) for nm in STD}

    grid = {}
    for a in CFG["alphas"]:
        for lam in CFG["lambdas"]:
            P = sig2 / (nc ** a) + (lam ** 2) * pen_unit
            ok = 0
            for nm in STD:
                r = float(spearmanr(w(M[nm]), w(P)).statistic)
                if (r > 0) == (sT[nm] > 0):
                    ok += 1
            grid[f"{a}|{lam}"] = int(ok)

    n_c = [len(R) for R in refs]
    print("\n" + "=" * 78)
    print(f"[{tag}]  C={len(refs)}  불균형 {max(n_c)/min(n_c):.1f}x   질의 {len(Q)}")
    print("=" * 78)
    print(f"  {'α \\ λ':>7}" + "".join(f"{l:>7.1f}" for l in CFG["lambdas"]))
    print("  " + "-" * (7 + 7 * len(CFG["lambdas"])))
    best = max(grid.values())
    for a in CFG["alphas"]:
        row = f"  {a:>7.2f}"
        for lam in CFG["lambdas"]:
            v = grid[f"{a}|{lam}"]
            th = (abs(a - 1.0) < 1e-9) and (abs(lam - cfg["hinge_lam"]) < 1e-9)
            mk = "*" if v == best else " "
            row += f"{v:>6}{'T' if th else mk}"
        print(row)
    print("  " + "-" * (7 + 7 * len(CFG["lambdas"])))
    print(f"  T = 이론값 (α=1, λ={cfg['hinge_lam']:.0f})   * = 이 설정의 최댓값 ({best}/11)")

    theory = grid[f"1.0|{float(cfg['hinge_lam'])}"]
    print(f"  ** 이론값 {theory}/11   최댓값 {best}/11   "
          f"{'** 이론이 최적 **' if theory == best else '** 이론이 최적 아님 **'}")

    return dict(tag=tag, C=len(refs), imbalance=float(max(n_c) / min(n_c)),
                grid=grid, best=best, theory=theory,
                theory_is_best=bool(theory == best))


def main():
    rng = np.random.default_rng(CFG["seed"])
    RES = []
    print("=" * 78)
    print("exponent_probe — P(α,λ) = σ_t²/n_c^α + λ²·Var[pen]")
    print("=" * 78)
    print("  ** 제곱근은 스피어만에 무영향 ** (단조변환). 따라서 α 와 λ 두 축만 훑는다.")
    print(f"  이론값: α = 1, λ = {DP.CFG['hinge_lam']:.0f}")

    for which in ["c100", "c10"]:
        got = build_cifar(rng, which)
        if got is None:
            continue
        g, refs = got
        RES.append(evaluate(g, refs, dict(DP.CFG), rng,
                            f"CIFAR-{'100' if which == 'c100' else '10'}"))

    for min_n in CFG["min_n_list"]:
        g, refs = build_derma(rng, min_n)
        cfg = dict(DP.CFG); cfg["min_n"] = min_n
        RES.append(evaluate(g, refs, cfg, rng, f"Derma C={len(refs)}"))

    print("\n" + "=" * 78)
    print("합산 — 5 설정 x 11 지표 = 55")
    print("=" * 78)
    tot = {}
    for a in CFG["alphas"]:
        for lam in CFG["lambdas"]:
            tot[f"{a}|{lam}"] = sum(r["grid"][f"{a}|{lam}"] for r in RES)
    best = max(tot.values())
    th = tot[f"1.0|{float(DP.CFG['hinge_lam'])}"]

    print(f"  {'α \\ λ':>7}" + "".join(f"{l:>8.1f}" for l in CFG["lambdas"]))
    print("  " + "-" * (7 + 8 * len(CFG["lambdas"])))
    for a in CFG["alphas"]:
        row = f"  {a:>7.2f}"
        for lam in CFG["lambdas"]:
            v = tot[f"{a}|{lam}"]
            is_th = (abs(a - 1.0) < 1e-9) and (abs(lam - DP.CFG["hinge_lam"]) < 1e-9)
            mk = "T" if is_th else ("*" if v == best else " ")
            row += f"{v:>7}{mk}"
        print(row)
    print("  " + "-" * (7 + 8 * len(CFG["lambdas"])))
    print(f"  ** 이론값 (α=1, λ=5): {th}/55   최댓값: {best}/55 **")

    print(f"\n  ** α 단면 (λ=5 고정) — count 의 지수가 규칙에서 최적인가 **")
    print(f"  {'α':>6} {'일치/55':>9}")
    for a in CFG["alphas"]:
        v = tot[f"{a}|5.0"]
        print(f"  {a:>6.2f} {v:>6}/55" + ("   ** 이론 **" if abs(a - 1) < 1e-9 else ""))

    print(f"\n  ** λ 단면 (α=1 고정) — 탐지기의 λ 를 그대로 쓰는 것이 최적인가 **")
    print(f"  {'λ':>6} {'일치/55':>9}")
    for lam in CFG["lambdas"]:
        v = tot[f"1.0|{lam}"]
        print(f"  {lam:>6.1f} {v:>6}/55" +
              ("   ** 이론 (탐지기의 λ) **" if abs(lam - 5) < 1e-9 else ""))

    print("\n" + "=" * 78)
    print("판정")
    print("=" * 78)
    if th == best:
        print("  ** 이론값이 격자의 최댓값이다. **")
        print("     -> 규칙은 자유 파라미터가 없다. 탐지기의 λ 와 이론의 지수를 그대로 쓴다.")
        print("     -> '이론의 각 항이 규칙에서도 필요하다' 가 ** 곡선으로 ** 증명된다.")
    else:
        print(f"  ** 이론값 {th} < 최댓값 {best}. **")
        arg = [k for k, v in tot.items() if v == best]
        print(f"     최적: {arg}")
        print("     -> 규칙을 이론값에 고정하는 근거를 다시 써야 한다.")
        print("        (튜닝하면 자유 파라미터가 생긴다. 그건 논문의 강점을 깎는다.)")
    print("=" * 78)

    with open(CFG["out_json"], "w") as f:
        json.dump(dict(config=CFG, derma_cfg=DP.CFG,
                       results=RES, total=tot, best=best, theory=th), f,
                  indent=2, default=float)
    print(f"\n[saved] {CFG['out_json']}")


if __name__ == "__main__":
    main()
