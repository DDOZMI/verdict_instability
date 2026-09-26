import os, json, numpy as np

CFG = dict(
    cache_dir="./feat_cache_dino", seed=0,
    datasets=dict(
        c100=dict(prefix=None, held_in=list(range(50))),
        c10=dict(prefix="c10", held_in=[0, 1, 2, 3, 4]),
    ),
    n_query=800, ref_per_class=400,
    P_list=[1, 5, 10, 20, 50, 100],
    n_bins=20,
    ood_sources=["svhn", "dtd_lr32", "cifar10", "lsun", "isun", "places365_lr32"],
    synth=dict(d=100, n=400, C=10, n_q=2000,
               a_list=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
               ood_scale=[1.0, 1.5, 2.0, 3.0]),
    out_json="anisotropy_probe.json",
)


def get(cache, prefix, split):
    tag = f"{prefix}_{split}" if prefix else split
    return (np.load(os.path.join(cache, f"dino_feat_{tag}.npy")).astype(np.float64),
            np.load(os.path.join(cache, f"dino_lab_{tag}.npy")))


def load_ood(cache, name):
    p = os.path.join(cache, f"dino_feat_far_{name}.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def partial_rho(x, y, z, n_bins):
    from scipy.stats import spearmanr
    e = np.percentile(z, np.linspace(0, 100, n_bins + 1))
    e[0] -= 1e-9; e[-1] += 1e-9
    bi = np.clip(np.digitize(z, e) - 1, 0, n_bins - 1)
    num = den = 0.0
    for b in range(n_bins):
        m = bi == b
        if m.sum() < 20:
            continue
        a, c = x[m], y[m]
        if np.all(a == a[0]) or np.all(c == c[0]):
            continue
        r = spearmanr(a, c).statistic
        if not np.isfinite(r):
            continue
        num += m.sum() * r
        den += m.sum()
    return num / den if den > 0 else float("nan")


def participation_ratio(lam):
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    return float(s1 ** 2 / (s2 + 1e-12))


def analyze(Q, refs, P_list, n_bins, tag):
    from scipy.stats import spearmanr
    mus = np.vstack([R.mean(0) for R in refs])
    Dc = np.linalg.norm(Q[:, None, :] - mus[None, :, :], axis=2)
    nc = Dc.argmin(1)
    N = len(Q)

    r = np.empty(N); st = np.empty(N)
    alpha = {P: np.empty(N) for P in P_list}
    PRs = {}
    for c in np.unique(nc):
        m = nc == c
        W = refs[c] - mus[c]
        Cov = (W.T @ W) / max(len(W) - 1, 1)
        lam, V = np.linalg.eigh(Cov)
        o = np.argsort(lam)[::-1]
        lam, V = lam[o], V[:, o]
        PRs[int(c)] = participation_ratio(np.maximum(lam, 0))

        v = Q[m] - mus[c]
        rr = np.linalg.norm(v, axis=1)
        u = v / (rr[:, None] + 1e-12)
        r[m] = rr
        st[m] = (u @ W.T).std(axis=1)
        UV = u @ V
        UV2 = UV ** 2
        for P in P_list:
            alpha[P][m] = UV2[:, :P].sum(1)

    print(f"\n  [{tag}] 참여율 PR (클래스별): "
          f"중앙={np.median(list(PRs.values())):.1f}  "
          f"범위=[{min(PRs.values()):.1f}, {max(PRs.values()):.1f}]   (d={refs[0].shape[1]})")

    rho_r_st = float(spearmanr(r, st).statistic)
    print(f"  [{tag}] ρ(r, σ_t) = {rho_r_st:+.3f}")

    print(f"\n  {'P':>5} {'ρ(r, α_P)':>10} {'ρ(α_P, σ_t)':>12} "
          f"{'ρ(r,σ_t | α_P)':>14}  <- 매개 확인")
    A = {}
    for P in P_list:
        a = alpha[P]
        r1 = float(spearmanr(r, a).statistic)
        r2 = float(spearmanr(a, st).statistic)
        pr = partial_rho(r, st, a, n_bins)
        A[P] = dict(rho_r_alpha=r1, rho_alpha_st=r2, partial_rho_r_st=pr)
        print(f"  {P:>5} {r1:+10.3f} {r2:+12.3f} {pr:+14.3f}")
    print(f"  ** 부분ρ 가 0 에 가까워지면 α_P 가 매개변수 (가설 확증) **")

    return dict(rho_r_sigma_t=rho_r_st, PR_median=float(np.median(list(PRs.values()))),
                PR_per_class=PRs, alpha=A)


def synth_experiment(cfg, rng):
    from scipy.stats import spearmanr
    d, n, C, n_q = cfg["d"], cfg["n"], cfg["C"], cfg["n_q"]
    print("\n" + "=" * 80)
    print(f"[C] ** 인공 데이터 ** — 스펙트럼 λ_j ∝ j^(-a) 를 직접 조절")
    print(f"    d={d}, n={n}/클래스, C={C}, 질의 {n_q}")
    print("=" * 80)
    print(f"\n  {'a':>5} {'PR':>8} {'ρ(r,σ_t)':>10} {'ρ(r,α_10)':>11} "
          f"{'ρ(r,σ_t|α_10)':>14}")
    print("  " + "-" * 54)
    RES = {}
    for a in cfg["a_list"]:
        rs = np.random.default_rng(CFG["seed"])
        j = np.arange(1, d + 1)
        lam = j.astype(float) ** (-a)
        lam = lam / lam.sum() * d
        PR = participation_ratio(lam)

        Vv = np.linalg.qr(rs.normal(size=(d, d)))[0]
        Sig_half = Vv @ np.diag(np.sqrt(lam)) @ Vv.T

        mus = rs.normal(size=(C, d)) * 3.0
        refs = [mus[c] + rs.normal(size=(n, d)) @ Sig_half for c in range(C)]

        Qs = []
        for sc in cfg["ood_scale"]:
            c = rs.integers(0, C, n_q // len(cfg["ood_scale"]))
            base = mus[c] + rs.normal(size=(len(c), d)) @ Sig_half
            push = rs.normal(size=(len(c), d))
            push /= np.linalg.norm(push, axis=1, keepdims=True)
            Qs.append(base + (sc - 1.0) * 3.0 * push)
        Q = np.vstack(Qs)

        rr_ = analyze_synth(Q, refs, mus, Vv, 10)
        RES[a] = dict(PR=PR, **rr_)
        print(f"  {a:>5.2f} {PR:>8.1f} {rr_['rho_r_st']:+10.3f} "
              f"{rr_['rho_r_alpha']:+11.3f} {rr_['partial']:+14.3f}")

    print("\n  ** a=0 (등방) 에서 ρ(r,σ_t) ≈ 0,  a 클수록 강한 음수면 가설 확증 **")
    return RES


def analyze_synth(Q, refs, mus, V, P):
    from scipy.stats import spearmanr
    Dc = np.linalg.norm(Q[:, None, :] - mus[None, :, :], axis=2)
    nc = Dc.argmin(1)
    N = len(Q)
    r = np.empty(N); st = np.empty(N); al = np.empty(N)
    for c in np.unique(nc):
        m = nc == c
        W = refs[c] - refs[c].mean(0)
        v = Q[m] - mus[c]
        rr = np.linalg.norm(v, axis=1)
        u = v / (rr[:, None] + 1e-12)
        r[m] = rr
        st[m] = (u @ W.T).std(axis=1)
        al[m] = ((u @ V[:, :P]) ** 2).sum(1)
    return dict(rho_r_st=float(spearmanr(r, st).statistic),
                rho_r_alpha=float(spearmanr(r, al).statistic),
                rho_alpha_st=float(spearmanr(al, st).statistic),
                partial=partial_rho(r, st, al, 20))


def main():
    rng = np.random.default_rng(CFG["seed"])
    cache = CFG["cache_dir"]
    OUT = {}

    for name, spec in CFG["datasets"].items():
        tag = f"{spec['prefix']}_train" if spec["prefix"] else "train"
        if not os.path.exists(os.path.join(cache, f"dino_feat_{tag}.npy")):
            print(f"[skip] {name}")
            continue
        print("\n" + "=" * 80)
        print(f"[{name}] 실데이터")
        print("=" * 80)
        ftr, ytr = get(cache, spec["prefix"], "train")
        fte, yte = get(cache, spec["prefix"], "test")
        hi = spec["held_in"]
        ho = [c for c in np.unique(ytr) if c not in hi]
        N = CFG["ref_per_class"]

        r0 = np.random.default_rng(CFG["seed"])
        refs = []
        for c in hi:
            Xc = ftr[ytr == c]
            if len(Xc) > N:
                Xc = Xc[r0.choice(len(Xc), N, replace=False)]
            refs.append(Xc)

        def sub(x, m):
            return x if len(x) <= m else x[r0.choice(len(x), m, replace=False)]

        nq = CFG["n_query"]
        Qs = [sub(fte[np.isin(yte, hi)], nq), sub(fte[np.isin(yte, ho)], nq)]
        for s in CFG["ood_sources"]:
            if name == "c10" and s == "cifar10":
                continue
            F = load_ood(cache, s)
            if F is not None:
                Qs.append(sub(F, nq))
        Q = np.vstack(Qs)
        OUT[name] = analyze(Q, refs, CFG["P_list"], CFG["n_bins"], name)

    OUT["synth"] = synth_experiment(CFG["synth"], rng)

    with open(CFG["out_json"], "w") as f:
        json.dump(dict(config=CFG, results=OUT), f, indent=2, default=float)
    print(f"\n[saved] {CFG['out_json']}")

    print("\n" + "=" * 80)
    print("판정")
    print("=" * 80)
    print("  [A] ρ(r, α_P) < 0  &  ρ(α_P, σ_t) > 0  &  부분ρ(r,σ_t | α_P) ≈ 0")
    print("      -> ** α_P(주성분 정렬도)가 매개변수. 가설 확증 **")
    print("         ID 는 다양체(고분산 방향)에 정렬, OOD 는 이탈해 저분산 방향으로 기운다.")
    print("  [B] CIFAR-10 의 PR 이 더 작으면(저차원) ρ(r,σ_t) 가 더 강한 음수")
    print("      -> §6.5 (CIFAR-10 부호 약화)와 연결")
    print("  [C] ** 인공: a=0 에서 ρ≈0, a 클수록 음수 **")
    print("      -> ** 스펙트럼 불균등도가 원인임을 인과적으로 확립 **")
    print("=" * 80)


if __name__ == "__main__":
    main()
