import os, json, numpy as np

CFG = dict(
    cache_dir="./feat_cache_dino", seed=0,
    min_n=400,
    n_query=700,
    k_lid=20, k_knn=5, vim_P=64, odin_T=1000.0, maha_shrink=0.3,
    k_frac=0.7,
    hinge_lam=5.0, hinge_tau_q=20,
    n_boot=200,
    gray_cut=75, coverages=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    flags=["knn_std", "lid", "energy", "msp", "maha", "random"],
    ood_far=["svhn_d28", "dtd_d28", "cifar10_d28"],
    out_json="derma_probe.json",
)

_trapz = getattr(np, "trapezoid", None) or np.trapz


def auc_cov(v, c):
    o = np.argsort(c)
    return float(_trapz(np.asarray(v)[o], np.asarray(c)[o]))


def get(split):
    d = CFG["cache_dir"]
    return (np.load(os.path.join(d, f"dino_feat_derma_{split}.npy")).astype(np.float64),
            np.load(os.path.join(d, f"dino_lab_derma_{split}.npy")))


def load_far(name):
    p = os.path.join(CFG["cache_dir"], f"dino_feat_far_{name}.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def rect_gauss_var(m, sd):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-12)
    a = m / sd
    Phi, phi = norm.cdf(a), norm.pdf(a)
    EY = m * Phi + sd * phi
    EY2 = (m ** 2 + sd ** 2) * Phi + m * sd * phi
    return np.maximum(EY2 - EY ** 2, 0.0)


class Space:
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

        Dc = np.linalg.norm(Q[:, None, :] - self.mus[None, :, :], axis=2)
        self.nc = Dc.argmin(1)
        self.d_cls = Dc[np.arange(N), self.nc]
        self.d_glob = np.linalg.norm(Q - self.mu_g, axis=1)
        self.n_assigned = self.n_c[self.nc]

        self.sigma_t = np.empty(N)
        self.r = np.empty(N)
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
            kk = max(3, int(round(cfg["k_frac"] * len(R))))
            kk = min(kk, len(R))
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
        self.maha = np.linalg.norm((Q @ Wm)[:, None, :] - (self.mus @ Wm)[None, :, :],
                                   axis=2).min(1)

        y = np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])
        self.probe = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(X, y)
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
        self.hinge = self.d_cls + cfg["hinge_lam"] * np.maximum(0.0,
                                                                self.tau - self.d_glob)

    def theory(self, use_nc=True):
        n_use = self.n_assigned if use_nc else np.full(len(self.Q), self.n_c.mean())
        var_cls = self.sigma_t ** 2 / n_use
        sd_D = self.sigma_w / np.sqrt(self.N_all)
        var_pen = (self.cfg["hinge_lam"] ** 2) * rect_gauss_var(
            self.tau - self.d_glob, sd_D)
        return np.sqrt(var_cls + var_pen)

    def metrics(self):
        return dict(knn_std=self.knn_std, lid=-self.lid,
                    d_cls=self.d_cls, hinge=self.hinge, knn=self.knn, maha=self.maha,
                    energy=self.energy, msp=self.msp, maxlogit=self.maxlogit,
                    entropy=self.entropy, odin=self.odin, vim=self.vim)

    def truth(self, n_boot, rng):
        B = np.zeros((n_boot, len(self.Q)))
        for b in range(n_boot):
            rb = [R[rng.integers(0, len(R), len(R))] for R in self.refs]
            mb = np.vstack([R.mean(0) for R in rb])
            mg = np.vstack(rb).mean(0)
            dc = np.linalg.norm(self.Q[:, None, :] - mb[None, :, :], axis=2).min(1)
            dg = np.linalg.norm(self.Q - mg, axis=1)
            B[b] = dc + self.cfg["hinge_lam"] * np.maximum(0.0, self.tau - dg)
            print(f"    boot {b+1}/{n_boot}", end="\r", flush=True)
        print()
        return B.std(0)


def main():
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(CFG["seed"])
    cache = CFG["cache_dir"]
    meta = json.load(open(os.path.join(cache, "DERMA_META.json")))
    names = meta["label_names"]

    ftr, ytr = get("train")
    fte, yte = get("test")
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}
    held_in = sorted([c for c, n in cnt.items() if n >= CFG["min_n"]])
    near_cls = sorted([c for c, n in cnt.items() if n < CFG["min_n"]])

    print("=" * 82)
    print("DermaMNIST (28px 통제)")
    print("=" * 82)
    print(f"  held-in ({len(held_in)}): " +
          ", ".join(f"{names[str(c)][:12]}(n={cnt[c]})" for c in held_in))
    print(f"  near    ({len(near_cls)}): " +
          ", ".join(f"{names[str(c)][:8]}({cnt[c]})" for c in near_cls))
    print(f"  ** n_c 범위: {min(cnt[c] for c in held_in)} ~ "
          f"{max(cnt[c] for c in held_in)}  "
          f"({max(cnt[c] for c in held_in)/min(cnt[c] for c in held_in):.1f}배) **")

    refs = [ftr[ytr == c] for c in held_in]
    C = len(refs)

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = CFG["n_query"]
    groups = {"in": sub(fte[np.isin(yte, held_in)], nq),
              "near": sub(fte[np.isin(yte, near_cls)], nq)}
    for s in CFG["ood_far"]:
        F = load_far(s)
        if F is not None:
            groups[s] = sub(F, nq)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    gnames = list(groups)
    print(f"  groups: {', '.join(f'{g}={len(v)}' for g, v in groups.items())}")

    S = Space(Q, refs, CFG)
    print(f"  프로브 정확도 = {S.probe.score(np.vstack(refs), np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])):.3f}")
    print(f"  ρ(r, σ_t) = {spearmanr(S.r, S.sigma_t).statistic:+.3f}")
    print(f"\n  truth 계산 (x{CFG['n_boot']})")
    truth = S.truth(CFG["n_boot"], rng)

    def within(x):
        o = x.copy().astype(float)
        for g in gnames:
            m = G == g
            o[m] -= x[m].mean()
        return o

    def zs(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    def r2(p, t):
        m = np.isfinite(p) & np.isfinite(t)
        lr = LinearRegression().fit(zs(p[m])[:, None], zs(t[m]))
        return float(lr.score(zs(p[m])[:, None], zs(t[m])))

    OUT = dict(held_in=held_in, near=near_cls, counts=cnt, C=C)

    print("\n" + "=" * 82)
    print("[E1] ** 1/√n_c 스케일링 (신규) **")
    print("=" * 82)
    th_nc = S.theory(use_nc=True)
    th_nbar = S.theory(use_nc=False)

    r_plain = float(spearmanr(truth, S.sigma_t).statistic)
    r_scaled = float(spearmanr(truth * np.sqrt(S.n_assigned), S.sigma_t).statistic)
    print(f"  (a) ρ(truth, σ_t)          = {r_plain:+.3f}")
    print(f"      ρ(truth·√n_c, σ_t)     = {r_scaled:+.3f}   "
          f"** {'상승' if r_scaled > r_plain else '하락'} ({r_scaled-r_plain:+.3f}) **")

    print(f"\n  (b) 클래스별 비교")
    print(f"      {'클래스':>14} {'n_c':>6} {'mean truth':>11} {'×√n_c':>9} {'mean σ_t':>9}")
    cls_rows = []
    for i, c in enumerate(held_in):
        m = S.nc == i
        if m.sum() < 10:
            continue
        t_ = truth[m].mean()
        s_ = S.sigma_t[m].mean()
        cls_rows.append(dict(cls=int(c), name=names[str(c)], n=int(S.n_c[i]),
                             truth=float(t_), scaled=float(t_ * np.sqrt(S.n_c[i])),
                             sigma_t=float(s_), n_q=int(m.sum())))
        print(f"      {names[str(c)][:14]:>14} {S.n_c[i]:>6} {t_:11.4f} "
              f"{t_*np.sqrt(S.n_c[i]):9.3f} {s_:9.3f}")
    print(f"      ** ×√n_c 가 σ_t 와 비슷해지면 스케일링 확증 **")

    print(f"\n  (c) ** R² 비교 (결정적) **")
    r2_nc = r2(th_nc, truth)
    r2_nbar = r2(th_nbar, truth)
    print(f"      σ_t/√n_c (올바름)  R² = {r2_nc:.3f}   비율 = "
          f"{np.median(truth/(th_nc+1e-12)):.2f}")
    print(f"      σ_t/√n̄  (n_c 무시) R² = {r2_nbar:.3f}   비율 = "
          f"{np.median(truth/(th_nbar+1e-12)):.2f}")
    print(f"      ** 차이 = {r2_nc - r2_nbar:+.3f} **")
    OUT["E1"] = dict(rho_plain=r_plain, rho_scaled=r_scaled,
                     r2_with_nc=r2_nc, r2_without_nc=r2_nbar,
                     gain=r2_nc - r2_nbar, per_class=cls_rows,
                     ratio_nc=float(np.median(truth / (th_nc + 1e-12))))

    print("\n" + "=" * 82)
    print("[E2] 부호 분리 재현")
    print("=" * 82)
    M = S.metrics()
    print(f"  {'지표':>10} {'ρ(M,σ_t)':>10} {'ρ(M,truth)':>11} {'|차|':>6} {'일치':>5}")
    print("  " + "-" * 46)
    sign = {}
    nm_ok = 0
    for nm, v in M.items():
        a = float(spearmanr(within(v), within(S.sigma_t)).statistic)
        b = float(spearmanr(within(v), within(truth)).statistic)
        ok = (a > 0) == (b > 0)
        nm_ok += ok
        sign[nm] = dict(rho_sigma_t=a, rho_truth=b, match=bool(ok))
        print(f"  {nm:>10} {a:+10.3f} {b:+11.3f} {abs(a-b):6.3f} "
              f"{'O' if ok else '** X **':>5}")
    print("  " + "-" * 46)
    print(f"  ** 부호 일치: {nm_ok}/{len(M)} **  "
          f"(CIFAR-100: 13/13, CIFAR-10: 12/13)")
    OUT["E2"] = dict(metrics=sign, n_match=nm_ok, n_total=len(M))

    print("\n" + "=" * 82)
    print("[E3] energy 보류 — 랜덤보다 나쁜가")
    print("=" * 82)
    F_all = dict(knn_std=S.knn_std, lid=-S.lid, energy=S.energy, msp=S.msp,
                 maha=S.maha, random=rng.random(len(Q)))
    print(f"  {'커버리지':>8}" + "".join(f"{n:>10}" for n in CFG["flags"]))
    RC = {n: [] for n in CFG["flags"]}
    for cov in CFG["coverages"]:
        print(f"  {cov:8.0%}", end="")
        for n in CFG["flags"]:
            f = F_all[n]
            keep = f <= np.percentile(f, cov * 100)
            RC[n].append(float(truth[keep].mean()))
            print(f"{truth[keep].mean():10.4f}", end="")
        print()
    print(f"  ^ 남은 샘플의 판정 변동 (낮을수록 좋음)")
    print(f"\n  AURC (낮을수록 좋음):")
    aurc = {}
    rnd = auc_cov(RC["random"], CFG["coverages"])
    for n in sorted(CFG["flags"], key=lambda n: auc_cov(RC[n], CFG["coverages"])):
        a = auc_cov(RC[n], CFG["coverages"])
        aurc[n] = a
        mark = "  <- 기준선" if n == "random" else (
            "  개선" if a < rnd else "  ** 악화 (랜덤보다 나쁨) **")
        print(f"    {n:>9}  {a:.5f}{mark}")
    OUT["E3"] = dict(aurc=aurc, curves={n: RC[n] for n in CFG["flags"]},
                     coverages=CFG["coverages"])

    print("\n" + "=" * 82)
    print(f"[E4] C={C} 추세 — 완전한 이론")
    print("=" * 82)
    r2_full = r2(th_nc, truth)
    r2_within = r2(within(th_nc), within(truth))
    print(f"  R² (완전한 이론)      = {r2_full:.3f}")
    print(f"  R² (within)           = {r2_within:.3f}")
    print(f"  비율                  = {np.median(truth/(th_nc+1e-12)):.2f}")
    print(f"\n  참고: CIFAR-100 (C=50) R²=0.81   CIFAR-10 (C=5) R²=0.91 (완전한 이론)")
    OUT["E4"] = dict(C=C, r2=r2_full, r2_within=r2_within,
                     rho_r_sigma_t=float(spearmanr(S.r, S.sigma_t).statistic))

    with open(CFG["out_json"], "w") as f:
        json.dump(dict(config=CFG, results=OUT), f, indent=2, default=float)
    print(f"\n[saved] {CFG['out_json']}")

    print("\n" + "=" * 82)
    print("판정")
    print("=" * 82)
    print("  [E1] R²(n_c 사용) > R²(n̄ 사용) -> ** 1/√n_c 스케일링 실증 **")
    print("       지금까지 n=400 고정이라 검증 못 했던 부분.")
    print("  [E2] 부호 일치 -> 의료 도메인에서도 재현")
    print("  [E3] energy 가 랜덤보다 나쁘면 -> downstream 일반화")
    print("  [E4] C=3 에서도 R² 유지 -> 이론이 C 에 무관")
    print("=" * 82)


if __name__ == "__main__":
    main()
