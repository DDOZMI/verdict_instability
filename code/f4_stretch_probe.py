import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derma_probe as DP
import c1_replication_probe as C1

EIG_FLOOR = 1e-8
N_SEEDS = int(os.environ.get("F4_SEEDS", 10))
N_BOOT = int(os.environ.get("F4_BOOT", DP.CFG["n_boot"]))
OUT_JSON = os.environ.get("F4_OUT", "f4_stretch_probe.json")

BASE = ["knn_std", "lid", "energy", "msp", "maha"]
CHAN = ["stretch_W", "size_d2", "maha_S"]

ELEVEN = ["knn_std", "lid", "d_cls", "knn", "maha", "vim",
          "energy", "msp", "maxlogit", "entropy", "odin"]


def tied_scatter(refs, shrink):
    X = np.vstack(refs)
    D = X.shape[1]
    Sw = np.zeros((D, D))
    for R in refs:
        Xc = R - R.mean(0)
        Sw += Xc.T @ Xc
    Sw /= max(len(X) - len(refs), 1)
    a = shrink
    Sw = (1 - a) * Sw + a * (np.trace(Sw) / D) * np.eye(D)
    return Sw


def channels(S, refs, shrink):
    Sw = tied_scatter(refs, shrink)
    lam, U = np.linalg.eigh(Sw)
    lam = np.maximum(lam, EIG_FLOOR)
    delta = S.Q - S.mus[S.nc]
    A = delta @ U
    nrm2 = (A ** 2).sum(1)
    P = (A ** 2) / (nrm2[:, None] + 1e-30)
    W = (P / lam[None, :]).sum(1)
    sig2_tied = (P * lam[None, :]).sum(1)
    return dict(stretch_W=W, size_d2=nrm2, maha_S=nrm2 * W), sig2_tied, P


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def part_A():
    print("\n" + "=" * 88)
    print("[A] seed 0, within-group centered  — §5.4 가 쓰는 관례 (bridge_probe 재현)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    groups, refs = C1.build_derma(rng)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    gnames = list(groups)
    print(f"  groups: {', '.join(f'{g}={len(v)}' for g, v in groups.items())}   n_q={len(Q)}")
    print(f"  n_c = {[len(R) for R in refs]}   (불균형 {max(len(R) for R in refs)/min(len(R) for R in refs):.1f}배)")

    S = DP.Space(Q, refs, dict(DP.CFG))
    T = C1.truth_fast(S, N_BOOT, rng)
    That = S.theory(use_nc=True)
    CH, sig2_tied, P = channels(S, refs, DP.CFG["maha_shrink"])

    def within(x):
        o = np.asarray(x, float).copy()
        for g in gnames:
            m = G == g
            o[m] -= o[m].mean()
        return o

    prod = CH["stretch_W"] * sig2_tied
    viol = int((prod < 1 - 1e-9).sum())
    rho_dual = rho(np.log(CH["stretch_W"]), -np.log(sig2_tied))
    print(f"\n  [sanity] W·σ² ≥ 1 위반 = {viol}/{len(prod)}   "
          f"ρ(log W, −log σ²) = {rho_dual:+.3f}   (원고: 0/3021, 0.93)")

    print(f"\n  [T̂ vs σ_t]  DermaMNIST 는 n_c 가 불균형하므로 둘은 같은 양이 아니다")
    print(f"    ρ(T̂, σ_t)  pooled   = {rho(That, S.sigma_t):+.3f}")
    print(f"    ρ(T̂, σ_t)  centered = {rho(within(That), within(S.sigma_t)):+.3f}")
    print(f"    ρ(T̂, T)    centered = {rho(within(That), within(T)):+.3f}")

    print(f"\n  {'채널':>10} {'ρ(M,σ_t)':>10} {'ρ(M,T̂)':>10} {'ρ(M,T)':>10} "
          f"{'eq:rule':>9} {'σ_t 대체':>9}")
    print("  " + "-" * 64)
    rows = {}
    for nm in CHAN:
        v = within(CH[nm])
        a = rho(v, within(S.sigma_t))
        h = rho(v, within(That))
        b = rho(v, within(T))
        rule_ok = (h > 0) == (b > 0)
        subst_ok = (a > 0) == (h > 0)
        rows[nm] = dict(rho_sigma_t=a, rho_That=h, rho_T=b,
                        rule_holds=bool(rule_ok), substitution_agrees=bool(subst_ok))
        print(f"  {nm:>10} {a:+10.3f} {h:+10.3f} {b:+10.3f} "
              f"{'O' if rule_ok else '** X **':>9} {'O' if subst_ok else '** X **':>9}")
    print("  " + "-" * 64)
    print("  ρ(M,σ_t) 열은 §5.4 가 현재 인쇄한 값(W: −0.703).  ρ(M,T̂) 열이 ** 빠져 있던 것 **.")
    print("  eq:rule 열 = 논문이 실제로 주장해야 하는 판정.  σ_t 대체 열 = 현재 문장이 옳았는지.")

    M = dict(S.metrics())
    eleven = {}
    for nm in ELEVEN:
        v = within(M[nm])
        eleven[nm] = dict(rho_sigma_t=rho(v, within(S.sigma_t)),
                          rho_That=rho(v, within(That)),
                          rho_T=rho(v, within(T)))
    n_rule = sum((e["rho_That"] > 0) == (e["rho_T"] > 0) for e in eleven.values())
    n_sub = sum((e["rho_sigma_t"] > 0) == (e["rho_That"] > 0) for e in eleven.values())
    print(f"\n  [참고] DermaMNIST 열한 점수:  eq:rule 성립 {n_rule}/11,  "
          f"σ_t 가 T̂ 와 같은 부호 {n_sub}/11")

    return dict(n_q=int(len(Q)), n_c=[int(len(R)) for R in refs],
                cs_violations=viol, rho_logW_neglogsig2=rho_dual,
                rho_That_sigma_t_pooled=rho(That, S.sigma_t),
                rho_That_sigma_t_centered=rho(within(That), within(S.sigma_t)),
                rho_That_T_centered=rho(within(That), within(T)),
                channels=rows, eleven=eleven,
                eleven_rule_ok=int(n_rule), eleven_subst_ok=int(n_sub))


def one_seed(builder, seed, tag):
    rng = np.random.default_rng(seed)
    groups, refs = builder(rng)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    gnames = list(groups)
    S = DP.Space(Q, refs, dict(DP.CFG))
    T = C1.truth_fast(S, N_BOOT, rng)
    That = S.theory(use_nc=True)
    CH, _, _ = channels(S, refs, DP.CFG["maha_shrink"])

    def within(x):
        o = np.asarray(x, float).copy()
        for g in gnames:
            m = G == g
            o[m] -= o[m].mean()
        return o

    M = dict(S.metrics())
    M.update(CH)
    keys = BASE + CHAN

    a = {k: C1.aurc(M[k], T) for k in keys}
    r_hat = {k: rho(M[k], That) for k in keys}
    r_true = {k: rho(M[k], T) for k in keys}
    r_hat_c = {k: rho(within(M[k]), within(That)) for k in keys}
    r_true_c = {k: rho(within(M[k]), within(T)) for k in keys}
    r_sig_c = {k: rho(within(M[k]), within(S.sigma_t)) for k in keys}

    rand = [C1.aurc(rng.random(len(Q)), T) for _ in range(C1.N_RAND)]
    return dict(seed=seed, tag=tag, n_q=int(len(Q)), C=len(refs), n_boot=N_BOOT,
                meanT=float(T.mean()), aurc=a,
                rho_That=r_hat, rho_T=r_true,
                rho_That_centered=r_hat_c, rho_T_centered=r_true_c,
                rho_sigma_t_centered=r_sig_c,
                aurc_random=float(np.mean(rand)),
                aurc_random_analytic=float(0.5 * T.mean()))


def summarize(rows, tag):
    keys = BASE + CHAN
    print("\n" + "=" * 88)
    print(f"[B] {tag}   seeds={len(rows)}  n_q={rows[0]['n_q']}  C={rows[0]['C']}  B={rows[0]['n_boot']}")
    print("=" * 88)
    lvl = np.array([r["aurc_random"] for r in rows])
    print(f"  random AURC = {lvl.mean():.4f} ± {lvl.std(ddof=1):.4f}   "
          f"(해석적 ½T̄ = {np.mean([r['aurc_random_analytic'] for r in rows]):.4f})")

    print(f"\n  ** tab:worse 형식 — pooled ρ, AURC, Δ(%) **")
    print(f"  {'score':>10} {'rho(M,That)':>12} {'rho(M,T)':>10} {'AURC':>9} "
          f"{'delta%':>9} {'95% CI':>18} {'sign':>7}")
    print("  " + "-" * 82)
    out = {}
    for k in keys:
        g = np.array([100 * (r["aurc"][k] / r["aurc_random"] - 1) for r in rows])
        m, sd = g.mean(), g.std(ddof=1)
        se = sd / np.sqrt(len(g))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        au = np.array([r["aurc"][k] for r in rows])
        rh = np.array([r["rho_That"][k] for r in rows])
        rt = np.array([r["rho_T"][k] for r in rows])
        nsign = int(np.sum(np.sign(g) == np.sign(m)))
        star = " **" if k in CHAN else ""
        print(f"  {k:>10} {rh.mean():+12.3f} {rt.mean():+10.3f} {au.mean():9.4f} "
              f"{m:+9.2f} [{lo:+7.2f},{hi:+7.2f}] {nsign:>3}/{len(g)}{star}")
        out[k] = dict(aurc=float(au.mean()), aurc_sd=float(au.std(ddof=1)),
                      delta_mean=float(m), delta_sd=float(sd), ci=[float(lo), float(hi)],
                      sign_agree=nsign, n=len(g), per_seed_delta=[float(v) for v in g],
                      rho_That=float(rh.mean()), rho_That_sd=float(rh.std(ddof=1)),
                      rho_T=float(rt.mean()), rho_T_sd=float(rt.std(ddof=1)),
                      rule_agree=int(np.sum(np.sign(rh) == np.sign(rt))))
    print("  " + "-" * 82)
    print(f"  {'random':>10} {0.0:+12.3f} {0.0:+10.3f} {lvl.mean():9.4f} "
          f"{'---':>9} {'---':>18}")

    print(f"\n  ** §5 관례 (within-group centered) — §5.4 문장 교체용 **")
    print(f"  {'score':>10} {'rho(M,sig_t)':>13} {'rho(M,That)':>12} {'rho(M,T)':>10} {'rule':>7}")
    print("  " + "-" * 56)
    cen = {}
    for k in keys:
        rs = np.array([r["rho_sigma_t_centered"][k] for r in rows])
        rh = np.array([r["rho_That_centered"][k] for r in rows])
        rt = np.array([r["rho_T_centered"][k] for r in rows])
        ok = int(np.sum(np.sign(rh) == np.sign(rt)))
        star = " **" if k in CHAN else ""
        print(f"  {k:>10} {rs.mean():+8.3f}±{rs.std(ddof=1):.3f} "
              f"{rh.mean():+7.3f}±{rh.std(ddof=1):.3f} "
              f"{rt.mean():+5.3f}±{rt.std(ddof=1):.3f} {ok:>3}/{len(rh)}{star}")
        cen[k] = dict(rho_sigma_t=float(rs.mean()), rho_sigma_t_sd=float(rs.std(ddof=1)),
                      rho_That=float(rh.mean()), rho_That_sd=float(rh.std(ddof=1)),
                      rho_T=float(rt.mean()), rho_T_sd=float(rt.std(ddof=1)),
                      rule_agree=ok, n=len(rh))
    return dict(tag=tag, level_mean=float(lvl.mean()), level_sd=float(lvl.std(ddof=1)),
                pooled=out, centered=cen, rows=rows)


def part_C():
    print("\n" + "=" * 88)
    print("[C] fig:eleven 캡션 점검 — CIFAR-100 에서 T̂ 는 σ_t 의 단조변환인가")
    print("=" * 88)
    rng = np.random.default_rng(0)
    groups, refs = C1.build_cifar100(rng)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    gnames = list(groups)
    S = DP.Space(Q, refs, dict(DP.CFG))
    T = C1.truth_fast(S, N_BOOT, rng)
    That = S.theory(use_nc=True)

    def within(x):
        o = np.asarray(x, float).copy()
        for g in gnames:
            m = G == g
            o[m] -= o[m].mean()
        return o

    ncs = sorted(set(int(n) for n in S.n_c))
    print(f"  n_c 집합 = {ncs}  (균일이면 원소 1개)   n_q={len(Q)}  C={len(refs)}")
    print(f"  ρ(T̂, σ_t) pooled   = {rho(That, S.sigma_t):+.4f}")
    print(f"  ρ(T̂, σ_t) centered = {rho(within(That), within(S.sigma_t)):+.4f}")
    print(f"  ** 1.000 이면 순위상 완전히 같은 양 -> fig:eleven 의 x 축 라벨은 무해 **")

    print(f"\n  {'score':>10} {'ρ(M,σ_t)':>10} {'ρ(M,T̂)':>10} {'ρ(M,T)':>10} "
          f"{'축 라벨':>8} {'eq:rule':>9}")
    print("  " + "-" * 62)
    tab, n_ax, n_rule = {}, 0, 0
    for nm in ELEVEN:
        v = within(S.metrics()[nm])
        a = rho(v, within(S.sigma_t))
        h = rho(v, within(That))
        b = rho(v, within(T))
        ax_ok = (a > 0) == (h > 0)
        rl_ok = (h > 0) == (b > 0)
        n_ax += ax_ok
        n_rule += rl_ok
        tab[nm] = dict(rho_sigma_t=a, rho_That=h, rho_T=b,
                       axis_label_safe=bool(ax_ok), rule_holds=bool(rl_ok))
        print(f"  {nm:>10} {a:+10.3f} {h:+10.3f} {b:+10.3f} "
              f"{'O' if ax_ok else '** X **':>8} {'O' if rl_ok else '** X **':>9}")
    print("  " + "-" * 62)
    print(f"  σ_t 축이 T̂ 축과 같은 부호: {n_ax}/11    eq:rule 성립: {n_rule}/11")
    return dict(n_c_set=ncs, n_q=int(len(Q)), C=len(refs),
                rho_That_sigma_t_pooled=rho(That, S.sigma_t),
                rho_That_sigma_t_centered=rho(within(That), within(S.sigma_t)),
                eleven=tab, axis_label_safe=int(n_ax), rule_ok=int(n_rule))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT = {}
    if os.path.exists(OUT_JSON):
        try:
            OUT = json.load(open(OUT_JSON))
        except (ValueError, OSError):
            OUT = {}

    print("=" * 88)
    print(f"f4_stretch_probe — B={N_BOOT}, seeds={N_SEEDS}   (원고 미변경, 진단 전용)")
    print("=" * 88)

    if which in ("all", "A", "derma"):
        t0 = time.time()
        OUT["A_seed0_centered"] = part_A()
        print(f"  [A 완료 {time.time()-t0:.0f}s]")
        json.dump(OUT, open(OUT_JSON, "w"), indent=2, default=float)

    if which in ("all", "B", "derma"):
        rows = []
        for s in range(N_SEEDS):
            t0 = time.time()
            r = one_seed(C1.build_derma, s, "derma")
            rows.append(r)
            print(f"  [derma seed {s}] meanT={r['meanT']:.4f} rand={r['aurc_random']:.4f} "
                  f"W={r['aurc']['stretch_W']:.4f} "
                  f"({100*(r['aurc']['stretch_W']/r['aurc_random']-1):+.2f}%)  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        OUT["B_derma"] = summarize(rows, "DermaMNIST C=3")
        json.dump(OUT, open(OUT_JSON, "w"), indent=2, default=float)

    if which in ("all", "B", "cifar"):
        rows = []
        for s in range(N_SEEDS):
            t0 = time.time()
            r = one_seed(C1.build_cifar100, s, "cifar100")
            rows.append(r)
            print(f"  [cifar seed {s}] meanT={r['meanT']:.4f} rand={r['aurc_random']:.4f} "
                  f"W={r['aurc']['stretch_W']:.4f} "
                  f"({100*(r['aurc']['stretch_W']/r['aurc_random']-1):+.2f}%)  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        OUT["B_cifar100"] = summarize(rows, "CIFAR-100 n_ref=400")
        json.dump(OUT, open(OUT_JSON, "w"), indent=2, default=float)

    if which in ("all", "C", "cifar"):
        t0 = time.time()
        OUT["C_fig_eleven"] = part_C()
        print(f"  [C 완료 {time.time()-t0:.0f}s]")
        json.dump(OUT, open(OUT_JSON, "w"), indent=2, default=float)

    json.dump(OUT, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\n[saved] {OUT_JSON}  (keys: {', '.join(sorted(OUT))})")


if __name__ == "__main__":
    main()
