import os, json, numpy as np

import derma_probe as DP

CFG = dict(
    min_n_list=[400, 200, 80],
    modes=["fixed", "natural"],
    n_boot=DP.CFG["n_boot"],
    seed=DP.CFG["seed"],
    out_json="derma_minn_sweep_probe.json",
)


def penalty_share(S):
    var_cls = S.sigma_t ** 2 / S.n_assigned
    sd_D = S.sigma_w / np.sqrt(S.N_all)
    var_pen = (S.cfg["hinge_lam"] ** 2) * DP.rect_gauss_var(S.tau - S.d_glob, sd_D)
    tot = var_cls + var_pen
    return (float(var_pen.mean() / (tot.mean() + 1e-30)),
            float(np.median(var_pen / (tot + 1e-30))),
            float(var_cls.mean()), float(var_pen.mean()))


def run_one(min_n, mode, ftr, ytr, fte, yte, cnt, names, held_ref, near_ref, rng):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    held_in = sorted([c for c, n in cnt.items() if n >= min_n])
    near_cls = sorted([c for c, n in cnt.items() if n < min_n])
    refs = [ftr[ytr == c] for c in held_in]
    C = len(refs)
    n_c = [len(R) for R in refs]
    imbal = max(n_c) / min(n_c)

    def sub(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    nq = DP.CFG["n_query"]
    if mode == "fixed":
        q_held, q_near = held_ref, near_ref
    else:
        q_held, q_near = held_in, near_cls

    groups = {}
    groups["in"] = sub(fte[np.isin(yte, q_held)], nq)
    if len(q_near) > 0:
        Qn = fte[np.isin(yte, q_near)]
        if len(Qn) > 0:
            groups["near"] = sub(Qn, nq)
    for s in DP.CFG["ood_far"]:
        F = DP.load_far(s)
        if F is not None:
            groups[s] = sub(F, nq)

    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])

    contam = sorted(set(q_near) & set(held_in))

    cfg = dict(DP.CFG); cfg["min_n"] = min_n
    S = DP.Space(Q, refs, cfg)
    truth = S.truth(CFG["n_boot"], rng)

    def w(x, keys=G):
        o = np.asarray(x, float).copy()
        for k in np.unique(keys):
            m = keys == k
            o[m] -= o[m].mean()
        return o

    def zs(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    def r2(p, t):
        m = np.isfinite(p) & np.isfinite(t)
        lr = LinearRegression().fit(zs(p[m])[:, None], zs(t[m]))
        return float(lr.score(zs(p[m])[:, None], zs(t[m])))

    th_nc = S.theory(use_nc=True)
    th_nbar = S.theory(use_nc=False)
    r2_nc, r2_nbar = r2(th_nc, truth), r2(th_nbar, truth)

    M = S.metrics()
    sign = {}
    for nm, v in M.items():
        a = float(spearmanr(w(v), w(S.sigma_t)).statistic)
        b = float(spearmanr(w(v), w(truth)).statistic)
        sign[nm] = dict(rho_sigma_t=a, rho_truth=b, match=bool((a > 0) == (b > 0)))
    n_match = sum(s["match"] for s in sign.values())

    pen_off = S.d_glob >= S.tau
    rows = []
    for i, c in enumerate(held_in):
        m = S.nc == i
        mp = m & pen_off
        if m.sum() < 20:
            rows.append(dict(cls=int(c), name=names[str(c)], n_c=int(S.n_c[i]),
                             n_q=int(m.sum()), n_q_penoff=int(mp.sum()),
                             ratio=None, ratio_penoff=None))
            continue
        t_, s_ = truth[m].mean(), S.sigma_t[m].mean()
        if mp.sum() >= 20:
            tp, sp = truth[mp].mean(), S.sigma_t[mp].mean()
            rp = float(tp * np.sqrt(S.n_c[i]) / (sp + 1e-12))
        else:
            rp = None
        rows.append(dict(cls=int(c), name=names[str(c)], n_c=int(S.n_c[i]),
                         n_q=int(m.sum()), n_q_penoff=int(mp.sum()),
                         mean_truth=float(t_), mean_sigma_t=float(s_),
                         ratio=float(t_ * np.sqrt(S.n_c[i]) / (s_ + 1e-12)),
                         ratio_penoff=rp))

    p_mean, p_med, v_cls, v_pen = penalty_share(S)

    return dict(
        min_n=min_n, mode=mode, C=C, held_in=held_in, near=near_cls,
        n_c=n_c, imbalance=float(imbal),
        groups={g: int(len(v)) for g, v in groups.items()},
        near_contaminated_by=contam,
        probe_acc=float(S.probe.score(
            np.vstack(refs),
            np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)]))),
        rho_r_sigma_t=float(spearmanr(S.r, S.sigma_t).statistic),
        rho_r_by_group={g: float(spearmanr(S.r[G == g], S.sigma_t[G == g]).statistic)
                        for g in groups},
        r2=r2_nc, r2_within=r2(w(th_nc), w(truth)), r2_without_nc=r2_nbar,
        scaling_gain=r2_nc - r2_nbar,
        ratio=float(np.median(truth / (th_nc + 1e-12))),
        penalty_share_mean=p_mean, penalty_share_median=p_med,
        var_cls_mean=v_cls, var_pen_mean=v_pen,
        frac_pen_off=float(pen_off.mean()),
        n_match=int(n_match), n_total=len(M), sign=sign,
        per_class=rows,
        tau=float(S.tau),
    )


def main():
    rng = np.random.default_rng(CFG["seed"])
    cache = DP.CFG["cache_dir"]
    meta = json.load(open(os.path.join(cache, "DERMA_META.json")))
    names = meta["label_names"]

    ftr, ytr = DP.get("train")
    fte, yte = DP.get("test")
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}

    held_ref = sorted([c for c, n in cnt.items() if n >= 400])
    near_ref = sorted([c for c, n in cnt.items() if n < 400])

    print("=" * 88)
    print("min_n 스윕 — 같은 캐시로 C 를 3 -> 5 -> 7")
    print("=" * 88)
    print(f"  클래스별 n: " + ", ".join(f"{names[str(c)][:8]}={n}"
                                        for c, n in sorted(cnt.items())))
    print(f"  질의 고정 기준(min_n=400): held={held_ref}  near={near_ref}\n")
    for mn in CFG["min_n_list"]:
        hi = sorted([c for c, n in cnt.items() if n >= mn])
        ncs = [cnt[c] for c in hi]
        print(f"  min_n={mn:>3}  C={len(hi)}  n_c={ncs}  "
              f"불균형 {max(ncs)/min(ncs):.1f}배")

    RES = []
    for mode in CFG["modes"]:
        for mn in CFG["min_n_list"]:
            print("\n" + "=" * 88)
            print(f"[mode={mode}] min_n={mn}")
            print("=" * 88)
            r = run_one(mn, mode, ftr, ytr, fte, yte, cnt, names,
                        held_ref, near_ref, rng)
            RES.append(r)
            print(f"  C={r['C']}  n_c={r['n_c']}  불균형 {r['imbalance']:.1f}배")
            print(f"  groups: {', '.join(f'{g}={n}' for g, n in r['groups'].items())}")
            if r["near_contaminated_by"]:
                print(f"  ** near 오염: {r['near_contaminated_by']} 가 refs 에도 있음 "
                      f"(fixed 모드의 대가) **")
            print(f"  프로브 정확도 = {r['probe_acc']:.3f}   "
                  f"ρ(r,σ_t) = {r['rho_r_sigma_t']:+.3f}")
            print(f"  R² = {r['r2']:.3f}  (within {r['r2_within']:.3f})  "
                  f"비율 = {r['ratio']:.2f}")
            print(f"  R²(n̄) = {r['r2_without_nc']:.3f}   "
                  f"** 스케일링 이득 = {r['scaling_gain']:+.3f} **")
            print(f"  ** 페널티 지배율 = {r['penalty_share_mean']:.1%} "
                  f"(median {r['penalty_share_median']:.1%}) **   "
                  f"페널티 비활성 {r['frac_pen_off']:.0%}")
            print(f"  부호 일치 = {r['n_match']}/{r['n_total']}")
            print(f"  {'클래스':>16} {'n_c':>6} {'비율':>7} {'비율*':>7}")
            for row in r["per_class"]:
                rr = f"{row['ratio']:7.3f}" if row.get("ratio") else "      -"
                rp = f"{row['ratio_penoff']:7.3f}" if row.get("ratio_penoff") else "      -"
                print(f"  {row['name'][:16]:>16} {row['n_c']:>6} {rr} {rp}")
            print("  (* = 페널티 비활성 부분집합.  ** 1.0 에 붙어야 1/√n_c 확증 **)")

    print("\n" + "=" * 88)
    print("요약 — 네 예측")
    print("=" * 88)
    for mode in CFG["modes"]:
        print(f"\n  [mode={mode}]")
        print(f"  {'min_n':>6} {'C':>3} {'불균형':>7} {'R²':>7} {'R²(n̄)':>8} "
              f"{'이득':>7} {'페널티%':>8} {'일치':>7} {'ρ(r,σ_t)':>10}")
        print("  " + "-" * 72)
        for r in [x for x in RES if x["mode"] == mode]:
            print(f"  {r['min_n']:>6} {r['C']:>3} {r['imbalance']:>6.1f}x "
                  f"{r['r2']:>7.3f} {r['r2_without_nc']:>8.3f} "
                  f"{r['scaling_gain']:>+7.3f} {r['penalty_share_mean']:>7.1%} "
                  f"{r['n_match']:>4}/{r['n_total']:<2} {r['rho_r_sigma_t']:>+10.3f}")

    print("\n" + "=" * 88)
    print("판정")
    print("=" * 88)
    print("  (P1) 페널티% 가 C 에 대해 ** 단조 감소 ** -> §2.6 확증 "
          "(지금까지 CIFAR-10 의 72% 한 점뿐이었다)")
    print("  (P2) R² 가 C 와 무관하게 유지 -> 완전한 이론이 C 에 무관")
    print("  (P3) 불균형이 커질수록 ** 이득 증가 ** -> 1/√n_c 가 우연이 아니다")
    print("  (P4) df(n_c=80) 의 비율* 이 1.0 근처 -> ** 58.7배에서도 스케일링 성립 **")
    print("  fixed 와 natural 의 차이 = 그룹 구성 효과 (near 유무). "
          "결론이 같으면 그룹 구성에 강건하다.")
    print("=" * 88)

    with open(CFG["out_json"], "w") as f:
        json.dump(dict(config=CFG, derma_cfg=DP.CFG, counts=cnt, results=RES),
                  f, indent=2, default=float)
    print(f"\n[saved] {CFG['out_json']}")


if __name__ == "__main__":
    main()
