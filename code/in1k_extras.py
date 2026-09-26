import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys, json, time, traceback, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import derma_probe as DP
import in200_probe as IP
import gpu_logreg as GL

BENCH, REPR = "in1k", "dino"
COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
FL = ["knn_std", "lid", "energy", "msp", "maha"]
N_SEEDS = int(os.environ.get("X_SEEDS", 10))
STATUS = "in1k_extras_status.json"


def aurc(v, c):
    o = np.argsort(c)
    tz = getattr(np, "trapezoid", None) or np.trapz
    return float(tz(np.asarray(v)[o], np.asarray(c)[o]))


def curve(target, score):
    return aurc([target[score <= np.percentile(score, c * 100)].mean() for c in COVS], COVS)


def within(x, G):
    o = np.asarray(x, float).copy()
    for g in np.unique(G):
        o[G == g] -= o[G == g].mean()
    return o


def build_seed(s):
    rng = np.random.default_rng(s)
    groups, refs = IP.build(REPR, rng, BENCH)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    return rng, groups, refs, Q, G


def stage_stretch(st):
    out = "in1k_stretch.json"
    if os.path.exists(out):
        print("[skip] A stretch"); return
    import f4_stretch_probe as F4
    rows_c, rows_a = {}, {}
    for s in range(N_SEEDS):
        rng, groups, refs, Q, G = build_seed(s)
        S = IP.Space200(Q, refs, DP.CFG)
        T = S.truth(DP.CFG["n_boot"], rng)
        Th = S.theory(True)
        CH, sig2, _ = F4.channels(S, refs, DP.CFG["maha_shrink"])
        rnd = float(np.mean([curve(T, np.random.default_rng(1000 + s).random(len(Q)))
                             for _ in range(1)]))
        for k, v in CH.items():
            rows_c.setdefault(k, []).append(
                dict(rho_That=float(spearmanr(within(v, G), within(Th, G)).statistic),
                     rho_T=float(spearmanr(within(v, G), within(T, G)).statistic)))
            a = curve(T, v)
            rows_a.setdefault(k, []).append(
                dict(aurc=a, delta=(a / rnd - 1) * 100,
                     rho_That_pooled=float(spearmanr(v, Th).statistic)))
        rows_a.setdefault("random", []).append(dict(aurc=rnd, delta=0.0, rho_That_pooled=0.0))
        print(f"  [stretch seed {s}] rand={rnd:.4f} " +
              " ".join(f"{k}={rows_a[k][-1]['delta']:+.2f}%" for k in CH), flush=True)

    def summ(d):
        o = {}
        for k, v in d.items():
            o[k] = {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]}
            if "delta" in v[0]:
                dv = np.array([x["delta"] for x in v])
                o[k]["ci_half"] = float(1.96 * dv.std(ddof=1) / np.sqrt(len(dv))) if len(dv) > 1 else 0.0
                o[k]["sign"] = f"{int((dv > 0).sum() if dv.mean() > 0 else (dv < 0).sum())}/{len(dv)}"
        return o
    json.dump(dict(centered=summ(rows_c), pooled=summ(rows_a), n_seeds=N_SEEDS),
              open(out, "w"), indent=2)
    print(f"[saved] {out}")


def stage_orthogonal(st):
    out = "in1k_orthogonal.json"
    if os.path.exists(out):
        print("[skip] B orthogonal"); return
    cd = IP.CACHE_FMT.format(BENCH, REPR)
    fte = np.load(f"{cd}/{BENCH}_feat_test.npy").astype(np.float64)
    yte = np.load(f"{cd}/{BENCH}_lab_test.npy")
    ftr = np.load(f"{cd}/{BENCH}_feat_train.npy").astype(np.float64)
    ytr = np.load(f"{cd}/{BENCH}_lab_train.npy")
    C = int(ytr.max()) + 1
    rows, accs = {}, []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        refs = []
        for c in range(C):
            X = ftr[ytr == c]
            refs.append(X if len(X) <= IP.N_REF else X[rng.choice(len(X), IP.N_REF, replace=False)])
        idx = np.arange(len(fte)) if len(fte) <= IP.N_QUERY else \
            rng.choice(len(fte), IP.N_QUERY, replace=False)
        Qi, yi = fte[idx], yte[idx]
        S = IP.Space200(Qi, refs, DP.CFG)
        err = (S.probe.predict(Qi) != yi).astype(float)
        accs.append(1 - err.mean())
        M = S.metrics()
        rnd = float(np.mean([curve(err, np.random.default_rng(2000 + s * 37 + r).random(len(Qi)))
                             for r in range(20)]))
        for k in FL:
            rows.setdefault(k, []).append(curve(err, M[k]))
        rows.setdefault("random", []).append(rnd)
        print(f"  [orth seed {s}] acc={1-err.mean():.4f} rand={rnd:.4f} "
              f"msp={rows['msp'][-1]:.4f} knn_std={rows['knn_std'][-1]:.4f}", flush=True)
    json.dump(dict(misclass={k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
                             for k, v in rows.items()},
                   probe_acc_mean=float(np.mean(accs)), n_seeds=N_SEEDS,
                   misclass_level_analytic=float(np.mean([0.5 * (1 - a) for a in accs]))),
              open(out, "w"), indent=2)
    print(f"[saved] {out}")


def stage_sweep(st):
    out = "in1k_sweep.json"
    if os.path.exists(out):
        print("[skip] C sweep"); return
    cd = IP.CACHE_FMT.format(BENCH, REPR)
    ftr = np.load(f"{cd}/{BENCH}_feat_train.npy").astype(np.float64)
    ytr = np.load(f"{cd}/{BENCH}_lab_train.npy")
    C = int(ytr.max()) + 1
    res, meanT = {}, {}
    for n_ref in [25, 50, 100, 200, 400]:
        rng = np.random.default_rng(0)
        refs = []
        for c in range(C):
            X = ftr[ytr == c]
            refs.append(X if len(X) <= n_ref else X[rng.choice(len(X), n_ref, replace=False)])
        groups, _ = IP.build(REPR, np.random.default_rng(0), BENCH)
        Q = np.vstack([groups[g] for g in groups])
        S = IP.Space200(Q, refs, DP.CFG)
        T = S.truth(DP.CFG["n_boot"], rng)
        M = S.metrics()
        rnd = float(np.mean([curve(T, np.random.default_rng(3000 + r).random(len(Q)))
                             for r in range(20)]))
        meanT[str(n_ref)] = float(T.mean())
        for k in FL:
            res.setdefault(k, {})[str(n_ref)] = curve(T, M[k])
        res.setdefault("random", {})[str(n_ref)] = rnd
        print(f"  [sweep n_c={n_ref}] meanT={T.mean():.4f} rand={rnd:.4f} "
              + " ".join(f"{k}={res[k][str(n_ref)]:.4f}" for k in FL), flush=True)
    json.dump(dict(rows=res, meanT=meanT), open(out, "w"), indent=2)
    print(f"[saved] {out}")


def rebuild_query_selection():
    cd = IP.CACHE_FMT.format(BENCH, REPR)
    ytr = np.load(f"{cd}/{BENCH}_lab_train.npy")
    meta = json.load(open(f"{cd}/BENCH_META.json"))
    n_te = np.load(f"{cd}/{BENCH}_feat_test.npy", mmap_mode="r").shape[0]
    rng = np.random.default_rng(0)
    for c in range(int(ytr.max()) + 1):
        n = int((ytr == c).sum())
        if n > IP.N_REF:
            rng.choice(n, IP.N_REF, replace=False)
    sel = {}
    sel["in"] = (np.arange(n_te) if n_te <= IP.N_QUERY
                 else rng.choice(n_te, IP.N_QUERY, replace=False))
    for s in meta["far_sets"]:
        n = np.load(f"{cd}/{BENCH}_feat_far_{s}.npy", mmap_mode="r").shape[0]
        sel[s] = (np.arange(n) if n <= IP.N_QUERY
                  else rng.choice(n, IP.N_QUERY, replace=False))
    return sel, meta


def stage_odin(st):
    out = "in1k_odin_eps.json"
    if os.path.exists(out):
        print("[skip] D odin"); return
    import torch, imagenet_extract as IX, extract_features as EF
    from torch.utils.data import DataLoader

    EPS = [0.0, 0.0005, 0.001, 0.0014, 0.002, 0.004]
    cd = IP.CACHE_FMT.format(BENCH, REPR)
    sel, meta = rebuild_query_selection()
    idx1k = np.array(meta["held_in_idx1k"])

    EF.REPRS.setdefault(REPR, dict(model="vit_base_patch16_224.dino", cache=cd))
    model, tf, dev, cfg = EF.build_model(EF.REPRS[REPR]["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    std_t = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)

    src = {}
    hin, _, _ = IX.collect_val(idx1k, IX.rng_for_stage("val"))
    src["in"] = hin
    for name in IX.FAR:
        src[name] = IX.collect_far(name, IX.rng_for_stage(name))

    rng = np.random.default_rng(0)
    groups, refs = IP.build(REPR, rng, BENCH)

    print("  [verify] 복원한 질의 이미지 vs 캐시 특징", flush=True)
    worst = 0.0
    for g in groups:
        items = [src[g][i] for i in sel[g][:32]]
        dl = DataLoader(IX._ListDS(items, tf), batch_size=32, num_workers=4)
        with torch.no_grad():
            got = np.concatenate([model(xb.to(dev)).cpu().numpy() for xb in dl])
        d = float(np.abs(got.astype(np.float64) - groups[g][:32]).max())
        worst = max(worst, d)
        print(f"    {g:14} max|diff| = {d:.3e}")
    assert worst < 1e-3, f"질의 복원 실패 (max|diff|={worst:.3e}) — 중단"
    print(f"  [verify] 통과 (max {worst:.3e})", flush=True)

    X = np.vstack(refs)
    y = np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])
    pr = GL.SklearnCompatGPU(max_iter=1000, C=1.0).fit(X, y)
    W = torch.tensor(pr.coef_, dtype=torch.float32, device=dev)
    b = torch.tensor(pr.intercept_, dtype=torch.float32, device=dev)
    Tt = float(DP.CFG["odin_T"])

    scores = {e: [] for e in EPS}
    for g in groups:
        items = [src[g][i] for i in sel[g]]
        dl = DataLoader(IX._ListDS(items, tf), batch_size=32, num_workers=8)
        for xb in dl:
            x0 = xb.to(dev)
            x = x0.clone().requires_grad_(True)
            lt = (model(x) @ W.T + b) / Tt
            loss = torch.nn.functional.cross_entropy(lt, lt.argmax(1), reduction="sum")
            sgn = torch.autograd.grad(loss, x)[0].sign()
            with torch.no_grad():
                for e in EPS:
                    xp = x0 if e == 0.0 else x0 - e * sgn / std_t
                    lp = (model(xp) @ W.T + b) / Tt
                    scores[e].append((-torch.softmax(lp, 1).max(1).values).cpu().numpy())
        print(f"    [odin {g}] {len(items)} 질의", flush=True)

    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    S = IP.Space200(Q, refs, DP.CFG)
    T = S.truth(DP.CFG["n_boot"], np.random.default_rng(0))
    Th = S.theory(True)
    res = {}
    for e in EPS:
        v = np.concatenate(scores[e]).astype(np.float64)
        res[str(e)] = dict(
            rho_sigma_t=float(spearmanr(within(v, G), within(S.sigma_t, G)).statistic),
            rho_That=float(spearmanr(within(v, G), within(Th, G)).statistic),
            rho_T=float(spearmanr(within(v, G), within(T, G)).statistic))
        print(f"  eps={e}: " + str({k: round(x, 3) for k, x in res[str(e)].items()}), flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[saved] {out}")


STAGES = [("A_stretch", stage_stretch), ("B_orthogonal", stage_orthogonal),
          ("C_sweep", stage_sweep), ("D_odin", stage_odin)]


def main():
    GL.install()
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    t00 = time.time()
    for name, fn in STAGES:
        print("\n" + "=" * 78); print(f"[{name}]"); print("=" * 78, flush=True)
        t0 = time.time()
        try:
            fn(st); st[name] = "ok"
        except Exception:
            st[name] = traceback.format_exc()[-1200:]
            traceback.print_exc()
        print(f"[{name}] {(time.time()-t0)/60:.1f}분", flush=True)
        json.dump(st, open(STATUS, "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] 총 {(time.time()-t00)/60:.1f}분")
    for k, v in st.items():
        if v != "ok":
            print(f"  ** 실패 {k}: {str(v).splitlines()[-1][:110]}")


if __name__ == "__main__":
    main()
