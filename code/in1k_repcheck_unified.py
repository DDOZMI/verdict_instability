import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, gc, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BENCH = "in1k"
EPS = 0.0014
BS = 128
WORKERS = 4
SPEC = {"clip":   dict(model="vit_base_patch16_clip_224.openai", pinned=None),
        "resnet": dict(model="resnet50.a1_in1k",                 pinned="resnet_extract"),
        "dinov3": dict(model="vit_base_patch16_dinov3.lvd1689m", pinned="dinov3_extract")}


def real_odin(repr_key, groups, refs):
    import torch, imagenet_extract as IX, in1k_extras as IE, extract_features as EF
    import in200_probe as IP, derma_probe as DP, gpu_logreg as GL
    from torch.utils.data import DataLoader

    sp = SPEC[repr_key]
    EF.REPRS[repr_key] = dict(model=sp["model"], cache=IP.CACHE_FMT.format(BENCH, repr_key))
    build = EF.build_model
    if sp["pinned"]:
        mod = __import__(sp["pinned"])
        build = getattr(mod, "build_model_pinned", build)
    model, tf, dev, cfg = build(sp["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    std_t = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)

    IE.REPR = "dino"
    sel, meta = IE.rebuild_query_selection()
    src = {"in": IX.collect_val(np.array(meta["held_in_idx1k"]), IX.rng_for_stage("val"))[0]}
    for name in IX.FAR:
        src[name] = IX.collect_far(name, IX.rng_for_stage(name))
    print(f"  [수집] " + ", ".join(f"{k}:{len(v)}" for k, v in src.items()), flush=True)

    worst, hit, tot = 0.0, 0, 0
    for g in groups:
        take = sel[g][:BS]
        dl = DataLoader(IX._ListDS([src[g][i] for i in take], tf),
                        batch_size=BS, num_workers=WORKERS)
        with torch.no_grad():
            got = np.concatenate([model(xb.to(dev)).cpu().numpy() for xb in dl]).astype(np.float64)
        ref = np.asarray(groups[g], dtype=np.float64)
        worst = max(worst, float(np.abs(got - ref[:len(take)]).max()))
        d2 = (got ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2.0 * (got @ ref.T)
        hit += int((d2.argmin(1) == np.arange(len(take))).sum()); tot += len(take)
    print(f"  [verify] max|diff|={worst:.3e}  정체성 {hit}/{tot}", flush=True)
    assert hit == tot, f"{repr_key} 질의 복원 실패 ({hit}/{tot})"

    X = np.vstack(refs); y = np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])
    pr = GL.SklearnCompatGPU(max_iter=1000, C=1.0).fit(X, y)
    W = torch.tensor(pr.coef_, dtype=torch.float32, device=dev)
    b = torch.tensor(pr.intercept_, dtype=torch.float32, device=dev)
    Tt = float(DP.CFG["odin_T"])

    out = []
    for g in groups:
        dl = DataLoader(IX._ListDS([src[g][i] for i in sel[g]], tf),
                        batch_size=BS, num_workers=WORKERS)
        for xb in dl:
            x0 = xb.to(dev); x = x0.clone().requires_grad_(True)
            lt = (model(x) @ W.T + b) / Tt
            loss = torch.nn.functional.cross_entropy(lt, lt.argmax(1), reduction="sum")
            sgn = torch.autograd.grad(loss, x)[0].sign()
            with torch.no_grad():
                lp = (model(x0 - EPS * sgn / std_t) @ W.T + b) / Tt
                out.append((-torch.softmax(lp, 1).max(1).values).cpu().numpy())
        print(f"    [{g}] {len(sel[g])} 질의", flush=True)

    v = np.concatenate(out).astype(np.float64)
    del model, src, X, W, b, pr
    torch.cuda.empty_cache(); gc.collect()
    return v, worst


def stage_odin(repr_key, vec_path):
    import in200_probe as IP
    t0 = time.time()
    rng = np.random.default_rng(0)
    groups, refs = IP.build(repr_key, rng, BENCH)
    names = list(groups); sizes = [len(groups[g]) for g in names]
    print(f"[build] C={len(refs)} Q={sum(sizes)}  ({time.time()-t0:.0f}s)", flush=True)
    v, verr = real_odin(repr_key, groups, refs)
    np.savez(vec_path, v=v, names=np.array(names), sizes=np.array(sizes), verr=verr)
    print(f"[saved] {vec_path}  n={len(v)}  총 {(time.time()-t0)/60:.1f}분", flush=True)


def stage_probe(repr_key, n_boot, vec_path, out_json):
    import gpu_logreg; gpu_logreg.install()
    import in200_probe as IP, derma_probe as DP
    t0 = time.time()

    z = np.load(vec_path, allow_pickle=False)
    odin_v, verr = z["v"], float(z["verr"])

    rng = np.random.default_rng(0)
    groups, refs = IP.build(repr_key, rng, BENCH)
    names = list(groups); sizes = [len(groups[g]) for g in names]
    assert list(z["names"]) == names and list(z["sizes"]) == sizes, \
        f"질의 구성이 ODIN 단계와 다르다: {list(z['names'])} vs {names}"
    Q = np.vstack([groups[g] for g in names])
    G = np.concatenate([np.full(sizes[i], names[i]) for i in range(len(names))])
    assert len(odin_v) == len(Q), f"odin 길이 {len(odin_v)} != Q {len(Q)}"
    del groups; gc.collect()
    print(f"[build] C={len(refs)} Q={len(Q)}  ({time.time()-t0:.0f}s)", flush=True)

    S = IP.Space200(Q, refs, DP.CFG)
    print(f"[Space200] ({time.time()-t0:.0f}s)", flush=True)
    T = S.truth(n_boot, rng)
    Th = S.theory(True)
    print(f"[truth] B={n_boot}  ({time.time()-t0:.0f}s)", flush=True)

    M = S.metrics()
    M["odin"] = odin_v

    rows, agree = [], 0
    print(f"\n  {'score':>10} {'rho(M,s_t)':>11} {'rho(M,That)':>12} {'rho(M,T)':>10}  rule")
    for nm in IP.SCORES11:
        v = M[nm]
        a = float(spearmanr(IP.within(v, G), IP.within(S.sigma_t, G)).statistic)
        b = float(spearmanr(IP.within(v, G), IP.within(Th, G)).statistic)
        c = float(spearmanr(IP.within(v, G), IP.within(T, G)).statistic)
        ok = (b > 0) == (c > 0); agree += ok
        rows.append(dict(score=nm, rho_sigma_t=a, rho_that=b, rho_T=c, rule=bool(ok)))
        print(f"  {nm:>10} {a:+11.3f} {b:+12.3f} {c:+10.3f}  {'O' if ok else '** X **'}", flush=True)
    n_neg = sum(1 for r in rows if r["rho_T"] < 0)
    print(f"  -> eq:rule {agree}/11,  rho(M,T)<0 인 점수 {n_neg}/11")

    res = dict(repr=repr_key, bench=BENCH, eps=EPS, batch_size=BS, n_boot=n_boot,
               verify_max_abs_diff=verr,
               fit=dict(r2=IP.r2(Th, T), ratio=float(np.median(T / (Th + 1e-12)))),
               eleven=dict(rows=rows, rule_ok=int(agree), n_negative=int(n_neg)))
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"[saved] {out_json}   총 {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=list(SPEC))
    ap.add_argument("--stage", required=True, choices=["odin", "probe"])
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    vec = f"in1k_odinvec_{a.only}.npz"
    if a.stage == "odin":
        stage_odin(a.only, vec)
    else:
        stage_probe(a.only, a.boot, vec, f"in1k_repcheck_{a.only}.json")
