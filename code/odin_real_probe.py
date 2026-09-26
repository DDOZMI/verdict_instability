import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as EF
import derma_probe as DP
import exponent_probe as EP
import odin_real_align as AL

CACHE = DP.CFG["cache_dir"]
EPS_LIST = [0.0, 0.0005, 0.001, 0.0014, 0.002, 0.004]
EPS_MAIN = 0.0014
BATCH = 48
OUT = "odin_real_probe.npz"


def build_cifar_idx(rng, which):
    cd = CACHE
    pre = None if which == "c100" else "c10"
    held = list(range(50)) if which == "c100" else [0, 1, 2, 3, 4]
    hout = list(range(50, 100)) if which == "c100" else [5, 6, 7, 8, 9]
    extra = ["cifar10"] if which == "c100" else ["cifar100"]

    def p(k, s):
        t = f"{pre}_{s}" if pre else s
        return os.path.join(cd, f"dino_{k}_{t}.npy")

    ytr = np.load(p("lab", "train"))
    yte = np.load(p("lab", "test"))

    refs_idx = []
    for c in held:
        ii = np.where(ytr == c)[0]
        if len(ii) > EP.CFG["cifar_ref"]:
            ii = ii[rng.choice(len(ii), EP.CFG["cifar_ref"], replace=False)]
        refs_idx.append(ii)

    def sub(ii, n):
        return ii if len(ii) <= n else ii[rng.choice(len(ii), n, replace=False)]

    nq = EP.CFG["cifar_nq"]
    gi = {"in": sub(np.where(np.isin(yte, held))[0], nq),
          "near": sub(np.where(np.isin(yte, hout))[0], nq)}
    src = {"in": ("test" if which == "c100" else "c10_test"),
           "near": ("test" if which == "c100" else "c10_test")}
    for s in EP.CIFAR["ood"] + extra:
        f = os.path.join(cd, f"dino_feat_far_{s}.npy")
        if os.path.exists(f):
            gi[s] = sub(np.arange(len(np.load(f, mmap_mode="r"))), nq)
            src[s] = f"far_{s}"
    return gi, src, refs_idx, ("train" if which == "c100" else "c10_train")


def build_derma_idx(rng, min_n):
    ytr = np.load(os.path.join(CACHE, "dino_lab_derma_train.npy"))
    yte = np.load(os.path.join(CACHE, "dino_lab_derma_test.npy"))
    cnt = {int(c): int((ytr == c).sum()) for c in np.unique(ytr)}
    hr = sorted([c for c, n in cnt.items() if n >= 400])
    nr = sorted([c for c, n in cnt.items() if n < 400])
    hi = sorted([c for c, n in cnt.items() if n >= min_n])
    refs_idx = [np.where(ytr == c)[0] for c in hi]

    def sub(ii, n):
        return ii if len(ii) <= n else ii[rng.choice(len(ii), n, replace=False)]

    nq = DP.CFG["n_query"]
    gi = {"in": sub(np.where(np.isin(yte, hr))[0], nq),
          "near": sub(np.where(np.isin(yte, nr))[0], nq)}
    src = {"in": "derma_test", "near": "derma_test"}
    for s in DP.CFG["ood_far"]:
        f = os.path.join(CACHE, f"dino_feat_far_{s}.npy")
        if os.path.exists(f):
            gi[s] = sub(np.arange(len(np.load(f, mmap_mode="r"))), nq)
            src[s] = f"far_{s}"
    return gi, src, refs_idx


def burn_truth(rng, refs_idx, n_boot):
    for _ in range(n_boot):
        for ii in refs_idx:
            rng.integers(0, len(ii), len(ii))


def _lazy(name):
    if name == "test":
        return AL.lz_cifar100_test() + (AL.id_tf,)
    if name == "c10_test":
        return AL.lz_cifar10_test() + (AL.id_tf,)
    if name == "derma_test":
        import medmnist
        from medmnist import INFO
        DataClass = getattr(medmnist, INFO["dermamnist"]["python_class"])
        ds = DataClass(split="test", download=True, size=28)
        imgs = ds.imgs
        return len(imgs), (lambda i: Image.fromarray(imgs[int(i)])), AL.d28_tf
    ld, pre = AL.SOURCES[name]
    n, get = ld()
    return n, get, pre


class ImageBank:

    def __init__(self):
        self.h = {}

    def get(self, src, i):
        if src not in self.h:
            self.h[src] = _lazy(src)
        n, get, pre = self.h[src]
        return pre(get(i))


def odin_batch(imgs, model, tf, dev, W, b, T, eps_list, std_t):
    x0 = torch.stack([tf(im.convert("RGB")) for im in imgs]).to(dev)

    x = x0.clone().requires_grad_(True)
    z = model(x)
    lg = z @ W.T + b
    lt = lg / T
    loss = torch.nn.functional.cross_entropy(lt, lt.argmax(1), reduction="sum")
    g, = torch.autograd.grad(loss, x)
    sgn = g.sign()

    out = []
    with torch.no_grad():
        for eps in eps_list:
            xp = x0 if eps == 0.0 else x0 - eps * sgn / std_t
            lp = (model(xp) @ W.T + b) / T
            out.append((-torch.softmax(lp, 1).max(1).values).cpu().numpy())
    return np.stack(out)


def run_setting(tag, gi, src, refs_idx, ref_split, bank, model, tf, dev, align):
    from sklearn.linear_model import LogisticRegression

    F = np.load(os.path.join(CACHE, f"dino_feat_{ref_split}.npy")).astype(np.float64)
    X = np.vstack([F[ii] for ii in refs_idx])
    y = np.concatenate([np.full(len(ii), k) for k, ii in enumerate(refs_idx)])
    print(f"  probe 적합: X{X.shape} C={len(refs_idx)}", flush=True)
    pr = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(X, y)

    W = torch.tensor(pr.coef_, dtype=torch.float32, device=dev)
    b = torch.tensor(pr.intercept_, dtype=torch.float32, device=dev)
    if W.shape[0] == 1:
        W = torch.cat([-W, W]); b = torch.cat([-b, b])

    s = torch.tensor(EF_STD, device=dev).view(1, 3, 1, 1)

    rows = []
    for g in gi:
        s_name = src[g]
        for r in gi[g]:
            if s_name.startswith("far_"):
                j = int(align[s_name][int(r)])
            else:
                j = int(r)
            rows.append((g, s_name, j))
    print(f"  query {len(rows)}  (미정렬 {sum(1 for _, _, j in rows if j < 0)})",
          flush=True)

    S = np.full((len(EPS_LIST), len(rows)), np.nan, dtype=np.float64)
    valid = np.array([j >= 0 for _, _, j in rows])
    order = [i for i in range(len(rows)) if valid[i]]
    order.sort(key=lambda i: (rows[i][1], rows[i][2]))

    for s0 in range(0, len(order), BATCH):
        sel = order[s0:s0 + BATCH]
        imgs = [bank.get(rows[i][1], rows[i][2]) for i in sel]
        sc = odin_batch(imgs, model, tf, dev, W, b, DP.CFG["odin_T"],
                        EPS_LIST, s)
        for t, i in enumerate(sel):
            S[:, i] = sc[:, t]
        print(f"    {min(s0 + BATCH, len(order))}/{len(order)}", end="\r", flush=True)
    print()

    groups = np.array([g for g, _, _ in rows])
    return dict(scores=S, groups=groups, valid=valid,
                sources=np.array([s for _, s, _ in rows]),
                src_idx=np.array([j for _, _, j in rows]))


EF_MEAN = EF_STD = None


def main():
    global EF_MEAN, EF_STD
    model, tf, dev, cfg = EF.build_model(EF.REPRS["dino"]["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    EF_MEAN, EF_STD = list(cfg["mean"]), list(cfg["std"])
    align = {k: v for k, v in np.load("odin_real_align.npz").items()}
    bank = ImageBank()

    rng = np.random.default_rng(EP.CFG["seed"])
    out = {}

    for which in ["c100", "c10"]:
        tag = f"CIFAR-{'100' if which == 'c100' else '10'}"
        print(f"\n[{tag}]", flush=True)
        gi, src, refs_idx, ref_split = build_cifar_idx(rng, which)
        r = run_setting(tag, gi, src, refs_idx, ref_split, bank, model, tf, dev, align)
        for k, v in r.items():
            out[f"{tag}|{k}"] = v
        burn_truth(rng, refs_idx, EP.CFG["n_boot"])

    for min_n in EP.CFG["min_n_list"]:
        gi, src, refs_idx = build_derma_idx(rng, min_n)
        tag = f"Derma C={len(refs_idx)}"
        print(f"\n[{tag}]  min_n={min_n}", flush=True)
        r = run_setting(tag, gi, src, refs_idx, "derma_train", bank,
                        model, tf, dev, align)
        for k, v in r.items():
            out[f"{tag}|{k}"] = v
        burn_truth(rng, refs_idx, EP.CFG["n_boot"])

    out["eps_list"] = np.array(EPS_LIST)
    np.savez(OUT, **out)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
