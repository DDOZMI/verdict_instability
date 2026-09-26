import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as EF

CACHE = "./feat_cache_dino"
OUT_NPZ = "odin_real_align.npz"
OUT_JSON = "odin_real_align.json"
CHUNK = 256

LOWRES = 32
D28 = 28


def lz_hf(repo, config, split, key):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split) if config else \
        load_dataset(repo, split=split)
    return len(ds), (lambda i: ds[int(i)][key])


def lz_svhn():
    return lz_hf("ufldl-stanford/svhn", "cropped_digits", "test", "image")


def lz_cifar10_test():
    return lz_hf("uoft-cs/cifar10", None, "test", "img")


def lz_cifar100_test():
    return lz_hf("uoft-cs/cifar100", None, "test", "img")


def lz_dtd():
    import torchvision.datasets as tvd
    ds = tvd.DTD(root=os.path.join(EF.LOCAL_ROOT, "dtd"), split="test", download=True)
    return len(ds), (lambda i: ds[int(i)][0])


def lz_places365():
    import torchvision.datasets as tvd
    ds = tvd.Places365(root=os.path.join(EF.LOCAL_ROOT, "places365"),
                       split="val", small=True, download=False)
    return len(ds), (lambda i: ds[int(i)][0])


def _folder(name):
    root = os.path.join(EF.LOCAL_ROOT, name)
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1] in EF.IMG_EXT:
                paths.append(os.path.join(dp, fn))
    paths.sort()
    return len(paths), (lambda i: Image.open(paths[int(i)]))


def lz_lsun():
    return _folder("LSUN")


def lz_isun():
    return _folder("iSUN")


def id_tf(im):
    return im


def lr32_tf(im):
    return im.convert("RGB").resize((LOWRES, LOWRES), Image.BICUBIC)


def d28_tf(im):
    return im.convert("RGB").resize((D28, D28), Image.BICUBIC)


SOURCES = {
    "far_svhn":            (lz_svhn,          id_tf),
    "far_lsun":            (lz_lsun,          id_tf),
    "far_isun":            (lz_isun,          id_tf),
    "far_cifar10":         (lz_cifar10_test,  id_tf),
    "far_cifar100":        (lz_cifar100_test, id_tf),
    "far_dtd_lr32":        (lz_dtd,           lr32_tf),
    "far_places365_lr32":  (lz_places365,     lr32_tf),
    "far_svhn_d28":        (lz_svhn,          d28_tf),
    "far_dtd_d28":         (lz_dtd,           d28_tf),
    "far_cifar10_d28":     (lz_cifar10_test,  d28_tf),
}


def encode_chunk(imgs, model, tf, dev):
    with torch.no_grad():
        b = torch.stack([tf(im.convert("RGB")) for im in imgs]).to(dev)
        return model(b).cpu().numpy().astype(np.float32)


def align_one(cache_F, n_cand, get, pre, model, tf, dev):
    allF = []
    for s in range(0, n_cand, CHUNK):
        e = min(s + CHUNK, n_cand)
        allF.append(encode_chunk([pre(get(i)) for i in range(s, e)], model, tf, dev))
        print(f"    encode {e}/{n_cand}", end="\r", flush=True)
    print()
    allF = np.concatenate(allF)

    A = torch.from_numpy(cache_F).to(dev)
    B = torch.from_numpy(allF).to(dev)
    d1 = torch.empty(len(A), device=dev)
    d2 = torch.empty(len(A), device=dev)
    idx = torch.empty(len(A), dtype=torch.long, device=dev)
    with torch.no_grad():
        for s in range(0, len(A), 512):
            D = torch.cdist(A[s:s + 512], B)
            top = torch.topk(D, k=2, dim=1, largest=False)
            d1[s:s + 512] = top.values[:, 0]
            d2[s:s + 512] = top.values[:, 1]
            idx[s:s + 512] = top.indices[:, 0]
    return (idx.cpu().numpy().astype(np.int64),
            d1.cpu().numpy(), d2.cpu().numpy())


def main():
    model, tf, dev, cfg = EF.build_model(EF.REPRS["dino"]["model"])
    out, stats = {}, {}
    if os.path.exists(OUT_JSON):
        stats = json.load(open(OUT_JSON))
    if os.path.exists(OUT_NPZ):
        out = {k: v for k, v in np.load(OUT_NPZ).items()}

    for name, (loader, pre) in SOURCES.items():
        if stats.get(name, {}).get("complete"):
            print(f"[skip] {name}: 이미 완전 정렬됨")
            continue
        fp = os.path.join(CACHE, f"dino_feat_{name}.npy")
        if not os.path.exists(fp):
            print(f"[skip] {name}: 캐시 없음")
            continue
        cache_F = np.load(fp)
        print(f"\n[{name}] 캐시 {cache_F.shape}")
        n_cand, get = loader()
        print(f"  후보 {n_cand}장 -> 스트리밍 encode + NN 매칭")
        idx, d1, d2 = align_one(cache_F, n_cand, get, pre, model, tf, dev)

        ok = d1 < 0.5
        n_amb = int(((d2 < 20 * np.maximum(d1, 1e-12)) & ok).sum())
        n_ok = int(ok.sum())
        n_uni = int(len(np.unique(idx[ok])))
        print(f"  매칭 {n_ok}/{len(cache_F)}  고유 {n_uni}  중복이미지 {n_amb}  "
              f"d1_max {d1.max():.3g}  d2_min {d2.min():.3g}")
        idx[~ok] = -1

        out[name] = idx
        stats[name] = dict(n_cache=int(len(cache_F)), n_cand=int(n_cand),
                           n_matched=n_ok, n_unique=n_uni, n_duplicate_img=n_amb,
                           d1_max=float(d1.max()), d1_median=float(np.median(d1)),
                           d2_min=float(d2.min()),
                           complete=bool(n_ok == len(cache_F)))
        if not stats[name]["complete"]:
            print(f"  ** {len(cache_F) - n_ok}행 미확정 — 해당 query 는 제외 처리 **")
        np.savez(OUT_NPZ, **out)
        with open(OUT_JSON, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n[saved] {OUT_NPZ}, {OUT_JSON}")
    bad = [k for k, v in stats.items() if not v["complete"]]
    print("완전 정렬 실패:", bad if bad else "없음")


if __name__ == "__main__":
    main()
