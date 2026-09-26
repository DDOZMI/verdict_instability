import os, argparse, numpy as np

REPRS = {
    "dino":   dict(model="vit_base_patch16_224.dino",        cache="./feat_cache_dino"),
    "dinov2": dict(model="vit_base_patch14_dinov2.lvd142m",  cache="./feat_cache_dino_v2"),
}

FAR_N_MAX = 5000
BATCH = 128
SEED = 0


def build_model(model_name):
    import torch, timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [model] {model_name} on {dev}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(dev).eval()
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print(f"  [transform] input_size={cfg.get('input_size')} "
          f"mean={cfg.get('mean')} std={cfg.get('std')}")
    return model, tf, dev, cfg


def encode(imgs, model, tf, dev):
    import torch
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), BATCH):
            batch = torch.stack([tf(im.convert("RGB")) for im in imgs[i:i + BATCH]]).to(dev)
            feats.append(model(batch).cpu().numpy())
            if i % (BATCH * 10) == 0:
                print(f"    {i}/{len(imgs)}", end="\r")
    print()
    return np.concatenate(feats).astype(np.float32)


def load_cifar100(split):
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar100", split=split)
    return ds["img"], np.array(ds["fine_label"])


LOCAL_ROOT = "./data_ood"

OOD_SOURCES = {
    "svhn": dict(role="spec", loader="hf",
                 hf="ufldl-stanford/svhn", config="cropped_digits",
                 split="test", img_key="image", lab_key="label",
                 note="숫자 사진. 개념 겹침 없음."),
    "dtd": dict(role="spec", loader="tv", tv="DTD",
                note="텍스처 5640장. CIFAR 과 카테고리 겹침 없음 — 가장 깨끗한 far."),

    "cifar10": dict(role="ref", loader="hf",
                    hf="uoft-cs/cifar10", split="test",
                    img_key="img", lab_key="label",
                    note="자연물. CIFAR-100 과 개념 겹침 -> near 수준 예상(확인됨)."),
    "places365": dict(role="ref", loader="tv", tv="Places365",
                      note="장면 365 카테고리. CIFAR-100 장면 클래스와 겹침."),
    "lsun": dict(role="ref", loader="folder", folder="LSUN",
                 note="장면(침실/거실 등). 겹침. Dropbox tar 를 data_ood/LSUN 에 풀 것."),
    "isun": dict(role="ref", loader="folder", folder="iSUN",
                 note="SUN 자연장면. 겹침. tar 를 data_ood/iSUN 에 풀 것."),
}

SPEC_SOURCES = [k for k, v in OOD_SOURCES.items() if v["role"] == "spec"]
REF_SOURCES = [k for k, v in OOD_SOURCES.items() if v["role"] == "ref"]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


def _load_hf(spec, n_max, rng):
    from datasets import load_dataset
    if "config" in spec:
        ds = load_dataset(spec["hf"], spec["config"], split=spec["split"])
    else:
        ds = load_dataset(spec["hf"], split=spec["split"])
    imgs = ds[spec["img_key"]]
    labs = np.array(ds[spec["lab_key"]])
    if len(imgs) > n_max:
        idx = rng.choice(len(imgs), n_max, replace=False)
        imgs = [imgs[int(i)] for i in idx]
        labs = labs[idx]
    return imgs, labs


def _load_tv(spec, n_max, rng):
    import torchvision.datasets as tvd
    name = spec["tv"]
    root = os.path.join(LOCAL_ROOT, name.lower())
    os.makedirs(root, exist_ok=True)
    if name == "DTD":
        ds = tvd.DTD(root=root, split="test", download=True)
    elif name == "Places365":
        ds = tvd.Places365(root=root, split="val", small=True, download=True)
    else:
        raise ValueError(name)
    n = len(ds)
    idx = rng.choice(n, min(n_max, n), replace=False)
    imgs, labs = [], []
    for i in idx:
        im, lb = ds[int(i)]
        imgs.append(im)
        labs.append(lb if isinstance(lb, int) else -1)
    return imgs, np.array(labs)


def _load_folder(spec, n_max, rng):
    from PIL import Image
    root = os.path.join(LOCAL_ROOT, spec["folder"])
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} 없음.\n"
            f"  LSUN: https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz\n"
            f"  iSUN: https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz\n"
            f"  받아서 {LOCAL_ROOT}/ 아래에 풀어두세요.")
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1] in IMG_EXT:
                paths.append(os.path.join(dp, fn))
    if not paths:
        raise FileNotFoundError(f"{root} 에 이미지 없음")
    paths.sort()
    if len(paths) > n_max:
        idx = rng.choice(len(paths), n_max, replace=False)
        paths = [paths[int(i)] for i in idx]
    imgs = [Image.open(p) for p in paths]
    return imgs, np.full(len(imgs), -1)


def load_ood(name, n_max, rng):
    spec = OOD_SOURCES[name]
    ld = spec["loader"]
    if ld == "hf":
        return _load_hf(spec, n_max, rng)
    if ld == "tv":
        return _load_tv(spec, n_max, rng)
    if ld == "folder":
        return _load_folder(spec, n_max, rng)
    raise ValueError(ld)


def extract_all(repr_key, only=None):
    spec = REPRS[repr_key]
    cache = spec["cache"]
    os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"[{repr_key}] cache={cache}")
    model, tf, dev, cfg = build_model(spec["model"])

    targets = (["train", "test"] + list(OOD_SOURCES)) if only is None else [only]
    ok_srcs, fail_srcs = [], []

    for t in targets:
        if t in ("train", "test"):
            fp = os.path.join(cache, f"dino_feat_{t}.npy")
            lp = os.path.join(cache, f"dino_lab_{t}.npy")
            if os.path.exists(fp):
                print(f"  [skip] {t} 이미 있음 shape={np.load(fp).shape}")
                continue
            print(f"  [extract] CIFAR-100 {t} ...")
            imgs, labs = load_cifar100(t)
            F = encode(imgs, model, tf, dev)
            np.save(fp, F)
            np.save(lp, labs)
            print(f"  [saved] {fp} shape={F.shape}")

        elif t in OOD_SOURCES:
            fp = os.path.join(cache, f"dino_feat_far_{t}.npy")
            lp = os.path.join(cache, f"dino_lab_far_{t}.npy")
            if os.path.exists(fp):
                print(f"  [skip] far_{t} 이미 있음 shape={np.load(fp).shape}")
                ok_srcs.append(t)
                continue
            role = OOD_SOURCES[t]["role"]
            print(f"  [extract] OOD '{t}' (role={role}, n<={FAR_N_MAX}) ...")
            try:
                imgs, labs = load_ood(t, FAR_N_MAX, rng)
                F = encode(imgs, model, tf, dev)
                np.save(fp, F)
                np.save(lp, labs)
                print(f"  [saved] {fp} shape={F.shape}")
                ok_srcs.append(t)
            except Exception as e:
                print(f"  [FAIL] {t}: {e}")
                fail_srcs.append(t)

    import json
    meta = dict(repr=repr_key, model=spec["model"],
                input_size=str(cfg.get("input_size")),
                mean=[float(x) for x in cfg.get("mean", [])],
                std=[float(x) for x in cfg.get("std", [])],
                source="uoft-cs/cifar100 (train/test)",
                ood_sources={k: dict(role=v["role"], loader=v["loader"],
                                     note=v.get("note", ""))
                             for k in OOD_SOURCES for v in [OOD_SOURCES[k]]},
                seed=SEED, far_n_max=FAR_N_MAX)
    with open(os.path.join(cache, "CACHE_META.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  [meta] {cache}/CACHE_META.json — 이 캐시의 정체를 여기 기록")

    print("\n  === OOD 소스 요약 ===")
    print(f"  사양 대상(spec, far 여야 함): "
          f"{[s for s in ok_srcs if s in SPEC_SOURCES] or '없음'}")
    print(f"  참조점(ref, 중간지대 예상):  "
          f"{[s for s in ok_srcs if s in REF_SOURCES] or '없음'}")
    if fail_srcs:
        print(f"  실패: {fail_srcs}  -> 수동 다운로드 후 재실행")


def verify(repr_key):
    cache = REPRS[repr_key]["cache"]
    print(f"\n[verify] {cache}")
    core = ["dino_feat_train.npy", "dino_lab_train.npy",
            "dino_feat_test.npy", "dino_lab_test.npy"]
    ok = True
    for f in core:
        p = os.path.join(cache, f)
        if os.path.exists(p):
            a = np.load(p)
            print(f"  OK   {f:32s} shape={a.shape}")
        else:
            print(f"  !!   {f:32s} 없음")
            ok = False

    print("  -- 사양 대상 (필수) --")
    for k in SPEC_SOURCES:
        p = os.path.join(cache, f"dino_feat_far_{k}.npy")
        if os.path.exists(p):
            print(f"  OK   far_{k:27s} shape={np.load(p).shape}")
        else:
            print(f"  !!   far_{k:27s} 없음 (사양 검증 불가)")
            ok = False

    print("  -- 참조점 (선택) --")
    for k in REF_SOURCES:
        p = os.path.join(cache, f"dino_feat_far_{k}.npy")
        if os.path.exists(p):
            print(f"  OK   far_{k:27s} shape={np.load(p).shape}")
        else:
            print(f"  --   far_{k:27s} 없음 (건너뜀 가능)")

    mp = os.path.join(cache, "CACHE_META.json")
    if os.path.exists(mp):
        import json
        print(f"  META: {json.load(open(mp))['model']}")
    else:
        print("  !! CACHE_META.json 없음 — 정체 기록 안 됨")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=list(REPRS), default="dino")
    ap.add_argument("--only", choices=["train", "test"] + list(OOD_SOURCES), default=None)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only:
        verify(a.repr)
    else:
        extract_all(a.repr, a.only)
        verify(a.repr)
