import os, numpy as np
from PIL import Image

import extract_features as EF

DERMA_SIZE = 28
MIN_N = 400
FAR_N_MAX = 5000
SEED = 0


def to_size(im, size=DERMA_SIZE):
    return im.convert("RGB").resize((size, size), Image.BICUBIC)


def load_derma():
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise SystemExit("pip install medmnist  필요")
    info = INFO["dermamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    out = {}
    for split in ["train", "test"]:
        ds = DataClass(split=split, download=True, size=28)
        imgs = [Image.fromarray(x) for x in ds.imgs]
        labs = ds.labels.flatten().astype(int)
        out[split] = (imgs, labs)
    return out, info["label"]


def main(repr_key="dino"):
    spec = EF.REPRS[repr_key]
    cache = spec["cache"]
    rng = np.random.default_rng(SEED)
    os.makedirs(cache, exist_ok=True)

    print(f"[{repr_key}] cache={cache}")
    print(f"** 해상도 통제: 모든 이미지를 {DERMA_SIZE}x{DERMA_SIZE} 로 통일 **")

    print(f"\n[1] DermaMNIST 로드")
    derma, label_names = load_derma()
    tr_imgs, tr_labs = derma["train"]
    te_imgs, te_labs = derma["test"]
    print(f"    train {len(tr_imgs)}장, test {len(te_imgs)}장")
    print(f"    첫 이미지 크기: {tr_imgs[0].size}")

    print(f"\n    ** 클래스별 표본 수 (train) **")
    print(f"    {'id':>3} {'이름':>34} {'n':>6} {'비율':>7} {'held-in?':>9}")
    counts = {}
    for c in sorted(set(tr_labs.tolist())):
        n = int((tr_labs == c).sum())
        counts[c] = n
        name = label_names[str(c)]
        ok = "** IN **" if n >= MIN_N else "near-OOD"
        print(f"    {c:>3} {name[:34]:>34} {n:>6} {n/len(tr_labs)*100:6.1f}% {ok:>9}")

    held_in = sorted([c for c, n in counts.items() if n >= MIN_N])
    near_cls = sorted([c for c, n in counts.items() if n < MIN_N])
    print(f"\n    held-in  (n>={MIN_N}): {held_in}   "
          f"n = {[counts[c] for c in held_in]}")
    print(f"    near-OOD (n< {MIN_N}): {near_cls}  "
          f"n = {[counts[c] for c in near_cls]}")
    print(f"\n    ** n_c 범위: {min(counts[c] for c in held_in)} ~ "
          f"{max(counts[c] for c in held_in)} "
          f"({max(counts[c] for c in held_in)/min(counts[c] for c in held_in):.1f}배) **")
    print(f"    -> 1/sqrt(n_c) 스케일링을 클래스 간 비교로 검증 가능")

    model, tf, dev, cfg = EF.build_model(spec["model"])
    print(f"\n[2] 특징 추출  (transform input={cfg.get('input_size')})")

    for split, (imgs, labs) in [("train", derma["train"]), ("test", derma["test"])]:
        fp = os.path.join(cache, f"dino_feat_derma_{split}.npy")
        if os.path.exists(fp):
            print(f"    [skip] derma_{split} 있음 {np.load(fp).shape}")
            continue
        print(f"    [extract] DermaMNIST {split} ({len(imgs)}장)")
        F = EF.encode([to_size(im) for im in imgs], model, tf, dev)
        np.save(fp, F)
        np.save(os.path.join(cache, f"dino_lab_derma_{split}.npy"), labs)
        print(f"    [saved] {fp} {F.shape}")

    print(f"\n[3] OOD 추출 (** 전부 {DERMA_SIZE}px 로 통제 **)")
    for name in ["svhn", "dtd", "cifar10"]:
        fp = os.path.join(cache, f"dino_feat_far_{name}_d{DERMA_SIZE}.npy")
        if os.path.exists(fp):
            print(f"    [skip] {name}_d{DERMA_SIZE} 있음 {np.load(fp).shape}")
            continue
        print(f"    [extract] {name} -> {DERMA_SIZE}px")
        imgs, labs = EF.load_ood(name, FAR_N_MAX, rng)
        print(f"      원본 {len(imgs)}장, 첫 크기 {imgs[0].size}")
        F = EF.encode([to_size(im) for im in imgs], model, tf, dev)
        np.save(fp, F)
        print(f"    [saved] {fp} {F.shape}")

    print(f"\n[4] ** 해상도 통제 검증 ** — 특징 norm 이 비슷해야")
    print(f"    {'그룹':>20} {'norm mean':>10} {'norm std':>9} {'원본 해상도':>12}")
    rows = [("derma_train", "dino_feat_derma_train.npy", "28px (ID)"),
            ("svhn_d28", f"dino_feat_far_svhn_d{DERMA_SIZE}.npy", "32 -> 28"),
            ("dtd_d28", f"dino_feat_far_dtd_d{DERMA_SIZE}.npy", "고해상 -> 28"),
            ("cifar10_d28", f"dino_feat_far_cifar10_d{DERMA_SIZE}.npy", "32 -> 28")]
    for tag, fn, res in rows:
        p = os.path.join(cache, fn)
        if os.path.exists(p):
            nm = np.linalg.norm(np.load(p), axis=1)
            print(f"    {tag:>20} {nm.mean():10.2f} {nm.std():9.2f} {res:>12}")
    p = os.path.join(cache, "dino_feat_train.npy")
    if os.path.exists(p):
        nm = np.linalg.norm(np.load(p), axis=1)
        print(f"    {'[참고] cifar100_32':>20} {nm.mean():10.2f} {nm.std():9.2f} "
              f"{'32px':>12}")
    print(f"\n    ** norm 이 서로 가까우면 해상도 통제 성공 **")
    print(f"    (28px 로 통일했으므로 아티팩트가 같아야 한다)")

    import json
    meta = dict(id_dataset="dermamnist", id_size=DERMA_SIZE, min_n=MIN_N,
                held_in=held_in, near_classes=near_cls,
                class_counts={str(k): v for k, v in counts.items()},
                label_names=label_names,
                ood_28px=["svhn", "dtd", "cifar10"],
                model=spec["model"])
    with open(os.path.join(cache, "DERMA_META.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[meta] {cache}/DERMA_META.json")
    print(f"\n다음: derma_probe.py")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=list(EF.REPRS), default="dino")
    a = ap.parse_args()
    main(a.repr)
