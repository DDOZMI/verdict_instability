import os, numpy as np

import extract_features as EF

SEED = 0


def main(repr_key="dino"):
    from datasets import load_dataset
    spec = EF.REPRS[repr_key]
    cache = spec["cache"]
    print(f"[{repr_key}] cache={cache}")

    need = [("train", "c10_train"), ("test", "c10_test")]
    todo = [(s, t) for s, t in need
            if not os.path.exists(os.path.join(cache, f"dino_feat_{t}.npy"))]

    if todo:
        model, tf, dev, cfg = EF.build_model(spec["model"])
        print(f"  [transform] input={cfg.get('input_size')}")
        for split, tag in todo:
            print(f"\n  [extract] CIFAR-10 {split} ...")
            ds = load_dataset("uoft-cs/cifar10", split=split)
            imgs, labs = ds["img"], np.array(ds["label"])
            print(f"    {len(imgs)}장, 클래스 {sorted(set(labs.tolist()))}")
            F = EF.encode(imgs, model, tf, dev)
            np.save(os.path.join(cache, f"dino_feat_{tag}.npy"), F)
            np.save(os.path.join(cache, f"dino_lab_{tag}.npy"), labs)
            print(f"  [saved] dino_feat_{tag}.npy  shape={F.shape}")
    else:
        print("  [skip] CIFAR-10 train/test 이미 있음")

    src = os.path.join(cache, "dino_feat_test.npy")
    dst = os.path.join(cache, "dino_feat_far_cifar100.npy")
    if not os.path.exists(dst):
        if not os.path.exists(src):
            raise SystemExit(f"{src} 없음")
        F = np.load(src)
        rng = np.random.default_rng(SEED)
        if len(F) > EF.FAR_N_MAX:
            F = F[rng.choice(len(F), EF.FAR_N_MAX, replace=False)]
        np.save(dst, F)
        print(f"\n  [saved] dino_feat_far_cifar100.npy  shape={F.shape}")
        print("    (CIFAR-100 test 특징 재사용 — 같은 build_model/encode 산물)")
    else:
        print(f"\n  [skip] far_cifar100 이미 있음")

    print("\n[verify]")
    for f in ["dino_feat_c10_train.npy", "dino_lab_c10_train.npy",
              "dino_feat_c10_test.npy", "dino_lab_c10_test.npy",
              "dino_feat_far_cifar100.npy", "dino_feat_far_svhn.npy",
              "dino_feat_far_dtd_lr32.npy", "dino_feat_far_lsun.npy",
              "dino_feat_far_isun.npy", "dino_feat_far_places365_lr32.npy"]:
        p = os.path.join(cache, f)
        if os.path.exists(p):
            print(f"  OK   {f:34s} {np.load(p).shape}")
        else:
            print(f"  !!   {f:34s} 없음")

    print("\n다음: cifar10_id_probe.py")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=list(EF.REPRS), default="dino")
    a = ap.parse_args()
    main(a.repr)
