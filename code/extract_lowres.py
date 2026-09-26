import os, json, numpy as np
from PIL import Image

import extract_features as EF

LOWRES = 32
TARGETS = ["dtd", "places365"]
N_MAX = EF.FAR_N_MAX
SEED = 0


def to_lowres(im, size=LOWRES):
    return im.convert("RGB").resize((size, size), Image.BICUBIC)


def main(repr_key="dino"):
    spec = EF.REPRS[repr_key]
    cache = spec["cache"]
    rng = np.random.default_rng(SEED)

    print(f"[{repr_key}] cache={cache}")
    model, tf, dev, cfg = EF.build_model(spec["model"])
    print(f"  [transform] input={cfg.get('input_size')}")
    print(f"  ** 조작: 원본 -> {LOWRES}x{LOWRES} -> (transform 이 224 로 업샘플) **")

    for t in TARGETS:
        fp = os.path.join(cache, f"dino_feat_far_{t}_lr{LOWRES}.npy")
        if os.path.exists(fp):
            print(f"  [skip] {t}_lr{LOWRES} 이미 있음 shape={np.load(fp).shape}")
            continue
        print(f"\n  [extract] {t} -> 저해상도({LOWRES}px) 버전 ...")
        imgs, labs = EF.load_ood(t, N_MAX, rng)
        print(f"    원본 {len(imgs)}장. 첫 이미지 크기: {imgs[0].size}")
        lr = [to_lowres(im) for im in imgs]
        print(f"    -> {LOWRES}x{LOWRES} 로 다운샘플 완료. encode 가 224 로 다시 올린다.")
        F = EF.encode(lr, model, tf, dev)
        np.save(fp, F)
        np.save(os.path.join(cache, f"dino_lab_far_{t}_lr{LOWRES}.npy"), labs)
        print(f"  [saved] {fp} shape={F.shape}")

    print(f"\n[norm 비교] 원본 vs 저해상도")
    print(f"  {'group':>16} {'norm mean':>10} {'norm std':>9}")
    for t in TARGETS:
        for suf, lab in [("", "원본"), (f"_lr{LOWRES}", f"저해상도{LOWRES}")]:
            p = os.path.join(cache, f"dino_feat_far_{t}{suf}.npy")
            if os.path.exists(p):
                F = np.load(p)
                nm = np.linalg.norm(F, axis=1)
                print(f"  {t+' ('+lab+')':>16} {nm.mean():10.2f} {nm.std():9.2f}")
    ftr = np.load(os.path.join(cache, "dino_feat_train.npy"))
    nm = np.linalg.norm(ftr, axis=1)
    print(f"  {'CIFAR-100 train':>16} {nm.mean():10.2f} {nm.std():9.2f}")
    for t in ["svhn", "lsun", "isun"]:
        p = os.path.join(cache, f"dino_feat_far_{t}.npy")
        if os.path.exists(p):
            F = np.load(p)
            nm = np.linalg.norm(F, axis=1)
            print(f"  {t+' (32px 원래)':>16} {nm.mean():10.2f} {nm.std():9.2f}")

    print("\n  -> 저해상도판 norm 이 CIFAR/SVHN/LSUN 쪽에 가까워지면")
    print("     ** 해상도가 특징 통계를 지배한다 ** 는 뜻.")
    print("\n다음: resolution_confound_probe.py 로 far 판정이 유지되는지 확인")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=list(EF.REPRS), default="dino")
    a = ap.parse_args()
    main(a.repr)
