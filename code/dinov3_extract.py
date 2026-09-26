import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as EF
import odin_real_align as ORA
import extract_cifar10_id as EC10

DINOV3_MODEL = "vit_base_patch16_dinov3.lvd1689m"
DINO_MODEL = "vit_base_patch16_224.dino"
CACHE = "./feat_cache_dinov3"
ALIGN_NPZ = "odin_real_align.npz"
ALIGN_JSON = "odin_real_align.json"

FAR_NEEDED = ["far_svhn", "far_dtd_lr32", "far_lsun", "far_isun",
              "far_places365_lr32", "far_cifar10"]

CHUNK = 256

_ORIG_BUILD_MODEL = EF.build_model
_NATIVE_CFG = {}


def dino_data_config():
    import timm, timm.data
    m = timm.create_model(DINO_MODEL, pretrained=False, num_classes=0)
    return dict(timm.data.resolve_model_data_config(m))


def build_model_pinned(model_name):
    import torch, timm, timm.data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [model] {model_name} on {dev}")
    ref = dino_data_config()
    img = int(ref["input_size"][-1])
    model = timm.create_model(model_name, pretrained=True, num_classes=0,
                              img_size=img).to(dev).eval()
    print(f"  [pin] img_size -> {img} (DINO 값). 위치 임베딩 보간 발생.")
    cfg = dict(timm.data.resolve_model_data_config(model))
    native = dict(cfg)
    _NATIVE_CFG[model_name] = native
    for k in ["input_size", "interpolation", "crop_pct", "crop_mode"]:
        cfg[k] = ref[k]
    if native["crop_pct"] != ref["crop_pct"]:
        print(f"  [pin] crop_pct {native['crop_pct']} -> {ref['crop_pct']} (DINO 값)")
    tf = timm.data.create_transform(**cfg, is_training=False)
    print(f"  [transform] input_size={cfg.get('input_size')} "
          f"mean={cfg.get('mean')} std={cfg.get('std')}")
    return model, tf, dev, cfg


def assert_pixels_identical_to_dino(tf, cfg):
    import timm.data
    ref = dino_data_config()
    ref_tf = timm.data.create_transform(**ref, is_training=False)

    print("\n" + "-" * 78)
    print("검증 ① 전처리 동일성 (vs DINO)")
    print("-" * 78)
    ok = True
    for k in ["input_size", "interpolation", "crop_pct", "crop_mode"]:
        same = cfg[k] == ref[k]
        ok &= same
        print(f"  {k:>14} : {str(cfg[k]):>22}  vs DINO {str(ref[k]):>22}  "
              f"{'O' if same else '** X **'}")
    for k in ["mean", "std"]:
        same = tuple(float(x) for x in cfg[k]) == tuple(float(x) for x in ref[k])
        ok &= same
        print(f"  {k:>14} : {str([round(float(x),4) for x in cfg[k]]):>22}  "
              f"vs DINO {str([round(float(x),4) for x in ref[k]]):>22}  "
              f"{'O' if same else '** X **'}")
    s_a, s_b = " ".join(str(tf).split()), " ".join(str(ref_tf).split())
    same = s_a == s_b
    ok &= same
    print(f"  {'transform repr':>14} : {'문자열 완전 일치' if same else '** 불일치 **'}")
    if not same:
        print(f"    dinov3: {s_a}")
        print(f"    dino  : {s_b}")

    from PIL import Image
    import torch
    rs = np.random.default_rng(0)
    probe = Image.fromarray(rs.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    d = float((tf(probe.convert("RGB")) - ref_tf(probe.convert("RGB"))).abs().max())
    print(f"  {'tensor max|diff|':>14} : {d:.3e}  {'O' if d == 0.0 else '** X **'}")
    ok &= (d == 0.0)

    print("-" * 78)
    assert ok, "** 전처리가 DINO 와 다르다 — 중단. 이 상태의 결과는 쓰지 않는다. **"
    print("  ** 같은 픽셀이 두 backbone 에 들어간다. **\n")


def inject_dinov3():
    EF.REPRS["dinov3"] = dict(model=DINOV3_MODEL, cache=CACHE)
    return EF.REPRS["dinov3"]


def encode_by_index(idx, n_cand, get, pre, model, tf, dev):
    assert (idx >= 0).all(), "미정렬(-1) 인덱스가 있다"
    assert idx.max() < n_cand, f"인덱스 범위 초과: {idx.max()} >= {n_cand}"
    out = []
    for s in range(0, len(idx), CHUNK):
        e = min(s + CHUNK, len(idx))
        imgs = [pre(get(idx[i])) for i in range(s, e)]
        out.append(EF.encode(imgs, model, tf, dev))
        print(f"    {e}/{len(idx)}", end="\r", flush=True)
    print()
    return np.concatenate(out)


def main():
    spec = inject_dinov3()
    os.makedirs(CACHE, exist_ok=True)

    align = {k: v for k, v in np.load(ALIGN_NPZ).items()}
    astat = json.load(open(ALIGN_JSON))

    print("=" * 78)
    print("dinov3_extract — DINOv3 ViT-B/16, DINO 와 동일 이미지·동일 행 순서·동일 픽셀")
    print("=" * 78)
    model, tf, dev, cfg = build_model_pinned(spec["model"])
    print(f"  [dim] num_features = {model.num_features}   (DINO/CLIP 과 같아야 한다: 768)")
    assert_pixels_identical_to_dino(tf, cfg)

    EF.build_model = build_model_pinned

    prov = dict(model=spec["model"], cache=CACHE,
                align_source=ALIGN_NPZ, crop_pct_pinned_to=cfg["crop_pct"],
                num_features=int(model.num_features), sources={})

    for t in ["train", "test"]:
        fp = os.path.join(CACHE, f"dino_feat_{t}.npy")
        lp = os.path.join(CACHE, f"dino_lab_{t}.npy")
        if os.path.exists(fp):
            print(f"\n[skip] {t}: 이미 있음 shape={np.load(fp, mmap_mode='r').shape}")
            continue
        print(f"\n[{t}] CIFAR-100 {t} — 전체 split, 순서 그대로")
        imgs, labs = EF.load_cifar100(t)
        F = EF.encode(imgs, model, tf, dev)
        np.save(fp, F); np.save(lp, labs)
        print(f"  [saved] {fp} shape={F.shape}")
        prov["sources"][t] = dict(mode="full_split_in_order", n=int(len(F)))

    for name in FAR_NEEDED:
        short = name[len("far_"):]
        fp = os.path.join(CACHE, f"dino_feat_far_{short}.npy")
        if os.path.exists(fp):
            print(f"\n[skip] {short}: 이미 있음 shape={np.load(fp, mmap_mode='r').shape}")
            continue
        if name not in align:
            print(f"\n[FAIL] {short}: 정렬표에 없음 -> 건너뜀")
            continue
        st = astat.get(name, {})
        if not st.get("complete"):
            print(f"\n[FAIL] {short}: 정렬 미완료(complete={st.get('complete')}) -> 건너뜀")
            continue

        idx = align[name].astype(np.int64)
        loader, pre = ORA.SOURCES[name]
        print(f"\n[{short}] 정렬표 {len(idx)}행  (DINO 캐시와 동일 이미지)")
        print(f"  정렬 통계: matched={st['n_matched']}/{st['n_cache']} "
              f"d1_med={st['d1_median']:g} d1_max={st['d1_max']:.3g} "
              f"d2_min={st['d2_min']:.3g}")
        n_cand, get = loader()
        print(f"  후보 풀 {n_cand}장 -> 인덱스 순서대로 인코딩")
        F = encode_by_index(idx, n_cand, get, pre, model, tf, dev)
        np.save(fp, F)
        np.save(os.path.join(CACHE, f"dino_lab_far_{short}.npy"),
                np.full(len(F), -1, dtype=np.int64))
        print(f"  [saved] {fp} shape={F.shape}")
        prov["sources"][short] = dict(mode="align_index_reuse", n=int(len(F)),
                                      n_cand=int(n_cand),
                                      align_stats=st,
                                      idx_head=[int(x) for x in idx[:8]])

    print("\n" + "=" * 78)
    print("CIFAR-10 (ID) + far_cifar100 — rng 체인을 DINO 와 맞추기 위해")
    print("=" * 78)
    EC10.main("dinov3")

    import extract_derma as ED
    print("\n" + "=" * 78)
    print("DermaMNIST (ID) + d28 OOD — rng 규칙이 DINO 이력을 재현함이 확인됨")
    print("=" * 78)
    ED.main("dinov3")

    ref = dino_data_config()
    meta = dict(repr="dinov3", model=spec["model"],
                input_size=str(cfg.get("input_size")),
                mean=[float(x) for x in cfg.get("mean", [])],
                std=[float(x) for x in cfg.get("std", [])],
                crop_pct=float(cfg.get("crop_pct", 0)),
                interpolation=str(cfg.get("interpolation")),
                num_features=int(model.num_features),
                crop_pct_native=float(
                    _NATIVE_CFG[spec["model"]]["crop_pct"]),
                crop_pct_pinned_to_dino=float(ref["crop_pct"]),
                source="uoft-cs/cifar100 (train/test)",
                far_from_align=ALIGN_NPZ,
                note="기하 전처리(img_size 포함)를 DINO 에 고정. normalization 은 원래부터 동일(ImageNet). "
                     "위치 임베딩 256->224 보간. 사전학습 데이터 LVD-1689M. "
                     "far-OOD 행 순서는 odin_real_align.npz 인덱스 재사용으로 DINO 와 동일.")
    with open(os.path.join(CACHE, "CACHE_META.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CACHE, "ALIGN_PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("검증 ② 캐시 대조 (DINOv3 도 768 이므로 shape 가 완전히 같아야 한다)")
    print("=" * 78)
    dino_cache = "./feat_cache_dino"
    files = (["dino_feat_train.npy", "dino_feat_test.npy",
              "dino_feat_c10_train.npy", "dino_feat_c10_test.npy",
              "dino_feat_far_cifar100.npy"] +
             [f"dino_feat_far_{n[4:]}.npy" for n in FAR_NEEDED] +
             ["dino_feat_derma_train.npy", "dino_feat_derma_test.npy",
              "dino_feat_far_svhn_d28.npy", "dino_feat_far_dtd_d28.npy",
              "dino_feat_far_cifar10_d28.npy"])
    print(f"  {'file':>34} {'DINOv3':>16} {'DINO':>16}  n 일치")
    n_ok = n_bad = 0
    for f in files:
        pr_, pd = os.path.join(CACHE, f), os.path.join(dino_cache, f)
        ar = np.load(pr_, mmap_mode="r") if os.path.exists(pr_) else None
        ad = np.load(pd, mmap_mode="r") if os.path.exists(pd) else None
        sr = str(ar.shape) if ar is not None else "없음"
        sd = str(ad.shape) if ad is not None else "없음"
        if ar is not None and ad is not None:
            same_n = ar.shape[0] == ad.shape[0]
        else:
            same_n = False
        n_ok += same_n; n_bad += (not same_n)
        print(f"  {f:>34} {sr:>16} {sd:>16}  {'O' if same_n else '** X **'}")
    print(f"  -> 행 수 일치 {n_ok}/{n_ok + n_bad}")

    print(f"\n  [라벨 대조] 특징은 backbone 마다 다르지만 라벨은 같아야 한다")
    lab_ok = 0; lab_tot = 0
    for f in ["dino_lab_train.npy", "dino_lab_test.npy",
              "dino_lab_c10_train.npy", "dino_lab_c10_test.npy",
              "dino_lab_derma_train.npy", "dino_lab_derma_test.npy"]:
        pr_, pd = os.path.join(CACHE, f), os.path.join(dino_cache, f)
        if os.path.exists(pr_) and os.path.exists(pd):
            lab_tot += 1
            same = np.array_equal(np.load(pr_), np.load(pd))
            lab_ok += same
            print(f"  {f:>34} {'완전 동일' if same else '** 불일치 **'}")
    print(f"  -> 라벨 일치 {lab_ok}/{lab_tot}")

    for nm in ["DERMA_META.json"]:
        pr_, pd = os.path.join(CACHE, nm), os.path.join(dino_cache, nm)
        if os.path.exists(pr_) and os.path.exists(pd):
            a, b = json.load(open(pr_)), json.load(open(pd))
            keys = [k for k in b if k not in ("repr", "model", "cache")]
            same = all(a.get(k) == b.get(k) for k in keys)
            print(f"\n  [{nm}] held_in/near/class_counts 대조: "
                  f"{'완전 동일' if same else '** 불일치 **'}")
            if not same:
                for k in keys:
                    if a.get(k) != b.get(k):
                        print(f"    {k}: resnet={a.get(k)}  dino={b.get(k)}")

    print(f"\n  [norm] 특징 norm 평균 (스케일은 Spearman·T/T̂ 에 무영향)")
    print(f"  {'group':>22} {'ResNet':>10} {'DINO':>10}")
    for f in ["dino_feat_train.npy"] + [f"dino_feat_far_{n[4:]}.npy" for n in FAR_NEEDED]:
        pr_, pd = os.path.join(CACHE, f), os.path.join(dino_cache, f)
        if os.path.exists(pr_) and os.path.exists(pd):
            nr = np.linalg.norm(np.load(pr_), axis=1).mean()
            nd = np.linalg.norm(np.load(pd), axis=1).mean()
            print(f"  {f[10:-4]:>22} {nr:10.2f} {nd:10.2f}")

    print(f"\n  [meta] {CACHE}/CACHE_META.json")
    print(f"  [prov] {CACHE}/ALIGN_PROVENANCE.json")
    print("\n다음: resnet_stream_check.py")


if __name__ == "__main__":
    main()
