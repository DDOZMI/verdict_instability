import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as EF
import odin_real_align as ORA
import extract_cifar10_id as EC10

CLIP_MODEL = "vit_base_patch16_clip_224.openai"
CACHE = "./feat_cache_clip"
ALIGN_NPZ = "odin_real_align.npz"
ALIGN_JSON = "odin_real_align.json"

FAR_NEEDED = ["far_svhn", "far_dtd_lr32", "far_lsun", "far_isun",
              "far_places365_lr32", "far_cifar10"]

CHUNK = 256


def inject_clip():
    EF.REPRS["clip"] = dict(model=CLIP_MODEL, cache=CACHE)
    return EF.REPRS["clip"]


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
    spec = inject_clip()
    os.makedirs(CACHE, exist_ok=True)

    align = {k: v for k, v in np.load(ALIGN_NPZ).items()}
    astat = json.load(open(ALIGN_JSON))

    print("=" * 78)
    print("clip_extract — CLIP ViT-B/16, DINO 와 동일 이미지·동일 행 순서")
    print("=" * 78)
    model, tf, dev, cfg = EF.build_model(spec["model"])
    print(f"  [dim] num_features = {model.num_features}")

    prov = dict(model=spec["model"], cache=CACHE,
                align_source=ALIGN_NPZ, sources={})

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
    EC10.main("clip")

    import extract_derma as ED
    print("\n" + "=" * 78)
    print("DermaMNIST (ID) + d28 OOD — rng 규칙이 DINO 이력을 재현함이 확인됨")
    print("=" * 78)
    ED.main("clip")

    meta = dict(repr="clip", model=spec["model"],
                input_size=str(cfg.get("input_size")),
                mean=[float(x) for x in cfg.get("mean", [])],
                std=[float(x) for x in cfg.get("std", [])],
                crop_pct=float(cfg.get("crop_pct", 0)),
                interpolation=str(cfg.get("interpolation")),
                source="uoft-cs/cifar100 (train/test)",
                far_from_align=ALIGN_NPZ,
                note="far-OOD 행 순서는 DINO 캐시와 동일 (odin_real_align.npz 인덱스 재사용)")
    with open(os.path.join(CACHE, "CACHE_META.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CACHE, "ALIGN_PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    dino_cache = "./feat_cache_dino"
    print(f"  {'file':>34} {'CLIP':>16} {'DINO':>16}")
    for f in (["dino_feat_train.npy", "dino_feat_test.npy",
               "dino_feat_c10_train.npy", "dino_feat_c10_test.npy",
               "dino_feat_far_cifar100.npy"] +
              [f"dino_feat_far_{n[4:]}.npy" for n in FAR_NEEDED]):
        pc, pd = os.path.join(CACHE, f), os.path.join(dino_cache, f)
        sc = str(np.load(pc, mmap_mode="r").shape) if os.path.exists(pc) else "없음"
        sd = str(np.load(pd, mmap_mode="r").shape) if os.path.exists(pd) else "없음"
        flag = "" if sc == sd else "   ** 불일치 **"
        print(f"  {f:>34} {sc:>16} {sd:>16}{flag}")

    print(f"\n  [norm 비교] 특징 norm 평균 (스케일 차이는 detector 에 무영향 —")
    print(f"              점수 전체가 c 배 되고 Spearman·T/T̂ 는 불변)")
    print(f"  {'group':>22} {'CLIP':>10} {'DINO':>10} {'비':>7}")
    for f in (["dino_feat_train.npy"] +
              [f"dino_feat_far_{n[4:]}.npy" for n in FAR_NEEDED]):
        pc, pd = os.path.join(CACHE, f), os.path.join(dino_cache, f)
        if os.path.exists(pc) and os.path.exists(pd):
            nc = np.linalg.norm(np.load(pc), axis=1).mean()
            nd = np.linalg.norm(np.load(pd), axis=1).mean()
            print(f"  {f[10:-4]:>22} {nc:10.2f} {nd:10.2f} {nc/nd:7.3f}")

    print(f"\n  [meta] {CACHE}/CACHE_META.json")
    print(f"  [prov] {CACHE}/ALIGN_PROVENANCE.json")
    print("\n다음: clip_eleven_probe.py")


if __name__ == "__main__":
    main()
