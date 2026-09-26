import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys, json, time, traceback, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_features as EF

SPECS = {
    "clip":   dict(model="vit_base_patch16_clip_224.openai", pinned=None),
    "resnet": dict(model="resnet50.a1_in1k",                 pinned="resnet_extract"),
    "dinov3": dict(model="vit_base_patch16_dinov3.lvd1689m", pinned="dinov3_extract"),
}
STATUS = "in1k_backbones_status.json"


def prepare(name):
    sp = SPECS[name]
    EF.REPRS[name] = dict(model=sp["model"], cache=f"./feat_cache_in1k_{name}")
    if sp["pinned"]:
        mod = __import__(sp["pinned"])
        EF.build_model = mod.build_model_pinned
        print(f"  [pin] build_model -> {sp['pinned']}.build_model_pinned")
    else:
        print("  [pin] 기본 build_model (기하가 DINO 와 원래 일치)")
    return sp


def main(only):
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    t00 = time.time()
    names = [only] if only else list(SPECS)
    import imagenet_extract as IX
    import gpu_logreg as GL
    GL.install()

    for name in names:
        cache = f"./feat_cache_in1k_{name}"
        if os.path.exists(os.path.join(cache, "BENCH_META.json")):
            print(f"\n[skip] {name} 추출 — 이미 있음")
        else:
            print("\n" + "=" * 78)
            print(f"[extract] ImageNet-1k / {name}")
            print("=" * 78, flush=True)
            try:
                prepare(name)
                IX.main(name, False, "in1k")
                st[f"{name}_extract"] = "ok"
            except Exception:
                st[f"{name}_extract"] = traceback.format_exc()[-1200:]
                traceback.print_exc()
            json.dump(st, open(STATUS, "w"), indent=2, ensure_ascii=False)

        out = f"in1k_probe_{name}.json"
        if os.path.exists(out):
            print(f"[skip] {name} 프로브 — 이미 있음")
            continue
        if not os.path.exists(os.path.join(cache, "BENCH_META.json")):
            print(f"[skip] {name} 프로브 — 캐시 없음")
            continue
        print("\n" + "=" * 78)
        print(f"[probe] ImageNet-1k / {name}   (seed 0, 11 scores)")
        print("=" * 78, flush=True)
        try:
            import runpy
            sys.argv = ["in200_probe.py", "--repr", name, "--bench", "in1k", "--seeds", "1"]
            runpy.run_path(os.path.join(HERE, "in200_probe.py"), run_name="__main__")
            st[f"{name}_probe"] = "ok"
        except Exception:
            st[f"{name}_probe"] = traceback.format_exc()[-1200:]
            traceback.print_exc()
        json.dump(st, open(STATUS, "w"), indent=2, ensure_ascii=False)

    ok = sum(1 for v in st.values() if v == "ok")
    print("\n" + "=" * 78)
    print(f"[done] 성공 {ok} / {len(st)}   총 {(time.time()-t00)/60:.1f}분")
    for k, v in st.items():
        if v != "ok":
            print(f"  ** 실패 {k}: {str(v).splitlines()[-1][:110]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=list(SPECS))
    a = ap.parse_args()
    main(a.only)
