import os, re, sys, json, time, zlib, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as EF

IN_DIR   = "/mnt/d/datasets/imagenet/data"
OOD_ROOT = "/mnt/d/datasets/ood"
SPLITS   = f"{OOD_ROOT}/splits/splits/imagenet200"
CLASSES  = "/mnt/d/datasets/imagenet"

SEED       = 0
REF_POOL   = 500
FAR_N_MAX  = 5000
NEAR_N_MAX = 5000

FAR = {
    "ssb_hard":    (f"{OOD_ROOT}/ssb_hard",           "test_ssb_hard.txt",    ""),
    "ninco":       (f"{OOD_ROOT}/ninco",              "test_ninco.txt",       "NINCO_OOD_classes/"),
    "inaturalist": (f"{OOD_ROOT}/inaturalist",        "test_inaturalist.txt", ""),
    "texture":     (f"{OOD_ROOT}/texture",            "test_textures.txt",    ""),
    "openimage_o": (f"{OOD_ROOT}/openimage_o/images", "test_openimage_o.txt", ""),
}


def rng_for_stage(stage):
    return np.random.default_rng([SEED, zlib.crc32(stage.encode())])


def held_in_classes(bench):
    if bench == "in1k":
        sys.path.insert(0, CLASSES)
        from classes import IMAGENET2012_CLASSES
        wn = list(IMAGENET2012_CLASSES)
        return np.arange(1000), wn
    import pyarrow.parquet as pq, pyarrow.compute as pc, glob
    sys.path.insert(0, CLASSES)
    from classes import IMAGENET2012_CLASSES
    w2i = {w: i for i, w in enumerate(IMAGENET2012_CLASSES)}

    f2w = {}
    for f in sorted(glob.glob(f"{IN_DIR}/validation-*.parquet")):
        tb = pq.read_table(f, columns=["image", "label"])
        paths = pc.struct_field(tb.column("image"), "path").to_pylist()
        for p, l in zip(paths, tb.column("label").to_pylist()):
            m = re.match(r"(ILSVRC2012_val_\d+)_(n\d+)\.JPEG", p)
            assert w2i[m.group(2)] == l, "parquet label 이 canonical wnid 순서와 다르다"
            f2w[m.group(1)] = m.group(2)

    seen = {}
    for line in open(f"{SPLITS}/test_imagenet200.txt"):
        rel, y = line.rsplit(" ", 1)
        key = re.match(r"val/(ILSVRC2012_val_\d+)\.JPEG", rel).group(1)
        seen.setdefault(int(y), set()).add(f2w[key])
    assert len(seen) == 200 and all(len(v) == 1 for v in seen.values())
    wn = {y: next(iter(v)) for y, v in seen.items()}
    idx = np.array([w2i[wn[y]] for y in sorted(wn)])
    return idx, [wn[y] for y in sorted(wn)]


class _ListDS:
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        from PIL import Image, ImageFile
        import io
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        kind, payload = self.items[i]
        if kind == "bytes":
            im = Image.open(io.BytesIO(payload))
        else:
            im = Image.open(payload)
        return self.tf(im.convert("RGB"))


def encode_stream(items, model, tf, dev, tag="", workers=12, bs=128):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(_ListDS(items, tf), batch_size=bs, num_workers=workers,
                    shuffle=False, pin_memory=True)
    out, n, t0 = [], 0, time.time()
    with torch.no_grad():
        for xb in dl:
            out.append(model(xb.to(dev, non_blocking=True)).cpu().numpy())
            n += len(xb)
            print(f"    [{tag}] {n}/{len(items)}  {n/(time.time()-t0):.0f} img/s", end="\r")
    print()
    return np.concatenate(out).astype(np.float32)


def extract_train(idx1k, rng, model, tf, dev, chunk=8000):
    import pyarrow.parquet as pq, pyarrow.compute as pc, glob
    want = {int(c): j for j, c in enumerate(idx1k)}
    files = sorted(glob.glob(f"{IN_DIR}/train-*.parquet"))

    print(f"  [train] label 스캔 {len(files)} shards")
    per = {j: [] for j in want.values()}
    for si, f in enumerate(files):
        lab = pq.read_table(f, columns=["label"]).column("label").to_numpy()
        for r, l in enumerate(lab):
            j = want.get(int(l))
            if j is not None:
                per[j].append((si, r))
        print(f"    {si+1}/{len(files)}", end="\r")
    print()
    counts = np.array([len(v) for v in per.values()])
    print(f"  [train] 후보 n_c: min={counts.min()} max={counts.max()}")

    pick = {}
    for j, rows in per.items():
        sel = rows if len(rows) <= REF_POOL else [rows[i] for i in
              rng.choice(len(rows), REF_POOL, replace=False)]
        for s, r in sel:
            pick.setdefault(s, {})[r] = j
    total = sum(len(v) for v in pick.values())
    print(f"  [train] 선택 {total}장, {len(pick)} shards")

    feats, labs, buf, buf_lab, done = [], [], [], [], 0
    for si, f in enumerate(files):
        if si not in pick:
            continue
        need = pick[si]
        tb = pq.read_table(f, columns=["image"])
        byt = pc.struct_field(tb.column("image"), "bytes")
        for r in sorted(need):
            buf.append(("bytes", byt[r].as_py())); buf_lab.append(need[r])
        del tb, byt
        if len(buf) >= chunk:
            feats.append(encode_stream(buf, model, tf, dev, f"train {done}/{total}"))
            labs.extend(buf_lab); done += len(buf); buf, buf_lab = [], []
    if buf:
        feats.append(encode_stream(buf, model, tf, dev, f"train {done}/{total}"))
        labs.extend(buf_lab)
    return np.concatenate(feats), np.array(labs, dtype=np.int64)


def collect_val(idx1k, rng):
    import pyarrow.parquet as pq, pyarrow.compute as pc, glob
    want = {int(c): j for j, c in enumerate(idx1k)}
    hin, hin_lab, hout = [], [], []
    for f in sorted(glob.glob(f"{IN_DIR}/validation-*.parquet")):
        tb = pq.read_table(f, columns=["image", "label"])
        byt = pc.struct_field(tb.column("image"), "bytes")
        for r, l in enumerate(tb.column("label").to_numpy()):
            j = want.get(int(l))
            if j is not None:
                hin.append(("bytes", byt[r].as_py())); hin_lab.append(j)
            else:
                hout.append(("bytes", byt[r].as_py()))
    if len(hout) > NEAR_N_MAX:
        hout = [hout[i] for i in rng.choice(len(hout), NEAR_N_MAX, replace=False)]
    return hin, np.array(hin_lab, dtype=np.int64), hout


def collect_far(name, rng):
    base, lst, strip = FAR[name]
    rels = [l.rsplit(" ", 1)[0] for l in open(f"{SPLITS}/{lst}")]
    if strip:
        rels = [r.replace(strip, "", 1) for r in rels]
    def resolve(r):
        p = os.path.join(base, r)
        if os.path.exists(p):
            return p
        alt = os.path.join(base, r.replace(" ", "_"))
        if os.path.exists(alt):
            return alt
        raise FileNotFoundError(p)

    paths = [resolve(r) for r in rels]
    if len(paths) > FAR_N_MAX:
        paths = [paths[i] for i in rng.choice(len(paths), FAR_N_MAX, replace=False)]
    return [("path", p) for p in paths]


def main(repr_key, do_verify, bench):
    global SPLITS
    SPLITS = f"{OOD_ROOT}/splits/splits/imagenet{'1k' if bench == 'in1k' else '200'}"
    spec = EF.REPRS[repr_key]
    cache = f"./feat_cache_{bench}_{repr_key}"
    os.makedirs(cache, exist_ok=True)
    print(f"[{repr_key}] cache={cache}")
    model, tf, dev, cfg = EF.build_model(spec["model"])

    idx1k, wnids = held_in_classes(bench)
    print(f"  [classes] {len(idx1k)}개 확정, 1k 인덱스 {idx1k.min()}..{idx1k.max()}")

    def save(tag, arr):
        p = os.path.join(cache, f"{bench}_{tag}.npy")
        np.save(p, arr); print(f"  [saved] {os.path.basename(p)}  {arr.shape}")

    if do_verify:
        rng = np.random.default_rng(SEED)
        items, _, _ = collect_val(idx1k, rng)
        sub = items[:64]
        from PIL import Image; import io
        ref = EF.encode([Image.open(io.BytesIO(b)) for _, b in sub], model, tf, dev)
        got = encode_stream(sub, model, tf, dev, tag="verify", workers=4, bs=32)
        d = np.abs(ref - got).max()
        print(f"\n  [verify] encode_stream vs extract_features.encode  max|diff| = {d:.3e}")
        print(f"  [verify] {'통과' if d < 1e-4 else '** 실패 **'}")
        return

    def rng_for(stage):
        return np.random.default_rng([SEED, zlib.crc32(stage.encode())])

    t0 = time.time()

    if os.path.exists(os.path.join(cache, f"{bench}_feat_train.npy")):
        print("  [train] 이미 있음 — 건너뜀")
    else:
        F, labs = extract_train(idx1k, rng_for("train"), model, tf, dev)
        save("feat_train", F); save("lab_train", labs); del F

    if os.path.exists(os.path.join(cache, f"{bench}_lab_test.npy")):
        print("  [val] 이미 있음 — 건너뜀")
    else:
        hin, hin_lab, hout = collect_val(idx1k, rng_for("val"))
        save("feat_test", encode_stream(hin, model, tf, dev, "val-in")); save("lab_test", hin_lab)
        del hin
        if hout:
            save("feat_near", encode_stream(hout, model, tf, dev, "val-near"))
        else:
            print("  [val] held-out 클래스 없음 — near 그룹은 OpenOOD 공식 near 를 쓴다")
        del hout

    for name in FAR:
        if os.path.exists(os.path.join(cache, f"{bench}_feat_far_{name}.npy")):
            print(f"  [far {name}] 이미 있음 — 건너뜀"); continue
        save(f"feat_far_{name}",
             encode_stream(collect_far(name, rng_for(name)), model, tf, dev, name))

    json.dump(dict(repr=repr_key, model=spec["model"], input_size=str(cfg.get("input_size")),
                   mean=list(cfg.get("mean")), std=list(cfg.get("std")),
                   benchmark=f"OpenOOD v1.5 imagenet{'1k' if bench == 'in1k' else '200'}", seed=SEED, ref_pool=REF_POOL,
                   far_n_max=FAR_N_MAX, near_n_max=NEAR_N_MAX,
                   held_in_wnids=wnids, held_in_idx1k=[int(x) for x in idx1k],
                   far_sets=list(FAR), load_truncated_images=True),
              open(os.path.join(cache, "BENCH_META.json"), "w"), indent=2)
    print(f"\n[done] {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=list(EF.REPRS), default="dino")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", choices=["in200", "in1k"], default="in200")
    a = ap.parse_args()
    main(a.repr, a.verify, a.bench)
