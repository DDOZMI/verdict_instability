# Verdict Instability — reproduction package

Code behind every table in the paper. Result files are not included; the core numbers are in
[Core results](#core-results). Notation follows the paper.

---

## Layout

```
reproduce/
  code/        30 Python files
  README.md
```

Run from `code/` with the feature caches (`feat_cache_*`) alongside.

---

## Code

### Extraction

| Script | Produces |
|---|---|
| `extract_features.py` | CIFAR-10 / CIFAR-100 and their far-OOD sources, DINOv2 ViT-B/14 |
| `extract_lowres.py` | 32-pixel variants of the CIFAR OOD sources |
| `extract_cifar10_id.py` | CIFAR-10 held-in split |
| `extract_derma.py` | DermaMNIST at 28 pixels |
| `imagenet_extract.py` | ImageNet-1k and the five OpenOOD far sets |
| `clip_extract.py` `resnet_extract.py` `dinov3_extract.py` | the three backbone re-encodings |

### Detector and closed form

| Script | Contents |
|---|---|
| `derma_probe.py` | `Space`: the detector of Eq. (2), the 11 scores, bootstrap `truth()`, `rect_gauss_var` |
| `in200_probe.py` | `Space200` (same numerics, large $C$), `build()`, the ImageNet-1k driver, `SCORES11` |
| `space_lowmem.py` | `Space200LowMem`, for query sets too large for a full distance matrix |
| `exponent_probe.py` | the CIFAR and DermaMNIST builders |
| `odin_real_probe.py` `odin_real_align.py` | index builders, ODIN alignment |
| `in1k_extras.py` | `rebuild_query_selection()`, the remaining ImageNet-1k columns |
| `gpu_logreg.py` | GPU replacement for the multinomial logistic probe |

---

## Paper artifact to script

### Main text

| Artifact | Script |
|---|---|
| Table 1 — closed form against the bootstrap | `exponent_probe.py` (CIFAR), `in200_probe.py` (ImageNet-1k), `derma_minn_sweep_probe.py` (DermaMNIST) |
| Table 2 — allocation effective dimension | `in200_probe.py`, `exponent_probe.py`, `derma_minn_sweep_probe.py` |
| Table 3 — abstention against instability | `c1_replication_probe.py` then `c1_tables.py` (CIFAR, DermaMNIST), `in200_probe.py` (ImageNet-1k) |

### Appendix

| Artifact | Script |
|---|---|
| Table 4 — the argmin correction | `min_correction_probe.py` (CIFAR, ImageNet-1k), `derma_mincorr_paperpath.py` (DermaMNIST) |
| Penalty share, per-class ratio, count identification | `derma_minn_sweep_probe.py` |
| The 11 scores, tabulated | `in200_probe.py`, `exponent_probe.py` |
| ODIN perturbation sweep over $\varepsilon$ | `in1k_extras.py` |
| Falsification under stratification | `falsify_in1k_probe.py`, `anisotropy_probe.py` |
| The three backbone re-encodings | `in1k_backbones.py` for the 10 cached scores, `in1k_repcheck_unified.py` for the `odin` row |
| Reference-count control | `nc_control_probe.py` |
| The $\alpha$ – $\lambda$ grid | `exponent_probe.py`, `in200_probe.py` |
| Stretch $W$ and size $\lVert \delta \rVert^2$ channels | `f4_stretch_probe.py`, `in1k_extras.py` |
| Retained mean instability across the coverage grid | `instability_coverage_probe.py` |
| Seed detail and level decomposition | `c1_replication_probe.py` then `c1_tables.py`, `in200_probe.py` |
| Reference-count sweep over $n_c$ | `sweep_values.py` (CIFAR), `in1k_extras.py` (ImageNet-1k) |
| Instability against misclassification | `c1_orthogonal_probe.py` (CIFAR), `in1k_extras.py` (ImageNet-1k) |

---

## Core results

### Table 1 — closed form against the bootstrap

$R^2 = \mathrm{corr}(T, \widehat{T})^2$; the last column is the median of $T / \widehat{T}$.

| Dataset | $C$ | $n_c$ | $R^2$ | $T / \widehat{T}$ |
|---|---:|---|---:|---:|
| CIFAR-10 | $5$ | $400$ | $0.918$ | $0.963$ |
| CIFAR-100 | $50$ | $400$ | $0.820$ | $0.953$ |
| ImageNet-1k | $1000$ | $400$ | $0.910$ | $0.976$ |
| DermaMNIST | $3$ | $769$ – $4693$ | $\mathbf{0.974}$ | $1.005$ |
| DermaMNIST | $7$ | $80$ – $4693$ | $0.923$ | $0.976$ |

### Table 2 — allocation effective dimension

Allocation effective dimension is $1 / \sum_i p_i^2$.

| Query group | $\rho(r, \sigma_t)$ | Alloc. eff. dim. |
|---|---:|---:|
| **CIFAR-100** ($C = 50$) | | |
| in | $-0.200$ | $36.5$ |
| near | $-0.269$ | $54.2$ |
| SVHN | $-0.138$ | $50.3$ |
| DTD | $-0.265$ | $76.2$ |
| LSUN | $-0.020$ | $52.5$ |
| iSUN | $-0.137$ | $55.4$ |
| Places365 | $-0.184$ | $56.3$ |
| CIFAR-10 | $-0.332$ | $56.1$ |
| *pooled* | $\mathbf{-0.306}$ | — |
| **ImageNet-1k** ($C = 1000$) | | |
| in | $-0.210$ | $35.1$ |
| SSB-hard | $-0.065$ | $40.2$ |
| NINCO | $-0.105$ | $53.0$ |
| iNaturalist | $-0.225$ | $47.1$ |
| Textures | $-0.254$ | $67.8$ |
| OpenImage-O | $-0.147$ | $58.0$ |
| *pooled* | $\mathbf{-0.223}$ | — |
| **DermaMNIST** ($C = 3$) | | |
| in | $+0.15$ | $\mathbf{8.5}$ |
| near | $+0.12$ | $12.9$ |
| SVHN | $-0.24$ | $34.7$ |
| DTD | $-0.61$ | $64.9$ |
| CIFAR-10 | $-0.47$ | $\mathbf{75.5}$ |
| *pooled* | $\mathbf{-0.83}$ | — |

### The 11 scores, tabulated

Correlations are centered within query group.

| Family | Score | CIFAR-100 $\rho(M, \sigma_t)$ | $\rho(M, \widehat{T})$ | $\rho(M, T)$ | ImageNet-1k $\rho(M, \sigma_t)$ | $\rho(M, \widehat{T})$ | $\rho(M, T)$ |
|---|---|---:|---:|---:|---:|---:|---:|
| Dispersion | `knn_std` | $+0.767$ | $+0.734$ | $+0.695$ | $+0.652$ | $+0.652$ | $+0.654$ |
| | `lid` | $+0.532$ | $+0.509$ | $+0.483$ | $+0.512$ | $+0.512$ | $+0.510$ |
| Distance | `d_cls` | $-0.202$ | $-0.309$ | $-0.334$ | $-0.167$ | $-0.167$ | $-0.239$ |
| | `knn` | $-0.419$ | $-0.495$ | $-0.481$ | $-0.405$ | $-0.405$ | $-0.439$ |
| | `maha` | $-0.265$ | $-0.390$ | $-0.355$ | $-0.150$ | $-0.151$ | $-0.193$ |
| | `vim` | $-0.324$ | $-0.317$ | $-0.371$ | $-0.187$ | $-0.187$ | $-0.203$ |
| Logit | `energy` | $-0.248$ | $-0.182$ | $-0.262$ | $-0.203$ | $-0.203$ | $-0.223$ |
| | `maxlogit` | $-0.248$ | $-0.183$ | $-0.263$ | $-0.200$ | $-0.200$ | $-0.220$ |
| | `odin` | $-0.153$ | $-0.123$ | $-0.179$ | $-0.079$ | $-0.079$ | $-0.087$ |
| | `msp` | $-0.167$ | $-0.130$ | $-0.184$ | $-0.059$ | $-0.059$ | $-0.074$ |
| | `entropy` | $-0.182$ | $-0.138$ | $-0.197$ | $-0.071$ | $-0.071$ | $-0.087$ |

### Table 3 — abstention against instability

$\Delta_M = \mathrm{AURC}_M / \mathrm{AURC}_{\mathrm{rand}} - 1$, in percent. Entries average 10
seeds and carry half the width of a $95\%$ interval over them.

| Score | CIFAR-100 $\rho(M, \widehat{T})$ | AURC | $\Delta$ (%) | ImageNet-1k $\rho(M, \widehat{T})$ | AURC | $\Delta$ (%) | DermaMNIST $\rho(M, \widehat{T})$ | AURC | $\Delta$ (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `knn_std` | $+0.714$ | $0.1201$ | $-6.99 \pm 0.14$ | $+0.631$ | $0.1376$ | $-8.82 \pm 0.17$ | $+0.747$ | $0.0500$ | $-19.70 \pm 0.20$ |
| `lid` | $+0.541$ | $0.1220$ | $-5.57 \pm 0.16$ | $+0.545$ | $0.1406$ | $-6.81 \pm 0.13$ | $+0.714$ | $0.0500$ | $-19.68 \pm 0.20$ |
| `energy` | $-0.108$ | $0.1317$ | $\mathbf{+1.95} \pm 0.20$ | $-0.220$ | $0.1554$ | $\mathbf{+3.00} \pm 0.17$ | $+0.530$ | $0.0533$ | $-14.26 \pm 0.30$ |
| `msp` | $-0.193$ | $0.1321$ | $\mathbf{+2.30} \pm 0.11$ | $-0.134$ | $0.1537$ | $\mathbf{+1.87} \pm 0.13$ | $+0.320$ | $0.0580$ | $-6.84 \pm 0.38$ |
| `maha` | $-0.534$ | $0.1342$ | $\mathbf{+3.93} \pm 0.11$ | $-0.206$ | $0.1557$ | $\mathbf{+3.19} \pm 0.14$ | $-0.696$ | $0.0704$ | $\mathbf{+13.09} \pm 0.18$ |
| *random* | $+0.000$ | $0.1291$ | — | $+0.000$ | $0.1509$ | — | $+0.000$ | $0.0622$ | — |

---

## Notes

**Bootstrap.** $T$ is drawn over 200 bootstrap replicates; $\sigma_t$ and $\widehat{T}$ are
deterministic given the reference set. A different RNG stream moves $\rho(M, T)$ by $0.003$ to
$0.007$. Follow each setting's existing call order and assert that the stored rows reproduce to
`0.00e+00` before trusting a new column.

**Re-encoding batch size.** The ImageNet-1k caches were written at $128$. ResNet-50 is convolutional
and cuDNN picks its algorithm from the batch shape: $32$ drifts $2.2 \times 10^{-3}$, $64$ drifts
$3.5 \times 10^{-4}$, $128$ is bit-exact. Verify by identity, each re-encoded row nearest to its own
cached row, rather than by absolute tolerance — the two rows sitting in each group's final short
batch drift either way.

**Memory.** Raw ImageNet-1k queries hold about $7.5$ GB and are not returned to the operating system
on `del`. Use one process per backbone, and for ResNet-50 split into
`in1k_repcheck_unified.py --stage odin` then `--stage probe`.

**Settings.** `gpu_logreg.install()` swaps in a float64 torch LBFGS probe. Hyperparameters are fixed
in `derma_probe.CFG`; ODIN uses temperature $1000$ and $\varepsilon = 0.0014$.
