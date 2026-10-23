# Patches applied to third_party/stylegan3

Submodule: `NVlabs/stylegan3` pinned at `c233a919a6faee6e36a316ddd4eddababad1adf9`
(2023-04-26, "Fix custom ops bug for pytorch 1.12 and onwards"). The submodule is never
modified in git; patches live in `third_party/patches/` and are applied with
`python scripts/apply_patches.py` (idempotent) after `git submodule update --init`.

## 0001-resume-shape-tolerant.patch

**Problem.** `training_loop.py` resumes with `misc.copy_params_and_buffers(..., require_all=False)`,
which copies tensors by name and calls `tensor.copy_()`; a shape mismatch raises. Fine-tuning a
class-conditional network (`--cond=1`) from an unconditional pickle (FFHQ-256) mismatches in:

| tensor | source | target (c_dim=5) | handling |
|---|---|---|---|
| `G.mapping.fc0.weight` | [512, 512] | [512, 1024] (z + label embedding) | partial copy into columns 0..511 |
| `G.mapping.embed.*` | absent | new | fresh init |
| `D.mapping.*` | absent | new | fresh init |
| `D.b4.out.{weight,bias}` | [1, 512] / [1] | [512, 512] / [512] (cmap projection) | skipped, fresh init |

**Change.** Adds `copy_params_and_buffers_tolerant()` to `torch_utils/misc.py` (exact-shape copy,
column-slice partial copy for 2-D weights with equal leading dim, skip otherwise, prints a
summary) and makes `training_loop.py` call it for G, D and G_ema. Behaviour for same-shape
resumes (our own snapshots) is identical to upstream.

**Verification.** `python scripts/check_resume_compat.py` builds conditional G/D from the FFHQ
pickle's `init_kwargs`, runs the tolerant copy and one forward pass, and prints the matching
`train.py` flags (`--cbase=16384 --cmax=512 --map-depth=8`).

## 0002-modern-python-imports.patch

**Problem.** Kaggle's image runs Python 3.13. `dnnlib/util.py` imports `strtobool` from `distutils`
(removed from the standard library in 3.12; it only works while setuptools injects a shim), and
`torch_utils/ops/conv2d_gradfix.py` / `grid_sample_gradfix.py` import `parse_version` from
`pkg_resources` (deprecated, removed from setuptools 81+). Either missing module would stop every
training or metrics run at import time.

**Change.** `strtobool` falls back to a local implementation; `parse_version` comes from
`packaging.version` with `pkg_resources` only as a fallback. Verified: all training, network and
metrics modules import under Python 3.13 with warnings promoted to errors.

## 0003-custom-ops-returned-module.patch

**Problem.** `torch_utils/custom_ops.py` compiles a plugin with `torch.utils.cpp_extension.load()`,
ignores the returned module and then calls `importlib.import_module(module_name)`. That only worked
while loading an extension registered it in `sys.modules`; with Kaggle's Python 3.13 + PyTorch 2.11
the build succeeds and the import fails with `ModuleNotFoundError: No module named 'bias_act_plugin'`
(first Kaggle run, `check_resume_compat.py`).

**Change.** Use the module returned by `load()`; fall back to the import only if it returned None.
Covers `bias_act`, `upfirdn2d` and `filtered_lrelu`, which all go through `get_plugin`.

**Verification.** ADA's augment pipe with `grid_sample_gradfix` and `conv2d_gradfix` enabled runs
forward plus the double backward that R1 needs on PyTorch 2.14 (CPU).

## 0004-infinite-sampler-init.patch

**Problem.** `misc.InfiniteSampler.__init__` calls `super().__init__(dataset)`. PyTorch removed
`Sampler.__init__(data_source)`, so the call reaches `object.__init__` and training dies at
"Loading training set..." with `TypeError: object.__init__() takes exactly one argument`.

**Change.** `super().__init__()`.

**Verification.** All four patches apply with `git apply` to a pristine LF checkout of the pinned commit.

## 0005-scipy-sqrtm-no-disp.patch

**Problem.** `metrics/frechet_inception_distance.py` calls `scipy.linalg.sqrtm(..., disp=False)` and
unpacks two values. SciPy 1.18 (June 2026, Python >= 3.12) removed `disp`, so FID raises
`TypeError: sqrtm() got an unexpected keyword argument 'disp'` after all features are computed.
Kaggle currently ships SciPy 1.16.3 (works with a DeprecationWarning), so the bug is latent until
the image or a dependency upgrades SciPy. `src/gan/metrics_per_class.py` had the same call.

**Change.** `s = scipy.linalg.sqrtm(...)`: without `disp`, every SciPy version returns the matrix only.

**Verification.** Reproduced the TypeError with the SciPy 1.18.1 cp313 wheel; the patched function
returns the same FID as the original on SciPy 1.16.3 and 1.17.1.

## Not patched (checked)

- `torch_utils/ops/grid_sample_gradfix.py` already handles PyTorch >= 1.12 (`op[0]`), so ADA runs
  on PyTorch 2.x without changes.
- Custom CUDA ops (`bias_act`, `upfirdn2d`) compile on Kaggle (nvcc present). On machines without
  a compiler they fall back to the slow reference implementation with a warning; inference only.

## Known upstream behaviour (kept; mention in the report's method section)

Found by a CPU run of the full training loop (both Python 3.11/torch 2.14 and 3.13/torch 2.10,
including 2-rank DDP, same-shape resume and the conditional-from-unconditional resume):

- **fc0 transfer is not function-preserving.** `FullyConnectedLayer` scales by
  `lr_multiplier / sqrt(in_features)`, so widening `mapping.fc0` from 512 to 1024 inputs shrinks the
  copied z-columns' gain by 1/sqrt(2), and the fresh label columns add an equal-variance term. At
  init the transferred mapping's W has a centred cosine of about 0.5 to FFHQ's. Fine-tuning absorbs
  this; every fraction's GAN uses the same procedure, so the comparison stays fair.
- **Adam state and the path-length mean `pl_mean` are not stored in snapshots**, so each Kaggle
  session restarts them (a short transient after each resume).
- **`--seed=0` every session** replays the same sampler and z stream after a resume.
- `augment_pipe.p` is a registered buffer and is broadcast from rank 0 at "Distributing across
  GPUs", so 0001's restore on every rank is redundant but harmless.
