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

## Not patched (checked)

- `torch_utils/ops/grid_sample_gradfix.py` already handles PyTorch >= 1.12 (`op[0]`), so ADA runs
  on PyTorch 2.x without changes.
- Custom CUDA ops (`bias_act`, `upfirdn2d`) compile on Kaggle (nvcc present). On machines without
  a compiler they fall back to the slow reference implementation with a warning; inference only.
