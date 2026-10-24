"""CPU test of the GAN-evaluation side on small fake data (no GPU, no downloads, ~1-2 min).

  python tests/test_gan_eval.py

KID estimator (exact closed forms, unbiasedness, SE calibration), import_synth routing and refusals,
metrics_per_class with the real-vs-real floor, gan_test with a random EfficientNet-B0 judge,
memorization and fraction_table on a fake split, and the eval notebook builder. The Inception detector
is replaced by --detector stub (a fixed random projection); the default path uses the real one.
Everything runs in a temp dir through the GENAI_* env vars.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
SPLIT_GRADES = {"train": [0] * 20 + [1] * 6 + [2] * 10 + [3] * 6 + [4] * 8, "val": [0, 1, 2, 3, 4] * 4, "test": [0, 1, 2, 3, 4] * 2}


def run(args: list, env: dict, ok: bool = True) -> str:
    p = subprocess.run([PY, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    assert (p.returncode == 0) == ok, f"rc {p.returncode} for {args}\n{out[-3000:]}"
    return out


def fake_eye(rng: np.random.Generator) -> np.ndarray:
    """256 px disc with an independent random texture (fine 'vessels' + coarse shading)."""
    yy, xx = np.mgrid[:256, :256]
    disc = (xx - 127.5) ** 2 + (yy - 127.5) ** 2 < 120 ** 2
    fine = cv2.GaussianBlur(rng.normal(size=(256, 256)).astype(np.float32), (0, 0), 2.0)
    coarse = cv2.GaussianBlur(rng.normal(size=(256, 256)).astype(np.float32), (0, 0), 24.0)
    tex = 30 * fine / fine.std() + 40 * coarse / coarse.std()
    img = np.stack([170 + tex, 80 + tex, 30 + tex / 3], -1).clip(0, 255)
    return np.where(disc[..., None], img, 0).astype(np.uint8)


def kid_checks() -> None:
    from src.gan.metrics_per_class import kid, poly_kernel

    def brute(a, b):
        kaa, kbb, kab = poly_kernel(a, a), poly_kernel(b, b), poly_kernel(a, b)
        m, n = len(a), len(b)
        return (kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (n * (n - 1)) - 2 * kab.mean()

    rng = np.random.default_rng(0)
    fr, fg = np.abs(rng.normal(size=(7, 16))), np.abs(rng.normal(size=(11, 16))) + 0.2
    est, se_r, se_g = kid(fr, fg)
    loo = np.array([brute(np.delete(fr, i, 0), fg) for i in range(len(fr))])
    assert np.isclose(est, brute(fr, fg)) and np.isclose(se_r, np.sqrt(6 / 7 * np.square(loo - loo.mean()).sum())), (est, se_r)
    x, y = fg[:7], fr                                   # stylegan3's kid50k formula for equal subset sizes
    a, b = poly_kernel(x, x) + poly_kernel(y, y), poly_kernel(x, y)
    assert np.isclose(kid(fr, fg[:7])[0], ((a.sum() - np.trace(a)) / 6 - b.sum() * 2 / 7) / 7)
    for shift, lo, hi in ((0.0, 0.8, 2.5), (0.15, 0.7, 1.4)):     # same distribution / different distributions
        ests, ses = [], []
        for r in range(200):
            e, s1, s2 = kid(np.abs(rng.normal(size=(25, 32))), np.abs(rng.normal(size=(100, 32))) + shift, n_half=50, seed=r)
            ests.append(e)
            ses.append(np.hypot(s1, s2))
        ests, sd = np.array(ests), np.std(ests, ddof=1)
        assert np.isfinite(ses).all() and min(ses) > 0 and lo < np.mean(ses) / sd < hi, (shift, np.mean(ses), sd)
        z = ests.mean() / (sd / np.sqrt(len(ests)))
        assert (abs(z) < 4) if shift == 0 else (z > 20), (shift, z)
        print(f"kid shift {shift}: mean {ests.mean():+.5f}  sd {sd:.5f}  mean se {np.mean(ses):.5f}")


def make_data(data: Path, splits: Path, rng) -> tuple[pd.DataFrame, np.ndarray]:
    rows = [(s, g) for s, gs in SPLIT_GRADES.items() for g in gs]
    df = pd.DataFrame({"image_id": [f"{i:012x}" for i in rng.choice(2 ** 40, len(rows), replace=False)],
                       "split": [s for s, _ in rows], "grade": [g for _, g in rows]})
    df["frac_rank"] = -1
    tr = df.split == "train"
    df.loc[tr, "frac_rank"] = rng.permutation(int(tr.sum()))
    splits.mkdir(parents=True)
    df[["image_id", "split", "frac_rank"]].to_csv(splits / "aptos_splits.csv", index=False)
    (data / "raw" / "aptos").mkdir(parents=True)
    df.rename(columns={"image_id": "id_code", "grade": "diagnosis"})[["id_code", "diagnosis"]].to_csv(data / "raw" / "aptos" / "train.csv", index=False)
    arr = np.stack([fake_eye(rng) for _ in range(len(df))])
    (data / "aptos256").mkdir()
    for i, a in zip(df.image_id, arr):
        Image.fromarray(a).save(data / "aptos256" / f"{i}.png")
    (data / "cache").mkdir()
    np.save(data / "cache" / "aptos256.npy", arr)
    (data / "cache" / "aptos256.ids.txt").write_text("\n".join(df.image_id) + "\n", encoding="utf-8")
    return df, arr


def sg3_zip(path: Path, sub: pd.DataFrame, data: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for i in sub.image_id:
            z.write(data / "aptos256" / f"{i}.png", f"{i}.png")
        z.writestr("dataset.json", json.dumps({"labels": [[f"{i}.png", int(g)] for i, g in zip(sub.image_id, sub.grade)]}))


def synth_zip(path: Path, imgs: dict, psi: float, grade_meta=None, generator=None, trunc_center="class") -> None:
    from src.gan.watermark import save_synthetic
    src = path.with_suffix("")
    generator = generator or path.name.split("_")[1].split("-")[0] + "-ada"  # synth_f100-ada[_psi1].zip -> f100-ada
    for g, lst in imgs.items():
        for s, a in enumerate(lst):
            save_synthetic(Image.fromarray(a), src / f"g{g}" / f"g{g}_s{s:05d}.png", visible=False, generator=generator,
                           grade=grade_meta if grade_meta is not None else g, seed=s, psi=psi, trunc_center=trunc_center)
    shutil.make_archive(str(src), "zip", src)


def notebook_checks() -> None:
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "notebooks" / "build_notebooks.py")
    bn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bn)
    for kind, src in bn.EVAL_CELLS:
        if kind == "code":
            compile(src, "eval_cell", "exec")
    code = "\n".join(src for kind, src in bn.EVAL_CELLS if kind == "code")
    assert "PSI_CHECK_TAGS" in code and '"--real-train"' in code and 'find_zip("f100"' in code and "_psi1" in code
    built = json.dumps(bn.nb(bn.EVAL_CELLS), indent=1)
    assert built == (ROOT / "notebooks" / "kaggle_gan_eval.ipynb").read_text(encoding="utf-8"), "rebuild: python notebooks/build_notebooks.py"


def main() -> None:
    kid_checks()
    notebook_checks()
    tmp = Path(tempfile.mkdtemp(prefix="gan_eval_test_"))
    data, splits, results, ckpt, dl = tmp / "data", tmp / "splits", tmp / "results", tmp / "ckpt", tmp / "downloads"
    env = {**os.environ, "GENAI_DATA": str(data), "GENAI_SPLITS": str(splits), "GENAI_RESULTS": str(results),
           "GENAI_CHECKPOINTS": str(ckpt), "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "2"}
    rng = np.random.default_rng(1)
    df, arr = make_data(data, splits, rng)
    train3 = df[(df.split == "train") & (df.grade == 3)].index[0]
    train4 = df[(df.split == "train") & (df.grade == 4)].index[0]

    # fraction_table: nested rare-grade counts, grades 0-2 always full
    run(["-m", "src.analysis.fraction_table"], env)
    ft = pd.read_csv(results / "tables" / "fraction_label_shares.csv").set_index("tag")
    assert ft.loc[["f010", "f025", "f050", "f100"], ["n_g3", "n_g4"]].values.tolist() == [[1, 1], [2, 2], [3, 4], [6, 8]], ft
    assert (ft[["n_g0", "n_g1", "n_g2"]] == [20, 6, 10]).all().all() and np.allclose(ft.filter(like="share_").sum(axis=1), 1, atol=1e-4)
    assert "| f010 | 38 | 20 (52.63%)" in (results / "tables" / "fraction_label_shares.md").read_text(encoding="utf-8")

    # import_synth: main sets, a psi-1.0 set, the noaug ablation; one planted copy of a real training eye
    dl.mkdir()
    f100 = {3: [arr[train3]] + [fake_eye(rng) for _ in range(5)],          # planted copy of a training eye
            4: [np.ascontiguousarray(arr[train4][:, ::-1])] + [fake_eye(rng) for _ in range(5)]}  # and a mirrored one
    synth_zip(dl / "synth_f100-ada.zip", f100, 0.8)
    synth_zip(dl / "synth_f100-ada_psi1.zip", {g: [fake_eye(rng) for _ in range(4)] for g in (3, 4)}, 1.0)
    synth_zip(dl / "synth_f010-ada.zip", {g: [fake_eye(rng) for _ in range(4)] for g in (3, 4)}, 0.8)
    synth_zip(dl / "synth_f010-noaug.zip", {g: [fake_eye(rng) for _ in range(2)] for g in (3, 4)}, 0.8)
    zips = [dl / f"synth_{t}.zip" for t in ("f100-ada", "f100-ada_psi1", "f010-ada", "f010-noaug")]
    out = run(["-m", "src.gan.import_synth", *zips], env)
    synth = data / "synth"
    assert "skip synth_f010-noaug.zip" in out and sorted(p.name for p in synth.iterdir()) == ["f010", "f100", "f100_psi1"], out
    assert [len(list((synth / s / g).glob("*.png"))) for s in ("f100", "f100_psi1", "f010") for g in ("g3", "g4")] == [6, 6, 4, 4, 4, 4]
    assert not list(synth.rglob("*.tmp"))
    run(["-m", "src.gan.import_synth", str(dl / "synth_f*.zip")], env)    # idempotent; pattern expanded by the tool
    assert "no synth_f*.zip files match" in run(["-m", "src.gan.import_synth", str(dl / "nothing_*.zip")], env, ok=False)
    for name, imgs, psi, gm, kw, msg, dest in (
            ("conflict/synth_f100-ada.zip", {3: [fake_eye(rng) for _ in range(7)]}, 0.8, None, {}, "another generation run", synth / "f100" / "g3" / "g3_s00006.png"),
            ("bad/synth_f010-ada_psi1.zip", {3: [fake_eye(rng)]}, 0.8, None, {}, "must have psi 1.0", synth / "f010_psi1"),
            ("bad/synth_f025-ada.zip", {3: [fake_eye(rng)]}, 1.0, None, {}, "belong in synth_<tag>_psi1.zip", synth / "f025"),
            ("bad2/synth_f050-ada.zip", {3: [fake_eye(rng)]}, 0.8, 4, {}, "grade metadata 4 but folder g3", synth / "f050"),
            ("bad3/synth_f050-ada.zip", {3: [fake_eye(rng)]}, 0.8, None, {"generator": "f100-ada"}, "expected f050-ada", synth / "f050"),
            ("bad4/synth_f050-ada.zip", {3: [fake_eye(rng)]}, 0.8, None, {"trunc_center": "global"}, "conditional truncation", synth / "f050")):
        (dl / name).parent.mkdir(exist_ok=True)
        synth_zip(dl / name, imgs, psi, gm, **kw)
        assert msg in run(["-m", "src.gan.import_synth", dl / name], env, ok=False) and not dest.exists(), name

    # metrics_per_class: per-class KID rows + the real-vs-real floor, merged into an existing CSV
    sg3 = data / "sg3"
    sg3.mkdir()
    sg3_zip(sg3 / "aptos256_val.zip", df[df.split == "val"], data)
    sg3_zip(sg3 / "aptos256_f100.zip", df[df.split == "train"], data)
    gm_csv = results / "gan_metrics.csv"
    pd.DataFrame([dict(gan="f100-ada", grade="3", kid=9.9), dict(gan="f100-ada", grade="all", fid=12.3, kid=0.01)]).to_csv(gm_csv, index=False)
    cmd = ["-m", "src.gan.metrics_per_class", "--gan", "f100-ada", "--synth-dir", synth / "f100", "--real", sg3 / "aptos256_val.zip",
           "--real-train", sg3 / "aptos256_f100.zip", "--out", gm_csv, "--detector", "stub"]
    run(cmd, env)
    run(cmd, env)
    gm = pd.read_csv(gm_csv, dtype={"grade": str})
    assert sorted(zip(gm.gan, gm.grade)) == [("f100-ada", "3"), ("f100-ada", "4"), ("f100-ada", "all"),
                                             ("real-f100-train", "3"), ("real-f100-train", "4")], gm
    per = gm[gm.grade != "all"]
    assert per.fid.isna().all() and np.isfinite(per[["kid", "kid_std", "kid_se_real", "kid_se_gen"]]).all().all(), per
    assert (per.n_real == 4).all() and per.set_index(["gan", "grade"]).n_gen.to_dict() == {
        ("f100-ada", "3"): 6, ("f100-ada", "4"): 6, ("real-f100-train", "3"): 6, ("real-f100-train", "4"): 8}
    assert gm[gm.grade == "all"].fid.item() == 12.3

    # gan_test: random EfficientNet-B0 judge saved like train.py's best.pt
    import torch
    from src.classify.model import create_model
    (ckpt / "c5_f100_r0_s0").mkdir(parents=True)
    runs_csv = results / "classifier_runs.csv"
    pd.DataFrame([dict(run_id="c5_f100_r0_s0", dataset="aptos_test", timestamp="2026-10-24T00:00:00+00:00")]).to_csv(runs_csv, index=False)
    torch.save({"model": create_model(pretrained=False).state_dict(), "epoch": 0, "val": {}, "run_id": "c5_f100_r0_s0",
                "model_name": "efficientnet_b0"}, ckpt / "c5_f100_r0_s0" / "best.pt")
    run(["-m", "src.analysis.gan_test"], env)
    pd.DataFrame([dict(run_id="c5_f100_r0_s0", dataset="aptos_test", timestamp="2099-01-01T00:00:00+00:00")]).to_csv(runs_csv, index=False)
    assert "is not the finished c5_f100_r0_s0" in run(["-m", "src.analysis.gan_test", "--out-csv", tmp / "x.csv"], env, ok=False)
    gt = pd.read_csv(results / "tables" / "gan_test.csv")
    assert sorted(zip(gt["set"], gt.grade)) == [(s, g) for s in ("f010", "f100", "f100_psi1", "real val") for g in (3, 4)], gt
    assert np.allclose(gt.filter(like="pred_").sum(axis=1), 1, atol=1e-3) and gt[gt["set"] == "real val"].n.tolist() == [4, 4]
    assert gt[gt["set"] == "f100_psi1"].psi.tolist() == [1.0, 1.0] and gt[gt["set"] == "f100"].n.tolist() == [6, 6]
    assert (gt.share_as_grade == gt.apply(lambda r: r[f"pred_{r.grade}"], axis=1)).all()
    text = (results / "tables" / "gan_test.csv").read_text(encoding="utf-8")
    assert not any(i in text for i in df.image_id), "gan_test.csv must not carry image ids"
    assert (results / "figures" / "gan_test.png").stat().st_size > 10_000

    # memorization: the planted copy is found in feature and pixel space, nothing else is
    run(["-m", "src.analysis.memorization", "--detector", "stub"], env)
    ms = pd.read_csv(results / "tables" / "memorization_summary.csv").set_index(["set", "grade"])
    assert sorted(ms.index) == [(s, g) for s in ("f010", "f100", "f100_psi1") for g in (3, 4)], ms
    for g in (3, 4):  # the plain copy (g3) and the mirrored copy (g4) are both found
        hit = ms.loc[("f100", g)]
        assert hit.n_vessel_copies == 1 and hit.n_vessel_above_ref >= 1 and hit.n_below_p1 >= 1 and hit.max_vessel > 0.99, (g, hit)
    rest = ms.drop(index=[("f100", 3), ("f100", 4)])
    assert (rest.n_vessel_copies == 0).all() and (ms.ref_max_vessel < 0.9).all() and (ms.n_ref == 20).all(), ms
    assert ms.loc[("f010", 3)].n_train == 38 and ms.loc[("f100", 3)].n_train == 50
    per_img = pd.read_csv(results / "tables" / "memorization_f100.csv")
    copy = per_img[per_img.vessel_copy].set_index("file")
    assert sorted(copy.index) == ["g3_s00000.png", "g4_s00000.png"], copy
    assert copy.loc["g3_s00000.png", "vessel_image_id"] == copy.loc["g3_s00000.png", "nn_image_id"] == df.image_id[train3]
    assert not copy.loc["g3_s00000.png", "vessel_mirrored"] and copy.loc["g4_s00000.png", "vessel_mirrored"], copy
    assert copy.loc["g4_s00000.png", "vessel_image_id"] == df.image_id[train4] and copy.loc["g4_s00000.png", "nn_mirrored"], copy
    assert "unknown synthetic sets ['f100_psi']" in run(["-m", "src.analysis.memorization", "--detector", "stub", "--sets", "f010,f100_psi"], env, ok=False)
    assert all((results / "figures" / f"memorization_{s}.png").exists() for s in ("f010", "f100", "f100_psi1"))
    cache = data / "cache" / "stub_trainval.npy"
    stamp = cache.stat().st_mtime_ns
    run(["-m", "src.analysis.memorization", "--detector", "stub", "--sets", "f010"], env)
    assert cache.stat().st_mtime_ns == stamp and len(pd.read_csv(results / "tables" / "memorization_summary.csv")) == 6

    shutil.rmtree(tmp, ignore_errors=True)
    print("GAN_EVAL_OK")


if __name__ == "__main__":
    main()
