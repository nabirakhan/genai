"""Fast CPU test of the pre-registered checks (no training, ~30 s): FundusRandAugment op subset, run_id suffixes,
synthetic-variant folders, queue --checks, and src.analysis.checks on synthetic result CSVs (variant adopted,
baseline kept, ties, missing runs, --apply). Settings go to a temp config via GENAI_CLASSIFY_CONFIG; the tracked
src/classify/config.json is never written (checked at the end).

  python tests/test_checks.py
"""
import collections
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRACKED = ROOT / "src" / "classify" / "config.json"
SUBSET = {"Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Brightness", "Color", "Contrast", "Sharpness", "AutoContrast"}


def write_config(ra_magnitude=9, synth_variant=""):
    CONFIG.write_text(json.dumps({"ra_magnitude": ra_magnitude, "synth_variant": synth_variant}) + "\n", encoding="utf-8")


def invoke(cmd, args, ok=True) -> str:
    r = CliRunner().invoke(cmd, [str(a) for a in args])
    if ok:
        assert r.exit_code == 0, (args, r.output, r.exception)
    else:
        assert r.exit_code == 1 and isinstance(r.exception, SystemExit), (args, r.output, r.exception)  # ClickException
    return r.output


def check_randaugment():
    assert set(FundusRandAugment._AUGMENTATION_SPACE) == SUBSET
    seen = collections.Counter()

    class Recording(FundusRandAugment):
        def _apply_image_or_video_transform(self, image, transform_id, *a, **k):
            seen[transform_id] += 1
            return super()._apply_image_or_video_transform(image, transform_id, *a, **k)

    torch.manual_seed(0)
    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    ra = Recording(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=BILINEAR, fill=0)
    for _ in range(400):
        assert ra(x).shape == x.shape
    assert set(seen) == SUBSET and sum(seen.values()) == 800, seen

    write_config(9)
    for cond, mag, want in ((4, None, None), (5, None, 9), (5, 15, 15), (6, 15, 15)):
        ops = build_transform(cond, True, mag).transforms
        ras = [op for op in ops if isinstance(op, FundusRandAugment)]
        assert [r.magnitude for r in ras] == ([want] if want else []), (cond, ras)
        assert all(r.num_ops == 2 and r.num_magnitude_bins == 31 and r.interpolation == BILINEAR and r.fill == 0 for r in ras)
        assert all(op.interpolation == BILINEAR for op in ops if isinstance(op, T.RandomRotation))
    assert not any(isinstance(op, (FundusRandAugment, T.RandomRotation)) for op in build_transform(6, False).transforms)
    # what a spawned Windows DataLoader worker does: unpickle the transform in a fresh interpreter
    p = subprocess.run([sys.executable, "-c", "import pickle, sys; t = pickle.loads(sys.stdin.buffer.read()); "
                        "print(type(t.transforms[4]).__module__, t.transforms[4].magnitude)"],
                       input=pickle.dumps(build_transform(6, True, 15)), capture_output=True, cwd=ROOT)
    assert p.returncode == 0 and p.stdout.split() == [b"src.classify.data", b"15"], p


def check_run_ids_and_synth_dirs():
    write_config(9, "")
    assert run_id_of(5, 0.1, 0.0, 0) == run_id_of(5, 0.1, 0.0, 0, ra_magnitude=9) == "c5_f010_r0_s0"
    assert run_id_of(5, 0.1, 0.0, 0, ra_magnitude=15, eval_split="val") == "c5_f010_r0_s0_m15_val"
    assert run_id_of(6, 1.0, 1.0, 2, 9, "psi1", "val") == "c6_f100_r1_s2_psi1_val"
    assert run_id_of(6, 0.25, 0.5, 1, 9, "") == "c6_f025_r0.5_s1"
    assert run_id_of(4, 1.0, 0.0, 0, eval_split="val") == "c4_f100_r0_s0_val"
    write_config(15, "psi1")
    assert run_id_of(5, 0.1, 0.0, 0, ra_magnitude=15) == "c5_f010_r0_s0"
    assert run_id_of(5, 0.1, 0.0, 0, ra_magnitude=9) == "c5_f010_r0_s0_m9"
    assert run_id_of(6, 1.0, 1.0, 0, 15, "") == "c6_f100_r1_s0_psi08"
    for folder in ("f100", "f100_psi1"):
        for g in (3, 4):
            d = TMP / "data" / "synth" / folder / f"g{g}"
            d.mkdir(parents=True)
            for k in range(3):
                (d / f"g{g}_s{k:05d}.png").write_bytes(b"")
    for variant, folder in (("psi1", "f100_psi1"), ("", "f100"), (None, "f100_psi1")):  # None: config.json (psi1 here)
        paths, labels = synth_files(1.0, 1.0, {3: 2, 4: 3}, variant)
        assert {p.parent.parent.name for p in paths} == {folder} and labels == [3, 3, 4, 4, 4], (variant, paths)
    try:
        synth_files(0.1, 1.0, {3: 1, 4: 1}, "psi1")
        raise AssertionError("missing synth/f010_psi1 not detected")
    except FileNotFoundError as e:
        assert "f010_psi1" in str(e)


def cfg_json(**kw) -> str:
    base = dict(epochs=20, batch_size=64, lr=3e-4, weight_decay=1e-4, patience=6, mixup_alpha=0.4, model="efficientnet_b0",
                pretrained=True, ra_magnitude=9, synth_variant=None, eval_split="test")
    return json.dumps({**base, **kw}, sort_keys=True)


def row(run_id, cond, frac, ratio, seed, dataset, f1=0.5, r3=0.5, r4=0.5, **cfg) -> dict:
    r = {c: 0.5 for c in ROW_COLS}
    r.update(run_id=run_id, condition=cond, condition_name="x", fraction=frac, ratio=ratio, seed=seed, dataset=dataset,
             val_macro_f1=f1, val_recall_3=r3, val_recall_4=r4, model="efficientnet_b0", config=cfg_json(**cfg),
             git_sha="0abc123", timestamp="2026-10-24T00:00:00+00:00")
    return r


def world(name: str, base_f1, var_f1, base_rec=None, var_rec=None, drop_variant=None, odd_variant=None, **common) -> Path:
    """Main CSV (+ preds) with the 6 baseline runs and unrelated runs, checks CSV with the 6 val-only variant runs.
    *_f1 / *_rec: {fraction: [seed 0, 1, 2]}; rare recall is put in recall_3 and recall_4 alike."""
    c = queue.CHECKS[name]
    res = TMP / f"res_{name}_{len(list(TMP.glob('res_*')))}" / "classifier_runs.csv"
    (res.parent / "preds").mkdir(parents=True)
    other = {"synth_variant": "" if c["condition"] == 6 else None, **common}
    main, var = [], []
    for f in queue.CHECK_FRACTIONS:
        for s in queue.SEEDS:
            br, vr = (base_rec or {}).get(f, [0.5] * 3)[s], (var_rec or {}).get(f, [0.5] * 3)[s]
            rid = run_id_of(c["condition"], f, c["ratio"], s)
            for ds in ("aptos_test", "idrid"):
                main.append(row(rid, c["condition"], f, c["ratio"], s, ds, base_f1[f][s], br, br, **{**other, c["key"]: c["baseline"]}))
            if (f, s) != drop_variant:
                vid = run_id_of(c["condition"], f, c["ratio"], s, **{c["key"]: c["variant"]}, eval_split="val")
                kw = {**other, c["key"]: c["variant"], "eval_split": "val", **({"epochs": 10} if (f, s) == odd_variant else {})}
                var.append(row(vid, c["condition"], f, c["ratio"], s, "val", var_f1[f][s], vr, vr, **kw))
    for ds in ("aptos_test", "idrid"):  # unaffected by either check, and (magnitude only) a stale c5 run at 50%
        main.append(row("c1_f100_r0_s0", 1, 1.0, 0.0, 0, ds, ra_magnitude=None))
        main.append(row("c5_f050_r0_s0", 5, 0.5, 0.0, 0, ds))
    pd.DataFrame(main, columns=ROW_COLS).to_csv(res, index=False)
    for r in main:
        (res.parent / "preds" / f"{r['run_id']}_{r['dataset']}.csv").write_text("image_id,y_pred\nx,0\n")
    (res.parent / "checks").mkdir()
    pd.DataFrame(var, columns=ROW_COLS).to_csv(res.parent / "checks" / "classifier_runs.csv", index=False)
    return res


def decision(res: Path, name: str) -> dict:
    return json.loads((res.parent / "checks" / f"decision_{name}.json").read_text(encoding="utf-8"))


BASE = {0.1: [0.50, 0.52, 0.54], 1.0: [0.60, 0.62, 0.64]}  # sd 0.02 at both fractions
PLUS = lambda d: {f: [v + d for v in vs] for f, vs in BASE.items()}  # noqa: E731


def check_magnitude_adopted():
    write_config(9, "")
    res = world("magnitude", BASE, PLUS(0.05))
    out = invoke(queue.main, ["--checks", "magnitude", "--dry-run", "--results-csv", res])
    assert "12 selected, 12 done, 0 pending" in out, out
    out = invoke(checks.main, ["magnitude", "--results-csv", res])
    assert "ADOPT variant ra_magnitude=15" in out and "not applied" in out, out
    d = decision(res, "magnitude")
    assert d["variant_adopted"] and not d["applied"] and d["rule"] == checks.RULE and len(d["runs"]) == 12, d
    assert json.loads(CONFIG.read_text()) == {"ra_magnitude": 9, "synth_variant": ""} and len(pd.read_csv(res)) == 16

    out = invoke(checks.main, ["magnitude", "--apply", "--results-csv", res])
    assert "superseded 14 rows" in out, out
    assert json.loads(CONFIG.read_text()) == {"ra_magnitude": 15, "synth_variant": ""}
    main, sup = pd.read_csv(res), pd.read_csv(res.parent / "checks" / "superseded_runs.csv")
    assert set(main.run_id) == {"c1_f100_r0_s0"} and len(main) == 2, main
    want = {run_id_of(5, f, 0.0, s) for f in (0.1, 1.0) for s in (0, 1, 2)} | {"c5_f050_r0_s0"}
    assert set(sup.run_id) == want and len(sup) == 14 and set(sup.superseded_by) == {"magnitude"}
    moved = {p.name for p in (res.parent / "checks" / "preds_superseded").iterdir()}
    assert moved == {f"{r}_{ds}.csv" for r in want for ds in ("aptos_test", "idrid")}
    assert {p.name for p in (res.parent / "preds").iterdir()} == {"c1_f100_r0_s0_aptos_test.csv", "c1_f100_r0_s0_idrid.csv"}
    d = decision(res, "magnitude")
    assert d["applied"] and d["superseded"] == sorted(want) and d["config"]["ra_magnitude"] == 15

    out = invoke(queue.main, ["--dry-run", "--condition", "5", "--fraction", "0.1", "--results-csv", res])
    assert "3 selected, 0 done, 3 pending" in out and "c5_f010_r0_s0\n" in out, out  # main queue re-runs them, plain ids
    out = invoke(queue.main, ["--checks", "magnitude", "--dry-run", "--results-csv", res], ok=False)
    assert "is decided" in out, out
    out = invoke(checks.main, ["magnitude", "--results-csv", res])  # recomputed from superseded rows, stays applied
    assert "ADOPT variant" in out and decision(res, "magnitude")["applied"], out
    out = invoke(checks.main, ["magnitude", "--apply", "--results-csv", res])  # idempotent
    assert "rows of" not in out and decision(res, "magnitude")["superseded"] == sorted(want), out
    assert len(pd.read_csv(res.parent / "checks" / "superseded_runs.csv")) == 14 and len(pd.read_csv(res)) == 2


def check_magnitude_kept():
    for label, var_f1, var_rec in (
            ("gain below sd at 10%", {0.1: [v + 0.01 for v in BASE[0.1]], 1.0: PLUS(0.05)[1.0]}, None),
            ("gain equals sd", PLUS(0.02), None),
            ("rare recall lower at 100%", PLUS(0.05), {1.0: [0.5, 0.5, 0.45]})):
        write_config(9, "")
        res = world("magnitude", BASE, var_f1, var_rec=var_rec)
        out = invoke(checks.main, ["magnitude", "--apply", "--results-csv", res])
        assert "KEEP baseline ra_magnitude=9" in out, (label, out)
        d = decision(res, "magnitude")
        assert d["applied"] and not d["variant_adopted"] and d["superseded"] == [], (label, d)
        assert json.loads(CONFIG.read_text())["ra_magnitude"] == 9 and len(pd.read_csv(res)) == 16, label
        assert not (res.parent / "checks" / "superseded_runs.csv").exists(), label


def check_errors():
    write_config(9, "")
    res = world("magnitude", BASE, PLUS(0.05), drop_variant=(1.0, 2))
    out = invoke(checks.main, ["magnitude", "--apply", "--results-csv", res], ok=False)
    assert "c5_f100_r0_s2_m15_val (0 rows)" in out and not (res.parent / "checks" / "decision_magnitude.json").exists(), out
    res = world("magnitude", BASE, PLUS(0.05), odd_variant=(0.1, 0))
    out = invoke(checks.main, ["magnitude", "--results-csv", res], ok=False)
    assert "differ in more than ra_magnitude" in out, out
    assert json.loads(CONFIG.read_text())["ra_magnitude"] == 9


def applied_magnitude(res: Path, adopted: bool) -> None:
    (res.parent / "checks" / "decision_magnitude.json").write_text(json.dumps(dict(applied=True, variant_adopted=adopted,
                                                                                  decided_at="2026-10-24T00:00:00+00:00")))


def check_order_guards():
    write_config(9, "")
    res = world("magnitude", BASE, PLUS(0.05))
    out = invoke(queue.main, ["--dry-run", "--results-csv", res])  # grid: nothing runs before the magnitude decision
    assert "wait for the pre-registered check(s): magnitude (S8a), psi (S10a)" in out and "\n  c" not in out, out
    out = invoke(queue.main, ["--checks", "psi", "--dry-run", "--results-csv", res], ok=False)
    assert "run S8a first" in out, out
    invoke(checks.main, ["magnitude", "--apply", "--results-csv", res])  # adopts 15
    out = invoke(queue.main, ["--dry-run", "--results-csv", res])
    assert "psi (S10a)" in out and "magnitude (S8a)" not in out and "\n  c5_f010_r0_s0" in out and "\n  c6_" not in out, out
    write_config(9, "")  # config reverted after an applied adoption: every queue refuses
    out = invoke(queue.main, ["--dry-run", "--results-csv", res], ok=False)
    assert "restore and commit src/classify/config.json" in out, out
    write_config(15, "")
    var = pd.read_csv(res.parent / "checks" / "classifier_runs.csv")  # rows changed after the decision: verdict flips
    var["val_macro_f1"] = 0.0
    var.to_csv(res.parent / "checks" / "classifier_runs.csv", index=False)
    sup = pd.read_csv(res.parent / "checks" / "superseded_runs.csv")
    before = decision(res, "magnitude")
    out = invoke(checks.main, ["magnitude", "--results-csv", res], ok=False)
    assert "opposite verdict" in out and decision(res, "magnitude") == before and len(sup) == 14, out
    res = world("psi", BASE, PLUS(0.05), ra_magnitude=15)  # psi decision without an applied magnitude decision
    out = invoke(checks.main, ["psi", "--results-csv", res], ok=False)
    assert "decide the magnitude check first" in out, out


def check_psi_adopted():
    write_config(15, "")  # magnitude check adopted 15 earlier: both psi arms ran with it; equal rare recall means is "not lower"
    res = world("psi", BASE, PLUS(0.05), base_rec={0.1: [0.4, 0.5, 0.6]}, var_rec={0.1: [0.5, 0.5, 0.5]}, ra_magnitude=15)
    applied_magnitude(res, adopted=True)
    out = invoke(checks.main, ["psi", "--apply", "--results-csv", res])
    assert "ADOPT variant synth_variant='psi1'" in out and "superseded 12 rows" in out, out
    assert json.loads(CONFIG.read_text()) == {"ra_magnitude": 15, "synth_variant": "psi1"}
    assert set(pd.read_csv(res).run_id) == {"c1_f100_r0_s0", "c5_f050_r0_s0"}  # psi only affects condition 6
    d = decision(res, "psi")
    assert d["applied"] and d["superseded"] == sorted(run_id_of(6, f, 1.0, s) for f in (0.1, 1.0) for s in (0, 1, 2)), d


def main() -> None:
    before = TRACKED.read_bytes()
    try:
        check_randaugment()
        check_run_ids_and_synth_dirs()
        check_magnitude_adopted()
        check_magnitude_kept()
        check_errors()
        check_order_guards()
        check_psi_adopted()
    finally:
        assert TRACKED.read_bytes() == before, "the tracked src/classify/config.json was modified"
        shutil.rmtree(TMP, ignore_errors=True)
    print("CHECKS_OK")


if __name__ == "__main__":  # all side effects here: paths are fixed when src is imported, and pytest collects nothing
    TMP = Path(tempfile.mkdtemp(prefix="genai_checks_"))
    CONFIG = TMP / "config.json"
    os.environ.update(GENAI_CLASSIFY_CONFIG=str(CONFIG), GENAI_DATA=str(TMP / "data"), GENAI_RESULTS=str(TMP / "results"),
                      GENAI_SPLITS=str(TMP / "splits"), GENAI_CHECKPOINTS=str(TMP / "ckpt"))
    sys.path.insert(0, str(ROOT))
    from src.analysis import checks
    from src.classify import queue
    from src.classify.data import BILINEAR, FundusRandAugment, build_transform, synth_files
    from src.classify.train import ROW_COLS, run_id_of
    main()
