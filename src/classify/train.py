"""Train one classifier run and evaluate it on APTOS test + IDRiD.

  python -m src.classify.train --condition 5 --fraction 1.0 --seed 0
  python -m src.classify.train --condition 6 --fraction 0.1 --ratio 2 --seed 1

Resumable: checkpoints/{run_id}/last.pt is written every epoch and picked up automatically.
Idempotent: if results/classifier_runs.csv already has this run_id, exits unless --force.
Appends one row per eval dataset to results/classifier_runs.csv and writes per-image
predictions to results/preds/{run_id}_{dataset}.csv (used for bootstrap CIs).
"""
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from filelock import FileLock

from src.classify.data import CONDITIONS, NUM_CLASSES, make_loaders
from src.classify.metrics import METRIC_COLS, compute_metrics
from src.classify.model import DEFAULT_MODEL, create_model
from src.genai.paths import CHECKPOINTS, FRACTIONS, RESULTS, frac_tag

RESULTS_CSV = RESULTS / "classifier_runs.csv"
ROW_COLS = ["run_id", "condition", "condition_name", "fraction", "ratio", "seed", "dataset", *METRIC_COLS,
            "n_train", "n_synth", "best_epoch", "epochs_run", "best_val_qwk", "train_time_s", "model", "config", "git_sha", "timestamp"]
KEEP_WEIGHTS = {"c5_f100_r0_s0", "c6_f100_r1_s0"}  # best.pt needed by gradcam (S14) and api.export_assets (S15)


def run_id_of(condition: int, fraction: float, ratio: float, seed: int) -> str:
    return f"c{condition}_{frac_tag(fraction)}_r{ratio:g}_s{seed}"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(gen: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "loader": gen.get_state()}


def set_rng_state(s: dict, gen: torch.Generator) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])
    gen.set_state(s["loader"].cpu())


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def mixup(x: torch.Tensor, y_onehot: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


@torch.no_grad()
def predict(model, loader, device, use_amp: bool):
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        p = logits.float().softmax(dim=1)
        ys.append(y.numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def already_done(results_csv: Path, run_id: str) -> bool:
    if not results_csv.exists():
        return False
    df = pd.read_csv(results_csv, usecols=["run_id", "dataset"])
    return ((df.run_id == run_id) & (df.dataset == "aptos_test")).any()


def append_rows(results_csv: Path, rows: list[dict]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(results_csv) + ".lock"):
        new = pd.DataFrame(rows, columns=ROW_COLS)
        if results_csv.exists():
            old = pd.read_csv(results_csv, dtype={"git_sha": str})
            keys = set(zip(new.run_id, new.dataset))  # --force re-run replaces its old rows
            old = old[[k not in keys for k in zip(old.run_id, old.dataset)]]
            new = pd.concat([old, new], ignore_index=True)
        tmp = results_csv.with_name(results_csv.name + ".tmp")
        new.to_csv(tmp, index=False)
        os.replace(tmp, results_csv)


@click.command()
@click.option("--condition", type=click.IntRange(1, 6), required=True)
@click.option("--fraction", type=float, required=True, help="rare-grade fraction: 0.1, 0.25, 0.5, 1.0")
@click.option("--ratio", type=float, default=None, help="synthetic ratio for condition 6 (default 1.0); 0 otherwise")
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=20)
@click.option("--batch-size", type=int, default=64)
@click.option("--workers", type=int, default=6)
@click.option("--lr", type=float, default=3e-4)
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--patience", type=int, default=6, help="early-stop patience on val QWK")
@click.option("--mixup-alpha", type=float, default=0.4)
@click.option("--model-name", default=DEFAULT_MODEL)
@click.option("--no-pretrained", is_flag=True, help="random init (smoke tests only)")
@click.option("--keep-weights", is_flag=True, help="keep checkpoints/{run_id}/best.pt after the run")
@click.option("--force", is_flag=True, help="re-run even if results exist")
@click.option("--allow-missing-idrid", is_flag=True, help="run without the IDRiD external test (the run then never gets an idrid row)")
@click.option("--allow-cpu", is_flag=True, help="allow training without CUDA (smoke tests only)")
@click.option("--results-csv", type=click.Path(path_type=Path), default=RESULTS_CSV)
def main(condition, fraction, ratio, seed, epochs, batch_size, workers, lr, weight_decay, patience, mixup_alpha,
         model_name, no_pretrained, keep_weights, force, allow_missing_idrid, allow_cpu, results_csv):
    assert any(abs(fraction - f) < 1e-9 for f in FRACTIONS), f"fraction must be one of {FRACTIONS}"
    if ratio is None:
        ratio = 1.0 if condition == 6 else 0.0
    if condition != 6:
        ratio = 0.0
    run_id = run_id_of(condition, fraction, ratio, seed)
    cfg = dict(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience,
               mixup_alpha=mixup_alpha, model=model_name, pretrained=not no_pretrained)
    if already_done(results_csv, run_id) and not force:
        click.echo(f"{run_id}: already in {results_csv.name}, skipping (use --force to redo)")
        return

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise click.ClickException("CUDA not available (driver / torch wheel, see S0): refusing a CPU run; pass --allow-cpu for smoke tests")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    t0 = time.time()

    loaders, info = make_loaders(condition, fraction, ratio, batch_size, workers, seed)
    if "idrid" not in loaders and not allow_missing_idrid:
        raise click.ClickException("IDRiD cache/labels missing (S2, S3, S5): the run would be marked done without "
                                   "its external-test row and its weights deleted; pass --allow-missing-idrid to override")
    click.echo(f"{run_id}: device={device} train={info['n_train']} synth={info['n_synth']} counts={info['counts']}")

    model = create_model(model_name, pretrained=not no_pretrained).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(loaders["train"]))
    total = epochs * steps_per_epoch
    warm = steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    class_weights = info["class_weights"].to(device) if info["class_weights"] is not None else None

    ckpt_dir = CHECKPOINTS / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    start_epoch, best_qwk, best_epoch, bad, best_state, best_val = 0, -1.0, -1, 0, None, None
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if "cfg" in ck and ck["cfg"] != cfg:
            raise click.ClickException(f"{last_path} was written with {ck['cfg']}, now {cfg}: rerun with the same options or delete it")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_qwk, best_epoch, bad = ck["epoch"] + 1, ck["best_qwk"], ck["best_epoch"], ck["bad"]
        if "best_model" in ck:
            best_state, best_val = ck["best_model"], ck["best_val"]
        else:  # last.pt written before this change keeps its best weights in best.pt
            b = torch.load(best_path, map_location=device, weights_only=False)
            best_state, best_val = b["model"], b["val"]
        if "rng" in ck:  # absent from a last.pt written before this change
            set_rng_state(ck["rng"], loaders["train"].generator)
        if bad >= patience:
            epochs = start_epoch  # it had already early-stopped; only the final evaluation was lost
        click.echo(f"resumed from epoch {start_epoch} (best val QWK {best_qwk:.4f} @ {best_epoch})")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        model.train()
        running, n_b = 0.0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            soft = condition >= 5
            if soft:
                y_soft = F.one_hot(y, NUM_CLASSES).float()
                if mixup_alpha > 0:
                    x, y_soft = mixup(x, y_soft, mixup_alpha)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = soft_cross_entropy(logits, y_soft) if soft else F.cross_entropy(logits.float(), y, weight=class_weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_b += 1
        y_true, y_pred, _ = predict(model, loaders["val"], device, use_amp)
        val = compute_metrics(y_true, y_pred)
        improved = val["qwk"] > best_qwk
        if improved:
            best_qwk, best_epoch, bad = val["qwk"], epoch, 0
            best_state, best_val = {k: v.detach().clone() for k, v in model.state_dict().items()}, val
        else:
            bad += 1
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_qwk": best_qwk, "best_epoch": best_epoch,
                     "bad": bad, "best_model": best_state, "best_val": best_val,
                     "rng": rng_state(loaders["train"].generator), "cfg": cfg}, last_path)
        click.echo(f"ep {epoch:02d} loss {running / max(1, n_b):.4f} val qwk {val['qwk']:.4f} f1 {val['macro_f1']:.4f} "
                   f"r3 {val['recall_3']:.2f} r4 {val['recall_4']:.2f} {'*' if improved else ''}")
        if bad >= patience:
            click.echo("early stop")
            break

    model.load_state_dict(best_state)
    train_time = time.time() - t0
    rows = []
    for ds_name in ("aptos_test", "idrid"):
        if ds_name not in loaders:
            click.echo(f"WARNING: {ds_name} not available, skipping")
            continue
        y_true, y_pred, probs = predict(model, loaders[ds_name], device, use_amp)
        m = compute_metrics(y_true, y_pred)
        rows.append({"run_id": run_id, "condition": condition, "condition_name": CONDITIONS[condition],
                     "fraction": fraction, "ratio": ratio, "seed": seed, "dataset": ds_name, **m,
                     "n_train": info["n_train"], "n_synth": info["n_synth"], "best_epoch": best_epoch,
                     "epochs_run": epoch + 1, "best_val_qwk": best_qwk, "train_time_s": round(train_time, 1),
                     "model": model_name, "config": json.dumps(cfg, sort_keys=True), "git_sha": git_sha(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        preds = pd.DataFrame({"image_id": info["ids"][ds_name], "y_pred": y_pred})  # labels joined at analysis time
        for k in range(NUM_CLASSES):
            preds[f"p{k}"] = np.round(probs[:, k], 4)
        (results_csv.parent / "preds").mkdir(parents=True, exist_ok=True)
        preds.to_csv(results_csv.parent / "preds" / f"{run_id}_{ds_name}.csv", index=False)
        click.echo(f"{ds_name}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    append_rows(results_csv, rows)

    last_path.unlink(missing_ok=True)
    if keep_weights or run_id in KEEP_WEIGHTS:
        atomic_save({"model": best_state, "epoch": best_epoch, "val": best_val, "run_id": run_id,
                     "model_name": model_name}, best_path)
        click.echo(f"kept {best_path}")
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    click.echo(f"{run_id} done in {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
