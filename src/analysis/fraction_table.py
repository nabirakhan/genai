"""Per fraction: real training images per grade and each grade's share of the GAN's label sampling.

  python -m src.analysis.fraction_table      # needs splits/aptos_splits.csv + raw/aptos/train.csv only (S4)

stylegan3 with --cond=1 draws the label of every generated training sample uniformly over the training
set's images (training_loop.py: training_set.get_label(randint(len(training_set))); --mirror duplicates
images, not shares), and D sees real images in the same mix. So a grade's share of the f<ppp> train set
is the share of G's training samples conditioned on it: at f010 grades 3 and 4 get about 0.6% and 0.9%.

results/tables/fraction_label_shares.csv / .md: aggregate counts only, fine to commit.
"""
from pathlib import Path

import click
import pandas as pd

from src.data.split import fraction_mask, load_splits
from src.genai.paths import FRACTIONS, GRADES, RESULTS, frac_tag


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in FRACTIONS:
        n = df[fraction_mask(df, p)].grade.value_counts().reindex(GRADES, fill_value=0)
        rows.append({"fraction": p, "tag": frac_tag(p), "n_train": int(n.sum()), **{f"n_g{g}": int(n[g]) for g in GRADES},
                     **{f"share_g{g}": round(float(n[g] / n.sum()), 5) for g in GRADES}})
    return pd.DataFrame(rows)


def markdown(t: pd.DataFrame) -> str:
    lines = ["Real training images per grade (share of the GAN's conditioning labels) per rare-grade fraction.", "",
             "| fraction | train images | " + " | ".join(f"grade {g}" for g in GRADES) + " |",
             "|---|---|" + "---|" * len(GRADES)]
    for r in t.to_dict("records"):
        cells = [f"{r[f'n_g{g}']} ({r[f'share_g{g}']:.2%})" for g in GRADES]
        lines.append(f"| {r['tag']} | {r['n_train']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@click.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=RESULTS / "tables")
def main(out_dir):
    t = table(load_splits())
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "fraction_label_shares.csv", index=False)
    md = markdown(t)
    (out_dir / "fraction_label_shares.md").write_text(md, encoding="utf-8")
    click.echo(md)


if __name__ == "__main__":
    main()
