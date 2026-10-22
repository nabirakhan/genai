# Figure and table manifest

| id | file | produced by |
|---|---|---|
| Table data | `splits/aptos_splits.csv` counts (crosstab printed by `python -m src.data.split`) | `src/data/split.py` |
| Table main | `results/tables/main_{aptos_test,idrid}_{macro_f1,qwk,recall_3,recall_4}.md` | `src/analysis/aggregate.py` |
| Table delta | `results/tables/delta_6_vs_5_{dataset}.csv`, CIs in `bootstrap_delta.csv` | `aggregate.py`, `bootstrap.py` |
| Table ratio | `results/tables/ratio_{dataset}.csv` | `aggregate.py` |
| Table GAN | `results/gan_metrics.csv` (FID/KID overall + per class; ADA vs noaug at f010) | Kaggle eval notebook, `src/gan/metrics_per_class.py` |
| Fig samples | `samples_{tag}.png` from the Kaggle eval notebook output | `notebooks/kaggle_gan_eval.ipynb` |
| Fig learning curves | `results/figures/learning_curve_{dataset}_{metric}.png` | `src/analysis/plots.py` |
| Fig ratio map | `results/figures/ratio_map_{dataset}.png` | `plots.py` |
| Fig recall bars | `results/figures/recall_bars_{dataset}_f{010,100}.png` | `plots.py` |
| Fig GAN KID | `results/figures/gan_kid.png` | `plots.py` |
| Fig memorisation | `results/figures/memorization_{tag}.png`, table `memorization_summary.csv` | `src/analysis/memorization.py` |
| Fig RPS | `results/figures/rps_hist.png`, tables `rps_summary.csv`, `rps_ks.csv` | `src/analysis/rps.py` |
| Fig Grad-CAM | `results/figures/gradcam_examples.png` | `src/analysis/gradcam.py` |
| Fig training curves (optional) | `fakes*.png`, `stats.jsonl` in the Kaggle train outputs | stylegan3 |
