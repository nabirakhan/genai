import { useEffect, useState } from "react";
import { api } from "../api";
import type { ResultsTables } from "../types";

const FIGURES: { file: string; caption: string }[] = [
  { file: "learning_curve_aptos_test_macro_f1.png", caption: "Macro F1 vs real rare-grade fraction, APTOS test" },
  { file: "learning_curve_idrid_macro_f1.png", caption: "Macro F1 vs fraction, IDRiD (external)" },
  { file: "learning_curve_aptos_test_recall_3.png", caption: "Recall for grade 3, APTOS test" },
  { file: "learning_curve_aptos_test_recall_4.png", caption: "Recall for grade 4, APTOS test" },
  { file: "ratio_map_aptos_test.png", caption: "GAN minus strong augmentation, by fraction and synthetic ratio" },
  { file: "ratio_map_idrid.png", caption: "Same map on IDRiD" },
];
const METRICS = ["macro_f1", "qwk", "recall_3", "recall_4"] as const;
const METRIC_LABEL: Record<(typeof METRICS)[number], string> = {
  macro_f1: "macro F1",
  qwk: "quadratic weighted kappa",
  recall_3: "recall · grade 3",
  recall_4: "recall · grade 4",
};

export default function Results() {
  const [t, setT] = useState<ResultsTables | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dataset, setDataset] = useState("aptos_test");
  const [metric, setMetric] = useState<(typeof METRICS)[number]>("macro_f1");
  useEffect(() => {
    api.results().then(setT).catch((e) => setErr(String(e)));
  }, []);

  const table = t?.main[dataset]?.[metric];
  const datasets = t ? Object.keys(t.main) : [];
  const fracs = table ? Object.keys(Object.values(table)[0] ?? {}) : [];

  return (
    <section className="section" id="results">
      <div className="section__head">
        <div>
          <span className="tag">03 · Results</span>
          <h2>When does the GAN help?</h2>
          <p>
            Six training conditions × four amounts of real rare-grade data × three seeds, scored on the APTOS test split
            and on IDRiD without retraining. Mean ± standard deviation over seeds.
          </p>
        </div>
        {t && <span className="muted">{t.n_runs} runs · updated {new Date(t.generated_at).toLocaleDateString()}</span>}
      </div>

      {err && (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            Results are not exported yet. They appear here once the lab runs finish and{" "}
            <code>src/analysis/aggregate.py</code> and <code>plots.py</code> have written <code>public/results/</code>.
          </p>
        </div>
      )}

      {t && (
        <div className="card">
          <div className="form-row" style={{ marginBottom: 16 }}>
            <label className="field">
              dataset
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              metric
              <select value={metric} onChange={(e) => setMetric(e.target.value as (typeof METRICS)[number])}>
                {METRICS.map((m) => (
                  <option key={m} value={m}>
                    {METRIC_LABEL[m]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>condition</th>
                  {fracs.map((f) => (
                    <th key={f}>{f} real grade-3/4</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table &&
                  Object.entries(table).map(([cond, cells]) => (
                    <tr key={cond}>
                      <td>{cond}</td>
                      {fracs.map((f) => (
                        <td key={f}>{cells[f] ?? "—"}</td>
                      ))}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="figures" style={{ marginTop: 20 }}>
        {FIGURES.map((f) => (
          <figure key={f.file}>
            <img src={`/results/${f.file}`} alt={f.caption} loading="lazy" onError={(e) => ((e.currentTarget.parentElement as HTMLElement).style.display = "none")} />
            <figcaption>{f.caption}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
