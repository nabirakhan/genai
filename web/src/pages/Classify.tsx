import { useState } from "react";
import { api, b64png } from "../api";
import type { ClassifyResponse, ModelKey } from "../types";

const MODELS: { key: ModelKey; label: string; note: string }[] = [
  { key: "baseline", label: "Strong augmentation", note: "condition 5 · RandAugment + mixup" },
  { key: "gan", label: "Strong augmentation + GAN", note: "condition 6 · + synthetic grades 3/4 at 1×" },
];
const GRADE_VARS = ["--grade-0", "--grade-1", "--grade-2", "--grade-3", "--grade-4"];

export default function Classify() {
  const [res, setRes] = useState<ClassifyResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setName(file.name);
    try {
      setRes(await api.classify(file));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="section" id="classify">
      <div className="section__head">
        <div>
          <span className="tag">02 · Classify</span>
          <h2>Two classifiers, one fundus photo</h2>
          <p>
            The same EfficientNet-B0 trained with the strongest cheap fix, with and without GAN images. Grad-CAM shows
            where each model looks. Research demo, not a diagnosis.
          </p>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <label className="dropzone">
            <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} disabled={busy} />
            <strong>{busy ? "running both models…" : "Drop or choose a fundus image"}</strong>
            <span>PNG or JPEG, up to 5 MB. It is cropped and resized to 256 px exactly like the training data.</span>
            {name && <span className="muted">{name}</span>}
          </label>
          {err && (
            <p className="alert" role="alert">
              {err}
            </p>
          )}
        </div>
        <div className="card">
          <h3>Preprocessed input</h3>
          <div className={`img-frame${busy ? " scanning" : ""}`} style={{ maxWidth: 260 }}>
            {res ? <img src={b64png(res.preprocessed_png_b64)} alt="preprocessed fundus" /> : <div className="img-placeholder">256 × 256 crop</div>}
          </div>
        </div>
      </div>

      {res && (
        <div className="grid-2" style={{ marginTop: 20 }}>
          {MODELS.map(({ key, label, note }) => {
            const probs = res.probs[key];
            const top = res.pred[key];
            return (
              <article className="card" key={key}>
                <h3>{label}</h3>
                <p className="muted">{note}</p>
                <div className="grid-2" style={{ alignItems: "start" }}>
                  <div className="img-frame">
                    <img src={b64png(res.gradcam_png_b64[key])} alt={`Grad-CAM for ${label}`} />
                    <span className="img-frame__label">grad-cam</span>
                  </div>
                  <div>
                    <p style={{ margin: "0 0 4px" }}>
                      prediction <strong style={{ color: `var(${GRADE_VARS[top]})` }}>{res.grade_names[top]}</strong>
                    </p>
                    <ul className="bars">
                      {probs.map((p, i) => (
                        <li className={`bar${i === top ? " bar--top" : ""}`} key={i}>
                          <span>{res.grade_names[i]}</span>
                          <span className="bar__track">
                            <span className="bar__fill" style={{ transform: `scaleX(${p.toFixed(3)})`, background: `var(${GRADE_VARS[i]})` }} />
                          </span>
                          <span className="bar__val">{(p * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
