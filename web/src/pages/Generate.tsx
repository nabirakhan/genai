import { useState } from "react";
import { api, b64png } from "../api";
import type { GenerateResponse, Grade } from "../types";

const GRADES: { v: Grade; label: string }[] = [
  { v: 0, label: "0 · no DR" },
  { v: 1, label: "1 · mild" },
  { v: 2, label: "2 · moderate" },
  { v: 3, label: "3 · severe" },
  { v: 4, label: "4 · proliferative" },
];

export default function Generate() {
  const [grade, setGrade] = useState<Grade>(3);
  const [seed, setSeed] = useState(0);
  const [psi, setPsi] = useState(0.8);
  const [res, setRes] = useState<GenerateResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const go = async () => {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.generate({ grade, seed, psi }));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="section" id="generate">
      <div className="section__head">
        <div>
          <span className="tag">01 · Generate</span>
          <h2>Ask the generator for a grade</h2>
          <p>
            One class-conditional StyleGAN2-ADA, fine-tuned on all five grades, so the rare severe grades borrow retinal
            structure from the common ones. Every output is labelled SYNTHETIC and shown next to its nearest real
            training image.
          </p>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="form-row">
            <label className="field">
              grade
              <select value={grade} onChange={(e) => setGrade(Number(e.target.value) as Grade)}>
                {GRADES.map((g) => (
                  <option key={g.v} value={g.v}>
                    {g.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              seed
              <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label className="field">
              truncation ψ
              <input type="number" step="0.05" min="0.3" max="1" value={psi} onChange={(e) => setPsi(Number(e.target.value))} />
            </label>
            <button className="btn" onClick={go} disabled={busy}>
              {busy ? "generating…" : "generate"}
            </button>
          </div>
          <p className="muted" style={{ marginTop: 14 }}>
            Lower ψ gives safer, more average images; 1.0 gives the most variety. Seeds are deterministic.
          </p>
          {err && (
            <p className="alert" role="alert">
              {err}
            </p>
          )}
        </div>

        <div className="card">
          <div className="grid-2">
            <figure style={{ margin: 0 }}>
              <div className={`img-frame${busy ? " scanning" : ""}`}>
                {res ? (
                  <>
                    <img src={b64png(res.image_png_b64)} alt={`synthetic fundus, grade ${res.grade}`} />
                    <span className="img-frame__label">synthetic</span>
                  </>
                ) : (
                  <div className="img-placeholder">{busy ? "rendering on CPU…" : "synthetic image"}</div>
                )}
              </div>
              <figcaption className="muted" style={{ marginTop: 8 }}>
                {res ? `grade ${res.grade} · seed ${res.seed} · ψ ${res.psi} · ${res.seconds}s` : "nothing generated yet"}
              </figcaption>
            </figure>
            <figure style={{ margin: 0 }}>
              <div className="img-frame">
                {res?.nearest_real?.thumb_png_b64 ? (
                  <>
                    <img src={b64png(res.nearest_real.thumb_png_b64)} alt="nearest real training image (thumbnail)" />
                    <span className="img-frame__label">nearest real</span>
                  </>
                ) : (
                  <div className="img-placeholder">nearest real image</div>
                )}
              </div>
              <figcaption className="muted" style={{ marginTop: 8 }}>
                {res?.nearest_real
                  ? `${res.nearest_real.image_id} · cosine distance ${res.nearest_real.distance.toFixed(3)}`
                  : "memorisation check: distance to the closest real image"}
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </section>
  );
}
