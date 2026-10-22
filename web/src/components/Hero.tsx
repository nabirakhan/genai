import { Suspense, lazy } from "react";

const EyeScene = lazy(() => import("./Eye"));

export default function Hero() {
  return (
    <section className="hero" aria-label="Introduction">
      <div className="hero__canvas" aria-hidden="true">
        <Suspense fallback={null}>
          <EyeScene />
        </Suspense>
      </div>
      <div className="hero__copy">
        <p className="eyebrow">Art inspired by Shatter Me</p>
        <h1 className="hero__title">
          When does a <em>GAN</em> help?
        </h1>
        <p className="hero__lede">
          Severe diabetic retinopathy is rare in training data, so classifiers miss the eyes that need urgent referral.
          This project pits GAN-generated retinal images against cheap fixes, and maps when the generator is worth it.
        </p>
        <div className="hero__actions">
          <a className="btn" href="#generate">
            Generate an image
          </a>
          <a className="btn btn--ghost" href="#results">
            See the results
          </a>
        </div>
        <ul className="hero__facts" aria-label="Setup">
          <li>
            <strong>StyleGAN2-ADA</strong> class-conditional, 256 px
          </li>
          <li>
            <strong>84</strong> classifier runs, 6 conditions
          </li>
          <li>
            <strong>IDRiD</strong> external test, never trained on
          </li>
        </ul>
      </div>
      <div className="hero__scroll" aria-hidden="true">
        scroll
      </div>
    </section>
  );
}
