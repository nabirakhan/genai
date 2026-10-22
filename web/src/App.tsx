import Hero from "./components/Hero";
import Generate from "./pages/Generate";
import Classify from "./pages/Classify";
import Results from "./pages/Results";

const NAV = [
  { href: "#generate", label: "Generate" },
  { href: "#classify", label: "Classify" },
  { href: "#results", label: "Results" },
];

export default function App() {
  return (
    <>
      <header className="nav">
        <a className="nav__brand" href="#top">
          <span className="nav__dot" aria-hidden="true" />
          When does a GAN help?
        </a>
        <nav className="nav__links" aria-label="Sections">
          {NAV.map((n) => (
            <a key={n.href} href={n.href}>
              {n.label}
            </a>
          ))}
        </nav>
      </header>
      <main id="top">
        <Hero />
        <Generate />
        <Classify />
        <Results />
      </main>
      <footer className="footer">
        <p>
          AI4009 Generative AI, FAST-NUCES Karachi, Fall 2026. Nabira Khan. Generated images are experimental,
          labelled SYNTHETIC and not for clinical use. APTOS 2019 and IDRiD are used under their own terms.
        </p>
      </footer>
    </>
  );
}
