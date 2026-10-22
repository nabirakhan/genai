# Frontend design notes

Palettes chosen by Nabira (22 Oct 2026); tokens live in `src/theme.css`.

## Concept (brief from Nabira, 22 Oct 2026)

One scrollable page. Hero: a **realistic 3D eye** that opens and closes, with **glittery eyelashes**,
and a **scan line sweeping over the eyeball** (Shatter Me book-cover feel: iridescent iris, sparkle
on the lashes, a tear-drop highlight is optional). The eye **follows the mouse**. Background and
chrome use **palette B** (dark glass `#110E15` → `#411D2B`) with the neon pink accent for the scan.
Below the hero: Generate, Classify, Results as sections of the same page (no separate routes needed;
keep `src/pages/*` as section components).

### Implementation plan (week 7)

- **Stack**: `three`, `@react-three/fiber`, `@react-three/drei` (+ `@types/three`); `<Canvas>` only in
  the hero, `dpr={[1, 1.5]}`, `frameloop="always"` while the hero is on screen, `demand` otherwise.
- **Eyeball**: sphere, procedural iris in a fragment shader (radial fibres + noise, palette B tints
  with a pink/violet rim) or a 2k iris texture; cornea = slightly larger transparent sphere with
  `MeshPhysicalMaterial` (transmission, clearcoat, ior 1.4) for the wet highlight; pupil dilates with
  scroll or hover.
- **Eyelids**: upper/lower lids as partial spheres (`thetaStart/thetaLength`) scaled around the
  eyeball; blink = animate `thetaLength` with a spring (drei `useSpring` from `@react-spring/three` or
  a `useFrame` easing), random blink every 3–6 s plus a slow open on load.
- **Lashes**: `InstancedMesh` of thin curved tubes along the lid edge (60–90 per lid), with a sparse
  `Points` layer of glitter sprites (additive blending, sine-flicker in a shader, pink/gold).
- **Scan**: a thin emissive ring/plane that sweeps across the eyeball on a loop (shader with
  `uTime`), plus a faint hex/grid overlay on the iris; colour `--accent`.
- **Mouse follow**: eyeball group `lookAt` a target lerped from pointer NDC → world (clamped to
  ±25°), lids follow slightly. On touch devices: gentle idle sway instead.
- **Performance budget**: < 150k triangles, one shadow-free directional + one point light, bloom via
  `@react-three/postprocessing` only if FPS stays > 50 on an integrated GPU; reduce-motion media
  query disables the blink loop and scan.
- **Fallback**: static poster render of the eye for `prefers-reduced-motion` and WebGL failure.

## Palette A: wine gradient (brand, hero, gradients)

| hex | role |
|---|---|
| `#46000D` | deepest wine, gradient start |
| `#5E0009` | wine red |
| `#720137` | magenta, brand mid-tone |
| `#590054` | purple |
| `#42002E` | plum, gradient end |

Use as a sweep (`--gradient-wine`), for card headers and the Generate page hero. These tones are
all low-luminance, so they are **backgrounds/fills, not text colours**.

## Palette B: dark glass (app surfaces)

| hex | role |
|---|---|
| `#110E15` | page background `--bg` |
| `#211720` | elevated panels `--bg-elevated` |
| `#2A0A11` | cards `--surface` |
| `#411D2B` | hover / secondary cards `--surface-2` |
| `#430E18` | borders, dividers with tint `--surface-3` |

The reference image's neon pink edge lines become `--accent #FF2E6E` (buttons, links, focus
rings, active grade). Keep it sparse: one accent per view.

## Contrast (WCAG AA = 4.5:1 for text)

- `--text #F6EBF0` on `--bg`: ~16:1. On `--surface-2`: ~11:1.
- `--text-muted #C9A9B6` on `--bg`: ~9:1.
- `--accent #FF2E6E` on `--bg`: ~6.3:1 (fine for buttons/links); on `--surface-2` ~4.6:1 (borderline, use bold or larger text).
- Wine tones (`#5E0009`, `#720137`) on `--bg`: < 2:1. Never use for text or icons.

## Fundus imagery

Retinal photos are orange-red; on this palette they sit naturally. Put them on `--bg-elevated` with
a 1 px `--border` and no saturated frames so the SYNTHETIC label (yellow on black, drawn into the
image by the API) stays the most visible element. Grad-CAM overlays use jet; keep UI chrome around
them neutral.

## Grade colours (charts, probability bars)

`--grade-0` green → `--grade-4` pink/red, so severity reads at a glance and still matches the palette.
