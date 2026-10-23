import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Sparkles } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

/* Shatter-Me style eye. An almond aperture cut into a painted face plane, a large iridescent
   eyeball (with branching blood vessels) behind it that follows the cursor. The camera looks
   straight at the eye and the frame is lens-shifted to place it on the right, so the sclera
   fills both corners. The inner (right) corner is a small rounded notch where the tears drain.
   Every lash on both lids IS a branch: a gnarled axis with side branches, forks and twiggy
   clumps, radiating outward from the lid line. Five big limbs rise from the upper lid. Leaves
   grow along the twigs, with sparse crystals and a few flowers at the ends. A handful of
   crystallised tears, each a smooth water body with its own neck/bulb profile, hang from the
   lower lid. Blink = the aperture closes and the lashes bend. */

const W = 1.25; // half width of the eye aperture (world units)
const Z_EYE_R = 1.55; // eyeball radius: big enough that the aperture corners never look past it
const Z_FACE = Z_EYE_R + 0.15;
const Z_LASH = Z_FACE + 0.07;
const IRIS_PROJ = 1.4 * Math.sin(0.34); // projected iris radius kept from the earlier tuning
const IRIS_T = Math.asin(IRIS_PROJ / Z_EYE_R); // polar angle of the iris edge on this sphere
const ANG_SCALE = 0.34 / IRIS_T; // the shader keeps its 0.34-based thresholds
const CANTHUS_R = 0.055; // rounded inner-corner notch, in units of W
const CAM_Z = 9;
const CAM_FOV = 18.2;

// rounded cap at the inner corner (nx near +1); 0 elsewhere
const innerCap = (nx: number) => {
  const c = 1 - CANTHUS_R;
  if (nx <= c) return 0;
  const u = (nx - c) / CANTHUS_R;
  return CANTHUS_R * W * Math.sqrt(Math.max(0, 1 - u * u));
};
const upperLid = (x: number) => {
  const nx = THREE.MathUtils.clamp(x / W, -1, 1);
  const b = 1 - nx * nx;
  return Math.max(0.5 * Math.pow(b, 0.75) + 0.12 * nx * b, innerCap(nx));
};
const lowerLid = (x: number) => {
  const nx = THREE.MathUtils.clamp(x / W, -1, 1);
  const b = 1 - nx * nx;
  return Math.min(-(0.34 * Math.pow(b, 1.1) - 0.05 * nx * b), -innerCap(nx) * 0.75);
};
// outward normal of a lid curve, as an angle in the (sin a, cos a) convention (0 = straight up)
const lidNormal = (lid: (x: number) => number, x: number, down: boolean) => {
  const h = 0.01;
  const dy = (lid(Math.min(W, x + h)) - lid(Math.max(-W, x - h))) / (2 * h);
  return down ? Math.atan2(dy, -1) : Math.atan2(-dy, 1);
};

/* ---------- shaders ---------------------------------------------------------------------- */

const NOISE = /* glsl */ `
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float hashp(vec2 i, float per) { i.x = mod(i.x, per); return hash(i); }
float pnoise(vec2 p, float per) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hashp(i, per), hashp(i + vec2(1.0, 0.0), per), f.x),
             mix(hashp(i + vec2(0.0, 1.0), per), hashp(i + vec2(1.0, 1.0), per), f.x), f.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) { float v = 0.0, a = 0.5; for (int k = 0; k < 4; k++) { v += a * noise(p); p *= 2.03; a *= 0.5; } return v; }
`;

const LID_GLSL = /* glsl */ `
const float W = ${W.toFixed(3)};
const float CR = ${CANTHUS_R.toFixed(3)};
float innerCap(float nx) {
  float c = 1.0 - CR;
  if (nx <= c) return 0.0;
  float u = (nx - c) / CR;
  return CR * W * sqrt(max(0.0, 1.0 - u * u));
}
float upperLid(float x) { float nx = clamp(x / W, -1.0, 1.0); float b = 1.0 - nx * nx; return max(0.5 * pow(b, 0.75) + 0.12 * nx * b, innerCap(nx)); }
float lowerLid(float x) { float nx = clamp(x / W, -1.0, 1.0); float b = 1.0 - nx * nx; return min(-(0.34 * pow(b, 1.1) - 0.05 * nx * b), -innerCap(nx) * 0.75); }
`;

const EYE_VERT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vPos;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const EYE_FRAG = /* glsl */ `
uniform float uTime;
uniform float uPupil;
uniform float uAngScale;
uniform vec3 uIrisA;
uniform vec3 uIrisB;
uniform vec3 uIrisC;
varying vec3 vNormal;
varying vec3 vPos;
${NOISE}
// one wiggly vessel centred on angle t0 (0..1 around the eye), present for ang > start,
// thickening toward the corners; returns 0..1 intensity
float vein(float t, float ang, float t0, float amp, float freq, float ph, float w0, float start) {
  float tc = t0 + amp * sin(ang * freq + ph) + amp * 0.35 * sin(ang * freq * 2.7 + ph * 1.9);
  float dt = abs(fract(t - tc + 0.5) - 0.5);
  float arc = dt * 6.2831853 * sin(ang);
  float w = w0 * (0.4 + smoothstep(start, start + 0.6, ang));
  return exp(-pow(arc / w, 2.0)) * smoothstep(start - 0.04, start + 0.12, ang);
}
void main() {
  vec3 p = normalize(vPos);
  float ang = acos(clamp(p.z, -1.0, 1.0)) * uAngScale;
  float t = atan(p.y, p.x) / 6.2831853 + 0.5;   // 0.5 = inner corner (+x), 0/1 = outer corner (-x)
  float r = ang / 0.34;

  vec3 sclera = vec3(0.96, 0.94, 0.95);
  // branching blood vessels: run in from the corners, wiggle, thin out and vanish near the iris
  float v = 0.0;
  for (int i = 0; i < 10; i++) {
    float fi = float(i);
    float side = i < 5 ? 0.5 : 0.0;
    float t0 = side + (hash(vec2(fi, 1.7)) - 0.5) * 0.24;
    float amp = 0.008 + 0.022 * hash(vec2(fi, 2.3));
    float freq = 3.0 + 7.0 * hash(vec2(fi, 3.1));
    float ph = 6.2831853 * hash(vec2(fi, 4.7));
    float w0 = 0.0035 + 0.0045 * hash(vec2(fi, 5.9));
    float start = 0.36 + 0.16 * hash(vec2(fi, 6.1));
    float strength = 0.45 + 0.55 * hash(vec2(fi, 7.3));
    v += vein(t, ang, t0, amp, freq, ph, w0, start) * strength;
    v += vein(t, ang, t0 + 0.035 * (hash(vec2(fi, 8.2)) - 0.5), amp * 1.4, freq * 1.3, ph + 1.3, w0 * 0.55, start + 0.18) * strength * 0.8;
    v += vein(t, ang, t0 - 0.03 * hash(vec2(fi, 9.4)), amp * 0.8, freq * 1.7, ph + 2.9, w0 * 0.4, start + 0.3) * strength * 0.6;
  }
  float capillary = smoothstep(0.84, 1.0, pnoise(vec2(t * 23.0, ang * 30.0), 23.0)) * smoothstep(0.5, 1.3, ang) * 0.25;
  v = clamp(v + capillary, 0.0, 1.0);
  sclera = mix(sclera, vec3(0.72, 0.14, 0.26), v * 0.6);
  // the inner corner (+x): sclera turns fleshy pink and sits in shadow near the tear duct
  float innerD = abs(t - 0.5);
  float innerCorner = exp(-pow(innerD / 0.11, 2.0)) * smoothstep(0.5, 1.05, ang);
  sclera = mix(sclera, vec3(0.88, 0.58, 0.62), innerCorner * 0.6);
  sclera *= 1.0 - 0.28 * innerCorner;
  sclera *= 1.0 - 0.2 * smoothstep(0.7, 1.5, ang);

  float fib = pnoise(vec2(t * 48.0, r * 5.0), 48.0) * 0.6 + pnoise(vec2(t * 140.0, r * 2.5 + uTime * 0.02), 140.0) * 0.4;
  vec3 iris = mix(uIrisA, uIrisB, smoothstep(0.05, 0.95, r));
  iris = mix(iris, uIrisC, fib * 0.8);
  iris *= 0.45 + 0.9 * fib;
  vec2 cell = vec2(t * 170.0, r * 46.0);
  vec2 cf = fract(cell) - 0.5;
  float fleck = step(0.975, hash(floor(cell))) * smoothstep(0.42, 0.1, length(cf)) * smoothstep(0.25, 0.9, r);
  iris += vec3(1.0, 0.92, 0.97) * fleck * 0.8;
  iris = mix(iris, vec3(0.08, 0.0, 0.06), smoothstep(0.82, 1.0, r));
  float pupil = smoothstep(uPupil, uPupil + 0.07, r);
  vec3 col = mix(vec3(0.0), iris, pupil);
  col = mix(col, sclera, smoothstep(0.97, 1.06, r));

  float diff = 0.5 + 0.5 * max(dot(normalize(vNormal), normalize(vec3(0.4, 0.8, 1.0))), 0.0);
  col *= diff;
  // far outside the aperture the eyeball must disappear into the painted face, not glow through it
  col = mix(col, vec3(0.1, 0.025, 0.05), smoothstep(1.08, 1.35, ang));
  gl_FragColor = vec4(col, 1.0);
}`;

const FACE_VERT = /* glsl */ `
varying vec2 vP;
void main() {
  vP = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FACE_FRAG = /* glsl */ `
uniform float uOpen;
uniform float uTime;
varying vec2 vP;
${NOISE}
${LID_GLSL}
void main() {
  vec2 p = vP;
  float up = upperLid(p.x) * uOpen;
  float lo = lowerLid(p.x) * uOpen;
  float edge = min(up - p.y, p.y - lo);
  if (abs(p.x) > W) edge = -1.0;
  float inside = smoothstep(-0.006, 0.006, edge);

  float n = fbm(p * 1.4 + vec2(0.0, uTime * 0.012));
  vec3 skin = mix(vec3(0.15, 0.035, 0.06), vec3(0.34, 0.1, 0.21), n);
  skin = mix(skin, vec3(0.24, 0.0, 0.17), smoothstep(0.55, 0.9, fbm(p * 2.6 + 7.0)) * 0.55);
  skin = mix(skin, vec3(0.42, 0.14, 0.28), smoothstep(0.6, 0.95, fbm(p * 2.1 - 3.0)) * 0.45);
  vec2 g = floor(p * 150.0);
  float h = hash(g);
  float sp = step(0.99, h) * (0.5 + 0.5 * sin(uTime * 3.0 + h * 40.0));
  float near = 1.0 - smoothstep(0.0, 0.9, -edge);
  skin += sp * vec3(1.0, 0.78, 0.86) * (0.25 + 0.6 * near);

  // liner along the lid line: black, turning fleshy pink into the inner corner
  float liner = 1.0 - smoothstep(0.0, 0.1, -edge);
  float innerX = smoothstep(0.55, 0.98, p.x / W);
  vec3 linerCol = mix(vec3(0.02, 0.0, 0.015), vec3(0.6, 0.2, 0.28), innerX);
  skin = mix(skin, linerCol, liner * 0.92);
  float crease = exp(-pow((p.y - (up + 0.24)) / 0.05, 2.0)) * step(abs(p.x), W * 1.05) * uOpen;
  skin = mix(skin, vec3(0.04, 0.0, 0.03), crease * 0.5);

  float rr = length(p * vec2(0.7, 1.0)) + (fbm(p * 1.8) - 0.5) * 0.7;
  float faceA = 1.0 - smoothstep(1.75, 2.6, rr);

  // inside the aperture: lid shadow on the eyeball plus a wet pink waterline along the lower lid
  float shadow = (1.0 - smoothstep(0.0, 0.17, up - p.y)) * 0.62;
  shadow += (1.0 - smoothstep(0.0, 0.06, p.y - lo)) * 0.25;
  float waterline = (1.0 - smoothstep(0.0, 0.016, p.y - lo)) * uOpen;
  vec3 insideCol = mix(vec3(0.02, 0.0, 0.015), vec3(0.86, 0.46, 0.54), waterline);
  float insideA = max(min(shadow, 0.8), waterline * 0.6);

  vec3 col = mix(skin, insideCol, inside);
  float a = mix(faceA, insideA, inside);
  gl_FragColor = vec4(col, a);
}`;

/* ---------- helpers ---------------------------------------------------------------------- */

const pointer = { x: 0, y: 0 };
const easeOut = (x: number) => 1 - Math.pow(1 - x, 3);
const smooth = (x: number) => x * x * (3 - 2 * x);
const X_AXIS = new THREE.Vector3(1, 0, 0);
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const Z_AXIS = new THREE.Vector3(0, 0, 1);

function useMedia(query: string) {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatch(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return match;
}

function makeRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/* ---------- growth system: every lash is a tree ----------------------------------------- */

type Seg = { root: number; ox: number; oy: number; oz: number; a: number; len: number; th: number };
type Crystal = { root: number; ox: number; oy: number; oz: number; sx: number; sy: number; rot: number; tilt: number; color: THREE.Color; spin: number };
type Leaf = { root: number; ox: number; oy: number; oz: number; dir: number; len: number; width: number; curl: number; turn: number; phase: number; color: THREE.Color };
type Flower = { root: number; ox: number; oy: number; oz: number; size: number; rot: number; n: number; layers: number; cup: number; tiltX: number; tiltY: number; color: THREE.Color; center: THREE.Color; spin: number };
type Tear = { root: number; ox: number; oy: number; variant: number; R: number; L: number; phase: number; sway: number };
type Root = { x: number; aOpen: number; aClosed: number; upper: boolean; follow: number; sway: number };
type Forest = { roots: Root[]; segs: Seg[]; tears: Tear[]; crystals: Crystal[]; leaves: Leaf[]; flowers: Flower[] };
type Rng = () => number;
type Style = {
  curl: number;        // how gnarled each piece is
  tropism: number;     // pull toward `lean` per piece
  lean: number;        // direction the whole lash wants to grow toward
  lateralProb: number; // chance of a side branch at each piece
  spreadMin: number;
  spreadVar: number;
  forkProb: number;    // chance the axis splits in two instead of continuing
  apical: number;      // length ratio of the continuing axis
  lateral: number;     // length ratio of side branches
  twigs: number;       // extra micro twigs at the ends
  leafy: number;       // leaf density multiplier for this lash
};

const CRYSTAL_COLORS = ["#fff4f8", "#ffd1e0", "#f3b6ff", "#ffe9a8", "#ff9fc0"].map((c) => new THREE.Color(c));
const FLOWER_COLORS = ["#ffb0c8", "#fff0f5", "#ff6fa0", "#d9a7ff", "#ffd9b0", "#ff8fb3"].map((c) => new THREE.Color(c));
const CENTER_COLORS = ["#ffd36b", "#ffb347", "#fff1a8", "#ff7aa2"].map((c) => new THREE.Color(c));
const LEAF_COLORS = ["#2e6b3b", "#5d7a2c", "#1f5c4c", "#3b7d5a", "#6a1f3a", "#8a2a3d", "#4a6b2a"].map((c) => new THREE.Color(c));

// water-drip profiles (variant 0 is the bead at the lid line): bulb centre position along the drop
const TEAR_BULB_POS = [0.5, 0.78, 0.84, 0.72, 0.88, 0.8];

function pick<T>(arr: T[], rand: Rng): T {
  return arr[Math.floor(rand() * arr.length) % arr.length];
}

function addLeaf(f: Forest, root: number, cx: number, cy: number, ca: number, rand: Rng) {
  const side = rand() < 0.5 ? 1 : -1;
  const len = 0.025 + rand() * 0.045;
  f.leaves.push({
    root, ox: cx, oy: cy, oz: 0.02 + rand() * 0.04,
    dir: ca + side * (0.5 + rand() * 0.9),
    len, width: len * (0.35 + rand() * 0.35),
    curl: (rand() - 0.5) * 0.9,
    turn: (rand() - 0.5) * 1.4,
    phase: rand() * Math.PI * 2,
    color: pick(LEAF_COLORS, rand),
  });
}

function decorateEnd(f: Forest, root: number, cx: number, cy: number, rand: Rng) {
  const roll = rand();
  if (roll < 0.03) {
    const bud = rand() < 0.25;
    const big = !bud && rand() < 0.4;
    f.flowers.push({
      root, ox: cx, oy: cy, oz: 0.04 + rand() * 0.03,
      size: bud ? 0.009 + rand() * 0.006 : big ? 0.022 + rand() * 0.016 : 0.013 + rand() * 0.012,
      rot: rand() * Math.PI * 2,
      n: bud ? 4 : 5 + Math.floor(rand() * 4),
      layers: big ? 2 : 1,
      cup: bud ? 1.1 : 0.25 + rand() * 0.55,
      tiltX: (rand() - 0.5) * 0.9,
      tiltY: (rand() - 0.5) * 0.9,
      color: pick(FLOWER_COLORS, rand),
      center: pick(CENTER_COLORS, rand),
      spin: (rand() - 0.5) * 0.25,
    });
  } else if (roll < 0.19) {
    const kind = rand();
    const base = 0.006 + rand() * 0.02;
    const shard = kind < 0.45;
    const cluster = kind > 0.85;
    const count = cluster ? 2 + Math.floor(rand() * 2) : 1;
    for (let i = 0; i < count; i++) {
      f.crystals.push({
        root, ox: cx + (i ? (rand() - 0.5) * 0.03 : 0), oy: cy + (i ? (rand() - 0.5) * 0.03 : 0), oz: 0.02 + rand() * 0.03,
        sx: base * (shard ? 0.45 : 0.8 + rand() * 0.5) * (i ? 0.7 : 1),
        sy: base * (shard ? 1.8 + rand() * 1.6 : 1 + rand() * 0.5) * (i ? 0.7 : 1),
        rot: rand() * Math.PI * 2,
        tilt: (rand() - 0.5) * 1.4,
        color: pick(CRYSTAL_COLORS, rand),
        spin: shard ? (rand() - 0.5) * 0.3 : (rand() - 0.5) * 1.2,
      });
    }
  }
}

function growBranch(root: number, x: number, y: number, a: number, len: number, th: number, depth: number, st: Style,
                    f: Forest, rand: Rng) {
  const { segs } = f;
  const pieces = 2 + Math.floor(rand() * 3);
  let cx = x, cy = y, ca = a;
  for (let k = 0; k < pieces; k++) {
    ca += st.curl * (rand() - 0.5) * 2 + (st.lean - ca) * st.tropism;
    const l = (len / pieces) * (0.7 + 0.6 * rand());
    segs.push({ root, ox: cx, oy: cy, oz: (rand() - 0.5) * 0.08, a: ca, len: l * 1.08, th: Math.max(0.0025, th * (1 - 0.4 * (k / pieces))) });
    cx += Math.sin(ca) * l;
    cy += Math.cos(ca) * l;
    const leafProb = (depth <= 1 ? 0.18 : depth === 2 ? 0.1 : 0.025) * st.leafy;
    if (rand() < leafProb) addLeaf(f, root, cx, cy, ca, rand);
    if (depth > 1 && k >= 1 && rand() < st.lateralProb) {
      const side = rand() < 0.5 ? 1 : -1;
      growBranch(root, cx, cy, ca + side * (st.spreadMin + rand() * st.spreadVar), len * st.lateral * (0.6 + 0.8 * rand()), th * 0.5, depth - 2, st, f, rand);
    }
  }
  if (depth > 0) {
    if (rand() < st.forkProb) {
      const sp = st.spreadMin * 0.7 + rand() * st.spreadVar * 0.6;
      const side = rand() < 0.5 ? 1 : -1;
      growBranch(root, cx, cy, ca + side * sp, len * st.apical * (0.7 + 0.5 * rand()), th * 0.62, depth - 1, st, f, rand);
      growBranch(root, cx, cy, ca - side * sp * (0.6 + 0.8 * rand()), len * st.apical * (0.5 + 0.6 * rand()), th * 0.55, depth - 2, st, f, rand);
    } else {
      growBranch(root, cx, cy, ca + (rand() - 0.5) * 0.35, len * st.apical * (0.8 + 0.4 * rand()), th * 0.68, depth - 1, st, f, rand);
    }
  }
  if (depth <= 1) {
    const n = 2 + Math.floor(rand() * st.twigs);
    for (let i = 0; i < n; i++) {
      const ta = ca + (rand() - 0.5) * 2.4;
      segs.push({ root, ox: cx, oy: cy, oz: (rand() - 0.5) * 0.04, a: ta, len: 0.02 + rand() * 0.06, th: 0.0025 });
    }
    decorateEnd(f, root, cx, cy, rand);
  }
}

// a crystallised tear: one smooth water body hanging from the lid line, thin neck swelling into a
// bulb (profile variant 1-5), plus a small meniscus bead (variant 0) and sometimes a detached
// falling drop below. Kinds: 0 bead only, 1 short run, 2 medium, 3 long ooze.
function growTear(f: Forest, root: number, kind: 0 | 1 | 2 | 3, rand: Rng) {
  const bead = 0.016 + rand() * 0.012;
  f.tears.push({ root, ox: 0, oy: 0.012, variant: 0, R: bead, L: bead * (1.1 + rand() * 0.5), phase: rand() * 6.28, sway: 0 });
  if (kind === 0) return;
  const L = kind === 1 ? 0.07 + rand() * 0.09 : kind === 2 ? 0.18 + rand() * 0.16 : 0.4 + rand() * 0.35;
  const R = kind === 1 ? 0.016 + rand() * 0.008 : kind === 2 ? 0.02 + rand() * 0.012 : 0.026 + rand() * 0.016;
  const variant = 1 + Math.floor(rand() * 5);
  f.tears.push({ root, ox: 0, oy: 0.004, variant, R, L, phase: rand() * 6.28, sway: 0.02 + rand() * 0.03 });
  const bulbPos = TEAR_BULB_POS[variant];
  const facets = kind === 1 ? 1 : 1 + Math.floor(rand() * 3);
  for (let i = 0; i < facets; i++) {
    f.crystals.push({ root, ox: (rand() - 0.5) * R * 1.3, oy: -L * bulbPos - R * (0.5 + rand() * 1.2), oz: 0.05 + rand() * 0.02,
                      sx: 0.005 + rand() * 0.006, sy: 0.014 + rand() * 0.02, rot: rand() * Math.PI * 2, tilt: (rand() - 0.5) * 1.2,
                      color: pick(CRYSTAL_COLORS, rand), spin: (rand() - 0.5) * 0.5 });
  }
  if (kind === 3 && rand() < 0.6) {
    const d = 0.009 + rand() * 0.007;
    f.tears.push({ root, ox: (rand() - 0.5) * 0.02, oy: -L - 0.06 - rand() * 0.1, variant: 1 + Math.floor(rand() * 5), R: d, L: d * (1.8 + rand() * 0.8),
                   phase: rand() * 6.28, sway: 0.01 });
  }
}

function randomStyle(rand: Rng, lean: number): Style {
  return {
    curl: 0.12 + rand() * 0.3,
    tropism: 0.03 + rand() * 0.12,
    lean,
    lateralProb: 0.35 + rand() * 0.45,
    spreadMin: 0.3 + rand() * 0.2,
    spreadVar: 0.4 + rand() * 0.35,
    forkProb: 0.25 + rand() * 0.35,
    apical: 0.65 + rand() * 0.2,
    lateral: 0.4 + rand() * 0.25,
    twigs: 2 + Math.floor(rand() * 3),
    leafy: rand() < 0.3 ? 0 : rand() < 0.5 ? 0.6 : 1.4,
  };
}

function buildForest(detail: number): Forest {
  const f: Forest = { roots: [], segs: [], tears: [], crystals: [], leaves: [], flowers: [] };
  const roots = f.roots;
  const rand = makeRng(4242);
  const addRoot = (r: Root) => roots.push(r) - 1;

  // the five big limbs of the upper lid: outer sweep, tallest, top, inner, inner sweep
  const giants: [number, number][] = [[-0.9, 0.8], [-0.45, 1.3], [-0.05, 1.05], [0.45, 1.0], [0.84, 0.85]];
  for (const [base, height] of giants) {
    const nx = base + (rand() - 0.5) * 0.05;
    const x = nx * W;
    const n = lidNormal(upperLid, x, false);
    const aOpen = n * 0.85 + nx * 0.2;
    const r = addRoot({ x, aOpen, aClosed: Math.sign(nx || 1) * (Math.PI - 0.6 * Math.abs(nx)), upper: true, follow: 0.2, sway: 0.01 + rand() * 0.008 });
    const st = randomStyle(rand, n * 0.6 + nx * 0.25);
    growBranch(r, 0, 0, aOpen, height * 0.38, 0.034 * (0.9 + 0.2 * rand()), Math.round(5 * detail) + 1, st, f, rand);
  }

  // every other upper lash: medium or small tree, radiating along the lid normal
  const nUpper = Math.round(66 * detail);
  for (let i = 0; i < nUpper; i++) {
    const u = (i + 0.5) / nUpper;
    const x = (2 * u - 1) * W * 0.985;
    const nx = x / W;
    const n = lidNormal(upperLid, x, false);
    const medium = rand() < 0.42;
    const aOpen = n * 0.9 + nx * 0.25 + (rand() - 0.5) * 0.3;
    const r = addRoot({ x, aOpen, aClosed: Math.sign(nx || 1) * (Math.PI - 0.6 * Math.abs(nx)) + (rand() - 0.5) * 0.3, upper: true,
                        follow: medium ? 0.45 : 0.8, sway: medium ? 0.006 : 0 });
    const st = randomStyle(rand, n * 0.7 + nx * 0.2);
    const height = medium ? 0.3 + rand() * 0.35 : 0.1 + rand() * 0.18;
    growBranch(r, 0, 0, aOpen, height * 0.4, (medium ? 0.019 : 0.012) * (0.8 + 0.4 * rand()), medium ? 4 : 2, st, f, rand);
  }

  // lower lid: shorter trees hanging down and out; sparser toward the inner corner where the tears run
  const nLower = Math.round(40 * detail);
  for (let i = 0; i < nLower; i++) {
    const u = (i + 0.5) / nLower;
    const x = (2 * u - 1) * W * 0.98;
    const nx = x / W;
    if (nx > 0.2 && rand() < 0.5) continue;
    const n = lidNormal(lowerLid, x, true);
    const big = nx < -0.35 && rand() < 0.35;
    const aOpen = n + nx * 0.15 + (rand() - 0.5) * 0.3;
    const r = addRoot({ x, aOpen, aClosed: aOpen, upper: false, follow: 0.5, sway: big ? 0.006 : 0 });
    const st = randomStyle(rand, n * 0.8 + nx * 0.15);
    const height = big ? 0.4 + rand() * 0.3 : 0.1 + rand() * 0.2;
    growBranch(r, 0, 0, aOpen, height * 0.4, (big ? 0.02 : 0.011) * (0.8 + 0.4 * rand()), big ? 4 : 2, st, f, rand);
  }

  // crystallised tears: beads, short runs, medium runs and long oozes, draining from the inner corner
  const tearPlan: [number, 0 | 1 | 2 | 3][] = [[0.4, 1], [0.47, 0], [0.53, 3], [0.6, 2], [0.68, 0], [0.74, 3], [0.8, 1], [0.86, 2], [0.91, 3]];
  for (const [base, kind] of tearPlan) {
    if (detail < 1 && kind === 0 && rand() < 0.5) continue;
    const nx = base + (rand() - 0.5) * 0.03;
    const r = addRoot({ x: nx * W, aOpen: Math.PI, aClosed: Math.PI, upper: false, follow: 0.7, sway: 0 });
    growTear(f, r, kind, rand);
  }

  return f;
}

/* ---------- geometries ------------------------------------------------------------------- */

const SEG_GEOM = new THREE.CylinderGeometry(0.42, 1, 1, 5, 1).translate(0, 0.5, 0);
const CRYSTAL_GEOM = new THREE.OctahedronGeometry(1, 0).scale(1, 1, 0.45);
const CENTER_GEOM = new THREE.SphereGeometry(1, 10, 10).scale(1, 1, 0.6);

// water drip hanging from the origin down to y = -1: neck radius `neck` (relative to the bulb),
// swelling smoothly into a bulb centred at `bulbPos`, rounded bottom. Variant 0 is a bead.
const TEAR_VARIANTS = [
  { neck: 0.9, bulbPos: 0.5, w: 0.6 },
  { neck: 0.12, bulbPos: 0.78, w: 0.3 },
  { neck: 0.2, bulbPos: 0.84, w: 0.42 },
  { neck: 0.09, bulbPos: 0.72, w: 0.22 },
  { neck: 0.16, bulbPos: 0.88, w: 0.5 },
  { neck: 0.25, bulbPos: 0.8, w: 0.36 },
].map(({ neck, bulbPos, w }) => {
  const pts: THREE.Vector2[] = [];
  const N = 32;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    let r: number;
    if (t <= bulbPos) {
      r = neck + (1 - neck) * Math.exp(-Math.pow((t - bulbPos) / w, 2));
      if (t < 0.06) r *= 0.6 + 0.4 * (t / 0.06); // slight pinch right under the lid
    } else {
      const u = (t - bulbPos) / (1 - bulbPos);
      r = Math.sqrt(Math.max(0, 1 - u * u));
    }
    pts.push(new THREE.Vector2(Math.max(0.002, r), -t));
  }
  return new THREE.LatheGeometry(pts, 18);
});

const LEAF_GEOM = (() => {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.quadraticCurveTo(0.42, 0.35, 0.08, 1);
  s.quadraticCurveTo(-0.3, 0.5, 0, 0);
  return new THREE.ShapeGeometry(s, 6);
})();

const PETAL_GEOM = (() => {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.quadraticCurveTo(0.5, 0.3, 0.32, 0.8);
  s.quadraticCurveTo(0.05, 1.08, 0, 1.0);
  s.quadraticCurveTo(-0.05, 1.08, -0.32, 0.8);
  s.quadraticCurveTo(-0.5, 0.3, 0, 0);
  return new THREE.ShapeGeometry(s, 8);
})();

/* ---------- forest renderer -------------------------------------------------------------- */

function Forest({ openRef, detail }: { openRef: React.MutableRefObject<number>; detail: number }) {
  const segRef = useRef<THREE.InstancedMesh>(null!);
  const tearRefs = useRef<(THREE.InstancedMesh | null)[]>([]);
  const crystalRef = useRef<THREE.InstancedMesh>(null!);
  const leafRef = useRef<THREE.InstancedMesh>(null!);
  const petalRef = useRef<THREE.InstancedMesh>(null!);
  const centerRef = useRef<THREE.InstancedMesh>(null!);
  const forest = useMemo(() => buildForest(detail), [detail]);
  const petals = useMemo(
    () => forest.flowers.flatMap((fl, fi) =>
      Array.from({ length: fl.n * fl.layers }, (_, j) => ({ fi, k: j % fl.n, layer: Math.floor(j / fl.n) }))),
    [forest]
  );
  const tearGroups = useMemo(() => TEAR_VARIANTS.map((_, vi) => forest.tears.filter((t) => t.variant === vi)), [forest]);
  const n = forest.roots.length;
  const tmp = useMemo(
    () => ({ m: new THREE.Matrix4(), q: new THREE.Quaternion(), q2: new THREE.Quaternion(), q3: new THREE.Quaternion(),
             p: new THREE.Vector3(), s: new THREE.Vector3(), e: new THREE.Euler(),
             cos: new Float32Array(n), sin: new Float32Array(n), rootY: new Float32Array(n), delta: new Float32Array(n) }),
    [n]
  );

  useEffect(() => {
    const paint = (mesh: THREE.InstancedMesh, colors: THREE.Color[]) => {
      colors.forEach((c, i) => mesh.setColorAt(i, c));
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    };
    paint(crystalRef.current, forest.crystals.map((c) => c.color));
    paint(leafRef.current, forest.leaves.map((l) => l.color));
    paint(petalRef.current, petals.map((pt) => forest.flowers[pt.fi].color));
    paint(centerRef.current, forest.flowers.map((fl) => fl.center));
  }, [forest, petals]);

  useFrame((state) => {
    const open = openRef.current;
    const now = state.clock.elapsedTime;
    const { m, q, q2, q3, p, s, e, cos, sin, rootY, delta } = tmp;
    const { roots, segs, crystals, leaves, flowers } = forest;
    for (let i = 0; i < n; i++) {
      const r = roots[i];
      const d = r.follow * (1 - open) * (r.aClosed - r.aOpen) + r.sway * Math.sin(now * 0.7 + i);
      delta[i] = d;
      cos[i] = Math.cos(d);
      sin[i] = Math.sin(d);
      rootY[i] = (r.upper ? upperLid : lowerLid)(r.x) * open;
    }
    // position of an attachment point (ox, oy) on root k after the root's blink rotation
    const px = (k: number, ox: number, oy: number) => roots[k].x + ox * cos[k] + oy * sin[k];
    const py = (k: number, ox: number, oy: number) => rootY[k] + oy * cos[k] - ox * sin[k];

    const segMesh = segRef.current;
    for (let i = 0; i < segs.length; i++) {
      const sg = segs[i];
      const k = sg.root;
      p.set(px(k, sg.ox, sg.oy), py(k, sg.ox, sg.oy), Z_LASH + sg.oz);
      e.set(0, 0, -(sg.a + delta[k]));
      q.setFromEuler(e);
      s.set(sg.th, sg.len, sg.th);
      m.compose(p, q, s);
      segMesh.setMatrixAt(i, m);
    }
    segMesh.instanceMatrix.needsUpdate = true;

    for (let vi = 0; vi < tearGroups.length; vi++) {
      const mesh = tearRefs.current[vi];
      const list = tearGroups[vi];
      if (!mesh) continue;
      for (let i = 0; i < list.length; i++) {
        const t = list[i];
        p.set(px(t.root, t.ox, t.oy), py(t.root, t.ox, t.oy), Z_LASH + 0.03);
        e.set(0, 0, -delta[t.root] * 0.3 + t.sway * Math.sin(now * 1.3 + t.phase));
        q.setFromEuler(e);
        const wob = 1 + 0.03 * Math.sin(now * 1.9 + t.phase);
        s.set(t.R * wob, t.L, t.R * wob);
        m.compose(p, q, s);
        mesh.setMatrixAt(i, m);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }

    for (let i = 0; i < crystals.length; i++) {
      const c = crystals[i];
      p.set(px(c.root, c.ox, c.oy), py(c.root, c.ox, c.oy), Z_LASH + c.oz);
      e.set(c.tilt, now * c.spin + c.rot, c.rot - delta[c.root]);
      q.setFromEuler(e);
      const tw = 0.9 + 0.1 * Math.sin(now * 2.2 + c.rot * 7);
      s.set(c.sx * tw, c.sy * tw, c.sx * tw);
      m.compose(p, q, s);
      crystalRef.current.setMatrixAt(i, m);
    }
    crystalRef.current.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < leaves.length; i++) {
      const lf = leaves[i];
      p.set(px(lf.root, lf.ox, lf.oy), py(lf.root, lf.ox, lf.oy), Z_LASH + lf.oz);
      q.setFromAxisAngle(Z_AXIS, -(lf.dir + delta[lf.root]));
      q2.setFromAxisAngle(Y_AXIS, lf.turn + 0.18 * Math.sin(now * 1.6 + lf.phase));
      q3.setFromAxisAngle(X_AXIS, lf.curl);
      q.multiply(q2).multiply(q3);
      s.set(lf.width, lf.len, 1);
      m.compose(p, q, s);
      leafRef.current.setMatrixAt(i, m);
    }
    leafRef.current.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < petals.length; i++) {
      const { fi, k, layer } = petals[i];
      const fl = flowers[fi];
      p.set(px(fl.root, fl.ox, fl.oy), py(fl.root, fl.ox, fl.oy), Z_LASH + fl.oz + layer * 0.004);
      const dir = fl.rot + now * fl.spin + (k + layer * 0.5) * ((Math.PI * 2) / fl.n) - delta[fl.root];
      q.setFromAxisAngle(X_AXIS, fl.tiltX).multiply(q2.setFromAxisAngle(Y_AXIS, fl.tiltY));
      q.multiply(q2.setFromAxisAngle(Z_AXIS, dir)).multiply(q3.setFromAxisAngle(X_AXIS, fl.cup + layer * 0.25));
      const sz = fl.size * (layer ? 0.65 : 1);
      s.set(sz * 0.9, sz, 1);
      m.compose(p, q, s);
      petalRef.current.setMatrixAt(i, m);
    }
    petalRef.current.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < flowers.length; i++) {
      const fl = flowers[i];
      p.set(px(fl.root, fl.ox, fl.oy), py(fl.root, fl.ox, fl.oy), Z_LASH + fl.oz + 0.012);
      q.setFromAxisAngle(X_AXIS, fl.tiltX).multiply(q2.setFromAxisAngle(Y_AXIS, fl.tiltY));
      const sz = fl.size * (fl.cup > 1 ? 0.2 : 0.3);
      s.set(sz, sz, sz);
      m.compose(p, q, s);
      centerRef.current.setMatrixAt(i, m);
    }
    centerRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <>
      <instancedMesh ref={segRef} args={[SEG_GEOM, undefined, forest.segs.length]} frustumCulled={false}>
        <meshStandardMaterial color="#0a0206" roughness={0.65} metalness={0.08} />
      </instancedMesh>
      {TEAR_VARIANTS.map((geom, vi) => (
        <instancedMesh
          key={vi}
          ref={(el) => {
            tearRefs.current[vi] = el;
          }}
          args={[geom, undefined, Math.max(1, tearGroups[vi].length)]}
          count={tearGroups[vi].length}
          frustumCulled={false}
          renderOrder={2}
        >
          <meshPhysicalMaterial color="#fff0f6" transparent opacity={0.86} roughness={0.03} metalness={0.05} clearcoat={1} clearcoatRoughness={0.02} emissive="#ff9fc0" emissiveIntensity={0.22} />
        </instancedMesh>
      ))}
      <instancedMesh ref={crystalRef} args={[CRYSTAL_GEOM, undefined, forest.crystals.length]} frustumCulled={false}>
        <meshPhysicalMaterial color="#ffffff" emissive="#ff7fae" emissiveIntensity={0.25} roughness={0.12} clearcoat={1} transparent opacity={0.9} />
      </instancedMesh>
      <instancedMesh ref={leafRef} args={[LEAF_GEOM, undefined, forest.leaves.length]} frustumCulled={false}>
        <meshStandardMaterial color="#ffffff" roughness={0.7} metalness={0} side={THREE.DoubleSide} />
      </instancedMesh>
      <instancedMesh ref={petalRef} args={[PETAL_GEOM, undefined, petals.length]} frustumCulled={false}>
        <meshStandardMaterial color="#ffffff" emissive="#ff4f8a" emissiveIntensity={0.12} roughness={0.55} side={THREE.DoubleSide} />
      </instancedMesh>
      <instancedMesh ref={centerRef} args={[CENTER_GEOM, undefined, forest.flowers.length]} frustumCulled={false}>
        <meshStandardMaterial color="#ffffff" emissive="#ffb347" emissiveIntensity={0.3} roughness={0.6} />
      </instancedMesh>
    </>
  );
}

/* ---------- the eye ---------------------------------------------------------------------- */

function EyeRig({ scale, reduced, detail }: { scale: number; reduced: boolean; detail: number }) {
  const eye = useRef<THREE.Group>(null!);
  const scan = useRef<THREE.Mesh>(null!);
  const scanGlow = useRef<THREE.Mesh>(null!);
  const ring = useRef<THREE.Mesh>(null!);
  const openRef = useRef(0);
  const blink = useRef({ t: -1, next: 3 });

  const eyeMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: EYE_VERT,
        fragmentShader: EYE_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uPupil: { value: 0.36 },
          uAngScale: { value: ANG_SCALE },
          uIrisA: { value: new THREE.Color("#ff8fb3") },
          uIrisB: { value: new THREE.Color("#4a1a6b") },
          uIrisC: { value: new THREE.Color("#ffe3ee") },
        },
      }),
    []
  );
  const faceMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: FACE_VERT,
        fragmentShader: FACE_FRAG,
        uniforms: { uOpen: { value: 0 }, uTime: { value: 0 } },
        transparent: true,
        depthWrite: false,
      }),
    []
  );

  useFrame((state, dt) => {
    const now = state.clock.elapsedTime;
    eyeMat.uniforms.uTime.value = now;
    eyeMat.uniforms.uPupil.value = 0.34 + 0.03 * Math.sin(now * 0.7);
    faceMat.uniforms.uTime.value = now;

    const tx = THREE.MathUtils.clamp(-pointer.y * 0.22, -0.25, 0.25);
    const ty = THREE.MathUtils.clamp(pointer.x * 0.33, -0.38, 0.38);
    eye.current.rotation.x += (tx - eye.current.rotation.x) * Math.min(1, dt * 6);
    eye.current.rotation.y += (ty - eye.current.rotation.y) * Math.min(1, dt * 6);

    let open = 1;
    if (!reduced) {
      const b = blink.current;
      if (b.t < 0 && now > b.next) b.t = 0;
      if (b.t >= 0) {
        b.t += dt;
        const d = 0.32;
        if (b.t > d) {
          b.t = -1;
          b.next = now + 2.5 + Math.random() * 4.5;
        } else {
          const ph = b.t / d;
          open = ph < 0.4 ? 1 - smooth(ph / 0.4) : smooth((ph - 0.4) / 0.6);
        }
      }
      open *= easeOut(Math.min(1, now / 2.4));
    }
    openRef.current = open;
    faceMat.uniforms.uOpen.value = open;

    const s = reduced ? 0.3 : Math.sin(now * 0.8) * 0.75;
    scan.current.rotation.x = s;
    scanGlow.current.rotation.x = s;
    ring.current.rotation.z = now * 0.25;
    (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.3 + 0.25 * Math.sin(now * 2.0);
  });

  return (
    <group position={[0, -0.22, 0]} scale={scale}>
      <group ref={eye}>
        <mesh material={eyeMat}>
          <sphereGeometry args={[Z_EYE_R, 128, 128]} />
        </mesh>
        <mesh>
          <sphereGeometry args={[Z_EYE_R + 0.02, 96, 96]} />
          <meshPhysicalMaterial color="#ffffff" transparent opacity={0.14} roughness={0.03} metalness={0} clearcoat={1} clearcoatRoughness={0.02} depthWrite={false} />
        </mesh>
        <mesh ref={ring} position={[0, 0, Z_EYE_R * Math.cos(IRIS_T) + 0.004]}>
          <torusGeometry args={[Z_EYE_R * Math.sin(IRIS_T), 0.004, 8, 128]} />
          <meshBasicMaterial color="#ff2e6e" transparent opacity={0.45} blending={THREE.AdditiveBlending} depthWrite={false} />
        </mesh>
        <mesh ref={scan}>
          <sphereGeometry args={[Z_EYE_R + 0.03, 128, 2, 0, Math.PI, Math.PI / 2 - 0.01, 0.02]} />
          <meshBasicMaterial color="#ff5c8d" transparent opacity={0.95} blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
        <mesh ref={scanGlow}>
          <sphereGeometry args={[Z_EYE_R + 0.028, 128, 6, 0, Math.PI, Math.PI / 2 - 0.1, 0.2]} />
          <meshBasicMaterial color="#ff2e6e" transparent opacity={0.1} blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      </group>

      <mesh position={[0, 0, Z_FACE]} material={faceMat} renderOrder={1}>
        <planeGeometry args={[9, 6]} />
      </mesh>

      <Forest openRef={openRef} detail={detail} />

      <Sparkles count={30} scale={[1.1, 0.8, 0.3]} position={[W * 0.65, -0.7, Z_LASH + 0.12]} size={1.6} speed={0.4} opacity={0.8} color="#ffffff" />
      <Sparkles count={70} scale={[3.0, 1.8, 0.5]} position={[0.2, 1.1, Z_LASH + 0.12]} size={1.6} speed={0.3} opacity={0.6} color="#ffd6e6" />

      <ambientLight intensity={0.45} />
      <pointLight position={[2.5, 3, 4]} intensity={60} color="#ffe6ee" />
      <pointLight position={[-3.5, -1.5, 3]} intensity={25} color="#b14bd6" />
      <pointLight position={[0, -3, 2.5]} intensity={12} color="#ff2e6e" />
    </group>
  );
}

/* Lens shift: the camera looks straight at the eye (no corner distortion) while the frame is
   slid so the eye sits `shift` of the canvas width to the right of centre. */
function Framing({ shift }: { shift: number }) {
  const { camera, size } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    if (shift) cam.setViewOffset(size.width, size.height, -shift * size.width, 0, size.width, size.height);
    else cam.clearViewOffset();
    cam.updateProjectionMatrix();
  }, [camera, size, shift]);
  return null;
}

export default function EyeScene() {
  const reduced = useMedia("(prefers-reduced-motion: reduce)");
  const narrow = useMedia("(max-width: 820px)");
  useEffect(() => {
    const on = (e: PointerEvent) => {
      pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
      pointer.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("pointermove", on, { passive: true });
    return () => window.removeEventListener("pointermove", on);
  }, []);
  return (
    <Canvas dpr={[1, 1.5]} camera={{ position: [0, 0, CAM_Z], fov: CAM_FOV }} gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}>
      <Framing shift={narrow ? 0 : 0.225} />
      <EyeRig scale={narrow ? 0.6 : 0.94} reduced={reduced} detail={narrow ? 0.6 : 1} />
    </Canvas>
  );
}
