// Scene catalogue. Each scene rasterizes its body onto the mask from a set of
// user-adjustable parameters and reports the reference length used for Re and
// the force coefficients. Sizes scale with the tunnel height so a scene looks
// the same at every resolution.
import {
  arc,
  fillCircle,
  fillPolygon,
  nacaOutline,
  parseNaca,
  place,
  rectangle,
  setSpin,
  thinAirfoilLift,
  zeroLiftAngle,
  type Mask,
  type Naca,
  type Point,
} from './physics/geometry.ts';
import type { WallMode } from './physics/lattice.ts';
import type { ViewMode } from './gpu/renderer.ts';

export interface RangeControl {
  kind: 'range';
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
}

export interface TextControl {
  kind: 'naca';
  key: string;
  label: string;
  value: string;
}

export type Control = RangeControl | TextControl;
export type Values = Record<string, number | string>;

export interface Built {
  length: number | null; // reference length in cells; null = measure frontal height
  lengthLabel: string; // what the length is, for the readout ("chord", "D", …)
  bodies?: number; // forces are reported per body when there are several
  noForces?: boolean; // channel scenes: the solid is a wall, not a test body
  airfoil?: { naca: Naca; alpha: number; alphaZeroLift: number };
  spin?: { centre: Point; radius: number; ratio: number };
  step?: { end: number; height: number };
}

export type Metric = 'cl' | 'cd' | 'st';

// A parameter sweep the scene offers: `key` is one of its controls, or 're'.
export interface Study {
  key: string;
  label: string;
  from: number;
  to: number;
  step: number;
  settle: number; // convective times
  average: number;
  series: Metric[];
  theory?: { label: string; metric: Metric; at: (x: number, built: Built) => number };
}

export interface Scene {
  id: string;
  name: string;
  blurb: string;
  re: number;
  walls: WallMode;
  view: ViewMode;
  contrast?: number; // log2 display contrast to start from
  origin: Point; // body position as a fraction of the tunnel
  movable: boolean;
  study?: Study;
  controls: Control[];
  build: (mask: Mask, v: Values, at: Point, u0: number) => Built;
}

const num = (v: Values, k: string) => Number(v[k]);
const rad = (deg: number) => (deg * Math.PI) / 180;

const airfoil = (mask: Mask, v: Values, at: Point): Built => {
  const naca = parseNaca(String(v.naca)) ?? parseNaca('0012')!;
  const chord = Math.round(mask.ny * num(v, 'chord'));
  const alpha = num(v, 'alpha');
  // Positive angle of attack is nose up: rotate clockwise about the quarter chord.
  fillPolygon(mask, place(nacaOutline(naca, 160), { at, pivot: { x: 0.25, y: 0 }, scale: chord, angle: -rad(alpha) }));
  return {
    length: chord,
    lengthLabel: 'chord',
    airfoil: { naca, alpha, alphaZeroLift: zeroLiftAngle(naca) },
  };
};

const airfoilControls = (code: string, alpha: number): Control[] => [
  { kind: 'naca', key: 'naca', label: 'NACA', value: code },
  { kind: 'range', key: 'alpha', label: 'Angle of attack', min: -10, max: 25, step: 0.5, value: alpha, unit: '°' },
  { kind: 'range', key: 'chord', label: 'Chord', min: 0.2, max: 0.6, step: 0.01, value: 0.42, unit: 'H' },
];

// Side profile of the Ahmed body: rounded nose, flat roof, rear slant.
// Proportions follow the 1984 reference body (L 1044, H 288, nose radius 100,
// slant length 222 mm); `at` is the bottom of the nose.
const ahmedOutline = (h: number, slantDeg: number): Point[] => {
  const s = h / 288;
  const len = 1044 * s;
  const r = 100 * s;
  const slant = 222 * s;
  const a = rad(slantDeg);
  const sx = slant * Math.cos(a);
  const sy = slant * Math.sin(a);
  return [
    ...arc({ x: r, y: r }, r, Math.PI * 1.5, Math.PI, 10),
    ...arc({ x: r, y: h - r }, r, Math.PI, Math.PI * 0.5, 10),
    { x: len - sx, y: h },
    { x: len, y: h - sy },
    { x: len, y: 0 },
  ];
};

export const scenes: Scene[] = [
  {
    id: 'cylinder',
    name: 'Cylinder',
    blurb:
      'Above Re ≈ 47 the twin vortices behind a cylinder become unstable and start shedding alternately: a Kármán vortex street. The Strouhal number St = fD/U settles near 0.2 across a wide range of Re.',
    re: 160,
    walls: 'slip',
    view: 'smoke',
    origin: { x: 0.22, y: 0.5 },
    movable: true,
    study: {
      key: 're',
      label: 'Re',
      from: 60,
      to: 300,
      step: 40,
      settle: 25,
      average: 20,
      series: ['st', 'cd'],
      theory: { label: 'Williamson & Brown St = 0.2665 − 1.018/√Re', metric: 'st', at: (re) => 0.2665 - 1.018 / Math.sqrt(re) },
    },
    controls: [{ kind: 'range', key: 'd', label: 'Diameter', min: 0.05, max: 0.3, step: 0.01, value: 0.12, unit: 'H' }],
    build: (mask, v, at) => {
      const d = mask.ny * num(v, 'd');
      fillCircle(mask, at, d / 2);
      return { length: d, lengthLabel: 'D' };
    },
  },
  {
    id: 'airfoil',
    name: 'Airfoil',
    blurb:
      'A NACA 4-digit section. The first digit is camber in % chord, the second where it peaks in tenths, the last two the thickness. Lift is measured directly from momentum exchange on the surface.',
    re: 1500,
    walls: 'slip',
    view: 'smoke',
    origin: { x: 0.24, y: 0.5 },
    movable: true,
    study: {
      key: 'alpha',
      label: 'α',
      from: -4,
      to: 14,
      step: 2,
      settle: 4,
      average: 5,
      series: ['cl', 'cd'],
      theory: {
        label: 'thin airfoil 2π(α − α₀)',
        metric: 'cl',
        at: (a, b) => thinAirfoilLift(rad(a), b.airfoil?.alphaZeroLift ?? 0),
      },
    },
    controls: airfoilControls('2412', 6),
    build: (mask, v, at) => airfoil(mask, v, at),
  },
  {
    id: 'stall',
    name: 'Stalled wing',
    blurb:
      'Past the stall angle the boundary layer can no longer follow the suction side. It separates near the leading edge, lift collapses, drag jumps and the wake turns into large shed vortices.',
    re: 3000,
    walls: 'slip',
    view: 'vorticity',
    origin: { x: 0.24, y: 0.52 },
    movable: true,
    study: {
      key: 'alpha',
      label: 'α',
      from: 0,
      to: 24,
      step: 3,
      settle: 4,
      average: 5,
      series: ['cl', 'cd'],
      theory: {
        label: 'thin airfoil 2π(α − α₀)',
        metric: 'cl',
        at: (a, b) => thinAirfoilLift(rad(a), b.airfoil?.alphaZeroLift ?? 0),
      },
    },
    controls: airfoilControls('0012', 18),
    build: (mask, v, at) => airfoil(mask, v, at),
  },
  {
    id: 'magnus',
    name: 'Magnus effect',
    blurb:
      'A spinning cylinder drags fluid around with it, speeding the flow on one side and slowing it on the other. The pressure difference is a lift force; at high spin ratios ΩR/U shedding stops altogether.',
    re: 160,
    walls: 'slip',
    view: 'smoke',
    origin: { x: 0.22, y: 0.5 },
    movable: true,
    study: {
      key: 'spin',
      label: 'ΩR/U',
      from: 0,
      to: 3,
      step: 0.5,
      settle: 12,
      average: 12,
      series: ['cl', 'cd'],
      theory: { label: 'potential flow 2π·ΩR/U', metric: 'cl', at: (x) => 2 * Math.PI * x },
    },
    controls: [
      { kind: 'range', key: 'spin', label: 'Spin ratio ΩR/U', min: -4, max: 4, step: 0.1, value: 1.5 },
      { kind: 'range', key: 'd', label: 'Diameter', min: 0.05, max: 0.25, step: 0.01, value: 0.12, unit: 'H' },
    ],
    build: (mask, v, at, u0) => {
      const r = (mask.ny * num(v, 'd')) / 2;
      const ratio = num(v, 'spin');
      fillCircle(mask, at, r);
      // Positive ratio spins clockwise (top surface moving with the stream): lift up.
      setSpin(mask, at, r, (-ratio * u0) / r);
      return { length: 2 * r, lengthLabel: 'D', spin: { centre: at, radius: r, ratio } };
    },
  },
  {
    id: 'plate',
    name: 'Flat plate',
    blurb:
      'A thin plate at an angle. Square to the stream it is the bluffest shape there is (Cd ≈ 2 in 2D) with a huge wake; turned edge-on its drag is almost all skin friction.',
    re: 400,
    walls: 'slip',
    view: 'vorticity',
    origin: { x: 0.24, y: 0.5 },
    movable: true,
    study: { key: 'angle', label: 'angle', from: 0, to: 90, step: 10, settle: 8, average: 10, series: ['cl', 'cd'] },
    controls: [
      { kind: 'range', key: 'angle', label: 'Angle to flow', min: 0, max: 90, step: 1, value: 90, unit: '°' },
      { kind: 'range', key: 'len', label: 'Length', min: 0.08, max: 0.4, step: 0.01, value: 0.16, unit: 'H' },
    ],
    build: (mask, v, at) => {
      const len = mask.ny * num(v, 'len');
      const thick = Math.max(2.5, mask.ny / 120);
      fillPolygon(mask, place(rectangle(len, thick), { at, angle: rad(num(v, 'angle')) }));
      return { length: null, lengthLabel: 'frontal h' };
    },
  },
  {
    id: 'tandem',
    name: 'Tandem cylinders',
    blurb:
      'The downstream cylinder sits in its neighbour’s wake. Close together the gap holds a trapped vortex pair and the rear one feels thrust; beyond about 3.5 D the front one sheds onto it.',
    re: 220,
    walls: 'slip',
    view: 'vorticity',
    origin: { x: 0.18, y: 0.5 },
    movable: true,
    controls: [
      { kind: 'range', key: 'gap', label: 'Spacing', min: 1.2, max: 7, step: 0.1, value: 3.5, unit: 'D' },
      { kind: 'range', key: 'd', label: 'Diameter', min: 0.05, max: 0.2, step: 0.01, value: 0.1, unit: 'H' },
    ],
    build: (mask, v, at) => {
      const d = mask.ny * num(v, 'd');
      fillCircle(mask, at, d / 2);
      fillCircle(mask, { x: at.x + num(v, 'gap') * d, y: at.y }, d / 2);
      return { length: d, lengthLabel: 'D', bodies: 2 };
    },
  },
  {
    id: 'tubes',
    name: 'Tube bank',
    blurb:
      'A staggered array of tubes, as in a heat exchanger or boiler. Each row breaks up the wakes of the last, so the flow inside the bank mixes far more than behind a single tube.',
    re: 300,
    walls: 'slip',
    view: 'speed',
    contrast: -0.8,
    origin: { x: 0.2, y: 0.5 },
    movable: true,
    controls: [
      { kind: 'range', key: 'cols', label: 'Rows deep', min: 1, max: 8, step: 1, value: 5 },
      { kind: 'range', key: 'pitch', label: 'Pitch', min: 1.4, max: 3.5, step: 0.1, value: 2.2, unit: 'D' },
      { kind: 'range', key: 'd', label: 'Diameter', min: 0.04, max: 0.12, step: 0.005, value: 0.07, unit: 'H' },
    ],
    build: (mask, v, at) => {
      const d = mask.ny * num(v, 'd');
      const pitch = num(v, 'pitch') * d;
      const cols = num(v, 'cols');
      const rows = Math.floor((mask.ny * 0.8) / pitch);
      let bodies = 0;
      for (let c = 0; c < cols; c++) {
        const shift = c % 2 ? pitch / 2 : 0;
        for (let r = 0; r < rows; r++) {
          const y = at.y + (r - (rows - 1) / 2) * pitch + shift;
          if (y - d / 2 > 2 && y + d / 2 < mask.ny - 2) {
            fillCircle(mask, { x: at.x + c * pitch * 0.866, y }, d / 2);
            bodies++;
          }
        }
      }
      return { length: d, lengthLabel: 'D', bodies };
    },
  },
  {
    id: 'ahmed',
    name: 'Ahmed body',
    blurb:
      'The standard simplified car used to validate automotive CFD. The rear slant angle decides how the flow leaves the roof: below about 30° it stays attached down the slant, steeper and it breaks away.',
    re: 2500,
    walls: 'slip',
    view: 'smoke',
    origin: { x: 0.14, y: 0.04 },
    movable: true,
    study: { key: 'slant', label: 'slant', from: 0, to: 40, step: 5, settle: 6, average: 8, series: ['cd', 'cl'] },
    controls: [
      { kind: 'range', key: 'slant', label: 'Slant angle', min: 0, max: 45, step: 1, value: 25, unit: '°' },
      { kind: 'range', key: 'h', label: 'Height', min: 0.1, max: 0.3, step: 0.01, value: 0.18, unit: 'H' },
      { kind: 'range', key: 'clear', label: 'Ground clearance', min: 0, max: 0.4, step: 0.01, value: 0.17, unit: 'h' },
    ],
    build: (mask, v, at) => {
      const h = mask.ny * num(v, 'h');
      const lift = num(v, 'clear') * h;
      fillPolygon(mask, place(ahmedOutline(h, num(v, 'slant')), { at: { x: at.x, y: at.y + lift } }));
      return { length: h, lengthLabel: 'h' };
    },
  },
  {
    id: 'step',
    name: 'Backward step',
    blurb:
      'Channel flow over a sudden expansion. The flow separates at the corner and reattaches downstream; the recirculation length x_r/h grows with Re, a classic benchmark for CFD codes.',
    re: 300,
    walls: 'noslip',
    view: 'streamlines',
    origin: { x: 0, y: 0 },
    movable: false,
    controls: [
      { kind: 'range', key: 'h', label: 'Step height', min: 0.2, max: 0.6, step: 0.01, value: 0.35, unit: 'H' },
      { kind: 'range', key: 'len', label: 'Inlet length', min: 0.04, max: 0.5, step: 0.01, value: 0.1, unit: 'L' },
    ],
    build: (mask, v) => {
      const h = Math.round(mask.ny * num(v, 'h'));
      const end = Math.round(mask.nx * num(v, 'len'));
      fillPolygon(mask, [
        { x: -1, y: -1 },
        { x: end, y: -1 },
        { x: end, y: h },
        { x: -1, y: h },
      ]);
      return { length: h, lengthLabel: 'h', noForces: true, step: { end, height: h } };
    },
  },
  {
    id: 'sandbox',
    name: 'Sandbox',
    blurb:
      'An empty tunnel. Draw your own shapes with the Draw tool; forces are measured on everything solid, with the frontal height as the reference length.',
    re: 1000,
    walls: 'slip',
    view: 'smoke',
    origin: { x: 0.25, y: 0.5 },
    movable: false,
    controls: [],
    build: () => ({ length: null, lengthLabel: 'frontal h' }),
  },
];

export const defaultValues = (scene: Scene): Values =>
  Object.fromEntries(scene.controls.map((c) => [c.key, c.value]));
