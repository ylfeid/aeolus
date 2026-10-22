// Obstacle geometry: NACA 4-digit airfoils, thin-airfoil theory, and rasterization
// of shapes onto the solid mask. Grid units throughout: cell (x, y) covers
// [x, x+1) × [y, y+1), row 0 is the bottom of the tunnel, y points up.

export interface Point {
  x: number;
  y: number;
}

export interface Mask {
  nx: number;
  ny: number;
  solid: Uint8Array;
  // Surface velocity of moving solids (x, y interleaved); zero for fixed walls.
  wall: Float32Array;
}

export const createMask = (nx: number, ny: number): Mask => ({
  nx,
  ny,
  solid: new Uint8Array(nx * ny),
  wall: new Float32Array(2 * nx * ny),
});

export const clearMask = (mask: Mask) => {
  mask.solid.fill(0);
  mask.wall.fill(0);
};

/* ---------------------------------------------------------------- NACA 4-digit */

export interface Naca {
  code: string;
  m: number; // maximum camber, fraction of chord
  p: number; // chordwise position of maximum camber
  t: number; // maximum thickness
}

export const parseNaca = (code: string): Naca | null => {
  if (!/^\d{4}$/.test(code)) return null;
  const m = Number(code[0]) / 100;
  const p = Number(code[1]) / 10;
  const t = Number(code.slice(2)) / 100;
  if (t === 0 || (m > 0 && p === 0)) return null;
  return { code, m, p, t };
};

export const nacaCamber = (naca: Naca, x: number) => {
  const { m, p } = naca;
  if (m === 0) return { yc: 0, slope: 0 };
  if (x < p) return { yc: (m / (p * p)) * (2 * p * x - x * x), slope: ((2 * m) / (p * p)) * (p - x) };
  const q = (1 - p) * (1 - p);
  return { yc: (m / q) * (1 - 2 * p + 2 * p * x - x * x), slope: ((2 * m) / q) * (p - x) };
};

// Half-thickness with the closed-trailing-edge coefficient (-0.1036).
export const nacaThickness = (t: number, x: number) =>
  5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4);

// Closed outline in chord units (leading edge at the origin), traced from the
// trailing edge over the upper surface and back along the lower one.
export const nacaOutline = (naca: Naca, points = 120): Point[] => {
  const upper: Point[] = [];
  const lower: Point[] = [];
  for (let k = 0; k <= points; k++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * k) / points)); // cosine spacing clusters at both edges
    const yt = nacaThickness(naca.t, x);
    const { yc, slope } = nacaCamber(naca, x);
    const th = Math.atan(slope);
    upper.push({ x: x - yt * Math.sin(th), y: yc + yt * Math.cos(th) });
    lower.push({ x: x + yt * Math.sin(th), y: yc - yt * Math.cos(th) });
  }
  return [...upper.reverse(), ...lower.slice(1)];
};

// Thin airfoil theory: α_L0 = -(1/π) ∫₀^π (dz/dx)(cos θ − 1) dθ with x = (1 − cos θ)/2.
export const zeroLiftAngle = (naca: Naca, samples = 4000) => {
  let sum = 0;
  const d = Math.PI / samples;
  for (let k = 0; k < samples; k++) {
    const th = (k + 0.5) * d;
    sum += nacaCamber(naca, 0.5 * (1 - Math.cos(th))).slope * (Math.cos(th) - 1) * d;
  }
  return -sum / Math.PI;
};

// Lift slope 2π per radian, offset by the zero-lift angle.
export const thinAirfoilLift = (alpha: number, alphaZeroLift: number) => 2 * Math.PI * (alpha - alphaZeroLift);

/* ------------------------------------------------------------------ transforms */

export interface Placement {
  at: Point; // where the pivot lands on the grid
  pivot?: Point; // pivot in shape coordinates
  scale?: number;
  angle?: number; // radians, counter-clockwise
}

export const place = (pts: Point[], { at, pivot = { x: 0, y: 0 }, scale = 1, angle = 0 }: Placement): Point[] => {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return pts.map((p) => {
    const x = (p.x - pivot.x) * scale;
    const y = (p.y - pivot.y) * scale;
    return { x: at.x + c * x - s * y, y: at.y + s * x + c * y };
  });
};

export const rectangle = (w: number, h: number): Point[] => [
  { x: -w / 2, y: -h / 2 },
  { x: w / 2, y: -h / 2 },
  { x: w / 2, y: h / 2 },
  { x: -w / 2, y: h / 2 },
];

export const arc = (c: Point, r: number, from: number, to: number, segments = 16): Point[] => {
  const out: Point[] = [];
  for (let k = 0; k <= segments; k++) {
    const a = from + ((to - from) * k) / segments;
    out.push({ x: c.x + r * Math.cos(a), y: c.y + r * Math.sin(a) });
  }
  return out;
};

/* --------------------------------------------------------------- rasterization */

// Even-odd scanline fill: a cell is solid when its centre lies inside the polygon.
export const fillPolygon = (mask: Mask, poly: Point[]) => {
  const { nx, ny, solid } = mask;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const p of poly) {
    ymin = Math.min(ymin, p.y);
    ymax = Math.max(ymax, p.y);
  }
  const y0 = Math.max(0, Math.floor(ymin));
  const y1 = Math.min(ny - 1, Math.ceil(ymax));
  const xs: number[] = [];
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    xs.length = 0;
    for (let k = 0; k < poly.length; k++) {
      const a = poly[k];
      const b = poly[(k + 1) % poly.length];
      if (a.y <= yc !== b.y <= yc) xs.push(a.x + ((yc - a.y) * (b.x - a.x)) / (b.y - a.y));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const from = Math.max(0, Math.ceil(xs[k] - 0.5));
      const to = Math.min(nx - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = from; x <= to; x++) solid[y * nx + x] = 1;
    }
  }
};

export const fillCircle = (mask: Mask, c: Point, r: number) => {
  const { nx, ny, solid } = mask;
  const x0 = Math.max(0, Math.floor(c.x - r));
  const x1 = Math.min(nx - 1, Math.ceil(c.x + r));
  const y0 = Math.max(0, Math.floor(c.y - r));
  const y1 = Math.min(ny - 1, Math.ceil(c.y + r));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - c.x;
      const dy = y + 0.5 - c.y;
      if (dx * dx + dy * dy <= r * r) solid[y * nx + x] = 1;
    }
  }
};

// Rigid rotation about c (rad per step, counter-clockwise) for the solid cells
// within reach of a body of radius r.
export const setSpin = (mask: Mask, c: Point, r: number, omega: number) => {
  const { nx, ny, solid, wall } = mask;
  const reach = r + 1.5;
  const x0 = Math.max(0, Math.floor(c.x - reach));
  const x1 = Math.min(nx - 1, Math.ceil(c.x + reach));
  const y0 = Math.max(0, Math.floor(c.y - reach));
  const y1 = Math.min(ny - 1, Math.ceil(c.y + reach));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * nx + x;
      if (!solid[idx]) continue;
      const dx = x + 0.5 - c.x;
      const dy = y + 0.5 - c.y;
      if (dx * dx + dy * dy > reach * reach) continue;
      wall[2 * idx] = -omega * dy;
      wall[2 * idx + 1] = omega * dx;
    }
  }
};

export const solidCount = (mask: Mask) => {
  let n = 0;
  for (let k = 0; k < mask.solid.length; k++) n += mask.solid[k];
  return n;
};

// Number of rows holding solid cells away from the tunnel walls: the projected
// height a uniform stream sees, used as the reference length for drawn shapes.
export const frontalHeight = (mask: Mask) => {
  const { nx, ny, solid } = mask;
  let rows = 0;
  for (let y = 1; y < ny - 1; y++) {
    for (let x = 1; x < nx; x++) {
      if (solid[y * nx + x]) {
        rows++;
        break;
      }
    }
  }
  return rows;
};
