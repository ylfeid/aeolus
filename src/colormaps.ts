// Colour maps shared by the WGSL renderer and the on-screen legend, so the two
// can never disagree. Stops are sRGB in [0, 1] at positions in [0, 1].

export type Stop = [number, number, number, number]; // position, r, g, b

// Dark-centred diverging map: cool negatives, warm positives.
export const diverging: Stop[] = [
  [0.0, 0.62, 0.9, 1.0],
  [0.22, 0.16, 0.5, 0.86],
  [0.42, 0.06, 0.12, 0.22],
  [0.5, 0.045, 0.047, 0.055],
  [0.58, 0.24, 0.07, 0.05],
  [0.78, 0.86, 0.34, 0.13],
  [1.0, 1.0, 0.86, 0.56],
];

// Perceptually ordered sequential map, close to inferno.
export const sequential: Stop[] = [
  [0.0, 0.0, 0.0, 0.016],
  [0.2, 0.26, 0.04, 0.41],
  [0.4, 0.58, 0.15, 0.4],
  [0.6, 0.87, 0.32, 0.23],
  [0.8, 0.99, 0.65, 0.04],
  [1.0, 0.99, 1.0, 0.64],
];

const f = (v: number) => v.toFixed(4);

// A WGSL function `name(t: f32) -> vec3f` that interpolates the stops.
export const wgslColormap = (name: string, stops: Stop[]) => {
  const lines = [`fn ${name}(t_in: f32) -> vec3f {`, '  let t = clamp(t_in, 0.0, 1.0);'];
  for (let k = 1; k < stops.length; k++) {
    const [p0, r0, g0, b0] = stops[k - 1];
    const [p1, r1, g1, b1] = stops[k];
    lines.push(
      `  if (t <= ${f(p1)}) { return mix(vec3f(${f(r0)}, ${f(g0)}, ${f(b0)}), vec3f(${f(r1)}, ${f(g1)}, ${f(b1)}), (t - ${f(p0)}) / ${f(p1 - p0)}); }`,
    );
  }
  const [, r, g, b] = stops[stops.length - 1];
  lines.push(`  return vec3f(${f(r)}, ${f(g)}, ${f(b)});`, '}');
  return lines.join('\n');
};

export const cssGradient = (stops: Stop[]) =>
  `linear-gradient(to right, ${stops
    .map(([p, r, g, b]) => `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)}) ${p * 100}%`)
    .join(', ')})`;
