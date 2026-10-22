// Post-processing of measured forces and flow fields: coefficients, shedding
// frequency, recirculation length, and a bounded sample history.

export const forceCoefficient = (force: number, u0: number, length: number, rho = 1) =>
  force / (0.5 * rho * u0 * u0 * length);

export const strouhal = (frequency: number, length: number, u0: number) => (frequency * length) / u0;

export const meanAndRms = (values: ArrayLike<number>, from = 0) => {
  const n = values.length - from;
  if (n <= 0) return { mean: NaN, rms: NaN };
  let sum = 0;
  for (let k = from; k < values.length; k++) sum += values[k];
  const mean = sum / n;
  let sq = 0;
  for (let k = from; k < values.length; k++) sq += (values[k] - mean) ** 2;
  return { mean, rms: Math.sqrt(sq / n) };
};

export interface Oscillation {
  frequency: number;
  rms: number;
  periods: number;
}

// Frequency from upward mean crossings, with a hysteresis band of half an rms
// so measurement noise doesn't register as extra crossings.
export const oscillation = (
  t: ArrayLike<number>,
  v: ArrayLike<number>,
  { from = 0, minRms = 1e-3, minPeriods = 2 } = {},
): Oscillation | null => {
  const { mean, rms } = meanAndRms(v, from);
  if (!(rms >= minRms)) return null;
  const band = 0.5 * rms;
  const crossings: number[] = [];
  let armed = false;
  for (let k = from + 1; k < v.length; k++) {
    const a = v[k - 1] - mean;
    const b = v[k] - mean;
    if (b < -band) armed = true;
    if (armed && a < 0 && b >= 0) {
      crossings.push(t[k - 1] + ((t[k] - t[k - 1]) * -a) / (b - a));
      armed = false;
    }
  }
  const periods = crossings.length - 1;
  if (periods < minPeriods) return null;
  return { frequency: periods / (crossings[periods] - crossings[0]), rms, periods };
};

// Distance from the step edge to where the near-wall streamwise velocity turns
// positive again behind the main recirculation bubble, in step heights. The
// main bubble is the longest run of reversed flow; the small counter-rotating
// eddy in the corner and noise further downstream are shorter.
export const reattachmentLength = (ux: ArrayLike<number>, stepEnd: number, height: number) => {
  let best: { from: number; to: number } | null = null;
  let from = -1;
  for (let x = Math.max(0, Math.ceil(stepEnd)); x <= ux.length; x++) {
    const reversed = x < ux.length && ux[x] < 0;
    if (reversed && from < 0) from = x;
    if (!reversed && from >= 0) {
      if (!best || x - from > best.to - best.from) best = { from, to: x };
      from = -1;
    }
  }
  if (!best || best.to >= ux.length) return null;
  const last = best.to - 1;
  const xr = last + 0.5 + ux[last] / (ux[last] - ux[best.to]);
  return (xr - stepEnd) / height;
};

// Parallel arrays of time-stamped samples, trimmed from the front when full.
export class History {
  readonly capacity: number;
  t: number[] = [];
  cd: number[] = [];
  cl: number[] = [];

  constructor(capacity = 6000) {
    this.capacity = capacity;
  }

  push(t: number, cd: number, cl: number) {
    this.t.push(t);
    this.cd.push(cd);
    this.cl.push(cl);
    if (this.t.length > this.capacity * 1.25) {
      const drop = this.t.length - this.capacity;
      this.t.splice(0, drop);
      this.cd.splice(0, drop);
      this.cl.splice(0, drop);
    }
  }

  clear() {
    this.t.length = this.cd.length = this.cl.length = 0;
  }

  get length() {
    return this.t.length;
  }

  // Index of the first sample at or after time `since`.
  indexAfter(since: number) {
    let lo = 0;
    let hi = this.t.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.t[mid] < since) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}
