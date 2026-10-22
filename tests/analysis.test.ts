// Force post-processing, shedding frequency, reattachment, history and polar sweeps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  History,
  forceCoefficient,
  meanAndRms,
  oscillation,
  reattachmentLength,
  strouhal,
} from '../src/physics/analysis.ts';
import { Sweep, sweepValues } from '../src/physics/sweep.ts';

const close = (a: number, b: number, tol: number, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b} ± ${tol}, got ${a}`);

test('coefficients and Strouhal number', () => {
  close(forceCoefficient(0.065, 0.1, 10), 1.3, 1e-12);
  close(strouhal(0.0005, 40, 0.1), 0.2, 1e-12);
  const { mean, rms } = meanAndRms([1, 3, 1, 3]);
  assert.equal(mean, 2);
  assert.equal(rms, 1);
});

test('recovers the frequency of a noisy sine', () => {
  const t: number[] = [];
  const v: number[] = [];
  let seed = 7;
  const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.2;
  for (let k = 0; k < 4000; k++) {
    const time = k * 0.05;
    t.push(time);
    v.push(0.3 + Math.sin(2 * Math.PI * 0.21 * time) + noise());
  }
  const osc = oscillation(t, v)!;
  close(osc.frequency, 0.21, 0.002);
  close(osc.rms, Math.SQRT1_2, 0.02);
});

test('a steady signal has no oscillation', () => {
  const t = Array.from({ length: 100 }, (_, k) => k);
  assert.equal(oscillation(t, t.map(() => 0.4)), null);
});

test('reattachment length from the near-wall velocity', () => {
  const ux: number[] = Array.from({ length: 100 }, (_, x) => (x < 20 ? 0 : x < 50 ? -0.01 : 0.02));
  ux[50] = 0.01; // crossing between cell 49 (-0.01) and 50 (+0.01), at x = 50
  close(reattachmentLength(ux, 20, 5)!, 6, 1e-9);
  // A corner eddy (positive) and a short reversed blip ahead of the main bubble.
  const withEddy = ux.slice();
  withEddy[21] = -0.001;
  withEddy[22] = 0.002;
  withEddy[60] = -0.001;
  close(reattachmentLength(withEddy, 20, 5)!, 6, 1e-9);
  assert.equal(reattachmentLength(ux.map(Math.abs), 20, 5), null);
});

test('history trims from the front and finds time windows', () => {
  const h = new History(100);
  for (let k = 0; k < 200; k++) h.push(k, k, -k);
  assert.ok(h.length <= 125 && h.length >= 100);
  assert.equal(h.t[h.length - 1], 199);
  assert.equal(h.t[h.indexAfter(150.5)], 151);
});

test('sweep values include both ends', () => {
  assert.deepEqual(sweepValues({ from: -4, to: 4, step: 2 }), [-4, -2, 0, 2, 4]);
  assert.deepEqual(sweepValues({ from: 4, to: 0, step: 2 }), [4, 2, 0]);
});

test('a sweep settles, averages, then moves on', () => {
  const sweep = new Sweep({ from: 0, to: 2, step: 2, settle: 1, average: 1 }, 0);
  assert.equal(sweep.value, 0);
  const changes: number[] = [];
  for (let k = 1; k <= 60; k++) {
    const t = k * 0.1;
    // During settling of each angle the signal is garbage; only averaging counts.
    const cl = sweep.phase === 'average' ? 0.1 * sweep.value + 0.2 : 99;
    const next = sweep.sample(t, cl, 0.05);
    if (next !== null) changes.push(next);
  }
  assert.deepEqual(changes, [2]);
  assert.equal(sweep.phase, 'done');
  assert.equal(sweep.points.length, 2);
  close(sweep.points[0].cl, 0.2, 1e-12);
  close(sweep.points[1].cl, 0.4, 1e-12);
  close(sweep.points[1].cd, 0.05, 1e-12);
  assert.equal(sweep.points[0].st, null);
  assert.equal(sweep.progress(100), 1);
});

test('a sweep reports the shedding frequency as a Strouhal number', () => {
  const sweep = new Sweep({ from: 0, to: 0, step: 1, settle: 1, average: 20 }, 0);
  for (let k = 1; k <= 2200; k++) {
    const t = k * 0.01;
    sweep.sample(t, Math.sin(2 * Math.PI * 0.2 * t), 1);
  }
  assert.equal(sweep.phase, 'done');
  close(sweep.points[0].st!, 0.2, 0.002);
});
