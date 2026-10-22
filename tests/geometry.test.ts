// NACA geometry, thin-airfoil theory and mask rasterization.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  createMask,
  fillCircle,
  fillPolygon,
  frontalHeight,
  nacaCamber,
  nacaOutline,
  nacaThickness,
  parseNaca,
  place,
  rectangle,
  setSpin,
  solidCount,
  thinAirfoilLift,
  zeroLiftAngle,
} from '../src/physics/geometry.ts';

const close = (a: number, b: number, tol: number, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b} ± ${tol}, got ${a}`);
const deg = (r: number) => (r * 180) / Math.PI;

test('parses 4-digit codes and rejects nonsense', () => {
  assert.deepEqual(parseNaca('2412'), { code: '2412', m: 0.02, p: 0.4, t: 0.12 });
  assert.equal(parseNaca('0000'), null);
  assert.equal(parseNaca('2012'), null);
  assert.equal(parseNaca('241'), null);
  assert.equal(parseNaca('abcd'), null);
});

test('NACA 0012 is symmetric with 12% thickness near 30% chord', () => {
  const naca = parseNaca('0012')!;
  let best = 0;
  let at = 0;
  for (let x = 0; x <= 1; x += 0.001) {
    const t = 2 * nacaThickness(naca.t, x);
    if (t > best) [best, at] = [t, x];
  }
  close(best, 0.12, 0.0005);
  close(at, 0.3, 0.01);
  close(nacaThickness(naca.t, 1), 0, 1e-12, 'closed trailing edge');
  const pts = nacaOutline(naca, 80);
  const n = pts.length;
  for (let k = 0; k < n; k++) close(pts[k].y, -pts[n - 1 - k].y, 1e-12);
});

test('camber line peaks at (p, m) with zero slope there', () => {
  const naca = parseNaca('4415')!;
  close(nacaCamber(naca, 0.4).yc, 0.04, 1e-12);
  close(nacaCamber(naca, 0.4).slope, 0, 1e-12);
  close(nacaCamber(naca, 1).yc, 0, 1e-12);
});

test('zero-lift angles match thin airfoil theory', () => {
  close(deg(zeroLiftAngle(parseNaca('0012')!)), 0, 1e-9);
  close(deg(zeroLiftAngle(parseNaca('2412')!)), -2.077, 0.01);
  close(deg(zeroLiftAngle(parseNaca('4412')!)), -4.154, 0.02);
  close(thinAirfoilLift(0, 0), 0, 0);
  close(thinAirfoilLift((5 * Math.PI) / 180, 0), 0.548, 0.001);
});

test('rasterized circle area is close to πr²', () => {
  const mask = createMask(100, 100);
  fillCircle(mask, { x: 50, y: 50 }, 20);
  close(solidCount(mask) / (Math.PI * 400), 1, 0.01);
  assert.equal(frontalHeight(mask), 40);
});

test('polygon fill covers exactly the cells whose centres are inside', () => {
  const mask = createMask(40, 40);
  fillPolygon(mask, place(rectangle(10, 6), { at: { x: 20, y: 20 } }));
  assert.equal(solidCount(mask), 60);
  assert.equal(mask.solid[17 * 40 + 15], 1);
  assert.equal(mask.solid[17 * 40 + 14], 0);
});

test('place rotates counter-clockwise about the pivot', () => {
  const [p] = place([{ x: 1, y: 0 }], { at: { x: 10, y: 10 }, angle: Math.PI / 2, scale: 2 });
  close(p.x, 10, 1e-12);
  close(p.y, 12, 1e-12);
});

test('spin gives the surface a tangential velocity ωr', () => {
  const mask = createMask(60, 60);
  const c = { x: 30, y: 30 };
  fillCircle(mask, c, 10);
  setSpin(mask, c, 10, 0.01);
  const idx = 39 * 60 + 30; // top of the cylinder
  close(mask.wall[2 * idx], -0.01 * 9.5, 1e-7);
  close(mask.wall[2 * idx + 1], 0.01 * 0.5, 1e-7);
});
