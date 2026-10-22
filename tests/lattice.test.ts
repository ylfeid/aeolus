// Checks the D2Q9 constants and the CPU reference solver against analytic results.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CX,
  CY,
  CpuLattice,
  MIRROR_Y,
  OPP,
  Q,
  W,
  equilibrium,
  latticeSpeedFromMach,
  smagorinskyFactor,
  smagorinskyTau,
  spongeTau,
  tauFromReynolds,
} from '../src/physics/lattice.ts';
import { createMask, fillCircle } from '../src/physics/geometry.ts';

const close = (a: number, b: number, tol: number, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b} ± ${tol}, got ${a}`);

test('weights sum to one and directions pair up', () => {
  close(W.reduce((a, b) => a + b, 0), 1, 1e-15);
  for (let i = 0; i < Q; i++) {
    assert.equal(CX[OPP[i]] + CX[i], 0);
    assert.equal(CY[OPP[i]] + CY[i], 0);
    assert.equal(CX[MIRROR_Y[i]], CX[i]);
    assert.equal(CY[MIRROR_Y[i]] + CY[i], 0);
  }
});

test('equilibrium reproduces density, momentum and the Euler stress', () => {
  const rho = 1.07;
  const ux = 0.08;
  const uy = -0.05;
  let m0 = 0;
  let mx = 0;
  let my = 0;
  let pxx = 0;
  let pxy = 0;
  for (let i = 0; i < Q; i++) {
    const f = equilibrium(i, rho, ux, uy);
    m0 += f;
    mx += CX[i] * f;
    my += CY[i] * f;
    pxx += CX[i] * CX[i] * f;
    pxy += CX[i] * CY[i] * f;
  }
  close(m0, rho, 1e-14);
  close(mx, rho * ux, 1e-14);
  close(my, rho * uy, 1e-14);
  close(pxx, rho / 3 + rho * ux * ux, 1e-14);
  close(pxy, rho * ux * uy, 1e-14);
});

test('unit conversions', () => {
  close(tauFromReynolds(100, 0.1, 50), 0.65, 1e-12);
  close(latticeSpeedFromMach(Math.sqrt(3) * 0.1), 0.1, 1e-12);
  assert.equal(smagorinskyTau(0.6, 1, 0.5, 0), 0.6);
  assert.ok(smagorinskyTau(0.51, 1, 1e-3, smagorinskyFactor(0.16)) > 0.51);
  assert.equal(spongeTau(10, 100, 80, 1), 0);
  close(spongeTau(100, 100, 80, 1), 1, 1e-12);
});

test('a uniform stream through an empty tunnel stays uniform', () => {
  const lat = new CpuLattice({ nx: 40, ny: 20, u0: 0.1, tau0: 0.6, walls: 'slip' });
  lat.run(200);
  for (const [x, y] of [[1, 0], [20, 10], [39, 19], [5, 19]]) {
    const { rho, ux, uy } = lat.moments(x, y);
    close(rho, 1, 1e-9);
    close(ux, 0.1, 1e-9);
    close(uy, 0, 1e-9);
  }
});

test('absorbing layers leave the free stream alone and damp a pressure pulse', () => {
  const layers = { inlet: 12, outlet: 20, strength: 0.05 };
  const quiet = new CpuLattice({ nx: 60, ny: 30, u0: 0.08, tau0: 0.52, layers });
  quiet.run(100);
  close(quiet.moments(30, 15).ux, 0.08, 1e-9);
  close(quiet.moments(2, 2).ux, 0.08, 1e-9);

  // A plane pressure pulse running along a slender tunnel: with the layers it
  // leaves through the ends instead of bouncing back and forth.
  const ringing = (l?: typeof layers) => {
    const nx = 160;
    const ny = 8;
    const lat = new CpuLattice({ nx, ny, u0: 0.05, tau0: 0.505, layers: l });
    lat.reset();
    const f = (lat as unknown as { f: Float64Array }).f;
    for (let y = 0; y < ny; y++) for (let i = 0; i < Q; i++) f[i * lat.n + y * nx + 80] *= 1.05;
    lat.run(800);
    let dev = 0;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) dev += Math.abs(lat.moments(x, y).rho - 1);
    return dev;
  };
  assert.ok(ringing(layers) < 0.5 * ringing(), 'layers absorb sound');
});

test('fluid at rest in a closed-off tunnel keeps its mass', () => {
  const lat = new CpuLattice({ nx: 30, ny: 16, u0: 0, tau0: 0.8, walls: 'noslip' });
  for (let y = 0; y < 16; y++) lat.solid[y * 30 + 29] = 1;
  lat.reset();
  const m0 = lat.totalMass();
  lat.run(300);
  close(lat.totalMass(), m0, 1e-9);
});

test('channel flow develops the Poiseuille profile', () => {
  const ny = 15;
  const lat = new CpuLattice({ nx: 160, ny, u0: 0.05, tau0: 0.8, walls: 'noslip' });
  lat.run(9000);
  const x = 110;
  const profile = Array.from({ length: ny }, (_, y) => lat.moments(x, y).ux);
  const umax = Math.max(...profile);
  const mean = profile.reduce((a, b) => a + b, 0) / ny;
  // Halfway bounce-back puts the walls at the cell faces y = 0 and y = H.
  for (let y = 0; y < ny; y++) {
    const yc = y + 0.5;
    const shape = (4 * yc * (ny - yc)) / (ny * ny);
    close(profile[y] / umax, shape / (4 * (ny / 2) ** 2 / (ny * ny)), 0.02, `row ${y}`);
  }
  close(umax / mean, 1.5, 0.03, 'peak / mean');
});

test('a cylinder feels drag, and spinning it clockwise lifts it (Magnus)', () => {
  const nx = 140;
  const ny = 60;
  const r = 6;
  const u0 = 0.06;
  const run = (spin: number) => {
    const lat = new CpuLattice({ nx, ny, u0, tau0: tauFromReynolds(20, u0, 2 * r), walls: 'slip' });
    const mask = createMask(nx, ny);
    fillCircle(mask, { x: 40, y: ny / 2 }, r);
    lat.solid.set(mask.solid);
    const omega = (-spin * u0) / r;
    for (let idx = 0; idx < nx * ny; idx++) {
      if (!lat.solid[idx]) continue;
      const dx = (idx % nx) + 0.5 - 40;
      const dy = Math.floor(idx / nx) + 0.5 - ny / 2;
      lat.wall[2 * idx] = -omega * dy;
      lat.wall[2 * idx + 1] = omega * dx;
    }
    lat.reset();
    lat.run(2500);
    return lat.force();
  };
  const [fx, fy] = run(0);
  assert.ok(fx > 0, `drag ${fx}`);
  assert.ok(Math.abs(fy) < 0.02 * fx, `symmetric lift ${fy}`);
  const [, spunLift] = run(1.5);
  assert.ok(spunLift > 0.3 * fx, `Magnus lift ${spunLift} vs drag ${fx}`);
});
