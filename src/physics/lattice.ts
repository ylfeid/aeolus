// D2Q9 lattice Boltzmann constants, unit conversions and a CPU reference solver.
// CpuLattice mirrors src/shaders/lbm.wgsl step for step, so the tests in tests/
// exercise the same boundary conditions and collision the GPU runs.

export const Q = 9;
export const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1] as const;
export const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1] as const;
export const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36] as const;
export const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6] as const;
// Same direction with cy flipped: specular reflection off a horizontal wall.
export const MIRROR_Y = [0, 1, 4, 3, 2, 8, 7, 6, 5] as const;

export const CS2 = 1 / 3;
export const SMAGORINSKY_CS = 0.16;
// Velocities are clamped to this in the collision so a diverging cell can't take the run down.
export const MAX_SPEED = 0.35;

export type WallMode = 'slip' | 'noslip';

export const equilibrium = (i: number, rho: number, ux: number, uy: number) => {
  const cu = 3 * (CX[i] * ux + CY[i] * uy);
  return W[i] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy));
};

export const latticeSpeedFromMach = (mach: number) => mach * Math.sqrt(CS2);
export const viscosityFromReynolds = (re: number, u: number, length: number) => (u * length) / re;
export const tauFromViscosity = (nu: number) => nu / CS2 + 0.5;
export const tauFromReynolds = (re: number, u: number, length: number) =>
  tauFromViscosity(viscosityFromReynolds(re, u, length));

// Coefficient that multiplies |Π_neq|/ρ under the square root of the Smagorinsky
// relaxation time: τ = ½(τ₀ + √(τ₀² + 18√2·Cs²·|Π_neq|/ρ)).
export const smagorinskyFactor = (cs: number) => 18 * Math.SQRT2 * cs * cs;

export const smagorinskyTau = (tau0: number, rho: number, piNorm: number, factor: number) =>
  factor > 0 ? 0.5 * (tau0 + Math.sqrt(tau0 * tau0 + (factor * piNorm) / rho)) : tau0;

// Extra relaxation time ramped in quadratically over the last columns before the outlet.
export const spongeTau = (x: number, nx: number, start: number, strength: number) => {
  const s = Math.min(Math.max((x - start) / (nx - start), 0), 1);
  return strength * s * s;
};

export interface Layers {
  inlet: number; // widths in cells
  outlet: number;
  strength: number;
}

export const NO_LAYERS: Layers = { inlet: 0, outlet: 0, strength: 0 };

// Rate at which the absorbing layers relax a cell towards the free stream.
export const absorption = (x: number, nx: number, l: Layers) => {
  let r = 0;
  if (l.inlet > 0) r = Math.max(r, 1 - x / l.inlet);
  if (l.outlet > 0) r = Math.max(r, (x - (nx - l.outlet)) / l.outlet);
  const d = Math.min(Math.max(r, 0), 1);
  return l.strength * d * d;
};

export interface LatticeOptions {
  nx: number;
  ny: number;
  u0: number;
  tau0: number;
  smagorinsky?: number;
  walls?: WallMode;
  spongeWidth?: number;
  spongeStrength?: number;
  layers?: Layers;
}

export class CpuLattice {
  readonly nx: number;
  readonly ny: number;
  readonly n: number;
  readonly solid: Uint8Array;
  // Wall velocity per cell (x, y interleaved), used by moving-wall bounce-back.
  readonly wall: Float64Array;
  u0: number;
  tau0: number;
  smagorinsky: number;
  walls: WallMode;
  spongeStart: number;
  spongeStrength: number;
  layers: Layers;
  private f: Float64Array;
  private g: Float64Array;

  constructor(o: LatticeOptions) {
    this.nx = o.nx;
    this.ny = o.ny;
    this.n = o.nx * o.ny;
    this.u0 = o.u0;
    this.tau0 = o.tau0;
    this.smagorinsky = o.smagorinsky ?? 0;
    this.walls = o.walls ?? 'slip';
    this.spongeStart = o.nx - (o.spongeWidth ?? 0);
    this.spongeStrength = o.spongeStrength ?? 0;
    this.layers = o.layers ?? NO_LAYERS;
    this.solid = new Uint8Array(this.n);
    this.wall = new Float64Array(2 * this.n);
    this.f = new Float64Array(Q * this.n);
    this.g = new Float64Array(Q * this.n);
    this.reset();
  }

  reset(perturb = 0) {
    const { nx, n } = this;
    for (let idx = 0; idx < n; idx++) {
      const x = idx % nx;
      const solid = this.solid[idx] !== 0;
      const ux = solid ? 0 : this.u0;
      const uy = solid ? 0 : perturb * this.u0 * Math.sin((2 * Math.PI * 4 * x) / nx);
      for (let i = 0; i < Q; i++) this.f[i * n + idx] = equilibrium(i, 1, ux, uy);
    }
  }

  step() {
    const { nx, ny, n, f, g, solid, wall } = this;
    const factor = smagorinskyFactor(this.smagorinsky);
    const pop = new Float64Array(Q);
    const eq = new Float64Array(Q);

    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = y * nx + x;
        if (solid[idx]) {
          for (let i = 0; i < Q; i++) g[i * n + idx] = W[i];
          continue;
        }
        if (x === 0) {
          for (let i = 0; i < Q; i++) g[i * n + idx] = equilibrium(i, 1, this.u0, 0);
          continue;
        }

        // Pull streaming with boundary handling.
        for (let i = 0; i < Q; i++) {
          const sx = Math.min(Math.max(x - CX[i], 0), nx - 1);
          const sy = y - CY[i];
          if (sy < 0 || sy >= ny) {
            pop[i] = this.walls === 'slip' ? f[MIRROR_Y[i] * n + y * nx + sx] : f[OPP[i] * n + idx];
            continue;
          }
          const src = sy * nx + sx;
          if (solid[src]) {
            const cu = CX[i] * wall[2 * src] + CY[i] * wall[2 * src + 1];
            pop[i] = f[OPP[i] * n + idx] + 6 * W[i] * cu;
          } else {
            pop[i] = f[i * n + src];
          }
        }

        let rho = 0;
        let mx = 0;
        let my = 0;
        for (let i = 0; i < Q; i++) {
          rho += pop[i];
          mx += CX[i] * pop[i];
          my += CY[i] * pop[i];
        }
        let ux = mx / rho;
        let uy = my / rho;
        const speed = Math.hypot(ux, uy);
        if (speed > MAX_SPEED) {
          ux *= MAX_SPEED / speed;
          uy *= MAX_SPEED / speed;
        }

        let pxx = 0;
        let pyy = 0;
        let pxy = 0;
        for (let i = 0; i < Q; i++) {
          eq[i] = equilibrium(i, rho, ux, uy);
          const d = pop[i] - eq[i];
          pxx += CX[i] * CX[i] * d;
          pyy += CY[i] * CY[i] * d;
          pxy += CX[i] * CY[i] * d;
        }
        const piNorm = Math.sqrt(pxx * pxx + pyy * pyy + 2 * pxy * pxy);
        const tau =
          smagorinskyTau(this.tau0, rho, piNorm, factor) +
          spongeTau(x, nx, this.spongeStart, this.spongeStrength);
        const omega = 1 / tau;
        const sigma = absorption(x, nx, this.layers);
        for (let i = 0; i < Q; i++) {
          const post = pop[i] - omega * (pop[i] - eq[i]);
          g[i * n + idx] = post + sigma * (equilibrium(i, 1, this.u0, 0) - post);
        }
      }
    }
    this.f = g;
    this.g = f;
  }

  run(steps: number) {
    for (let s = 0; s < steps; s++) this.step();
  }

  moments(x: number, y: number) {
    const idx = y * this.nx + x;
    let rho = 0;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < Q; i++) {
      const v = this.f[i * this.n + idx];
      rho += v;
      mx += CX[i] * v;
      my += CY[i] * v;
    }
    return { rho, ux: mx / rho, uy: my / rho };
  }

  totalMass() {
    let m = 0;
    for (let k = 0; k < this.f.length; k++) m += this.f[k];
    return m;
  }

  // Momentum exchange on every solid cell; mirrors src/shaders/force.wgsl.
  force(): [number, number] {
    const { nx, ny, n, f, solid, wall } = this;
    let fx = 0;
    let fy = 0;
    for (let y = 0; y < ny; y++) {
      for (let x = 1; x < nx; x++) {
        const idx = y * nx + x;
        if (solid[idx]) continue;
        for (let i = 1; i < Q; i++) {
          const tx = x + CX[i];
          const ty = y + CY[i];
          if (tx < 0 || tx >= nx || ty < 0 || ty >= ny) continue;
          const t = ty * nx + tx;
          if (!solid[t]) continue;
          const cu = CX[i] * wall[2 * t] + CY[i] * wall[2 * t + 1];
          const transfer = 2 * f[i * n + idx] - 6 * W[i] * cu;
          fx += CX[i] * transfer;
          fy += CY[i] * transfer;
        }
      }
    }
    return [fx, fy];
  }
}
