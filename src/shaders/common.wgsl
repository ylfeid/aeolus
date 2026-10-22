// Shared by every simulation kernel: the parameter block, D2Q9 constants and
// small helpers. Prepended to each shader source in src/gpu/sim.ts.

struct Params {
  nx: u32,
  ny: u32,
  frame: u32,
  walls: u32,          // 0 = slip, 1 = no-slip
  u0: f32,             // inflow speed, lattice units
  tau0: f32,           // molecular relaxation time
  smag: f32,           // 18√2·Cs², 0 disables the LES term
  spongeStart: f32,    // first column of the outlet sponge
  spongeTau: f32,      // extra relaxation time at the outlet
  dt: f32,             // lattice steps per frame, for dye and tracers
  dyeDecay: f32,
  dyeSpacing: f32,     // rows between smoke streaks
  particleCount: u32,
  perturb: f32,        // transverse kick applied on reset
  dyeWidth: f32,
  layerIn: f32,        // absorbing layer widths in cells (0 = none)
  layerOut: f32,
  layerStrength: f32,  // relaxation rate towards the free stream at full depth
  _pad0: f32,
  _pad1: f32,
}

const CX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const W = array<f32, 9>(4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0);
const OPP = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const MIRROR_Y = array<u32, 9>(0u, 1u, 4u, 3u, 2u, 8u, 7u, 6u, 5u);
const MAX_SPEED = 0.35;

fn cvec(i: u32) -> vec2f {
  return vec2f(f32(CX[i]), f32(CY[i]));
}

fn feq(i: u32, rho: f32, u: vec2f) -> f32 {
  let cu = 3.0 * dot(cvec(i), u);
  return W[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * dot(u, u));
}

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) / 4294967296.0;
}
