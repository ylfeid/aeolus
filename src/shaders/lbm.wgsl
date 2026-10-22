// One lattice Boltzmann time step: pull streaming with boundary conditions, then
// BGK collision with a Smagorinsky eddy viscosity, an outlet sponge and
// absorbing layers.
// src/physics/lattice.ts (CpuLattice) is the reference for every branch here.
//
// Distributions are stored structure-of-arrays: f[i * n + cell]. The buffer
// always holds post-collision values.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var<storage, read> wall: array<vec2f>;

// Depth-weighted rate at which the absorbing layers pull a cell towards the free
// stream, so pressure waves leave the tunnel instead of ringing in it.
fn absorption(x: f32) -> f32 {
  var r = 0.0;
  if (P.layerIn > 0.0) { r = max(r, 1.0 - x / P.layerIn); }
  if (P.layerOut > 0.0) { r = max(r, (x - (f32(P.nx) - P.layerOut)) / P.layerOut); }
  let d = clamp(r, 0.0, 1.0);
  return P.layerStrength * d * d;
}

@compute @workgroup_size(16, 8)
fn init(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.nx || id.y >= P.ny) { return; }
  let n = P.nx * P.ny;
  let idx = id.y * P.nx + id.x;
  // A small transverse kick so symmetric wakes start shedding quickly.
  let kick = P.perturb * P.u0 * sin(6.2831853 * 4.0 * f32(id.x) / f32(P.nx));
  var u = vec2f(P.u0, kick);
  if (solid[idx] != 0u) { u = vec2f(0.0); }
  for (var i = 0u; i < 9u; i++) { fout[i * n + idx] = feq(i, 1.0, u); }
}

@compute @workgroup_size(16, 8)
fn step(@builtin(global_invocation_id) id: vec3u) {
  let nx = P.nx;
  let ny = P.ny;
  if (id.x >= nx || id.y >= ny) { return; }
  let n = nx * ny;
  let x = i32(id.x);
  let y = i32(id.y);
  let idx = id.y * nx + id.x;

  // Solid cells hold the rest state, so a cell uncovered by a moving body starts calm.
  if (solid[idx] != 0u) {
    for (var i = 0u; i < 9u; i++) { fout[i * n + idx] = W[i]; }
    return;
  }

  // Inlet column: free-stream equilibrium.
  if (x == 0) {
    let u = vec2f(P.u0, 0.0);
    for (var i = 0u; i < 9u; i++) { fout[i * n + idx] = feq(i, 1.0, u); }
    return;
  }

  // ---- Streaming (pull). Clamping sx makes the outlet zero-gradient.
  var f: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    let sx = clamp(x - CX[i], 0, i32(nx) - 1);
    let sy = y - CY[i];
    if (sy < 0 || sy >= i32(ny)) {
      if (P.walls == 0u) {
        // Specular reflection: free-slip tunnel wall.
        f[i] = fin[MIRROR_Y[i] * n + u32(y) * nx + u32(sx)];
      } else {
        // Halfway bounce-back: no-slip wall.
        f[i] = fin[OPP[i] * n + idx];
      }
      continue;
    }
    let src = u32(sy) * nx + u32(sx);
    if (solid[src] != 0u) {
      // Bounce-back off a body, with the moving-wall correction for spinning ones.
      f[i] = fin[OPP[i] * n + idx] + 6.0 * W[i] * dot(cvec(i), wall[src]);
    } else {
      f[i] = fin[i * n + src];
    }
  }

  // ---- Moments
  var rho = 0.0;
  var m = vec2f(0.0);
  for (var i = 0u; i < 9u; i++) {
    rho += f[i];
    m += f[i] * cvec(i);
  }
  var u = m / rho;
  let speed = length(u);
  if (speed > MAX_SPEED) { u *= MAX_SPEED / speed; }

  // ---- Non-equilibrium stress drives the Smagorinsky eddy viscosity
  var fe: array<f32, 9>;
  var pxx = 0.0;
  var pyy = 0.0;
  var pxy = 0.0;
  for (var i = 0u; i < 9u; i++) {
    fe[i] = feq(i, rho, u);
    let d = f[i] - fe[i];
    let c = cvec(i);
    pxx += c.x * c.x * d;
    pyy += c.y * c.y * d;
    pxy += c.x * c.y * d;
  }
  var tau = P.tau0;
  if (P.smag > 0.0) {
    let q = sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy);
    tau = 0.5 * (tau + sqrt(tau * tau + P.smag * q / rho));
  }
  let s = clamp((f32(x) - P.spongeStart) / (f32(nx) - P.spongeStart), 0.0, 1.0);
  tau += P.spongeTau * s * s;

  // ---- BGK collision, then the absorbing layers
  let omega = 1.0 / tau;
  let sigma = absorption(f32(x));
  let freeStream = vec2f(P.u0, 0.0);
  for (var i = 0u; i < 9u; i++) {
    let post = f[i] - omega * (f[i] - fe[i]);
    fout[i * n + idx] = mix(post, feq(i, 1.0, freeStream), sigma);
  }
}
