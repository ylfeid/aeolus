// Passive transport through the computed velocity field, once per frame:
// smoke (MacCormack advection, injected as streaks at the inlet) and tracer
// particles. Velocities are in cells per lattice step; P.dt steps pass per frame.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fields: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read> mid: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> particles: array<vec4f>;

fn cell(x: i32, y: i32) -> u32 {
  return u32(clamp(y, 0, i32(P.ny) - 1)) * P.nx + u32(clamp(x, 0, i32(P.nx) - 1));
}

// Bilinear velocity at a point in cell-centre coordinates.
fn velocity(p: vec2f) -> vec2f {
  let b = floor(p);
  let t = p - b;
  let x = i32(b.x);
  let y = i32(b.y);
  let a = mix(fields[cell(x, y)].yz, fields[cell(x + 1, y)].yz, t.x);
  let c = mix(fields[cell(x, y + 1)].yz, fields[cell(x + 1, y + 1)].yz, t.x);
  return mix(a, c, t.y);
}

fn sampleSrc(p: vec2f) -> vec4f {
  let b = floor(p);
  let t = p - b;
  let x = i32(b.x);
  let y = i32(b.y);
  return mix(mix(src[cell(x, y)], src[cell(x + 1, y)], t.x),
             mix(src[cell(x, y + 1)], src[cell(x + 1, y + 1)], t.x), t.y);
}

fn sampleMid(p: vec2f) -> vec4f {
  let b = floor(p);
  let t = p - b;
  let x = i32(b.x);
  let y = i32(b.y);
  return mix(mix(mid[cell(x, y)], mid[cell(x + 1, y)], t.x),
             mix(mid[cell(x, y + 1)], mid[cell(x + 1, y + 1)], t.x), t.y);
}

// Midpoint (RK2) characteristic trace over one frame; dir = -1 traces upstream.
fn trace(p: vec2f, dir: f32) -> vec2f {
  let h = dir * P.dt;
  let half = p + 0.5 * h * velocity(p);
  return p + h * velocity(half);
}

// Predictor: plain semi-Lagrangian step into `dst` (the scratch buffer).
@compute @workgroup_size(16, 8)
fn advect(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.nx || id.y >= P.ny) { return; }
  let p = vec2f(id.xy);
  dst[id.y * P.nx + id.x] = sampleSrc(trace(p, -1.0));
}

// Corrector: MacCormack error compensation, clamped to the upstream stencil so
// it can't overshoot, then smoke injection and removal inside solids.
@compute @workgroup_size(16, 8)
fn correct(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.nx || id.y >= P.ny) { return; }
  let idx = id.y * P.nx + id.x;
  let p = vec2f(id.xy);

  let back = trace(p, -1.0);
  let roundTrip = sampleMid(trace(p, 1.0));
  var phi = mid[idx] + 0.5 * (src[idx] - roundTrip);

  let b = floor(back);
  let x = i32(b.x);
  let y = i32(b.y);
  let c0 = src[cell(x, y)];
  let c1 = src[cell(x + 1, y)];
  let c2 = src[cell(x, y + 1)];
  let c3 = src[cell(x + 1, y + 1)];
  phi = clamp(phi, min(min(c0, c1), min(c2, c3)), max(max(c0, c1), max(c2, c3)));
  phi *= P.dyeDecay;

  // Channel x: smoke rake at the inlet, thin streaks every dyeSpacing rows.
  if (id.x < 3u) {
    let k = (f32(id.y) + 0.5 * P.dyeSpacing) % P.dyeSpacing;
    let streak = 1.0 - smoothstep(0.0, P.dyeWidth, abs(k - 0.5 * P.dyeSpacing));
    phi.x = max(phi.x, streak);
  }
  // Channel y: smoke bled from every body surface, which traces boundary
  // layers and their separation.
  let xi = i32(id.x);
  let yi = i32(id.y);
  let touching = fields[cell(xi + 1, yi)].w + fields[cell(xi - 1, yi)].w +
                 fields[cell(xi, yi + 1)].w + fields[cell(xi, yi - 1)].w;
  if (touching > 0.5) { phi.y = 1.0; }
  if (fields[idx].w > 0.5) { phi = vec4f(0.0); }
  dst[idx] = phi;
}

// Tracers: (x, y, frames left, frames lived). Expired or stuck particles
// respawn at a random fluid position.
@compute @workgroup_size(64)
fn tracers(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.particleCount) { return; }
  var p = particles[i];
  let size = vec2f(f32(P.nx), f32(P.ny));

  if (p.z <= 0.0) {
    let seed = pcg(i * 3u + pcg(P.frame));
    p = vec4f(rand01(seed) * (size.x - 2.0), rand01(seed + 1u) * size.y,
              120.0 + rand01(seed + 2u) * 480.0, 0.0);
  } else {
    p = vec4f(trace(p.xy, 1.0), p.z - 1.0, p.w + 1.0);
  }
  let inside = p.x >= 0.0 && p.x < size.x - 1.0 && p.y >= 0.0 && p.y < size.y - 1.0;
  if (!inside || fields[cell(i32(p.x + 0.5), i32(p.y + 0.5))].w > 0.5) {
    p.z = 0.0;
    p.w = 0.0;
  }
  particles[i] = p;
}
