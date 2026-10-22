// Draws the tunnel: a full-screen field pass (smoke, vorticity, speed, pressure
// or line-integral-convolution streamlines) and instanced tracer dots on top.
// The colour map functions `diverging` and `sequential` are generated from
// src/colormaps.ts and prepended at load time.

struct View {
  nx: f32,
  ny: f32,
  u0: f32,
  contrast: f32,
  mode: u32,       // 0 smoke, 1 vorticity, 2 speed, 3 pressure, 4 streamlines
  _pad0: u32,
  pointSize: f32,  // tracer diameter, cells
  noiseScale: f32, // LIC noise frequency, per cell
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> fields: array<vec4f>;
@group(0) @binding(2) var<storage, read> dye: array<vec4f>;
@group(0) @binding(3) var<storage, read> particles: array<vec4f>;

const BACKGROUND = vec3f(0.045, 0.047, 0.055);
const SOLID = vec3f(0.80, 0.80, 0.78);

struct FieldOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn fieldVertex(@builtin(vertex_index) vi: u32) -> FieldOut {
  let corner = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: FieldOut;
  out.pos = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  out.uv = corner; // y up: row 0 is the tunnel floor
  return out;
}

fn cell(x: i32, y: i32) -> u32 {
  let nx = i32(V.nx);
  let ny = i32(V.ny);
  return u32(clamp(y, 0, ny - 1) * nx + clamp(x, 0, nx - 1));
}

fn fieldAt(p: vec2f) -> vec4f {
  let b = floor(p);
  let t = p - b;
  let x = i32(b.x);
  let y = i32(b.y);
  return mix(mix(fields[cell(x, y)], fields[cell(x + 1, y)], t.x),
             mix(fields[cell(x, y + 1)], fields[cell(x + 1, y + 1)], t.x), t.y);
}

fn dyeAt(p: vec2f) -> vec4f {
  let b = floor(p);
  let t = p - b;
  let x = i32(b.x);
  let y = i32(b.y);
  return mix(mix(dye[cell(x, y)], dye[cell(x + 1, y)], t.x),
             mix(dye[cell(x, y + 1)], dye[cell(x + 1, y + 1)], t.x), t.y);
}

fn hash2(p: vec2i) -> f32 {
  let h = pcg(u32(p.x) * 1973u + pcg(u32(p.y) * 9277u));
  return f32(h) / 4294967296.0;
}

// Value noise with features about three pixels across, the texture LIC smears
// along streamlines.
fn noise(p: vec2f) -> f32 {
  let q = p * V.noiseScale;
  let b = floor(q);
  let t = smoothstep(vec2f(0.0), vec2f(1.0), q - b);
  let i = vec2i(b);
  return mix(mix(hash2(i), hash2(i + vec2i(1, 0)), t.x),
             mix(hash2(i + vec2i(0, 1)), hash2(i + vec2i(1, 1)), t.x), t.y);
}

fn lic(start: vec2f) -> f32 {
  var sum = noise(start);
  var weight = 1.0;
  for (var dir = -1.0; dir <= 1.0; dir += 2.0) {
    var p = start;
    for (var k = 1; k <= 26; k++) {
      let v = fieldAt(p).yz;
      let s = length(v);
      if (s < 1e-6) { break; }
      p += dir * 0.6 * v / s;
      let w = 1.0 - f32(k) / 27.0;
      sum += noise(p) * w;
      weight += w;
    }
  }
  return sum / weight;
}

@fragment
fn fieldFragment(in: FieldOut) -> @location(0) vec4f {
  let size = vec2f(V.nx, V.ny);
  let p = in.uv * size - 0.5;
  let here = fieldAt(p);
  let k = V.contrast;
  var col = BACKGROUND;

  switch (V.mode) {
    case 0u: {
      let d = dyeAt(p);
      col = mix(col, vec3f(0.93, 0.94, 0.96), 1.0 - exp(-2.4 * k * d.x));
      col = mix(col, vec3f(1.0, 0.6, 0.3), 0.92 * (1.0 - exp(-2.0 * k * d.y)));
    }
    case 1u: {
      let dx = fieldAt(p + vec2f(1.0, 0.0)).yz - fieldAt(p - vec2f(1.0, 0.0)).yz;
      let dy = fieldAt(p + vec2f(0.0, 1.0)).yz - fieldAt(p - vec2f(0.0, 1.0)).yz;
      let omega = 0.5 * (dx.y - dy.x);
      // Scaled by U/H so the look doesn't depend on resolution.
      col = diverging(0.5 + 0.5 * omega * V.ny * 0.05 / V.u0 * k);
    }
    case 2u: {
      col = sequential(length(here.yz) / (1.6 * V.u0) * k);
    }
    case 3u: {
      let cp = (here.x - 1.0) / 3.0 / (0.5 * V.u0 * V.u0);
      col = diverging(0.5 + 0.5 * cp / 1.5 * k);
    }
    default: {
      let l = clamp((lic(p) - 0.5) * 2.6 + 0.5, 0.0, 1.0);
      col = sequential(0.15 + 0.85 * length(here.yz) / (1.6 * V.u0) * k) * (0.3 + 0.85 * l);
    }
  }

  // Solids, antialiased by the bilinear solid fraction.
  let s = smoothstep(0.35, 0.65, here.w);
  col = mix(col, SOLID, s);
  return vec4f(col, 1.0);
}

struct DotOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) alpha: f32,
}

@vertex
fn particleVertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DotOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let p = particles[ii];
  let size = vec2f(V.nx, V.ny);
  let centre = (p.xy + 0.5) / size * 2.0 - 1.0;
  var out: DotOut;
  out.pos = vec4f(centre + c * V.pointSize / size, 0.0, 1.0);
  out.local = c;
  // Fade in after spawning and out before expiring.
  out.alpha = select(0.0, min(p.w / 24.0, 1.0) * min(p.z / 24.0, 1.0), p.z > 0.0);
  return out;
}

@fragment
fn particleFragment(in: DotOut) -> @location(0) vec4f {
  let a = in.alpha * 0.75 * (1.0 - smoothstep(0.5, 1.0, length(in.local)));
  return vec4f(vec3f(0.98, 0.97, 0.92) * a, a);
}
