// Macroscopic fields from the distributions, once per frame, for rendering
// and passive transport.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var<storage, read_write> macroOut: array<vec4f>;

// (ρ, ux, uy, solid) per cell.
@compute @workgroup_size(16, 8)
fn macroscopic(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.nx || id.y >= P.ny) { return; }
  let n = P.nx * P.ny;
  let idx = id.y * P.nx + id.x;
  if (solid[idx] != 0u) {
    macroOut[idx] = vec4f(1.0, 0.0, 0.0, 1.0);
    return;
  }
  var rho = 0.0;
  var m = vec2f(0.0);
  for (var i = 0u; i < 9u; i++) {
    let v = f[i * n + idx];
    rho += v;
    m += v * cvec(i);
  }
  macroOut[idx] = vec4f(rho, m / rho, 0.0);
}
