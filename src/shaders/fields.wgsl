// Per-frame diagnostics from the distributions: macroscopic fields for
// rendering and transport, and the momentum-exchange force on all solids.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> solid: array<u32>;
@group(0) @binding(3) var<storage, read> wall: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> macroOut: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> partials: array<vec4f>;

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

var<workgroup> scratch: array<vec2f, 256>;

// Each post-collision population heading into a solid bounces back next step;
// the body receives c_i (f_i + f_opp) = c_i (2 f_i − 6 w_i c_i·u_wall).
// One partial sum per workgroup, summed on the CPU.
@compute @workgroup_size(16, 16)
fn force(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wg: vec3u,
  @builtin(num_workgroups) groups: vec3u,
) {
  let n = P.nx * P.ny;
  var acc = vec2f(0.0);
  if (id.x > 0u && id.x < P.nx && id.y < P.ny) {
    let idx = id.y * P.nx + id.x;
    if (solid[idx] == 0u) {
      for (var i = 1u; i < 9u; i++) {
        let tx = i32(id.x) + CX[i];
        let ty = i32(id.y) + CY[i];
        if (tx < 0 || tx >= i32(P.nx) || ty < 0 || ty >= i32(P.ny)) { continue; }
        let t = u32(ty) * P.nx + u32(tx);
        if (solid[t] == 0u) { continue; }
        let c = cvec(i);
        acc += c * (2.0 * f[i * n + idx] - 6.0 * W[i] * dot(c, wall[t]));
      }
    }
  }
  scratch[li] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { scratch[li] += scratch[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    partials[wg.y * groups.x + wg.x] = vec4f(scratch[0], 0.0, 0.0);
  }
}
