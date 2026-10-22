// The lattice Boltzmann state and its time stepping on the GPU: distribution
// buffers, obstacle buffers, the parameter block, the step kernel and the
// momentum-exchange force with its readback. Everything that advances or
// measures the flow lives here, so the interactive Simulation and the adjoint
// solver run the same lattice.
import commonSource from '../shaders/common.wgsl?raw';
import lbmSource from '../shaders/lbm.wgsl?raw';
import forceSource from '../shaders/force.wgsl?raw';
import { smagorinskyFactor, type WallMode } from '../physics/lattice.ts';

export interface FlowSettings {
  u0: number;
  tau0: number;
  smagorinsky: number; // Cs; 0 turns LES off
  walls: WallMode;
}

export interface ForceSample {
  step: number;
  fx: number;
  fy: number;
}

// Fills the parameter fields owned by a caller (see Params in common.wgsl).
export type ParamFill = (u: Uint32Array, f: Float32Array) => void;

export const PARAM_BYTES = 80;
const SPONGE_FRACTION = 0.08;
const SPONGE_TAU = 1.2;
const LAYER_STRENGTH = 0.05;

export class LatticeCore {
  readonly device: GPUDevice;
  readonly nx: number;
  readonly ny: number;
  readonly n: number;
  readonly params: GPUBuffer;
  readonly solid: GPUBuffer;
  readonly wall: GPUBuffer;
  // Ping-pong distributions; f[current] holds the latest post-collision state.
  readonly f: GPUBuffer[];
  // Dispatch sizes: 16×8 groups for per-cell kernels, 16×16 for the force reduction.
  readonly groups: { x: number; y: number; fx: number; fy: number };
  current = 0;
  steps = 0;
  settings: FlowSettings;
  onForce: (s: ForceSample) => void = () => {};

  private readonly paramData = new ArrayBuffer(PARAM_BYTES);
  private readonly initPipeline: GPUComputePipeline;
  private readonly stepPipeline: GPUComputePipeline;
  private readonly forcePipeline: GPUComputePipeline;
  private readonly initGroup: GPUBindGroup;
  private readonly stepGroups: GPUBindGroup[];
  private readonly forceGroups: GPUBindGroup[];
  private readonly partials: GPUBuffer;
  private readonly forceRead: GPUBuffer;

  private forceQueued = false;
  private forcePending = false;
  private forceStep = 0;
  private needsInit = true;
  private perturb = 0.03;

  constructor(device: GPUDevice, nx: number, ny: number, settings: FlowSettings) {
    this.device = device;
    this.nx = nx;
    this.ny = ny;
    this.n = nx * ny;
    this.settings = settings;
    const n = this.n;
    // Read here, not at module scope: browsers without WebGPU lack the global,
    // and the import must still succeed so main.ts can say so.
    const usage = GPUBufferUsage;

    const storage = (bytes: number) =>
      device.createBuffer({ size: bytes, usage: usage.STORAGE | usage.COPY_DST | usage.COPY_SRC });
    this.f = [storage(9 * 4 * n), storage(9 * 4 * n)];
    this.solid = storage(4 * n);
    this.wall = storage(8 * n);
    this.params = device.createBuffer({ size: PARAM_BYTES, usage: usage.UNIFORM | usage.COPY_DST });

    this.groups = {
      x: Math.ceil(nx / 16),
      y: Math.ceil(ny / 8),
      fx: Math.ceil(nx / 16),
      fy: Math.ceil(ny / 16),
    };
    const partialBytes = 16 * this.groups.fx * this.groups.fy;
    this.partials = storage(partialBytes);
    this.forceRead = device.createBuffer({ size: partialBytes, usage: usage.MAP_READ | usage.COPY_DST });

    const module = (src: string, label: string) =>
      device.createShaderModule({ code: commonSource + '\n' + src, label });
    const lbm = module(lbmSource, 'lbm');
    const force = module(forceSource, 'force');
    const pipeline = (m: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint }, label: entryPoint });
    this.initPipeline = pipeline(lbm, 'init');
    this.stepPipeline = pipeline(lbm, 'step');
    this.forcePipeline = pipeline(force, 'force');

    const [fa, fb] = this.f;
    this.initGroup = this.bindGroup(this.initPipeline, { 0: this.params, 3: this.solid, 2: fa });
    this.stepGroups = [
      this.bindGroup(this.stepPipeline, { 0: this.params, 1: fa, 2: fb, 3: this.solid, 4: this.wall }),
      this.bindGroup(this.stepPipeline, { 0: this.params, 1: fb, 2: fa, 3: this.solid, 4: this.wall }),
    ];
    this.forceGroups = this.f.map((buf) =>
      this.bindGroup(this.forcePipeline, { 0: this.params, 1: buf, 2: this.solid, 3: this.wall, 5: this.partials }),
    );
  }

  bindGroup(p: GPUComputePipeline, entries: Record<number, GPUBuffer>) {
    return this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: Object.entries(entries).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
    });
  }

  // The latest post-collision distributions.
  get state() {
    return this.f[this.current];
  }

  setMask(solid: Uint8Array, wall: Float32Array) {
    const words = new Uint32Array(this.n);
    for (let k = 0; k < this.n; k++) words[k] = solid[k];
    this.device.queue.writeBuffer(this.solid, 0, words);
    this.device.queue.writeBuffer(this.wall, 0, wall);
  }

  // Back to a uniform stream at the next encodeInit().
  reset(perturb = 0.03) {
    this.needsInit = true;
    this.perturb = perturb;
    this.steps = 0;
  }

  // Write the lattice's parameter fields; `fill` sets the caller's own.
  writeParams(fill?: ParamFill) {
    const s = this.settings;
    const u = new Uint32Array(this.paramData);
    const f = new Float32Array(this.paramData);
    u[0] = this.nx;
    u[1] = this.ny;
    u[3] = s.walls === 'slip' ? 0 : 1;
    f[4] = s.u0;
    f[5] = s.tau0;
    f[6] = smagorinskyFactor(s.smagorinsky);
    f[7] = Math.floor(this.nx * (1 - SPONGE_FRACTION));
    f[8] = SPONGE_TAU;
    f[13] = this.perturb;
    // Absorbing layers at both ends only make sense for an open test section;
    // a channel (no-slip walls) keeps its developing profile.
    const open = s.walls === 'slip';
    f[15] = open ? Math.round(this.nx * 0.03) : 0;
    f[16] = open ? Math.round(this.nx * SPONGE_FRACTION) : 0;
    f[17] = LAYER_STRENGTH;
    fill?.(u, f);
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  // Encode a pending reset. Returns true when it did, so callers can clear
  // state of their own.
  encodeInit(encoder: GPUCommandEncoder) {
    if (!this.needsInit) return false;
    this.needsInit = false;
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.initGroup);
    pass.dispatchWorkgroups(this.groups.x, this.groups.y);
    pass.end();
    encoder.copyBufferToBuffer(this.f[0], 0, this.f[1], 0, this.f[0].size);
    this.current = 0;
    return true;
  }

  encodeSteps(pass: GPUComputePassEncoder, steps: number) {
    pass.setPipeline(this.stepPipeline);
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, this.stepGroups[this.current]);
      pass.dispatchWorkgroups(this.groups.x, this.groups.y);
      this.current = 1 - this.current;
    }
    this.steps += steps;
  }

  // Encode the force reduction unless a readback is still in flight. Returns
  // whether it did; if so, call encodeForceCopy() after the pass ends.
  encodeForce(pass: GPUComputePassEncoder) {
    if (this.forcePending || this.forceQueued) return false;
    pass.setPipeline(this.forcePipeline);
    pass.setBindGroup(0, this.forceGroups[this.current]);
    pass.dispatchWorkgroups(this.groups.fx, this.groups.fy);
    return true;
  }

  encodeForceCopy(encoder: GPUCommandEncoder) {
    encoder.copyBufferToBuffer(this.partials, 0, this.forceRead, 0, this.partials.size);
    this.forceQueued = true;
    this.forceStep = this.steps;
  }

  // Start the force readback queued by encodeForceCopy(); call right after submit.
  collect() {
    if (!this.forceQueued) return;
    this.forceQueued = false;
    this.forcePending = true;
    const step = this.forceStep;
    this.forceRead
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const data = new Float32Array(this.forceRead.getMappedRange());
        let fx = 0;
        let fy = 0;
        for (let k = 0; k < data.length; k += 4) {
          fx += data[k];
          fy += data[k + 1];
        }
        this.forceRead.unmap();
        this.onForce({ step, fx, fy });
      })
      .catch(() => {})
      .finally(() => (this.forcePending = false));
  }
}
