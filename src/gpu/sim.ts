// WebGPU side of the solver: buffers, compute pipelines and the per-frame
// command sequence (lattice steps → fields → forces → smoke → tracers), plus
// asynchronous readback of the force sums and of single rows of the field.
import commonSource from '../shaders/common.wgsl?raw';
import lbmSource from '../shaders/lbm.wgsl?raw';
import fieldsSource from '../shaders/fields.wgsl?raw';
import transportSource from '../shaders/transport.wgsl?raw';
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

const PARAM_BYTES = 80;
const SPONGE_FRACTION = 0.08;
const SPONGE_TAU = 1.2;
const LAYER_STRENGTH = 0.05;

export class Simulation {
  readonly device: GPUDevice;
  readonly nx: number;
  readonly ny: number;
  readonly n: number;
  readonly particleCount: number;
  readonly fields: GPUBuffer;
  readonly particles: GPUBuffer;
  readonly dye: GPUBuffer[];
  dyeIndex = 0;

  steps = 0;
  frame = 0;
  settings: FlowSettings;
  onForce: (s: ForceSample) => void = () => {};
  onRow: (y: number, ux: Float32Array) => void = () => {};

  private readonly f: GPUBuffer[];
  private current = 0;
  private readonly solid: GPUBuffer;
  private readonly wall: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly paramData = new ArrayBuffer(PARAM_BYTES);
  private readonly dyeScratch: GPUBuffer;
  private readonly partials: GPUBuffer;
  private readonly forceRead: GPUBuffer;
  private readonly rowRead: GPUBuffer;
  private readonly groups: { x: number; y: number; fx: number; fy: number };

  private readonly initPipeline: GPUComputePipeline;
  private readonly stepPipeline: GPUComputePipeline;
  private readonly macroPipeline: GPUComputePipeline;
  private readonly forcePipeline: GPUComputePipeline;
  private readonly advectPipeline: GPUComputePipeline;
  private readonly correctPipeline: GPUComputePipeline;
  private readonly tracerPipeline: GPUComputePipeline;

  private readonly initGroup: GPUBindGroup;
  private readonly stepGroups: GPUBindGroup[];
  private readonly macroGroups: GPUBindGroup[];
  private readonly forceGroups: GPUBindGroup[];
  private readonly advectGroups: GPUBindGroup[];
  private readonly correctGroups: GPUBindGroup[];
  private readonly tracerGroup: GPUBindGroup;

  private forceQueued = false;
  private forcePending = false;
  private forceStep = 0;
  private rowQueued: number | null = null;
  private rowPending = false;
  private needsInit = true;
  private perturb = 0.03;

  constructor(device: GPUDevice, nx: number, ny: number, settings: FlowSettings) {
    this.device = device;
    this.nx = nx;
    this.ny = ny;
    this.n = nx * ny;
    this.settings = settings;
    this.particleCount = Math.round(this.n / 14);
    const n = this.n;
    // Read here, not at module scope: browsers without WebGPU lack the global,
    // and the import must still succeed so main.ts can say so.
    const usage = GPUBufferUsage;

    const storage = (bytes: number, extra = 0) =>
      device.createBuffer({ size: bytes, usage: usage.STORAGE | usage.COPY_DST | usage.COPY_SRC | extra });
    this.f = [storage(9 * 4 * n), storage(9 * 4 * n)];
    this.solid = storage(4 * n);
    this.wall = storage(8 * n);
    this.fields = storage(16 * n);
    this.dye = [storage(16 * n), storage(16 * n)];
    this.dyeScratch = storage(16 * n);
    this.particles = storage(16 * this.particleCount);
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
    this.rowRead = device.createBuffer({ size: 16 * nx, usage: usage.MAP_READ | usage.COPY_DST });

    const module = (src: string, label: string) =>
      device.createShaderModule({ code: commonSource + '\n' + src, label });
    const lbm = module(lbmSource, 'lbm');
    const fieldsModule = module(fieldsSource, 'fields');
    const transport = module(transportSource, 'transport');
    const pipeline = (m: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint }, label: entryPoint });
    this.initPipeline = pipeline(lbm, 'init');
    this.stepPipeline = pipeline(lbm, 'step');
    this.macroPipeline = pipeline(fieldsModule, 'macroscopic');
    this.forcePipeline = pipeline(fieldsModule, 'force');
    this.advectPipeline = pipeline(transport, 'advect');
    this.correctPipeline = pipeline(transport, 'correct');
    this.tracerPipeline = pipeline(transport, 'tracers');

    const group = (p: GPUComputePipeline, entries: Record<number, GPUBuffer>) =>
      device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: Object.entries(entries).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
      });
    const [fa, fb] = this.f;
    this.initGroup = group(this.initPipeline, { 0: this.params, 3: this.solid, 2: fa });
    this.stepGroups = [
      group(this.stepPipeline, { 0: this.params, 1: fa, 2: fb, 3: this.solid, 4: this.wall }),
      group(this.stepPipeline, { 0: this.params, 1: fb, 2: fa, 3: this.solid, 4: this.wall }),
    ];
    this.macroGroups = this.f.map((buf) =>
      group(this.macroPipeline, { 0: this.params, 1: buf, 2: this.solid, 4: this.fields }),
    );
    this.forceGroups = this.f.map((buf) =>
      group(this.forcePipeline, { 0: this.params, 1: buf, 2: this.solid, 3: this.wall, 5: this.partials }),
    );
    this.advectGroups = this.dye.map((src) =>
      group(this.advectPipeline, { 0: this.params, 1: this.fields, 2: src, 4: this.dyeScratch }),
    );
    this.correctGroups = this.dye.map((src, k) =>
      group(this.correctPipeline, {
        0: this.params,
        1: this.fields,
        2: src,
        3: this.dyeScratch,
        4: this.dye[1 - k],
      }),
    );
    this.tracerGroup = group(this.tracerPipeline, { 0: this.params, 1: this.fields, 5: this.particles });
  }

  setMask(solid: Uint8Array, wall: Float32Array) {
    const words = new Uint32Array(this.n);
    for (let k = 0; k < this.n; k++) words[k] = solid[k];
    this.device.queue.writeBuffer(this.solid, 0, words);
    this.device.queue.writeBuffer(this.wall, 0, wall);
  }

  // Back to a uniform stream on the next frame.
  reset(perturb = 0.03) {
    this.needsInit = true;
    this.perturb = perturb;
    this.steps = 0;
  }

  requestRow(y: number) {
    if (!this.rowPending) this.rowQueued = Math.min(Math.max(Math.round(y), 0), this.ny - 1);
  }

  private writeParams(dt: number) {
    const s = this.settings;
    const u = new Uint32Array(this.paramData);
    const f = new Float32Array(this.paramData);
    u[0] = this.nx;
    u[1] = this.ny;
    u[2] = this.frame;
    u[3] = s.walls === 'slip' ? 0 : 1;
    f[4] = s.u0;
    f[5] = s.tau0;
    f[6] = smagorinskyFactor(s.smagorinsky);
    f[7] = Math.floor(this.nx * (1 - SPONGE_FRACTION));
    f[8] = SPONGE_TAU;
    f[9] = dt;
    f[10] = 0.9994;
    f[11] = Math.max(8, Math.round(this.ny / 22));
    u[12] = this.particleCount;
    f[13] = this.perturb;
    f[14] = Math.max(1, this.ny / 260);
    // Absorbing layers at both ends only make sense for an open test section;
    // a channel (no-slip walls) keeps its developing profile.
    const open = s.walls === 'slip';
    f[15] = open ? Math.round(this.nx * 0.03) : 0;
    f[16] = open ? Math.round(this.nx * SPONGE_FRACTION) : 0;
    f[17] = LAYER_STRENGTH;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  // Encode one frame of simulation work. `steps` may be 0 (paused): the
  // diagnostic passes still run so the display reflects mask edits.
  encode(encoder: GPUCommandEncoder, steps: number, measure: boolean) {
    this.frame++;
    this.writeParams(Math.max(steps, 0));
    const { x, y, fx, fy } = this.groups;

    if (this.needsInit) {
      this.needsInit = false;
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.initPipeline);
      pass.setBindGroup(0, this.initGroup);
      pass.dispatchWorkgroups(x, y);
      pass.end();
      encoder.copyBufferToBuffer(this.f[0], 0, this.f[1], 0, this.f[0].size);
      this.current = 0;
      for (const b of [...this.dye, this.dyeScratch, this.particles]) encoder.clearBuffer(b);
    }

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.stepPipeline);
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, this.stepGroups[this.current]);
      pass.dispatchWorkgroups(x, y);
      this.current = 1 - this.current;
    }
    this.steps += steps;

    pass.setPipeline(this.macroPipeline);
    pass.setBindGroup(0, this.macroGroups[this.current]);
    pass.dispatchWorkgroups(x, y);

    const measuring = measure && !this.forcePending && !this.forceQueued;
    if (measuring) {
      pass.setPipeline(this.forcePipeline);
      pass.setBindGroup(0, this.forceGroups[this.current]);
      pass.dispatchWorkgroups(fx, fy);
    }

    if (steps > 0) {
      pass.setPipeline(this.advectPipeline);
      pass.setBindGroup(0, this.advectGroups[this.dyeIndex]);
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.correctPipeline);
      pass.setBindGroup(0, this.correctGroups[this.dyeIndex]);
      pass.dispatchWorkgroups(x, y);
      this.dyeIndex = 1 - this.dyeIndex;

      pass.setPipeline(this.tracerPipeline);
      pass.setBindGroup(0, this.tracerGroup);
      pass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
    }
    pass.end();

    if (measuring) {
      encoder.copyBufferToBuffer(this.partials, 0, this.forceRead, 0, this.partials.size);
      this.forceQueued = true;
      this.forceStep = this.steps;
    }
    if (this.rowQueued !== null && !this.rowPending) {
      encoder.copyBufferToBuffer(this.fields, 16 * this.nx * this.rowQueued, this.rowRead, 0, 16 * this.nx);
    }
  }

  // Start the readbacks queued by encode(); call right after submit.
  collect() {
    if (this.forceQueued) {
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
    if (this.rowQueued !== null && !this.rowPending) {
      const row = this.rowQueued;
      this.rowQueued = null;
      this.rowPending = true;
      this.rowRead
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const data = new Float32Array(this.rowRead.getMappedRange());
          const ux = new Float32Array(this.nx);
          for (let x = 0; x < this.nx; x++) ux[x] = data[4 * x + 1];
          this.rowRead.unmap();
          this.onRow(row, ux);
        })
        .catch(() => {})
        .finally(() => (this.rowPending = false));
    }
  }
}
