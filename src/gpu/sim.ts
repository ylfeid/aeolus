// The interactive tunnel: a LatticeCore plus everything drawn on top of it,
// namely macroscopic fields, smoke and tracers, and readback of single rows
// of the field. Encodes one frame as lattice steps → fields → forces → smoke
// → tracers.
import commonSource from '../shaders/common.wgsl?raw';
import fieldsSource from '../shaders/fields.wgsl?raw';
import transportSource from '../shaders/transport.wgsl?raw';
import { LatticeCore, type FlowSettings, type ForceSample } from '../core/lattice-core.ts';

export type { FlowSettings, ForceSample } from '../core/lattice-core.ts';

export class Simulation {
  readonly core: LatticeCore;
  readonly device: GPUDevice;
  readonly nx: number;
  readonly ny: number;
  readonly n: number;
  readonly particleCount: number;
  readonly fields: GPUBuffer;
  readonly particles: GPUBuffer;
  readonly dye: GPUBuffer[];
  dyeIndex = 0;
  frame = 0;
  onRow: (y: number, ux: Float32Array) => void = () => {};

  private readonly dyeScratch: GPUBuffer;
  private readonly rowRead: GPUBuffer;

  private readonly macroPipeline: GPUComputePipeline;
  private readonly advectPipeline: GPUComputePipeline;
  private readonly correctPipeline: GPUComputePipeline;
  private readonly tracerPipeline: GPUComputePipeline;

  private readonly macroGroups: GPUBindGroup[];
  private readonly advectGroups: GPUBindGroup[];
  private readonly correctGroups: GPUBindGroup[];
  private readonly tracerGroup: GPUBindGroup;

  private rowQueued: number | null = null;
  private rowPending = false;

  constructor(device: GPUDevice, nx: number, ny: number, settings: FlowSettings) {
    const core = new LatticeCore(device, nx, ny, settings);
    this.core = core;
    this.device = device;
    this.nx = nx;
    this.ny = ny;
    this.n = core.n;
    this.particleCount = Math.round(this.n / 14);
    const n = this.n;
    const usage = GPUBufferUsage;

    const storage = (bytes: number) =>
      device.createBuffer({ size: bytes, usage: usage.STORAGE | usage.COPY_DST | usage.COPY_SRC });
    this.fields = storage(16 * n);
    this.dye = [storage(16 * n), storage(16 * n)];
    this.dyeScratch = storage(16 * n);
    this.particles = storage(16 * this.particleCount);
    this.rowRead = device.createBuffer({ size: 16 * nx, usage: usage.MAP_READ | usage.COPY_DST });

    const module = (src: string, label: string) =>
      device.createShaderModule({ code: commonSource + '\n' + src, label });
    const fieldsModule = module(fieldsSource, 'fields');
    const transport = module(transportSource, 'transport');
    const pipeline = (m: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint }, label: entryPoint });
    this.macroPipeline = pipeline(fieldsModule, 'macroscopic');
    this.advectPipeline = pipeline(transport, 'advect');
    this.correctPipeline = pipeline(transport, 'correct');
    this.tracerPipeline = pipeline(transport, 'tracers');

    const params = core.params;
    this.macroGroups = core.f.map((buf) =>
      core.bindGroup(this.macroPipeline, { 0: params, 1: buf, 2: core.solid, 4: this.fields }),
    );
    this.advectGroups = this.dye.map((src) =>
      core.bindGroup(this.advectPipeline, { 0: params, 1: this.fields, 2: src, 4: this.dyeScratch }),
    );
    this.correctGroups = this.dye.map((src, k) =>
      core.bindGroup(this.correctPipeline, { 0: params, 1: this.fields, 2: src, 3: this.dyeScratch, 4: this.dye[1 - k] }),
    );
    this.tracerGroup = core.bindGroup(this.tracerPipeline, { 0: params, 1: this.fields, 5: this.particles });
  }

  get settings() {
    return this.core.settings;
  }

  set settings(s: FlowSettings) {
    this.core.settings = s;
  }

  get steps() {
    return this.core.steps;
  }

  set onForce(handler: (s: ForceSample) => void) {
    this.core.onForce = handler;
  }

  setMask(solid: Uint8Array, wall: Float32Array) {
    this.core.setMask(solid, wall);
  }

  // Back to a uniform stream on the next frame.
  reset(perturb = 0.03) {
    this.core.reset(perturb);
  }

  requestRow(y: number) {
    if (!this.rowPending) this.rowQueued = Math.min(Math.max(Math.round(y), 0), this.ny - 1);
  }

  // Encode one frame of simulation work. `steps` may be 0 (paused): the
  // diagnostic passes still run so the display reflects mask edits.
  encode(encoder: GPUCommandEncoder, steps: number, measure: boolean) {
    const { core } = this;
    const dt = Math.max(steps, 0);
    this.frame++;
    core.writeParams((u, f) => {
      u[2] = this.frame;
      f[9] = dt;
      f[10] = 0.9994;
      f[11] = Math.max(8, Math.round(this.ny / 22));
      u[12] = this.particleCount;
      f[14] = Math.max(1, this.ny / 260);
    });
    const { x, y } = core.groups;

    if (core.encodeInit(encoder)) {
      for (const b of [...this.dye, this.dyeScratch, this.particles]) encoder.clearBuffer(b);
    }

    const pass = encoder.beginComputePass();
    core.encodeSteps(pass, steps);

    pass.setPipeline(this.macroPipeline);
    pass.setBindGroup(0, this.macroGroups[core.current]);
    pass.dispatchWorkgroups(x, y);

    const measuring = measure && core.encodeForce(pass);

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

    if (measuring) core.encodeForceCopy(encoder);
    if (this.rowQueued !== null && !this.rowPending) {
      encoder.copyBufferToBuffer(this.fields, 16 * this.nx * this.rowQueued, this.rowRead, 0, 16 * this.nx);
    }
  }

  // Start the readbacks queued by encode(); call right after submit.
  collect() {
    this.core.collect();
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
