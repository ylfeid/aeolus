// Draws the simulation into the tunnel canvas: field pass plus tracer dots.
import commonSource from '../shaders/common.wgsl?raw';
import renderSource from '../shaders/render.wgsl?raw';
import { diverging, sequential, wgslColormap } from '../colormaps.ts';
import type { Simulation } from './sim.ts';

export const VIEW_MODES = ['smoke', 'vorticity', 'speed', 'pressure', 'streamlines'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export interface ViewOptions {
  mode: ViewMode;
  contrast: number;
  tracers: boolean;
}

export class Renderer {
  readonly context: GPUCanvasContext;
  private readonly device: GPUDevice;
  private readonly sim: Simulation;
  private readonly uniform: GPUBuffer;
  private readonly data = new ArrayBuffer(32);
  private readonly fieldPipeline: GPURenderPipeline;
  private readonly dotPipeline: GPURenderPipeline;
  private readonly fieldGroups: GPUBindGroup[];
  private readonly dotGroup: GPUBindGroup;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement, sim: Simulation) {
    this.device = device;
    this.sim = sim;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext('webgpu')!;
    this.context.configure({ device, format, alphaMode: 'opaque' });

    const code = [commonSource, wgslColormap('diverging', diverging), wgslColormap('sequential', sequential), renderSource].join('\n');
    const module = device.createShaderModule({ code, label: 'render' });
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.fieldPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'fieldVertex' },
      fragment: { module, entryPoint: 'fieldFragment', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.dotPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'particleVertex' },
      fragment: {
        module,
        entryPoint: 'particleFragment',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.fieldGroups = sim.dye.map((dye) =>
      device.createBindGroup({
        layout: this.fieldPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: sim.fields } },
          { binding: 2, resource: { buffer: dye } },
        ],
      }),
    );
    this.dotGroup = device.createBindGroup({
      layout: this.dotPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 3, resource: { buffer: sim.particles } },
      ],
    });
  }

  encode(encoder: GPUCommandEncoder, view: ViewOptions, pixelsPerCell: number) {
    const f = new Float32Array(this.data);
    const u = new Uint32Array(this.data);
    f[0] = this.sim.nx;
    f[1] = this.sim.ny;
    f[2] = this.sim.settings.u0;
    f[3] = view.contrast;
    u[4] = VIEW_MODES.indexOf(view.mode);
    // Dots stay about 2 px across whatever the zoom.
    f[6] = Math.max(0.6, 2.2 / pixelsPerCell);
    f[7] = Math.min(Math.max(pixelsPerCell / 3, 0.25), 1.5);
    this.device.queue.writeBuffer(this.uniform, 0, this.data);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.fieldPipeline);
    pass.setBindGroup(0, this.fieldGroups[this.sim.dyeIndex]);
    pass.draw(3);
    if (view.tracers) {
      pass.setPipeline(this.dotPipeline);
      pass.setBindGroup(0, this.dotGroup);
      pass.draw(6, this.sim.particleCount);
    }
    pass.end();
  }
}
