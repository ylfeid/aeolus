// Entry point: WebGPU setup, then the App that ties scenes, controls, pointer
// tools, live force readouts, charts and parameter sweeps to the simulation.
import { Simulation, type ForceSample } from './gpu/sim.ts';
import { Renderer, VIEW_MODES, type ViewMode } from './gpu/renderer.ts';
import { defaultValues, scenes, type Built, type Metric, type Scene, type Study, type Values } from './scenes.ts';
import { clearMask, createMask, frontalHeight, parseNaca, type Mask, type Point } from './physics/geometry.ts';
import { SMAGORINSKY_CS, latticeSpeedFromMach, tauFromReynolds, type WallMode } from './physics/lattice.ts';
import { History, forceCoefficient, meanAndRms, oscillation, reattachmentLength } from './physics/analysis.ts';
import { Sweep, type SweepPoint } from './physics/sweep.ts';
import { Chart, type Series } from './charts.ts';
import { cssGradient, diverging, sequential } from './colormaps.ts';

type Tool = 'move' | 'draw' | 'erase';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const RE_MIN = 10;
const RE_MAX = 20000;
const reFromSlider = (s: number) => RE_MIN * (RE_MAX / RE_MIN) ** (s / 1000);
const sliderFromRe = (re: number) => (1000 * Math.log(re / RE_MIN)) / Math.log(RE_MAX / RE_MIN);
// Below this the BGK relaxation is too close to the stability limit even with LES.
const MIN_TAU = 0.5004;
// Readouts average over this many convective times; St looks further back.
const MEAN_WINDOW = 8;
const ST_WINDOW = 30;
const CHART_WINDOW = 40;
const GRID_WIDTHS = [512, 768, 1024, 1536, 2048];

const METRIC_LABEL: Record<Metric, string> = { cl: 'CL', cd: 'CD', st: 'St' };
const METRIC_COLOR: Record<Metric, string> = { cd: '--series-1', cl: '--series-2', st: '--series-4' };
// The impulsive start rings for a couple of convective times; charts skip it.
const STARTUP = 2;

const fmt = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(digits);

const download = (name: string, blob: Blob) => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

// Replaces the page with the WebGPU message; `reason` adds the specific cause.
const fail = (reason?: string) => {
  $('unsupported').hidden = false;
  document.body.classList.add('no-webgpu');
  const detail = $('unsupported-reason');
  detail.hidden = !reason;
  if (reason) detail.textContent = reason;
};

const pickGridWidth = (params: URLSearchParams) => {
  const requested = Number(params.get('res'));
  if (GRID_WIDTHS.includes(requested)) return requested;
  return window.innerWidth < 900 ? 768 : 1024;
};

const init = async () => {
  if (!navigator.gpu) return fail();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No suitable GPU adapter was found.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fail(`The GPU device was lost: ${info.message}`);
  });

  const params = new URLSearchParams(location.search);
  const nx = pickGridWidth(params);
  const ny = Math.round((nx * 3) / 8 / 8) * 8;
  const sim = new Simulation(device, nx, ny, { u0: 0.1, tau0: 0.6, smagorinsky: SMAGORINSKY_CS, walls: 'slip' });
  const renderer = new Renderer(device, $<HTMLCanvasElement>('sim'), sim);
  const app = new App(device, sim, renderer, params);
  app.start();
  // Handle for poking at the running app from the console during development.
  if (import.meta.env.DEV) Object.assign(window, { aeolus: app });
};

class App {
  private readonly device: GPUDevice;
  private readonly sim: Simulation;
  private readonly renderer: Renderer;
  private readonly params: URLSearchParams;
  private readonly simCanvas = $<HTMLCanvasElement>('sim');
  private readonly overlay = $<HTMLCanvasElement>('overlay');
  private readonly octx: CanvasRenderingContext2D;
  private readonly viewport = $('viewport');
  private readonly forceChart = new Chart($<HTMLCanvasElement>('chart-forces'));
  private readonly sweepChart = new Chart($<HTMLCanvasElement>('chart-sweep'));

  // Scene state
  private scene: Scene = scenes[0];
  private values: Values = {};
  private origin: Point = { x: 0, y: 0 };
  private built: Built = { length: null, lengthLabel: '' };
  private refLength = 0;
  private readonly mask: Mask;
  // Hand-drawn edits on top of the scene: 1 adds a wall, -1 removes one.
  private readonly paint: Int8Array;
  private maskDirty = true;
  private controlInputs = new Map<string, HTMLInputElement>();

  // Flow and display settings
  private re = 160;
  private mach = 0.17;
  private stepsPerFrame = 12;
  private les = true;
  private walls: WallMode = 'slip';
  private view: ViewMode = 'smoke';
  private contrastLog = 0;
  private tracers = false;
  private tool: Tool = 'move';
  private brush = 5;
  private paused = false;

  // Measurements
  private readonly history = new History();
  private tStar = 0;
  private reattachment: number | null = null;
  private sweep: Sweep | null = null;
  private sweepResult: { scene: Scene; study: Study; points: SweepPoint[] } | null = null;
  private stateMessage = '';

  // Interaction and bookkeeping
  private drag: { kind: 'move'; offset: Point } | { kind: 'paint'; value: 1 | -1; last: Point } | null = null;
  private hover: Point | null = null;
  private frame = 0;
  private lastReadout = 0;
  private perf = { t: 0, cells: 0, mlups: 0 };
  private shotRequested = false;
  private urlTimer = 0;
  private stopped = false;

  constructor(device: GPUDevice, sim: Simulation, renderer: Renderer, params: URLSearchParams) {
    this.device = device;
    this.sim = sim;
    this.renderer = renderer;
    this.params = params;
    this.octx = this.overlay.getContext('2d')!;
    this.mask = createMask(sim.nx, sim.ny);
    this.paint = new Int8Array(sim.nx * sim.ny);
  }

  start() {
    const { sim } = this;
    sim.onForce = (s) => this.onForce(s);
    sim.onRow = (y, ux) => this.onRow(y, ux);
    this.device.lost.then(() => (this.stopped = true));

    $('s-grid').textContent = `${sim.nx}×${sim.ny}`;
    this.viewport.style.aspectRatio = `${sim.nx} / ${sim.ny}`;
    this.viewport.style.maxWidth = `calc(68vh * ${sim.nx / sim.ny})`;
    this.viewport.style.marginInline = 'auto';

    this.buildSceneSelect();
    this.bindFlowControls();
    this.bindDisplayControls();
    this.bindStudy();
    this.bindPointer();
    this.bindKeys();
    this.observeSize();

    const ma = Number(this.params.get('ma'));
    if (ma >= 0.05 && ma <= 0.3) this.mach = ma;
    const initial = scenes.find((s) => s.id === this.params.get('scene')) ?? scenes[0];
    this.loadScene(initial, this.params);
    const view = this.params.get('view') as ViewMode | null;
    if (view && VIEW_MODES.includes(view)) this.setView(view);
    requestAnimationFrame(this.tick);
  }

  /* ------------------------------------------------------------------ scenes */

  private buildSceneSelect() {
    const select = $<HTMLSelectElement>('scene-select');
    for (const s of scenes) select.append(new Option(s.name, s.id));
    select.addEventListener('change', () => {
      const s = scenes.find((x) => x.id === select.value);
      if (s) this.loadScene(s);
      select.blur();
    });
  }

  private loadScene(scene: Scene, from?: URLSearchParams) {
    this.abortSweep();
    this.sweepResult = null;
    this.scene = scene;
    this.values = defaultValues(scene);
    for (const c of scene.controls) {
      const raw = from?.get(c.key);
      if (raw === null || raw === undefined) continue;
      if (c.kind === 'naca') {
        if (parseNaca(raw)) this.values[c.key] = raw;
      } else if (Number.isFinite(Number(raw))) {
        this.values[c.key] = Math.min(Math.max(Number(raw), c.min), c.max);
      }
    }
    const re = Number(from?.get('re'));
    this.re = re >= RE_MIN && re <= RE_MAX ? re : scene.re;
    this.origin = { x: scene.origin.x * this.sim.nx, y: scene.origin.y * this.sim.ny };
    this.paint.fill(0);
    this.walls = scene.walls;
    this.reattachment = null;

    $<HTMLSelectElement>('scene-select').value = scene.id;
    $('scene-name').textContent = scene.name;
    $('blurb').textContent = scene.blurb;
    this.buildSceneControls();
    this.syncFlowControls();
    this.contrastLog = scene.contrast ?? 0;
    $<HTMLInputElement>('in-contrast').value = String(this.contrastLog);
    this.setView(scene.view);
    this.setTool(scene.movable ? 'move' : 'draw');
    this.setupStudyPanel();
    this.rebuildMask();
    this.resetFlow();
  }

  private buildSceneControls() {
    const box = $('scene-controls');
    box.replaceChildren();
    this.controlInputs.clear();
    for (const c of this.scene.controls) {
      const label = document.createElement('label');
      const head = document.createElement('span');
      head.textContent = c.label;
      if (c.kind === 'naca') {
        label.className = 'field row';
        const input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 4;
        input.value = String(this.values[c.key]);
        input.spellcheck = false;
        input.autocomplete = 'off';
        input.addEventListener('input', () => {
          const ok = parseNaca(input.value) !== null;
          input.setAttribute('aria-invalid', String(!ok));
          if (ok) this.setValue(c.key, input.value);
        });
        label.append(head, input);
        this.controlInputs.set(c.key, input);
      } else {
        label.className = 'field';
        const out = document.createElement('output');
        head.append(out);
        const input = document.createElement('input');
        input.type = 'range';
        input.min = String(c.min);
        input.max = String(c.max);
        input.step = String(c.step);
        input.value = String(this.values[c.key]);
        const show = () => (out.textContent = this.formatControl(c.key, Number(input.value)));
        input.addEventListener('input', () => {
          show();
          this.setValue(c.key, Number(input.value));
        });
        show();
        label.append(head, input);
        this.controlInputs.set(c.key, input);
      }
      box.append(label);
    }
  }

  private formatControl(key: string, v: number) {
    const c = this.scene.controls.find((x) => x.key === key);
    if (!c || c.kind !== 'range') return String(v);
    const digits = c.step < 0.01 ? 3 : c.step < 0.1 ? 2 : c.step < 1 ? 1 : 0;
    const text = v.toFixed(digits);
    if (!c.unit) return text;
    return c.unit === '°' ? `${text}°` : `${text} ${c.unit}`;
  }

  // Set a scene parameter from code (sweeps, keys) or from its own control.
  private setValue(key: string, value: number | string) {
    if (key === 're') {
      this.re = Number(value);
      this.syncFlowControls();
      this.updateFlow();
      return;
    }
    this.values[key] = value;
    const input = this.controlInputs.get(key);
    if (input && input.value !== String(value)) {
      input.value = String(value);
      input.dispatchEvent(new Event('input'));
      return;
    }
    this.maskDirty = true;
    this.scheduleUrl();
  }

  private rebuildMask() {
    const { mask, paint, sim } = this;
    clearMask(mask);
    this.built = this.scene.build(mask, this.values, this.origin, latticeSpeedFromMach(this.mach));
    for (let k = 0; k < paint.length; k++) {
      if (paint[k] === 1) mask.solid[k] = 1;
      else if (paint[k] === -1) {
        mask.solid[k] = 0;
        mask.wall[2 * k] = mask.wall[2 * k + 1] = 0;
      }
    }
    this.refLength = this.built.length ?? frontalHeight(mask);
    sim.setMask(mask.solid, mask.wall);
    this.maskDirty = false;
    this.updateFlow();
  }

  private resetFlow() {
    this.sim.reset();
    this.history.clear();
    this.tStar = 0;
    this.reattachment = null;
    this.stateMessage = '';
  }

  /* -------------------------------------------------------------------- flow */

  private bindFlowControls() {
    $<HTMLInputElement>('in-re').addEventListener('input', (e) => {
      const re = reFromSlider(Number((e.target as HTMLInputElement).value));
      this.re = re < 100 ? Math.round(re) : Math.round(re / 10) * 10;
      this.syncFlowControls();
      this.updateFlow();
    });
    $<HTMLInputElement>('in-ma').addEventListener('input', (e) => {
      this.mach = Number((e.target as HTMLInputElement).value);
      this.syncFlowControls();
      this.maskDirty = true; // a spinning wall's speed is tied to U
    });
    $<HTMLInputElement>('in-steps').addEventListener('input', (e) => {
      this.stepsPerFrame = Number((e.target as HTMLInputElement).value);
      this.syncFlowControls();
    });
    $<HTMLInputElement>('in-les').addEventListener('change', (e) => {
      this.les = (e.target as HTMLInputElement).checked;
      this.updateFlow();
    });
    for (const b of $('walls').querySelectorAll('button')) {
      b.addEventListener('click', () => {
        this.walls = b.dataset.value as WallMode;
        this.syncFlowControls();
        this.updateFlow();
      });
    }
  }

  private syncFlowControls() {
    $<HTMLInputElement>('in-re').value = String(sliderFromRe(this.re));
    $('out-re').textContent = String(Math.round(this.re));
    $<HTMLInputElement>('in-ma').value = String(this.mach);
    $('out-ma').textContent = this.mach.toFixed(2);
    $<HTMLInputElement>('in-steps').value = String(this.stepsPerFrame);
    $('out-steps').textContent = String(this.stepsPerFrame);
    $<HTMLInputElement>('in-les').checked = this.les;
    for (const b of $('walls').querySelectorAll('button')) b.classList.toggle('active', b.dataset.value === this.walls);
  }

  // Forces are measured whenever there is a test body in the tunnel.
  private get measuring() {
    return this.refLength > 0 && !this.built.noForces;
  }

  private get lengthForRe() {
    return this.refLength > 0 ? this.refLength : this.sim.ny * 0.12;
  }

  private updateFlow() {
    const u0 = latticeSpeedFromMach(this.mach);
    const wanted = tauFromReynolds(this.re, u0, this.lengthForRe);
    const tau0 = Math.max(wanted, MIN_TAU);
    this.sim.settings = { u0, tau0, smagorinsky: this.les ? SMAGORINSKY_CS : 0, walls: this.walls };

    const effectiveRe = (u0 * this.lengthForRe) / ((tau0 - 0.5) / 3);
    $('s-re').textContent = String(Math.round(effectiveRe));
    $('s-ma').textContent = this.mach.toFixed(2);
    $('s-tau').textContent = tau0.toFixed(4);
    let note = '';
    if (wanted < MIN_TAU) note = `Re is capped at ${Math.round(effectiveRe)} for this body size and Mach number.`;
    else if (!this.les && tau0 < 0.52) note = 'τ is close to ½; without LES the run will probably diverge.';
    if (this.mach > 0.25) note += (note ? ' ' : '') + 'Above Ma ≈ 0.25 compressibility errors grow.';
    $('flow-note').textContent = note;
    this.scheduleUrl();
  }

  /* ----------------------------------------------------------------- display */

  private bindDisplayControls() {
    for (const b of $('views').querySelectorAll('button')) {
      b.addEventListener('click', () => this.setView(b.dataset.value as ViewMode));
    }
    $<HTMLInputElement>('in-contrast').addEventListener('input', (e) => {
      this.contrastLog = Number((e.target as HTMLInputElement).value);
      this.updateLegend();
    });
    $<HTMLInputElement>('in-tracers').addEventListener('change', (e) => {
      this.tracers = (e.target as HTMLInputElement).checked;
    });
    for (const b of $('tools').querySelectorAll('button')) {
      b.addEventListener('click', () => this.setTool(b.dataset.value as Tool));
    }
    $<HTMLInputElement>('in-brush').addEventListener('input', (e) => {
      this.brush = Number((e.target as HTMLInputElement).value);
      $('out-brush').textContent = `${this.brush} cells`;
    });
    $('out-brush').textContent = `${this.brush} cells`;
    $('btn-clear').addEventListener('click', () => this.clearPaint());
    $('btn-play').addEventListener('click', () => this.togglePause());
    $('btn-reset').addEventListener('click', () => this.resetFlow());
    $('btn-shot').addEventListener('click', () => (this.shotRequested = true));
    $('btn-csv-forces').addEventListener('click', () => this.exportHistory());
  }

  private setView(mode: ViewMode) {
    this.view = mode;
    for (const b of $('views').querySelectorAll('button')) b.classList.toggle('active', b.dataset.value === mode);
    this.updateLegend();
    this.scheduleUrl();
  }

  private get contrast() {
    return 2 ** this.contrastLog;
  }

  private updateLegend() {
    const k = this.contrast;
    $('out-contrast').textContent = `×${k.toFixed(2)}`;
    const bar = $('legend-bar');
    const set = (label: string, min: string, max: string, background: string) => {
      $('legend-label').textContent = label;
      $('legend-min').textContent = min;
      $('legend-max').textContent = max;
      bar.style.background = background;
      bar.hidden = background === '';
    };
    switch (this.view) {
      case 'smoke':
        set('smoke: inlet rake (white) · surface (orange)', '', '', '');
        break;
      case 'vorticity': {
        // Matches the shader: t = ½ + ½ · ω · H · 0.05 / U · k.
        const w = (20 / k / this.sim.ny) * this.lengthForRe;
        set('ω L/U', `−${fmt(w, 1)}`, `+${fmt(w, 1)}`, cssGradient(diverging));
        break;
      }
      case 'pressure':
        set('Cp', `−${fmt(1.5 / k, 2)}`, `+${fmt(1.5 / k, 2)}`, cssGradient(diverging));
        break;
      default:
        set('|u|/U', '0', fmt(1.6 / k, 2), cssGradient(sequential));
    }
  }

  private setTool(tool: Tool) {
    if (tool === 'move' && !this.scene.movable) tool = 'draw';
    this.tool = tool;
    for (const b of $('tools').querySelectorAll<HTMLButtonElement>('button')) {
      b.classList.toggle('active', b.dataset.value === tool);
      if (b.dataset.value === 'move') b.disabled = !this.scene.movable;
    }
    this.overlay.style.cursor = tool === 'move' ? 'grab' : 'none';
  }

  private clearPaint() {
    this.paint.fill(0);
    this.maskDirty = true;
  }

  private togglePause() {
    this.paused = !this.paused;
    $('btn-play').textContent = this.paused ? 'Play' : 'Pause';
  }

  /* -------------------------------------------------------------- pointer */

  private toGrid(e: PointerEvent): Point {
    const r = this.overlay.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * this.sim.nx,
      y: (1 - (e.clientY - r.top) / r.height) * this.sim.ny,
    };
  }

  private bindPointer() {
    const el = this.overlay;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      const p = this.toGrid(e);
      el.setPointerCapture(e.pointerId);
      const erase = e.button === 2 || this.tool === 'erase';
      if (this.tool === 'move' && !erase) {
        if (this.sweep) return;
        this.drag = { kind: 'move', offset: { x: this.origin.x - p.x, y: this.origin.y - p.y } };
        el.style.cursor = 'grabbing';
      } else {
        this.drag = { kind: 'paint', value: erase ? -1 : 1, last: p };
        this.stamp(p, p, erase ? -1 : 1);
      }
    });
    el.addEventListener('pointermove', (e) => {
      const p = this.toGrid(e);
      this.hover = p;
      const d = this.drag;
      if (!d) return;
      if (d.kind === 'move') {
        const { nx, ny } = this.sim;
        this.origin = {
          x: Math.min(Math.max(p.x + d.offset.x, nx * 0.04), nx * 0.85),
          y: Math.min(Math.max(p.y + d.offset.y, 0), ny),
        };
        this.maskDirty = true;
      } else {
        this.stamp(d.last, p, d.value);
        d.last = p;
      }
    });
    const end = () => {
      if (this.drag?.kind === 'move') this.overlay.style.cursor = 'grab';
      this.drag = null;
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => (this.hover = null));
  }

  // Paint a capsule of brush radius from a to b.
  private stamp(a: Point, b: Point, value: 1 | -1) {
    const { nx, ny } = this.sim;
    const r = this.brush;
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const n = Math.max(1, Math.ceil(len / Math.max(1, r / 2)));
    for (let s = 0; s <= n; s++) {
      const cx = a.x + ((b.x - a.x) * s) / n;
      const cy = a.y + ((b.y - a.y) * s) / n;
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(ny - 1, Math.ceil(cy + r)); y++) {
        for (let x = Math.max(1, Math.floor(cx - r)); x <= Math.min(nx - 1, Math.ceil(cx + r)); x++) {
          if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) this.paint[y * nx + x] = value;
        }
      }
    }
    this.maskDirty = true;
  }

  /* ------------------------------------------------------------------ keys */

  private bindKeys() {
    window.addEventListener('keydown', (e) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' && (target as HTMLInputElement).type !== 'range') return;
      if (target.tagName === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === ' ') this.togglePause();
      else if (key === 'r') this.resetFlow();
      else if (key === 's') this.shotRequested = true;
      else if (key === 'c') this.clearPaint();
      else if (key === 't') {
        this.tracers = !this.tracers;
        $<HTMLInputElement>('in-tracers').checked = this.tracers;
      } else if (key === 'm') this.setTool('move');
      else if (key === 'd') this.setTool('draw');
      else if (key === 'e') this.setTool('erase');
      else if (key >= '1' && key <= '5') this.setView(VIEW_MODES[Number(key) - 1]);
      else if (key === 'arrowup' || key === 'arrowdown') {
        if (!this.nudge(key === 'arrowup' ? 1 : -1, e.shiftKey ? 4 : 1)) return;
      } else return;
      e.preventDefault();
    });
  }

  // Arrow keys step the scene's main angle-like parameter.
  private nudge(dir: number, times: number) {
    const c = this.scene.controls.find(
      (x) => x.kind === 'range' && ['alpha', 'spin', 'angle', 'slant'].includes(x.key),
    );
    if (!c || c.kind !== 'range' || this.sweep?.phase === 'settle' || this.sweep?.phase === 'average') return false;
    const v = Math.min(Math.max(Number(this.values[c.key]) + dir * times * c.step, c.min), c.max);
    this.setValue(c.key, Math.round(v * 1000) / 1000);
    return true;
  }

  /* ------------------------------------------------------------ measuring */

  private onForce({ step, fx, fy }: ForceSample) {
    if (!Number.isFinite(fx) || !Number.isFinite(fy)) return this.diverged();
    if (!this.measuring) return;
    const L = this.refLength;
    const u0 = this.sim.settings.u0;
    const t = this.tStar - ((this.sim.steps - step) * u0) / L;
    const per = this.built.bodies ?? 1;
    const cd = forceCoefficient(fx / per, u0, L);
    const cl = forceCoefficient(fy / per, u0, L);
    this.history.push(t, cd, cl);

    const sweep = this.sweep;
    if (sweep) {
      const next = sweep.sample(t, cl, cd);
      if (next !== null) this.setValue(this.studyKey, next);
      if (sweep.phase === 'done') this.finishSweep();
    }
  }

  private onRow(y: number, ux: Float32Array) {
    for (let x = 0; x < ux.length; x += 7) if (!Number.isFinite(ux[x])) return this.diverged();
    const step = this.built.step;
    if (step && y === 0) this.reattachment = reattachmentLength(ux, step.end, step.height);
  }

  private diverged() {
    this.abortSweep();
    this.resetFlow();
    this.stateMessage = 'The run diverged and was restarted. Lower Re or Mach, or turn LES on.';
  }

  private updateReadouts() {
    const h = this.history;
    const ready = h.length > 4 && this.measuring;
    const now = h.length ? h.t[h.length - 1] : 0;
    const from = h.indexAfter(now - MEAN_WINDOW);
    const cd = ready ? meanAndRms(h.cd, from).mean : NaN;
    const clStats = ready ? meanAndRms(h.cl, from) : { mean: NaN, rms: NaN };
    const osc = ready ? oscillation(h.t, h.cl, { from: h.indexAfter(now - ST_WINDOW), minRms: 2e-3 }) : null;
    $('r-cd').textContent = fmt(cd);
    $('r-cl').textContent = fmt(clStats.mean);
    $('r-clrms').textContent = fmt(clStats.rms);
    $('r-st').textContent = fmt(osc?.frequency);
    $('r-ld').textContent = Math.abs(cd) > 1e-3 ? fmt(clStats.mean / cd, 2) : '–';

    // A theory value for the current setting, or the step's reattachment length.
    const extra = this.extraReadout();
    $('r-extra-wrap').hidden = !extra;
    if (extra) {
      $('r-extra-label').textContent = extra.label;
      $('r-extra').textContent = extra.value;
    }

    const len = this.refLength;
    const per = this.built.bodies ? `, averaged over ${this.built.bodies} bodies` : '';
    $('ref-note').textContent = this.built.noForces
      ? 'A channel has no test body. Re and t* use the step height h.'
      : len > 0
        ? `Per unit span${per}. Reference length: ${this.built.lengthLabel} = ${Math.round(len)} cells. Means over the last ${MEAN_WINDOW} t*.`
        : 'Draw something in the tunnel to measure forces on it.';
    $('s-time').textContent = this.tStar.toFixed(1);
    $('s-state').textContent = this.stateMessage;

    const first = h.indexAfter(Math.max(now - CHART_WINDOW, STARTUP));
    const t = h.t.slice(first);
    this.forceChart.draw({
      series: [
        { label: 'CD', color: METRIC_COLOR.cd, x: t, y: h.cd.slice(first) },
        { label: 'CL', color: METRIC_COLOR.cl, x: t, y: h.cl.slice(first) },
      ],
      xLabel: 't*',
      emptyText: this.measuring ? 'waiting for samples' : 'no test body',
    });
  }

  private extraReadout(): { label: string; value: string } | null {
    if (this.built.step) return { label: 'xᵣ / h', value: fmt(this.reattachment, 2) };
    const theory = this.scene.study?.theory;
    if (!theory) return null;
    const key = this.scene.study!.key;
    const x = key === 're' ? this.re : Number(this.values[key]);
    return { label: `${METRIC_LABEL[theory.metric]} theory`, value: fmt(theory.at(x, this.built)) };
  }

  /* --------------------------------------------------------------- sweeps */

  private get studyKey() {
    return this.scene.study?.key ?? '';
  }

  private bindStudy() {
    $('btn-sweep').addEventListener('click', () => (this.sweep ? this.abortSweep() : this.startSweep()));
    $('btn-csv-sweep').addEventListener('click', () => this.exportSweep());
  }

  private setupStudyPanel() {
    const study = this.scene.study;
    $('panel-study').hidden = !study;
    if (!study) return;
    $('study-title').textContent = `Sweep · ${study.label}`;
    $<HTMLInputElement>('in-sweep-from').value = String(study.from);
    $<HTMLInputElement>('in-sweep-to').value = String(study.to);
    $<HTMLInputElement>('in-sweep-step').value = String(study.step);
    const what = study.series.map((m) => METRIC_LABEL[m]).join(' and ');
    $('sweep-note').textContent =
      `Steps ${study.label} through the range. At each value the flow settles for ${study.settle} t*, ` +
      `then ${what} are averaged over ${study.average} t*.` +
      (study.theory ? ` Dashed: ${study.theory.label}.` : '');
    this.updateSweepUi();
  }

  private startSweep() {
    const study = this.scene.study;
    if (!study) return;
    const read = (id: string) => Number($<HTMLInputElement>(id).value);
    let from = read('in-sweep-from');
    let to = read('in-sweep-to');
    const step = Math.abs(read('in-sweep-step')) || study.step;
    const control = this.scene.controls.find((c) => c.key === study.key);
    const [lo, hi] = control?.kind === 'range' ? [control.min, control.max] : [RE_MIN, RE_MAX];
    from = Math.min(Math.max(from, lo), hi);
    to = Math.min(Math.max(to, lo), hi);
    if (this.paused) this.togglePause();
    this.sweep = new Sweep({ from, to, step, settle: study.settle, average: study.average }, this.tStar);
    this.sweepResult = { scene: this.scene, study, points: this.sweep.points };
    this.setValue(study.key, this.sweep.value);
    this.updateSweepUi();
  }

  private abortSweep() {
    if (!this.sweep) return;
    this.sweep = null;
    this.updateSweepUi();
  }

  private finishSweep() {
    this.sweep = null;
    this.updateSweepUi();
  }

  private updateSweepUi() {
    const running = this.sweep !== null;
    $('btn-sweep').textContent = running ? 'Stop' : 'Run sweep';
    $<HTMLButtonElement>('btn-csv-sweep').disabled = !this.sweepResult?.points.length;
    for (const id of ['in-sweep-from', 'in-sweep-to', 'in-sweep-step']) $<HTMLInputElement>(id).disabled = running;
    const input = this.controlInputs.get(this.studyKey);
    if (input) input.disabled = running;
    $<HTMLInputElement>('in-re').disabled = running && this.studyKey === 're';
    if (!running) $('sweep-progress').textContent = this.sweepResult?.points.length ? 'done' : '';
  }

  private drawSweep() {
    const study = this.scene.study;
    if (!study) return;
    const result = this.sweepResult?.scene === this.scene ? this.sweepResult : null;
    const points = result?.points ?? [];
    const x = points.map((p) => p.value);
    const series: Series[] = study.series.map((m) => ({
      label: METRIC_LABEL[m],
      color: METRIC_COLOR[m],
      x,
      y: points.map((p) => (m === 'st' ? (p.st ?? NaN) : p[m])),
      dots: true,
    }));
    const from = Number($<HTMLInputElement>('in-sweep-from').value);
    const to = Number($<HTMLInputElement>('in-sweep-to').value);
    const lo = Math.min(from, to);
    const hi = Math.max(from, to);
    if (study.theory && hi > lo) {
      const tx: number[] = [];
      const ty: number[] = [];
      for (let k = 0; k <= 60; k++) {
        const v = lo + ((hi - lo) * k) / 60;
        tx.push(v);
        ty.push(study.theory.at(v, this.built));
      }
      series.push({ label: 'theory', color: METRIC_COLOR[study.theory.metric], x: tx, y: ty, dashed: true });
    }
    this.sweepChart.draw({
      series,
      xLabel: study.label,
      xRange: hi > lo ? [lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04] : undefined,
      emptyText: 'run a sweep to fill this in',
    });
    if (this.sweep) {
      const s = this.sweep;
      const phase = s.phase === 'settle' ? 'settling' : 'averaging';
      $('sweep-progress').textContent =
        `${study.label} = ${s.value} · ${phase} · ${Math.round(s.progress(this.tStar) * 100)}%`;
    }
  }

  /* -------------------------------------------------------------- exports */

  private exportHistory() {
    const h = this.history;
    const rows = ['t_star,cd,cl'];
    for (let k = 0; k < h.length; k++) rows.push(`${h.t[k].toFixed(4)},${h.cd[k].toFixed(6)},${h.cl[k].toFixed(6)}`);
    download(`aeolus-${this.scene.id}-forces.csv`, new Blob([rows.join('\n')], { type: 'text/csv' }));
  }

  private exportSweep() {
    const r = this.sweepResult;
    if (!r) return;
    const rows = [`${r.study.key},cl,cd,cl_rms,st`];
    for (const p of r.points) {
      rows.push([p.value, p.cl.toFixed(6), p.cd.toFixed(6), p.clRms.toFixed(6), p.st?.toFixed(5) ?? ''].join(','));
    }
    download(`aeolus-${r.scene.id}-sweep.csv`, new Blob([rows.join('\n')], { type: 'text/csv' }));
  }

  private scheduleUrl() {
    clearTimeout(this.urlTimer);
    this.urlTimer = window.setTimeout(() => {
      const q = new URLSearchParams();
      q.set('scene', this.scene.id);
      for (const c of this.scene.controls) q.set(c.key, String(this.values[c.key]));
      q.set('re', String(Math.round(this.re)));
      if (this.mach !== 0.17) q.set('ma', String(this.mach));
      if (this.view !== this.scene.view) q.set('view', this.view);
      const res = this.params.get('res');
      if (res) q.set('res', res);
      if (`?${q}` !== location.search) history.replaceState(null, '', `?${q}`);
    }, 300);
  }

  /* -------------------------------------------------------------- drawing */

  private observeSize() {
    const resize = () => {
      const r = this.viewport.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const c of [this.simCanvas, this.overlay]) {
        c.width = Math.max(1, Math.round(r.width * dpr));
        c.height = Math.max(1, Math.round(r.height * dpr));
      }
    };
    new ResizeObserver(resize).observe(this.viewport);
    resize();
  }

  private drawOverlay() {
    const { octx: ctx, overlay, sim } = this;
    const sx = overlay.width / sim.nx;
    const sy = overlay.height / sim.ny;
    const X = (x: number) => x * sx;
    const Y = (y: number) => overlay.height - y * sy;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = Math.max(1, sx * 0.6);

    // Rotation arrow for the spinning cylinder.
    const spin = this.built.spin;
    if (spin && spin.ratio !== 0) {
      const r = (spin.radius + 6) * sx;
      const cx = X(spin.centre.x);
      const cy = Y(spin.centre.y);
      const cw = spin.ratio > 0;
      const sweep = Math.min(Math.abs(spin.ratio) / 4, 1) * Math.PI * 1.2 + 0.4;
      const a0 = -Math.PI / 2 - sweep / 2;
      const a1 = a0 + sweep;
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.arc(cx, cy, r, a0, a1);
      ctx.stroke();
      // Arrow head at the end the surface is moving towards.
      const a = cw ? a1 : a0;
      const tip = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
      const dir = cw ? a + Math.PI / 2 : a - Math.PI / 2;
      const s = 4 * sx + 3;
      ctx.beginPath();
      ctx.moveTo(tip.x + s * Math.cos(dir), tip.y + s * Math.sin(dir));
      ctx.lineTo(tip.x + s * 0.7 * Math.cos(dir + 2.2), tip.y + s * 0.7 * Math.sin(dir + 2.2));
      ctx.lineTo(tip.x + s * 0.7 * Math.cos(dir - 2.2), tip.y + s * 0.7 * Math.sin(dir - 2.2));
      ctx.fill();
    }

    // Reattachment marker on the floor behind the step.
    const step = this.built.step;
    if (step && this.reattachment !== null) {
      const x = X(step.end + this.reattachment * step.height);
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.beginPath();
      ctx.moveTo(x, overlay.height);
      ctx.lineTo(x, overlay.height - 14 * sy - 6);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = `${Math.round(11 * Math.min(window.devicePixelRatio || 1, 2))}px ui-monospace, Menlo, monospace`;
      ctx.fillText('xᵣ', x + 4, overlay.height - 14 * sy - 4);
    }

    // Brush outline.
    if (this.hover && this.tool !== 'move') {
      ctx.strokeStyle = this.tool === 'erase' ? 'rgba(255,140,120,0.9)' : 'rgba(255,255,255,0.85)';
      ctx.beginPath();
      ctx.arc(X(this.hover.x), Y(this.hover.y), Math.max(this.brush * sx, 2), 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  private tick = (now: number) => {
    if (this.stopped) return;
    const { sim, device } = this;
    if (this.maskDirty) this.rebuildMask();

    const steps = this.paused ? 0 : this.stepsPerFrame;
    this.frame++;
    if (this.frame % 15 === 0) sim.requestRow(this.built.step && this.frame % 30 === 0 ? 0 : sim.ny >> 1);

    const encoder = device.createCommandEncoder();
    sim.encode(encoder, steps, steps > 0 && this.measuring);
    this.renderer.encode(encoder, { mode: this.view, contrast: this.contrast, tracers: this.tracers }, this.simCanvas.width / sim.nx);
    device.queue.submit([encoder.finish()]);
    sim.collect();
    this.tStar += (steps * sim.settings.u0) / this.lengthForRe;

    if (this.shotRequested) {
      this.shotRequested = false;
      this.simCanvas.toBlob((b) => b && download(`aeolus-${this.scene.id}.png`, b));
    }

    // Throughput, measured on the CPU side over half-second windows.
    this.perf.cells += steps * sim.n;
    if (now - this.perf.t > 500) {
      this.perf.mlups = this.perf.t ? this.perf.cells / ((now - this.perf.t) * 1e3) : 0;
      this.perf.t = now;
      this.perf.cells = 0;
      $('s-mlups').textContent = this.paused ? '–' : this.perf.mlups.toFixed(0);
    }

    this.drawOverlay();
    if (now - this.lastReadout > 120) {
      this.lastReadout = now;
      this.updateReadouts();
      this.drawSweep();
    }
    requestAnimationFrame(this.tick);
  };
}

init().catch((err) => fail(String(err?.message ?? err)));
