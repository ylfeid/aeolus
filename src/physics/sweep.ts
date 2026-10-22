// Parameter sweep: step one input (angle of attack, Re, spin ratio, …) through
// a range; at each value let the flow settle, then average the force
// coefficients and measure the shedding frequency. Time is in convective
// units t* = steps · U / L, so a frequency in 1/t* is directly a Strouhal number.
import { oscillation } from './analysis.ts';

export interface SweepConfig {
  from: number;
  to: number;
  step: number;
  settle: number;
  average: number;
}

export interface SweepPoint {
  value: number;
  cl: number;
  cd: number;
  clRms: number;
  st: number | null;
}

export type SweepPhase = 'settle' | 'average' | 'done';

export const sweepValues = ({ from, to, step }: Pick<SweepConfig, 'from' | 'to' | 'step'>) => {
  const out: number[] = [];
  const dir = to >= from ? 1 : -1;
  const s = Math.abs(step) * dir;
  if (s === 0) return [from];
  for (let a = from; dir * (to - a) >= -1e-9; a += s) out.push(Math.round(a * 1e6) / 1e6);
  return out;
};

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

export class Sweep {
  readonly config: SweepConfig;
  readonly values: number[];
  readonly points: SweepPoint[] = [];
  index = 0;
  phase: SweepPhase = 'settle';
  private phaseStart: number;
  private t: number[] = [];
  private cl: number[] = [];
  private cd: number[] = [];

  constructor(config: SweepConfig, now: number) {
    this.config = config;
    this.values = sweepValues(config);
    this.phaseStart = now;
  }

  get value() {
    return this.values[Math.min(this.index, this.values.length - 1)];
  }

  // Overall progress in [0, 1].
  progress(now: number) {
    if (this.phase === 'done') return 1;
    const per = this.config.settle + this.config.average;
    const into =
      this.phase === 'average'
        ? this.config.settle + Math.min(now - this.phaseStart, this.config.average)
        : Math.min(now - this.phaseStart, this.config.settle);
    return Math.min((this.index + into / per) / this.values.length, 1);
  }

  // Feed one measurement. Returns the next value when the sweep moves on.
  sample(now: number, cl: number, cd: number): number | null {
    if (this.phase === 'done') return null;
    if (this.phase === 'settle') {
      if (now - this.phaseStart >= this.config.settle) {
        this.phase = 'average';
        this.phaseStart = now;
      }
      return null;
    }
    this.t.push(now);
    this.cl.push(cl);
    this.cd.push(cd);
    if (now - this.phaseStart < this.config.average) return null;

    const clMean = mean(this.cl);
    const osc = oscillation(this.t, this.cl, { minRms: 2e-3 });
    this.points.push({
      value: this.value,
      cl: clMean,
      cd: mean(this.cd),
      clRms: Math.sqrt(mean(this.cl.map((v) => (v - clMean) ** 2))),
      st: osc ? osc.frequency : null,
    });
    this.t = [];
    this.cl = [];
    this.cd = [];
    this.index++;
    this.phaseStart = now;
    if (this.index >= this.values.length) {
      this.phase = 'done';
      return null;
    }
    this.phase = 'settle';
    return this.value;
  }
}
