/**
 * Moteurs de repli 100 % navigateur.
 *
 * Si le serveur de calcul est injoignable (demo statique, coupure reseau), on
 * garde une simulation vivante en JavaScript scalaire, a resolution reduite.
 * Elle produit exactement le meme `SimFrame` que le serveur : les scenes
 * Three.js n'ont pas une ligne de code specifique. Le bandeau signale
 * clairement le mode degrade — le but est de rester utilisable, pas de
 * pretendre egaler les noyaux AVX2.
 */

import type { FrameHeader, SimFrame } from '../net/protocol';

export interface LocalEngine {
  step(dt: number): SimFrame;
  setParams(params: Record<string, unknown>): void;
  command(message: Record<string, unknown>): void;
  readonly rate: number;
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rnd: () => number): number {
  const u = Math.max(rnd(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}

function header(kind: string, tick: number, time: number, ms: number, shape: number[], stats: Record<string, unknown>): FrameHeader {
  return {
    magic: 'spearvm.sim.v1',
    kind,
    tick,
    time,
    compute_ms: ms,
    dtype: 'f32',
    shape,
    scale: 1,
    stats: { backend: 'js-fallback', ...stats },
  };
}

/* ------------------------------------------------------------------ */
class FlowLocal implements LocalEngine {
  readonly rate = 8;
  private n = 16;
  private hid = 32;
  private w1!: Float32Array;
  private w2!: Float32Array;
  private w3!: Float32Array;
  private pot!: Float32Array;
  private vel!: Float32Array;
  private tick = 0;
  private time = 0;
  private phase = 0;
  private morph = 0.35;
  private turbulence = 1;

  constructor(params: Record<string, unknown>) {
    this.setParams(params);
    this.build(7);
  }

  private build(seed: number): void {
    const rnd = mulberry32(seed * 977 + 13);
    const k = 8;
    this.w1 = new Float32Array(this.hid * k).map(() => gaussian(rnd) * Math.sqrt(2 / k));
    this.w2 = new Float32Array(this.hid * this.hid).map(() => gaussian(rnd) * Math.sqrt(2 / this.hid));
    this.w3 = new Float32Array(3 * this.hid).map(() => gaussian(rnd) * Math.sqrt(1 / this.hid));
    this.pot = new Float32Array(this.n ** 3 * 3);
    this.vel = new Float32Array(this.n ** 3 * 3);
  }

  setParams(params: Record<string, unknown>): void {
    if (typeof params.morph === 'number') this.morph = params.morph;
    if (typeof params.turbulence === 'number') this.turbulence = params.turbulence;
    if (typeof params.seed === 'number') this.build(params.seed);
  }

  command(message: Record<string, unknown>): void {
    if (message.type === 'reseed') this.build(Math.floor(Math.random() * 64) + 1);
  }

  step(dt: number): SimFrame {
    const t0 = performance.now();
    const { n, hid } = this;
    this.tick += 1;
    this.time += dt;
    this.phase += dt * this.morph;
    const st = Math.sin(this.phase);
    const ct = Math.cos(this.phase);
    const f = 2 * Math.PI * this.turbulence;
    const x = new Float32Array(8);
    const h1 = new Float32Array(hid);
    const h2 = new Float32Array(hid);

    let idx = 0;
    for (let i = 0; i < n; i += 1) {
      const cx = (i + 0.5) / n;
      for (let j = 0; j < n; j += 1) {
        const cy = (j + 0.5) / n;
        for (let l = 0; l < n; l += 1) {
          const cz = (l + 0.5) / n;
          x[0] = Math.sin(f * cx); x[1] = Math.cos(f * cx);
          x[2] = Math.sin(f * cy); x[3] = Math.cos(f * cy);
          x[4] = Math.sin(f * cz); x[5] = Math.cos(f * cz);
          x[6] = st; x[7] = ct;
          for (let o = 0; o < hid; o += 1) {
            let s = 0;
            for (let q = 0; q < 8; q += 1) s += x[q] * this.w1[o * 8 + q];
            h1[o] = gelu(s);
          }
          for (let o = 0; o < hid; o += 1) {
            let s = 0;
            for (let q = 0; q < hid; q += 1) s += h1[q] * this.w2[o * hid + q];
            h2[o] = gelu(s);
          }
          for (let o = 0; o < 3; o += 1) {
            let s = 0;
            for (let q = 0; q < hid; q += 1) s += h2[q] * this.w3[o * hid + q];
            this.pot[idx + o] = s;
          }
          idx += 3;
        }
      }
    }

    // rotationnel periodique (differences centrees)
    const at = (i: number, j: number, l: number, c: number) =>
      this.pot[((((i + n) % n) * n + ((j + n) % n)) * n + ((l + n) % n)) * 3 + c];
    const inv = n / 2;
    let p = 0;
    let maxAbs = 1e-6;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        for (let l = 0; l < n; l += 1) {
          const vx = (at(i, j + 1, l, 2) - at(i, j - 1, l, 2)) * inv - (at(i, j, l + 1, 1) - at(i, j, l - 1, 1)) * inv;
          const vy = (at(i, j, l + 1, 0) - at(i, j, l - 1, 0)) * inv - (at(i + 1, j, l, 2) - at(i - 1, j, l, 2)) * inv;
          const vz = (at(i + 1, j, l, 1) - at(i - 1, j, l, 1)) * inv - (at(i, j + 1, l, 0) - at(i, j - 1, l, 0)) * inv;
          this.vel[p] = vx; this.vel[p + 1] = vy; this.vel[p + 2] = vz;
          maxAbs = Math.max(maxAbs, Math.abs(vx), Math.abs(vy), Math.abs(vz));
          p += 3;
        }
      }
    }
    const gain = 1 / maxAbs;
    for (let i = 0; i < this.vel.length; i += 1) this.vel[i] = Math.tanh(this.vel[i] * gain * 1.4);

    const ms = performance.now() - t0;
    return {
      header: header('velocity_grid', this.tick, this.time, ms, [n, n, n, 3], {
        grid: n, hidden: hid, points: n ** 3, phase: Number(this.phase.toFixed(3)),
      }),
      data: this.vel,
      bytes: this.vel.byteLength,
    };
  }
}

/* ------------------------------------------------------------------ */
class WaveLocal implements LocalEngine {
  readonly rate = 24;
  private m = 96;
  private u: Float32Array;
  private prev: Float32Array;
  private next: Float32Array;
  private tick = 0;
  private time = 0;
  private courant = 0.45;
  private damping = 0.0025;
  private stiffness = 0.55;
  private pending: [number, number, number][] = [];

  constructor(params: Record<string, unknown>) {
    this.u = new Float32Array(this.m * this.m);
    this.prev = new Float32Array(this.m * this.m);
    this.next = new Float32Array(this.m * this.m);
    this.setParams(params);
    this.drop(0.5, 0.5, 1, 0.05);
  }

  setParams(params: Record<string, unknown>): void {
    if (typeof params.courant === 'number') this.courant = Math.min(0.7, params.courant);
    if (typeof params.damping === 'number') this.damping = params.damping;
    if (typeof params.stiffness === 'number') this.stiffness = params.stiffness;
  }

  command(message: Record<string, unknown>): void {
    if (message.type === 'pulse') {
      this.pending.push([Number(message.x ?? 0.5), Number(message.y ?? 0.5), Number(message.amp ?? 1)]);
    } else if (message.type === 'clear') {
      this.u.fill(0); this.prev.fill(0);
    }
  }

  private drop(cx: number, cy: number, amp: number, radius = 0.035): void {
    const { m } = this;
    for (let i = 0; i < m; i += 1) {
      const dx = (i + 0.5) / m - cx;
      for (let j = 0; j < m; j += 1) {
        const dy = (j + 0.5) / m - cy;
        this.u[i * m + j] += amp * Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
      }
    }
  }

  step(dt: number): SimFrame {
    const t0 = performance.now();
    this.tick += 1;
    this.time += dt;
    while (this.pending.length) {
      const [x, y, a] = this.pending.pop() as [number, number, number];
      this.drop(x, y, a);
    }
    const { m } = this;
    const c2 = this.courant * this.courant;
    const damp = this.damping;
    const stiff = this.stiffness;
    const invA = Math.max(stiff, 1e-3);
    for (let i = 0; i < m; i += 1) {
      const im = ((i - 1 + m) % m) * m;
      const ip = ((i + 1) % m) * m;
      const ic = i * m;
      for (let j = 0; j < m; j += 1) {
        const jm = (j - 1 + m) % m;
        const jp = (j + 1) % m;
        const u = this.u[ic + j];
        const lap = this.u[im + j] + this.u[ip + j] + this.u[ic + jm] + this.u[ic + jp] - 4 * u;
        let v = (2 - damp) * u - (1 - damp) * this.prev[ic + j] + c2 * lap;
        if (stiff > 0) {
          const a = 1 / invA;
          v = (1 - stiff) * v + stiff * a * Math.tanh(v / a);
        }
        this.next[ic + j] = v * 0.999;
      }
    }
    const tmp = this.prev;
    this.prev = this.u;
    this.u = this.next;
    this.next = tmp;

    let peak = 0;
    for (let i = 0; i < this.u.length; i += 1) peak = Math.max(peak, Math.abs(this.u[i]));
    const ms = performance.now() - t0;
    return {
      header: header('height_grid', this.tick, this.time, ms, [m, m], {
        size: m, peak: Number(peak.toFixed(4)), substeps: 1,
      }),
      data: this.u,
      bytes: this.u.byteLength,
    };
  }
}

export function createLocalEngine(simId: string, params: Record<string, unknown>): LocalEngine | null {
  if (simId === 'flowfield') return new FlowLocal(params);
  if (simId === 'wavefield') return new WaveLocal(params);
  return null; // trainer & kernel lab exigent les noyaux du serveur
}
