/**
 * Shell applicatif : onglets de cas d'usage, cycle de vie des scenes,
 * connexion au serveur de simulation, GUI de parametres (source de verite =
 * le schema renvoye par le serveur), telemetrie et bascule vers le moteur
 * local si le backend est injoignable.
 */

import GUI from 'lil-gui';
import './style.css';
import { Viewer } from './core/viewer';
import type { SceneApi, SimScene } from './core/types';
import { SimSocket, type HelloMessage, type LinkState, type ParamSpec, type SimDescription } from './net/simSocket';
import { createLocalEngine, type LocalEngine } from './local/engines';
import { FlowFieldScene } from './scenes/flowField';
import { ImplicitFieldScene } from './scenes/implicitField';
import { WaveFieldScene } from './scenes/waveField';
import { TrainerScene } from './scenes/trainer';
import { KernelLabScene } from './scenes/kernelLab';

interface TabDef {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  kernels: string[];
  factory: () => SimScene;
}

const TABS: TabDef[] = [
  {
    id: 'flowfield',
    icon: '◈',
    title: 'Champ de flux neuronal',
    subtitle: 'MLP GELU → potentiel vecteur → rotationnel',
    description:
      "Un MLP evalue sur une grille 3D periodique produit un potentiel vecteur ; son rotationnel donne un champ incompressible ou le GPU advecte les particules.",
    kernels: ['matmul_nt_gelu', 'matmul_nt', 'tanh'],
    factory: () => new FlowFieldScene(),
  },
  {
    id: 'implicit',
    icon: '◉',
    title: 'Champ implicite neuronal',
    subtitle: 'MLP par point → SDF → sphere tracing GPU',
    description:
      "Un MLP evalue sur chaque voxel sculpte une surface implicite ; le GPU la ray-marche. C'est le profil de forme (beaucoup de lignes, k tres court) ou les noyaux SpearVM passent devant OpenBLAS.",
    kernels: ['matmul_nt_gelu', 'matmul_nt'],
    factory: () => new ImplicitFieldScene(),
  },
  {
    id: 'wavefield',
    icon: '≈',
    title: 'Membrane non lineaire',
    subtitle: 'ondes 2D + saturation tanh SPEAR',
    description:
      "Equation des ondes en differences finies ; le noyau tanh AVX2 sature l'amplitude a chaque sous-pas. Cliquez la surface pour injecter une impulsion.",
    kernels: ['tanh'],
    factory: () => new WaveFieldScene(),
  },
  {
    id: 'trainer',
    icon: '↯',
    title: 'Entrainement live',
    subtitle: 'forward fusionne + backprop SpearVM',
    description:
      'Un reseau apprend une surface cible en direct : matmul_nt_gelu en forward, gelu_backward + matmul_backward en backward, Adam pour la mise a jour.',
    kernels: ['matmul_nt_gelu', 'gelu_backward', 'matmul_backward'],
    factory: () => new TrainerScene(),
  },
  {
    id: 'kernellab',
    icon: '⌗',
    title: 'Kernel Lab',
    subtitle: 'debit, precision, GFLOPS mesures en direct',
    description:
      "Banc d'essai execute sur la machine hote : speedup vs numpy, erreur vs IEEE sur [-4,4], GFLOPS matmul et gain de la fusion gelu.",
    kernels: ['tous'],
    factory: () => new KernelLabScene(),
  },
];

/* ------------------------------------------------------------------ */
/* HUD                                                                 */
/* ------------------------------------------------------------------ */
const el = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

function setPill(id: string, value: string, cls: '' | 'ok' | 'warn' | 'bad' = ''): void {
  const pill = el(id);
  const span = pill.querySelector('span');
  if (span) span.textContent = value;
  pill.className = `pill${cls ? ` ${cls}` : ''}`;
}

function toast(message: string, kind: 'ok' | 'warn' | 'error' = 'ok'): void {
  const host = el('toasts');
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  host.appendChild(node);
  window.setTimeout(() => {
    node.style.opacity = '0';
    node.style.transition = 'opacity .35s';
    window.setTimeout(() => node.remove(), 400);
  }, 4200);
}

function renderTelemetry(rows: Record<string, string>): void {
  const dl = el('telemetry');
  dl.innerHTML = Object.entries(rows)
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join('');
}

function humanBytes(bps: number): string {
  if (bps > 1e6) return `${(bps / 1e6).toFixed(2)} Mo/s`;
  if (bps > 1e3) return `${(bps / 1e3).toFixed(0)} ko/s`;
  return `${bps.toFixed(0)} o/s`;
}

/* ------------------------------------------------------------------ */
/* Application                                                         */
/* ------------------------------------------------------------------ */
function assertWebGL2(): boolean {
  // canvas jetable : on ne veut pas figer les attributs du contexte du viewer
  const probe = document.createElement('canvas').getContext('webgl2');
  if (probe) {
    probe.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  }
  el('loader').innerHTML =
    '<div style="max-width:32rem;text-align:center;line-height:1.7">'
    + '<b>WebGL 2 indisponible dans ce navigateur.</b><br>'
    + 'Les simulations utilisent les textures 3D et les render targets flottants de WebGL 2. '
    + "Activez l'acceleration materielle ou essayez un navigateur recent (Chrome, Firefox, Safari 15+).<br>"
    + "L'API de calcul reste accessible sur <code>/api/health</code> et <code>/api/bench</code>."
    + '</div>';
  setPill('pill-link', 'WebGL 2 requis', 'bad');
  return false;
}

class App {
  private viewer = new Viewer(el<HTMLCanvasElement>('viewport'));
  private scene: SimScene | null = null;
  private socket: SimSocket | null = null;
  private gui: GUI | null = null;
  private engine: LocalEngine | null = null;
  private engineTimer: number | null = null;
  private engineLast = 0;
  private descriptions = new Map<string, SimDescription>();
  private serverParams: Record<string, unknown> = {};
  private lastCompute = 0;
  private lastTick = 0;
  private backendLabel = '…';
  private currentTab: TabDef = TABS[0];

  async start(): Promise<void> {
    this.buildTabs();
    this.viewer.start();
    window.setInterval(() => this.refreshTelemetry(), 300);
    await this.loadCatalog();
    await this.activate(TABS[0]);
  }

  /* --------------------------- catalogue --------------------------- */
  private async loadCatalog(): Promise<void> {
    try {
      const res = await fetch('/api/simulations');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = (await res.json()) as {
        backend: Record<string, unknown>;
        simulations: SimDescription[];
      };
      payload.simulations.forEach((sim) => this.descriptions.set(sim.id, sim));
      this.applyBackend(payload.backend);
    } catch {
      setPill('pill-backend', 'hors ligne', 'bad');
      this.backendLabel = 'js-fallback';
      toast('serveur de calcul injoignable — bascule sur le moteur JS local', 'warn');
    }
  }

  private applyBackend(backend: Record<string, unknown>): void {
    const name = String(backend.backend ?? 'inconnu');
    const flags = Array.isArray(backend.cpu_flags) ? (backend.cpu_flags as string[]) : [];
    this.backendLabel = name;
    setPill('pill-backend', `${name}${flags.length ? ` · ${flags.join('/')}` : ''}`, backend.native ? 'ok' : 'warn');
    const sub = el('brand-sub');
    sub.textContent = `Three.js × ${name} · ${String(backend.threads ?? '?')} threads · numpy ${String(backend.numpy ?? '?')}`;
    if (!backend.native) {
      toast('noyaux natifs absents : le serveur calcule en numpy de reference', 'warn');
    }
  }

  /* ----------------------------- onglets --------------------------- */
  private buildTabs(): void {
    const host = el('tabs');
    TABS.forEach((tab) => {
      const button = document.createElement('button');
      button.className = 'tab';
      button.dataset.id = tab.id;
      button.innerHTML = `<i>${tab.icon}</i><span>${tab.title}</span>`;
      button.onclick = () => void this.activate(tab);
      host.appendChild(button);
    });
  }

  private markActiveTab(id: string): void {
    el('tabs')
      .querySelectorAll<HTMLButtonElement>('.tab')
      .forEach((btn) => btn.classList.toggle('active', btn.dataset.id === id));
  }

  /* --------------------------- activation -------------------------- */
  private async activate(tab: TabDef): Promise<void> {
    this.currentTab = tab;
    this.markActiveTab(tab.id);
    this.teardownLink();

    el('scene-title').textContent = tab.title;
    el('scene-desc').textContent = tab.description;
    el('scene-kernels').textContent = `noyaux : ${tab.kernels.join(' · ')}`;
    el('loader').classList.remove('hidden');

    const scene = tab.factory();
    this.scene = scene;
    el('scene-hint').textContent = scene.hint;

    const api: SceneApi = {
      command: (message) => {
        if (this.socket && this.socket.state === 'live') this.socket.command(message);
        else this.engine?.command(message);
      },
      setParams: (params) => this.pushParams(params),
      toast,
    };

    await this.viewer.setScene(scene, api);

    // panneau lateral
    const extra = el('extra-host');
    extra.innerHTML = '';
    scene.buildPanel?.(extra, api);

    if (scene.simId) {
      this.connect(scene.simId);
    } else {
      el('loader').classList.add('hidden');
      setPill('pill-link', 'REST', 'ok');
      this.rebuildGui(null);
    }
  }

  /* ------------------------------ lien ----------------------------- */
  private connect(simId: string): void {
    setPill('pill-link', 'connexion…', 'warn');
    this.socket = new SimSocket(simId, {
      onHello: (msg: HelloMessage) => {
        this.serverParams = { ...msg.params };
        this.descriptions.set(msg.sim.id, msg.sim);
        this.applyBackend(msg.backend);
        this.rebuildGui(msg.sim.params);
        this.stopLocalEngine();
        el('loader').classList.add('hidden');
        setPill('pill-link', `websocket ${msg.rate.toFixed(0)} Hz`, 'ok');
      },
      onFrame: (frame) => {
        this.lastCompute = frame.header.compute_ms;
        this.lastTick = frame.header.tick;
        this.scene?.frame?.(frame);
      },
      onParams: (params) => {
        this.serverParams = { ...this.serverParams, ...params };
      },
      onState: (state, detail) => this.onLinkState(state, detail, simId),
    });
  }

  private onLinkState(state: LinkState, detail: string | undefined, simId: string): void {
    if (state === 'live') {
      setPill('pill-link', 'websocket', 'ok');
      return;
    }
    if (state === 'retrying') {
      setPill('pill-link', `reconnexion… ${detail ?? ''}`.trim(), 'warn');
      return;
    }
    if (state === 'local') {
      setPill('pill-link', 'moteur JS local', 'bad');
      this.startLocalEngine(simId);
    }
  }

  /* ------------------------ moteur de repli ------------------------ */
  private startLocalEngine(simId: string): void {
    if (this.engine) return;
    const engine = createLocalEngine(simId, this.serverParams);
    el('loader').classList.add('hidden');
    if (!engine) {
      toast(`« ${this.currentTab.title} » necessite le serveur de calcul SpearVM`, 'error');
      setPill('pill-backend', 'serveur requis', 'bad');
      return;
    }
    this.engine = engine;
    this.engineLast = performance.now();
    this.rebuildGui(this.descriptions.get(simId)?.params ?? null);
    this.backendLabel = 'js-fallback';
    setPill('pill-backend', 'js-fallback (scalaire)', 'bad');
    toast('serveur injoignable : simulation reprise en JavaScript, resolution reduite', 'warn');
    this.engineTimer = window.setInterval(() => {
      const now = performance.now();
      const dt = Math.min((now - this.engineLast) / 1000, 0.2);
      this.engineLast = now;
      const frame = engine.step(dt);
      this.lastCompute = frame.header.compute_ms;
      this.lastTick = frame.header.tick;
      this.scene?.frame?.(frame);
    }, 1000 / engine.rate);
  }

  private stopLocalEngine(): void {
    if (this.engineTimer !== null) window.clearInterval(this.engineTimer);
    this.engineTimer = null;
    this.engine = null;
  }

  private teardownLink(): void {
    this.socket?.close();
    this.socket = null;
    this.stopLocalEngine();
    this.gui?.destroy();
    this.gui = null;
    this.lastCompute = 0;
    this.lastTick = 0;
  }

  /* ------------------------------- GUI ----------------------------- */
  private pushParams(params: Record<string, unknown>): void {
    this.serverParams = { ...this.serverParams, ...params };
    if (this.socket && this.socket.state === 'live') this.socket.setParams(params);
    else this.engine?.setParams(params);
  }

  private rebuildGui(specs: ParamSpec[] | null): void {
    this.gui?.destroy();
    const gui = new GUI({ container: el('gui-host'), title: 'Parametres' });
    this.gui = gui;

    if (specs?.length) {
      const folder = gui.addFolder('Simulation (serveur SIMD)');
      const state: Record<string, unknown> = {};
      specs.forEach((spec) => {
        state[spec.key] = this.serverParams[spec.key] ?? spec.default;
        let ctrl;
        if (spec.kind === 'range') {
          ctrl = folder
            .add(state, spec.key, spec.min ?? 0, spec.max ?? 1, spec.step ?? 0.01)
            .onFinishChange((value: number) => this.pushParams({ [spec.key]: value }));
        } else if (spec.kind === 'select') {
          ctrl = folder
            .add(state, spec.key, spec.options ?? [])
            .onChange((value: unknown) => this.pushParams({ [spec.key]: value }));
        } else {
          ctrl = folder
            .add(state, spec.key)
            .onChange((value: unknown) => this.pushParams({ [spec.key]: value }));
        }
        ctrl.name(spec.label);
        if (spec.help) ctrl.domElement.title = spec.help;
      });

      const link = { cadence: this.descriptions.get(this.currentTab.id)?.rate ?? 30 };
      folder
        .add(link, 'cadence', 1, 60, 1)
        .name('cadence (Hz)')
        .onFinishChange((value: number) => this.socket?.send({ type: 'rate', value }));
    }

    this.scene?.clientGui?.(gui);
  }

  /* --------------------------- telemetrie -------------------------- */
  private refreshTelemetry(): void {
    setPill('pill-fps', `${this.viewer.fps.toFixed(0)} fps`, this.viewer.fps > 45 ? 'ok' : 'warn');
    setPill(
      'pill-compute',
      this.lastCompute ? `${this.lastCompute.toFixed(1)} ms/tick` : '—',
      this.lastCompute && this.lastCompute < 40 ? 'ok' : '',
    );

    if (this.socket && this.socket.state === 'live') {
      setPill(
        'pill-net',
        `${humanBytes(this.socket.bytesPerSecond)} · ${this.socket.framesPerSecond.toFixed(0)} f/s · ${this.socket.latencyMs.toFixed(0)} ms rtt`,
        'ok',
      );
    } else if (this.engine) {
      setPill('pill-net', 'local (0 octet reseau)', 'warn');
    } else if (!this.scene?.simId) {
      setPill('pill-net', 'REST a la demande', '');
    }

    renderTelemetry({
      backend: this.backendLabel,
      tick: this.lastTick.toLocaleString('fr-FR'),
      'calcul serveur': this.lastCompute ? `${this.lastCompute.toFixed(2)} ms` : '—',
      rendu: `${this.viewer.fps.toFixed(0)} fps`,
      ...(this.scene?.telemetry?.() ?? {}),
    });
  }
}

if (assertWebGL2()) {
  void new App().start();
}
