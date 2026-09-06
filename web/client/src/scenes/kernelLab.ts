/**
 * Scene 4 — Kernel Lab.
 *
 * Scene autonome (pas de WebSocket) : elle interroge `/api/bench`, qui mesure
 * sur la machine hote le debit des noyaux, leur erreur vs IEEE, les GFLOPS
 * matmul et le gain de la fusion `gelu(A.B^T)`. Le resultat est rendu en 3D :
 * barres de speedup, murs de courbes d'erreur (echelle log), barres GFLOPS.
 */

import * as THREE from 'three';
import type GUI from 'lil-gui';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

interface ElementwiseEntry {
  key: string;
  label: string;
  n: number;
  spear_ms: number;
  baseline_ms: number;
  baseline_label: string;
  speedup: number | null;
  mele_per_s: number;
  accuracy: { linf: number; rmse: number; domain: [number, number] };
  curve: { x: number[]; y: number[]; err: number[] };
}

interface MatmulEntry {
  key: string;
  label: string;
  dtype: string;
  spear_gflops: number;
  baseline_gflops: number;
  speedup: number | null;
  max_abs_err: number;
}

interface BenchReport {
  elapsed_ms: number;
  quick: boolean;
  cached?: boolean;
  method?: string;
  noisy_host?: boolean;
  host: Record<string, unknown>;
  elementwise: ElementwiseEntry[];
  matmul: MatmulEntry[];
  fused_ffn: Record<string, number | Record<string, number>>;
  gradcheck: { max_abs_diff: number; passed: boolean };
}

const COLORS = [0x34e6b0, 0x7ecdf0, 0xffb648, 0xff6b6b, 0xb98cff, 0x8ef0c4];

function labelSprite(text: string, color = '#d9e6f2', size = 46): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  ctx.font = `${size}px ui-monospace, monospace`;
  canvas.width = Math.ceil(ctx.measureText(text).width) + 16;
  canvas.height = size * 1.5;
  const g = canvas.getContext('2d') as CanvasRenderingContext2D;
  g.font = `${size}px ui-monospace, monospace`;
  g.fillStyle = color;
  g.textBaseline = 'middle';
  g.fillText(text, 8, canvas.height / 2);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set((canvas.width / canvas.height) * 0.42, 0.42, 1);
  return sprite;
}

export class KernelLabScene implements SimScene {
  readonly simId = null;
  readonly hint = 'les mesures sont prises en direct sur la machine du serveur (min-of-N)';

  private api!: SceneApi;
  private group = new THREE.Group();
  private bars = new THREE.Group();
  private curves = new THREE.Group();
  private report: BenchReport | null = null;
  private tableHost: HTMLElement | null = null;
  private button: HTMLButtonElement | null = null;
  private stats: Record<string, string> = { etat: 'mesure en cours…' };

  opts = { autorotate: true };

  async init(ctx: SceneContext, api: SceneApi): Promise<void> {
    this.api = api;
    ctx.scene.add(this.group);
    ctx.camera.position.set(6.4, 4.6, 8.2);
    ctx.controls.target.set(0, 1.2, 0);

    this.group.add(this.bars, this.curves);
    const grid = new THREE.GridHelper(16, 16, 0x14303c, 0x0d1f28);
    this.group.add(grid);
    ctx.scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xbfe9ff, 1.5);
    key.position.set(4, 8, 6);
    ctx.scene.add(key);

    await this.run(true);
  }

  async run(quick: boolean): Promise<void> {
    if (this.button) this.button.disabled = true;
    this.stats = { etat: quick ? 'bench rapide…' : 'bench complet…' };
    try {
      const res = await fetch(`/api/bench?quick=${quick}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.report = (await res.json()) as BenchReport;
      this.build();
      this.api.toast(
        `bench ${quick ? 'rapide' : 'complet'} termine en ${this.report.elapsed_ms.toFixed(0)} ms`,
        'ok',
      );
    } catch (err) {
      this.stats = { etat: 'serveur injoignable' };
      this.api.toast(`bench indisponible : ${String(err)}`, 'error');
    } finally {
      if (this.button) this.button.disabled = false;
    }
  }

  /* ------------------------------------------------------------------ */
  private clearGroup(group: THREE.Group): void {
    group.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    });
    group.clear();
  }

  private build(): void {
    const report = this.report;
    if (!report) return;
    this.clearGroup(this.bars);
    this.clearGroup(this.curves);

    /* --- barres de speedup elementwise --- */
    const entries = report.elementwise;
    const spacing = 1.5;
    const x0 = (-(entries.length - 1) * spacing) / 2;
    entries.forEach((entry, i) => {
      const speed = entry.speedup ?? 1;
      const height = Math.max(0.12, Math.log10(Math.max(speed, 1.001)) * 2.4);
      const geo = new THREE.BoxGeometry(0.62, height, 0.62);
      const mat = new THREE.MeshStandardMaterial({
        color: COLORS[i % COLORS.length],
        emissive: COLORS[i % COLORS.length],
        emissiveIntensity: 0.22,
        roughness: 0.35,
        metalness: 0.15,
      });
      const bar = new THREE.Mesh(geo, mat);
      bar.position.set(x0 + i * spacing, height / 2, -1.6);
      this.bars.add(bar);

      const value = labelSprite(`×${speed.toFixed(1)}`, '#eaf6ff');
      value.position.set(bar.position.x, height + 0.32, -1.6);
      this.bars.add(value);

      const name = labelSprite(entry.key, '#7d90a4', 36);
      name.position.set(bar.position.x, -0.28, -1.6);
      this.bars.add(name);
    });

    const title = labelSprite('speedup vs baseline numpy (log)', '#34e6b0', 38);
    title.position.set(0, 3.4, -1.6);
    this.bars.add(title);

    /* --- courbes d'erreur (echelle log) --- */
    entries.forEach((entry, i) => {
      const pts: THREE.Vector3[] = [];
      const { x, err } = entry.curve;
      for (let j = 0; j < x.length; j += 1) {
        const e = Math.max(err[j], 1e-9);
        const y = (Math.log10(e) + 9) / 9; // 1e-9 -> 0, 1e0 -> 1
        pts.push(new THREE.Vector3((x[j] / 4) * 4.6, y * 2.6, 1.4 + i * 0.55));
      }
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: COLORS[i % COLORS.length], transparent: true, opacity: 0.9 }),
      );
      this.curves.add(line);

      const tag = labelSprite(`${entry.key}  L∞ ${entry.accuracy.linf.toExponential(1)}`, '#9fb4c8', 32);
      tag.position.set(-5.4, ((Math.log10(Math.max(entry.accuracy.linf, 1e-9)) + 9) / 9) * 2.6, 1.4 + i * 0.55);
      this.curves.add(tag);
    });

    const errTitle = labelSprite('erreur |approx − IEEE| sur [-4,4] (log10)', '#7ecdf0', 38);
    errTitle.position.set(0, 3.0, 2.6);
    this.curves.add(errTitle);

    /* --- matmul GFLOPS --- */
    report.matmul.forEach((mm, i) => {
      const pairs: [number, number][] = [
        [mm.spear_gflops, 0x34e6b0],
        [mm.baseline_gflops, 0x39566a],
      ];
      pairs.forEach(([gf, color], j) => {
        const h = Math.max(0.08, (gf / 200) * 2.4);
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.34, h, 0.34),
          new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2 }),
        );
        bar.position.set(-3.2 + i * 1.3 + j * 0.42, h / 2, 5.2);
        this.bars.add(bar);
      });
      const tag = labelSprite(`${mm.key}`, '#7d90a4', 30);
      tag.position.set(-3.0 + i * 1.3, -0.25, 5.2);
      this.bars.add(tag);
    });
    const mmTitle = labelSprite('GFLOPS matmul NT — SpearVM vs BLAS', '#ffb648', 36);
    mmTitle.position.set(0, 2.4, 5.2);
    this.bars.add(mmTitle);

    this.updateStats();
    this.renderTable();
  }

  private updateStats(): void {
    const r = this.report;
    if (!r) return;
    const ffn = r.fused_ffn as Record<string, number>;
    this.stats = {
      etat: r.cached ? 'cache serveur' : 'mesure fraiche',
      duree: `${r.elapsed_ms.toFixed(0)} ms`,
      'FFN fusionne': `${ffn.fused_ms?.toFixed?.(2) ?? '—'} ms`,
      'gain fusion': `×${ffn.fusion_gain ?? '—'}`,
      'FFN vs numpy': `×${ffn.vs_numpy ?? '—'}`,
      gradcheck: r.gradcheck.passed
        ? `ok (${r.gradcheck.max_abs_diff.toExponential(1)})`
        : `echec (${r.gradcheck.max_abs_diff.toExponential(1)})`,
    };
  }

  private renderTable(): void {
    const host = this.tableHost;
    const r = this.report;
    if (!host || !r) return;
    const rows = r.elementwise
      .map(
        (e) => `<tr>
          <td>${e.key}</td>
          <td>${e.spear_ms.toFixed(2)}</td>
          <td>${e.baseline_ms.toFixed(2)}</td>
          <td class="${(e.speedup ?? 0) >= 1 ? 'up' : 'down'}">×${(e.speedup ?? 0).toFixed(1)}</td>
          <td>${e.accuracy.linf.toExponential(1)}</td>
        </tr>`,
      )
      .join('');
    const mm = r.matmul
      .map(
        (m) => `<tr>
          <td>${m.key}</td>
          <td>${m.spear_gflops.toFixed(1)}</td>
          <td>${m.baseline_gflops.toFixed(1)}</td>
          <td class="${(m.speedup ?? 0) >= 1 ? 'up' : 'down'}">×${(m.speedup ?? 0).toFixed(2)}</td>
          <td>${m.max_abs_err.toExponential(0)}</td>
        </tr>`,
      )
      .join('');
    host.innerHTML = `
      <table class="bench">
        <thead><tr><th>noyau</th><th>spear ms</th><th>base ms</th><th>gain</th><th>L∞</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <table class="bench" style="margin-top:.6rem">
        <thead><tr><th>matmul</th><th>spear GF</th><th>BLAS GF</th><th>ratio</th><th>err</th></tr></thead>
        <tbody>${mm}</tbody>
      </table>
      <p class="note">Baselines : numpy vectorise (tanh-GELU, np.tanh, BLAS) ; erf compare a libm scalaire.
      ${r.method ?? ''} Machine : ${String((r.host as Record<string, unknown>).threads ?? '?')} threads,
      backend <b>${String((r.host as Record<string, unknown>).backend ?? '?')}</b>.
      ${r.noisy_host ? '<b>Peu de coeurs disponibles : attendez-vous a des valeurs bruitees d\'un run a l\'autre.</b>' : ''}</p>`;
  }

  update(dt: number): void {
    if (this.opts.autorotate) this.group.rotation.y += dt * 0.05;
  }

  clientGui(gui: GUI): void {
    gui.addFolder('Rendu (client / GPU)').add(this.opts, 'autorotate').name('rotation auto');
  }

  telemetry(): Record<string, string> {
    return this.stats;
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    this.api = api;
    const quick = document.createElement('button');
    quick.className = 'btn';
    quick.textContent = 'relancer le bench (rapide)';
    quick.onclick = () => void this.run(true);
    this.button = quick;

    const full = document.createElement('button');
    full.className = 'btn secondary';
    full.style.marginTop = '0.4rem';
    full.textContent = 'bench complet (2 s de CPU)';
    full.onclick = () => void this.run(false);

    const table = document.createElement('div');
    table.style.marginTop = '0.6rem';
    this.tableHost = table;

    host.append(quick, full, table);
    this.renderTable();
  }

  dispose(): void {
    this.clearGroup(this.bars);
    this.clearGroup(this.curves);
    this.group.clear();
    this.tableHost = null;
    this.button = null;
  }
}
