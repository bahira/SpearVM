/**
 * Scene 6 — Attention Lab.
 *
 * Le serveur execute un vrai bloc de decodeur SpearVM (attention multi-tetes
 * sur cache KV packe + FFN) token apres token, et pousse la carte d'attention
 * effectivement calculee : (tetes x positions du contexte), quantifiee en int16.
 *
 * Le client la depose dans une texture R32F lue au sommet — un vertex par
 * couple (tete, position), donc aucune interpolation ne masque ce que le noyau
 * a reellement produit. La tete selectionnee est mise en relief, les autres
 * restent lisibles en fond.
 */

import * as THREE from 'three';
import type GUI from 'lil-gui';
import type { SimFrame } from '../net/protocol';
import { statNumber, statString } from '../net/protocol';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

const WIDTH = 11.0;
const DEPTH = 3.4;

const VS = /* glsl */ `
precision highp float;
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform sampler2D uMap;
uniform float uAmp;
uniform float uFocus;
uniform int uCols;
uniform int uRows;
out float vW;
out float vFocus;
out vec3 vNormal;

float at(ivec2 c) {
  ivec2 cc = clamp(c, ivec2(0), ivec2(uCols - 1, uRows - 1));
  return texelFetch(uMap, cc, 0).r;
}

void main() {
  int cx = int(round(uv.x * float(uCols - 1)));
  int cy = int(round(uv.y * float(uRows - 1)));
  float w = at(ivec2(cx, cy));

  // la tete selectionnee est mise en relief, les autres restent visibles
  float focus = 1.0 - clamp(abs(float(cy) - uFocus), 0.0, 1.0);
  float gain = mix(0.42, 1.0, focus);

  float dx = at(ivec2(cx + 1, cy)) - at(ivec2(cx - 1, cy));
  float dy = at(ivec2(cx, cy + 1)) - at(ivec2(cx, cy - 1));
  vec3 n = normalize(vec3(-dx * uAmp * 40.0, -dy * uAmp * 4.0, 1.0));

  vW = w;
  vFocus = focus;
  vNormal = normalize(normalMatrix * n);
  vec3 p = vec3(position.x, position.y, w * uAmp * gain);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`;

const FS = /* glsl */ `
precision highp float;
in float vW;
in float vFocus;
in vec3 vNormal;
uniform float uPeak;
out vec4 outColor;

vec3 palette(float t) {
  vec3 a = vec3(0.05, 0.16, 0.28);
  vec3 b = vec3(0.16, 0.70, 0.78);
  vec3 c = vec3(0.98, 0.86, 0.40);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, (t - 0.5) * 2.0);
}

void main() {
  float t = clamp(vW / max(uPeak, 1e-6), 0.0, 1.0);
  vec3 base = palette(pow(t, 0.55));
  vec3 n = normalize(vNormal);
  float lam = clamp(dot(n, normalize(vec3(0.35, 0.5, 0.8))), 0.0, 1.0);
  vec3 col = base * (0.32 + 0.85 * lam);
  col *= mix(0.45, 1.0, vFocus);
  col = col / (col + vec3(0.9));
  outColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
`;

export class AttentionLabScene implements SimScene {
  readonly simId = 'attention';
  readonly hint = 'chaque ligne = une tete, chaque colonne = une position du contexte · '
    + 'le relief est le poids d\'attention reellement calcule';

  private ctx!: SceneContext;
  private group = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private material!: THREE.RawShaderMaterial;
  private texture: THREE.DataTexture | null = null;
  private cols = 0;
  private rows = 0;
  private stats: Record<string, string> = {};
  private opts = { amp: 4.5, autorotate: true };

  init(ctx: SceneContext): void {
    this.ctx = ctx;
    ctx.scene.add(this.group);
    ctx.camera.position.set(0.2, -7.4, 5.6);
    ctx.camera.up.set(0, 0, 1);
    ctx.controls.target.set(0, 0, 0.6);

    const grid = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(WIDTH, DEPTH)),
      new THREE.LineBasicMaterial({ color: 0x17414f, transparent: true, opacity: 0.6 }),
    );
    this.group.add(grid);

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VS,
      fragmentShader: FS,
      uniforms: {
        uMap: { value: null },
        uAmp: { value: this.opts.amp },
        uPeak: { value: 1 },
        uFocus: { value: 0 },
        uCols: { value: 1 },
        uRows: { value: 1 },
      },
      side: THREE.DoubleSide,
    });
  }

  private rebuild(cols: number, rows: number): void {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    this.texture?.dispose();
    // un vertex par (tete, position) : la grille de calcul est rendue telle quelle
    const geo = new THREE.PlaneGeometry(WIDTH, DEPTH, Math.min(cols, 2048) - 1,
                                        Math.max(rows - 1, 1));
    const tex = new THREE.DataTexture(new Float32Array(cols * rows), cols, rows);
    tex.format = THREE.RedFormat;
    tex.type = THREE.FloatType;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    this.texture = tex;
    this.cols = cols;
    this.rows = rows;
    this.material.uniforms.uMap.value = tex;
    this.material.uniforms.uCols.value = cols;
    this.material.uniforms.uRows.value = rows;
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.group.add(this.mesh);
  }

  frame(frame: SimFrame): void {
    const { header, data } = frame;
    if (header.kind !== 'attention_map' || !data) return;
    const rows = header.shape[0] ?? 0;
    const cols = header.shape[1] ?? 0;
    if (!rows || !cols) return;
    if (rows !== this.rows || cols !== this.cols) this.rebuild(cols, rows);

    const tex = this.texture as THREE.DataTexture;
    (tex.image.data as Float32Array).set(data.subarray(0, rows * cols));
    tex.needsUpdate = true;

    const peak = statNumber(header.stats, 'attention_max', 1);
    this.material.uniforms.uPeak.value = peak;
    this.material.uniforms.uFocus.value = statNumber(header.stats, 'focus');

    const fmt = statString(header.stats, 'format', 'f32');
    this.stats = {
      format: fmt,
      'tokens/s': statNumber(header.stats, 'tokens_par_s').toFixed(0),
      GFLOPS: statNumber(header.stats, 'gflops').toFixed(1),
      'poids en memoire': `${statNumber(header.stats, 'poids_Mo').toFixed(2)} Mo`,
      'cache KV': `${statNumber(header.stats, 'kv_Mo').toFixed(2)} Mo`,
      contexte: `${statNumber(header.stats, 'context')} tokens`,
      tetes: `${statNumber(header.stats, 'heads')} Q / ${statNumber(header.stats, 'kv_heads')} KV`,
      'entropie attention': statNumber(header.stats, 'entropie').toFixed(3),
      'poids max': peak.toFixed(3),
      'tokens generes': statNumber(header.stats, 'tokens').toFixed(0),
    };
  }

  update(dt: number): void {
    this.material.uniforms.uAmp.value = this.opts.amp;
    if (this.opts.autorotate) this.group.rotation.z += dt * 0.045;
  }

  telemetry(): Record<string, string> {
    return this.stats;
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    const button = document.createElement('button');
    button.className = 'btn';
    button.textContent = 'nouveaux poids (reseed)';
    button.onclick = () => {
      api.command({ type: 'reseed' });
      api.toast('nouveau bloc tire — la carte d\'attention change', 'ok');
    };
    const note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      'Le serveur decode reellement, token apres token : attention multi-tetes '
      + 'sur cache KV packe puis FFN. Changez le format des poids pour voir le '
      + 'debit bouger — f32, bf16 (÷2) et int8 (÷4, echelle par ligne). Seule la '
      + 'carte d\'attention transite : 8 Ko par frame.';
    host.append(button, note);
  }

  clientGui(gui: GUI): void {
    const folder = gui.addFolder('Rendu (client / GPU)');
    folder.add(this.opts, 'amp', 0.5, 14, 0.1).name('relief');
    folder.add(this.opts, 'autorotate').name('rotation auto');
  }

  dispose(): void {
    this.ctx.camera.up.set(0, 1, 0);
    this.texture?.dispose();
    this.material?.dispose();
    this.mesh?.geometry.dispose();
    this.group.clear();
  }
}
