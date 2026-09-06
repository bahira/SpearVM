/**
 * Scene 3 — Entrainement live.
 *
 * Surface predite (solide, coloree par l'erreur) + surface cible (fantome
 * filaire). Les deux hauteurs arrivent en Float32 depuis le serveur, qui
 * execute forward/backward via SpearVM. La courbe de perte est tracee dans le
 * panneau lateral.
 */

import * as THREE from 'three';
import type { SimFrame } from '../net/protocol';
import { statNumber, statString } from '../net/protocol';
import type GUI from 'lil-gui';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

const PLANE = 6.4;

const VS = /* glsl */ `
precision highp float;
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform sampler2D uField;
uniform sampler2D uTarget;
uniform float uAmp;
uniform int uSize;
out vec3 vNormal;
out float vHeight;
out float vError;

float sampleF(sampler2D tex, ivec2 c) {
  ivec2 cc = clamp(c, ivec2(0), ivec2(uSize - 1, uSize - 1));
  return texelFetch(tex, cc, 0).r;
}

void main() {
  ivec2 coord = ivec2(int(round(uv.x * float(uSize - 1))), int(round(uv.y * float(uSize - 1))));
  float h = sampleF(uField, coord);
  float dx = sampleF(uField, coord + ivec2(1, 0)) - sampleF(uField, coord - ivec2(1, 0));
  float dy = sampleF(uField, coord + ivec2(0, 1)) - sampleF(uField, coord - ivec2(0, 1));
  float cell = ${PLANE.toFixed(1)} / float(uSize - 1);

  vHeight = h;
  vError = abs(h - sampleF(uTarget, coord));
  vNormal = normalize(normalMatrix * normalize(vec3(-dx * uAmp / (2.0 * cell), -dy * uAmp / (2.0 * cell), 1.0)));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.x, position.y, h * uAmp, 1.0);
}
`;

const FS = /* glsl */ `
precision highp float;
in vec3 vNormal;
in float vHeight;
in float vError;
uniform int uMode;      // 0 = erreur, 1 = hauteur
uniform float uOpacity;
uniform vec3 uTint;
out vec4 outColor;

vec3 errorRamp(float e) {
  vec3 cold = vec3(0.10, 0.85, 0.65);
  vec3 warm = vec3(0.98, 0.78, 0.28);
  vec3 hot  = vec3(0.95, 0.28, 0.35);
  float t = clamp(e / 0.35, 0.0, 1.0);
  return t < 0.5 ? mix(cold, warm, t * 2.0) : mix(warm, hot, (t - 0.5) * 2.0);
}

void main() {
  vec3 n = normalize(vNormal);
  float diff = 0.35 + 0.75 * max(dot(n, normalize(vec3(0.35, 0.85, 0.4))), 0.0);
  vec3 base = uMode == 0
    ? errorRamp(vError)
    : mix(vec3(0.05, 0.25, 0.42), vec3(0.55, 0.95, 0.85), clamp(vHeight * 0.5 + 0.5, 0.0, 1.0));
  outColor = vec4(base * diff * uTint, uOpacity);
}
`;

export class TrainerScene implements SimScene {
  readonly simId = 'trainer';
  readonly hint = 'la surface pleine est la prediction (couleur = erreur), le filaire est la cible';

  private group = new THREE.Group();
  private predMesh: THREE.Mesh | null = null;
  private targetMesh: THREE.Mesh | null = null;
  private predTex: THREE.DataTexture | null = null;
  private targetTex: THREE.DataTexture | null = null;
  private predMat!: THREE.RawShaderMaterial;
  private targetMat!: THREE.RawShaderMaterial;
  private size = 0;
  private history: number[] = [];
  private stats: Record<string, string> = {};
  private canvas: HTMLCanvasElement | null = null;

  opts = { amplitude: 1.6, mode: 'erreur', showTarget: true, autorotate: true };

  init(ctx: SceneContext): void {
    ctx.scene.add(this.group);
    ctx.camera.position.set(5.2, 4.4, 6.2);

    const makeMat = (opacity: number, wireframe: boolean, tint: THREE.Vector3) =>
      new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: VS,
        fragmentShader: FS,
        uniforms: {
          uField: { value: null },
          uTarget: { value: null },
          uAmp: { value: this.opts.amplitude },
          uSize: { value: 2 },
          uMode: { value: 0 },
          uOpacity: { value: opacity },
          uTint: { value: tint },
        },
        transparent: opacity < 1,
        wireframe,
        side: THREE.DoubleSide,
        depthWrite: opacity >= 1,
      });

    this.predMat = makeMat(1.0, false, new THREE.Vector3(1, 1, 1));
    this.targetMat = makeMat(0.35, true, new THREE.Vector3(0.55, 0.8, 1.0));
    this.targetMat.uniforms.uMode.value = 1;

    const grid = new THREE.GridHelper(PLANE * 1.8, 18, 0x14303c, 0x0d1f28);
    grid.position.y = -2.0;
    this.group.add(grid);
  }

  private rebuild(size: number): void {
    this.size = size;
    for (const mesh of [this.predMesh, this.targetMesh]) {
      if (mesh) {
        this.group.remove(mesh);
        mesh.geometry.dispose();
      }
    }
    const geo = new THREE.PlaneGeometry(PLANE, PLANE, size - 1, size - 1);
    this.predMesh = new THREE.Mesh(geo, this.predMat);
    this.predMesh.rotation.x = -Math.PI / 2;
    this.targetMesh = new THREE.Mesh(geo, this.targetMat);
    this.targetMesh.rotation.x = -Math.PI / 2;
    this.group.add(this.predMesh, this.targetMesh);

    const mk = (): THREE.DataTexture => {
      const tex = new THREE.DataTexture(new Float32Array(size * size), size, size);
      tex.format = THREE.RedFormat;
      tex.type = THREE.FloatType;
      tex.internalFormat = 'R32F';
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
      return tex;
    };
    this.predTex?.dispose();
    this.targetTex?.dispose();
    this.predTex = mk();
    this.targetTex = mk();

    for (const mat of [this.predMat, this.targetMat]) {
      mat.uniforms.uSize.value = size;
      mat.uniforms.uTarget.value = this.targetTex;
    }
    this.predMat.uniforms.uField.value = this.predTex;
    this.targetMat.uniforms.uField.value = this.targetTex;
  }

  frame(frame: SimFrame): void {
    const { header, data } = frame;
    if (!data) return;
    const size = header.shape[0] ?? 0;
    if (!size) return;
    if (size !== this.size) this.rebuild(size);

    if (header.kind === 'target_grid') {
      (this.targetTex!.image.data as Float32Array).set(data);
      this.targetTex!.needsUpdate = true;
      this.history = [];
      return;
    }
    if (header.kind !== 'prediction_grid') return;

    (this.predTex!.image.data as Float32Array).set(data);
    this.predTex!.needsUpdate = true;

    const hist = header.stats.history;
    if (Array.isArray(hist)) this.history = hist as number[];

    const loss = statNumber(header.stats, 'loss');
    this.stats = {
      cible: statString(header.stats, 'target'),
      perte: loss.toExponential(3),
      'gain vs init': `×${statNumber(header.stats, 'improvement').toFixed(1)}`,
      rmse: statNumber(header.stats, 'rmse').toFixed(4),
      'erreur L∞': statNumber(header.stats, 'linf').toFixed(4),
      'pas SGD': statNumber(header.stats, 'steps').toLocaleString('fr-FR'),
      echantillons: statNumber(header.stats, 'samples').toLocaleString('fr-FR'),
      GFLOPS: statNumber(header.stats, 'gflops').toFixed(1),
    };
    this.drawChart();
  }

  private drawChart(): void {
    const canvas = this.canvas;
    if (!canvas || this.history.length < 2) return;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = canvas.clientWidth * dpr;
    const h = canvas.clientHeight * dpr;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const g = canvas.getContext('2d');
    if (!g) return;
    g.clearRect(0, 0, w, h);

    const values = this.history.map((v) => Math.log10(Math.max(v, 1e-9)));
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = Math.max(hi - lo, 0.3);

    g.strokeStyle = 'rgba(126,205,240,0.12)';
    g.lineWidth = 1;
    for (let i = 0; i <= 3; i += 1) {
      const y = (h * i) / 3;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(w, y);
      g.stroke();
    }

    g.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - lo) / span) * (h * 0.86) - h * 0.07;
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.strokeStyle = '#34e6b0';
    g.lineWidth = 1.6 * dpr;
    g.stroke();

    g.fillStyle = 'rgba(125,144,164,0.9)';
    g.font = `${10 * dpr}px ui-monospace, monospace`;
    g.fillText(`log10(loss) ${lo.toFixed(2)} → ${values[values.length - 1].toFixed(2)}`, 6 * dpr, 12 * dpr);
  }

  update(dt: number): void {
    this.predMat.uniforms.uAmp.value = this.opts.amplitude;
    this.targetMat.uniforms.uAmp.value = this.opts.amplitude;
    this.predMat.uniforms.uMode.value = this.opts.mode === 'erreur' ? 0 : 1;
    if (this.targetMesh) this.targetMesh.visible = this.opts.showTarget;
    if (this.opts.autorotate) this.group.rotation.y += dt * 0.08;
  }

  clientGui(gui: GUI): void {
    const folder = gui.addFolder('Rendu (client / GPU)');
    folder.add(this.opts, 'amplitude', 0.4, 4, 0.05).name('relief');
    folder.add(this.opts, 'mode', ['erreur', 'hauteur']).name('couleur');
    folder.add(this.opts, 'showTarget').name('afficher la cible');
    folder.add(this.opts, 'autorotate').name('rotation auto');
  }

  telemetry(): Record<string, string> {
    return this.stats;
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    const box = document.createElement('div');
    box.className = 'chartbox';
    const canvas = document.createElement('canvas');
    box.appendChild(canvas);
    this.canvas = canvas;

    const reset = document.createElement('button');
    reset.className = 'btn';
    reset.textContent = 'reinitialiser les poids';
    reset.onclick = () => {
      api.command({ type: 'reset' });
      this.history = [];
      api.toast('poids reinitialises — la convergence repart de zero', 'ok');
    };

    const note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      'Backward complet cote SpearVM : gelu_backward puis matmul_backward (dX, dW) ; '
      + 'Adam applique les gradients. La couleur mesure |prediction − cible|.';

    host.append(box, reset, note);
  }

  dispose(): void {
    this.predTex?.dispose();
    this.targetTex?.dispose();
    this.predMat?.dispose();
    this.targetMat?.dispose();
    this.predMesh?.geometry.dispose();
    this.group.clear();
    this.canvas = null;
  }
}
