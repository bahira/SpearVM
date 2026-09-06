/**
 * Scene 5 — Champ implicite neuronal (SDF ray-marche).
 *
 * Le serveur evalue un MLP sur chaque voxel d'une grille 3D et renvoie une
 * distance signee quantifiee en int16. Le client la charge dans une texture 3D
 * demi-flottante et fait du **sphere tracing** dans le fragment shader : a
 * chaque pas on avance exactement de la distance a la surface, ce qui est
 * correct tant que le champ est 1-lipschitzien — condition que le serveur
 * garantit et publie dans ses stats (`lipschitz <= 1`).
 *
 * Repartition : CPU SIMD = algebre lineaire du reseau, GPU = rendu.
 * Aucune geometrie n'est transmise : la surface n'existe que comme volume.
 */

import * as THREE from 'three';
import type GUI from 'lil-gui';
import type { SimFrame } from '../net/protocol';
import { statNumber } from '../net/protocol';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

const VS = /* glsl */ `
in vec3 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out vec3 vLocal;
void main() {
  vLocal = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FS = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vLocal;
out vec4 outColor;

uniform sampler3D uSdf;
uniform mat4  uInvModel;     // monde -> objet
uniform vec3  uCamPos;       // camera en coordonnees monde
uniform float uIso;
uniform float uVoxel;        // pas de grille en unites objet
uniform float uSteps;
uniform float uClip;         // plan de coupe sur x (objet)
uniform float uMode;         // 0 = matiere, 1 = normales
uniform float uGlow;

const vec3 BMIN = vec3(-1.0);
const vec3 BMAX = vec3(1.0);

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 inv = 1.0 / dir;
  vec3 t0v = (BMIN - orig) * inv;
  vec3 t1v = (BMAX - orig) * inv;
  vec3 tsm = min(t0v, t1v);
  vec3 tbg = max(t0v, t1v);
  return vec2(max(tsm.x, max(tsm.y, tsm.z)), min(tbg.x, min(tbg.y, tbg.z)));
}

float field(vec3 p) {
  return texture(uSdf, p * 0.5 + 0.5).r - uIso;
}

vec3 gradient(vec3 p) {
  float h = uVoxel;
  vec2 e = vec2(h, 0.0);
  return normalize(vec3(
    field(p + e.xyy) - field(p - e.xyy),
    field(p + e.yxy) - field(p - e.yxy),
    field(p + e.yyx) - field(p - e.yyx)
  ) + 1e-9);
}

/* occlusion ambiante : 5 sondes le long de la normale (Quilez) */
float ao(vec3 p, vec3 n) {
  float sum = 0.0;
  float w = 1.0;
  for (int i = 1; i <= 5; i++) {
    float d = 0.035 * float(i);
    sum += w * (d - field(p + n * d));
    w *= 0.62;
  }
  return clamp(1.0 - 2.2 * sum, 0.0, 1.0);
}

vec3 palette(float t) {
  vec3 a = vec3(0.06, 0.24, 0.38);
  vec3 b = vec3(0.20, 0.90, 0.70);
  vec3 c = vec3(0.98, 0.84, 0.42);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, (t - 0.5) * 2.0);
}

void main() {
  vec3 orig = (uInvModel * vec4(uCamPos, 1.0)).xyz;
  vec3 dir = normalize(vLocal - orig);

  vec2 tb = hitBox(orig, dir);
  if (tb.y <= max(tb.x, 0.0)) discard;

  float t = max(tb.x, 0.0) + uVoxel * 0.5;
  float surf = uVoxel * 0.6;
  bool hit = false;
  vec3 p = vec3(0.0);
  int steps = int(uSteps);

  for (int i = 0; i < 512; i++) {
    if (i >= steps || t > tb.y) break;
    p = orig + dir * t;
    if (p.x < uClip) {                 // plan de coupe : on saute la matiere
      t += uVoxel;
      continue;
    }
    float d = field(p);
    if (d < surf) { hit = true; break; }
    t += max(d, uVoxel * 0.75);        // sphere tracing (champ 1-lipschitzien)
  }

  if (!hit) discard;

  vec3 n = gradient(p);
  vec3 v = -dir;
  vec3 key = normalize(vec3(0.6, 0.85, 0.45));
  vec3 fill = normalize(vec3(-0.5, 0.15, -0.7));

  float diff = max(dot(n, key), 0.0);
  float back = max(dot(n, fill), 0.0) * 0.35;
  float spec = pow(max(dot(reflect(-key, n), v), 0.0), 42.0);
  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  float occ = ao(p, n);

  float depth = clamp((t - tb.x) / max(tb.y - tb.x, 1e-3), 0.0, 1.0);
  vec3 base = palette(clamp(0.25 + 0.75 * (0.5 + 0.5 * n.y) - 0.35 * depth, 0.0, 1.0));

  vec3 col = base * (0.18 + 0.9 * diff + back) * occ
           + vec3(0.9, 0.97, 1.0) * spec * 0.55
           + vec3(0.25, 0.75, 0.95) * fres * uGlow;

  if (uMode > 0.5) col = n * 0.5 + 0.5;

  col = col / (col + vec3(0.85));                 // tone mapping doux
  outColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
`;

export class ImplicitFieldScene implements SimScene {
  readonly simId = 'implicit';
  readonly hint = 'clic-glisser : orbite · molette : zoom · « iso » et « coupe » sont recalcules sur GPU, sans aller-retour serveur';

  private ctx!: SceneContext;
  private group = new THREE.Group();
  private mesh!: THREE.Mesh;
  private material!: THREE.RawShaderMaterial;
  private texture: THREE.Data3DTexture | null = null;
  private halves: Uint16Array | null = null;
  private gridSize = 0;
  private invModel = new THREE.Matrix4();
  private stats: Record<string, string> = {};

  private opts = { iso: 0, steps: 160, autorotate: true, mode: 'matiere', clip: -1.05, glow: 0.6 };

  init(ctx: SceneContext): void {
    this.ctx = ctx;
    ctx.scene.add(this.group);
    ctx.camera.position.set(3.1, 2.0, 3.4);

    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0x18404f, transparent: true, opacity: 0.5 }),
    );
    this.group.add(box);

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VS,
      fragmentShader: FS,
      uniforms: {
        uSdf: { value: null },
        uInvModel: { value: this.invModel },
        uCamPos: { value: new THREE.Vector3() },
        uIso: { value: this.opts.iso },
        uVoxel: { value: 0.05 },
        uSteps: { value: this.opts.steps },
        uClip: { value: this.opts.clip },
        uMode: { value: 0 },
        uGlow: { value: this.opts.glow },
      },
      side: THREE.BackSide,   // on rend la face de sortie : fonctionne aussi
      transparent: false,     // quand la camera est a l'interieur du volume
      depthWrite: true,
    });

    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), this.material);
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;    // rien a afficher tant qu'aucun volume n'est arrive
    this.group.add(this.mesh);
  }

  /* ---------------- flux serveur ---------------- */
  frame(frame: SimFrame): void {
    const { header, data } = frame;
    if (header.kind !== 'sdf_grid' || !data) return;
    const n = header.shape[0] ?? 0;
    if (!n) return;

    if (n !== this.gridSize) {
      this.gridSize = n;
      this.halves = new Uint16Array(n * n * n);
      this.texture?.dispose();
      const tex = new THREE.Data3DTexture(this.halves, n, n, n);
      tex.format = THREE.RedFormat;
      tex.type = THREE.HalfFloatType;      // 16 bits : pas de banding sur les normales
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.wrapR = THREE.ClampToEdgeWrapping;
      tex.unpackAlignment = 2;
      tex.needsUpdate = true;
      this.texture = tex;
      this.material.uniforms.uSdf.value = tex;
      this.material.uniforms.uVoxel.value = 2 / (n - 1);
      this.mesh.visible = true;
    }

    const halves = this.halves as Uint16Array;
    const total = n * n * n;
    const toHalf = THREE.DataUtils.toHalfFloat;
    for (let i = 0; i < total; i += 1) halves[i] = toHalf(data[i]);
    if (this.texture) this.texture.needsUpdate = true;

    const grad = statNumber(header.stats, 'grad_max');
    this.stats = {
      volume: `${n}³ = ${total.toLocaleString('fr-FR')} voxels`,
      'MLP par point': `${statNumber(header.stats, 'k_in')} → ${statNumber(header.stats, 'hidden')}² → 1`,
      'MFLOP / tick': statNumber(header.stats, 'mflop_per_tick').toFixed(1),
      'GFLOPS noyau': statNumber(header.stats, 'gflops').toFixed(1),
      'Lipschitz (rendu)': `${statNumber(header.stats, 'lipschitz').toFixed(2)} (brut ${grad.toFixed(2)})`,
      'voxels de surface': `${(statNumber(header.stats, 'surface_frac') * 100).toFixed(2)} %`,
    };
  }

  /* ---------------- boucle ---------------- */
  update(dt: number): void {
    if (!this.texture) return;
    if (this.opts.autorotate) this.group.rotation.y += dt * 0.14;

    this.mesh.updateWorldMatrix(true, false);
    this.invModel.copy(this.mesh.matrixWorld).invert();
    const u = this.material.uniforms;
    u.uInvModel.value = this.invModel;
    (u.uCamPos.value as THREE.Vector3).copy(this.ctx.camera.position);
    u.uIso.value = this.opts.iso;
    u.uSteps.value = this.opts.steps;
    u.uClip.value = this.opts.clip;
    u.uGlow.value = this.opts.glow;
    u.uMode.value = this.opts.mode === 'normales' ? 1 : 0;
  }

  telemetry(): Record<string, string> {
    return { rendu: `sphere tracing ${this.opts.steps} pas`, ...this.stats };
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    const button = document.createElement('button');
    button.className = 'btn';
    button.textContent = 'nouvelle sculpture (reseed)';
    button.onclick = () => {
      api.command({ type: 'reseed' });
      api.toast('nouveaux poids : la surface est entierement resculptee', 'ok');
    };
    const note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      "Le serveur evalue le reseau sur chaque voxel (grid³ lignes, k=14) : c'est le "
      + 'profil de forme ou les noyaux SpearVM depassent OpenBLAS. Le volume arrive '
      + 'quantifie en int16 ; le GPU fait le sphere tracing — aucune geometrie transmise.';
    host.append(button, note);
  }

  clientGui(gui: GUI): void {
    const folder = gui.addFolder('Rendu (client / GPU)');
    folder.add(this.opts, 'iso', -0.35, 0.35, 0.005).name('iso-surface');
    folder.add(this.opts, 'clip', -1.05, 1.05, 0.01).name('plan de coupe');
    folder.add(this.opts, 'steps', 40, 400, 10).name('pas de marche');
    folder.add(this.opts, 'glow', 0, 1.5, 0.05).name('halo de bord');
    folder.add(this.opts, 'mode', ['matiere', 'normales']).name('rendu');
    folder.add(this.opts, 'autorotate').name('rotation auto');
  }

  dispose(): void {
    this.texture?.dispose();
    this.material?.dispose();
    this.mesh?.geometry.dispose();
    this.group.clear();
  }
}
