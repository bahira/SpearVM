/**
 * Scene 2 — Membrane non lineaire.
 *
 * Le serveur avance l'equation des ondes (avec saturation `tanh` AVX2) et
 * pousse la hauteur quantifiee en int16. Le client la depose dans une texture
 * R32F lue au sommet : le maillage suit exactement la grille de calcul
 * (1 vertex = 1 cellule), donc aucune interpolation ne masque la physique.
 */

import * as THREE from 'three';
import type { SimFrame } from '../net/protocol';
import { statNumber } from '../net/protocol';
import type GUI from 'lil-gui';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

const PLANE = 7.2;

const VS = /* glsl */ `
precision highp float;
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform sampler2D uHeight;
uniform float uAmp;
uniform int uSize;
out vec3 vNormal;
out vec3 vView;
out float vHeight;

float sample(ivec2 c) {
  ivec2 cc = clamp(c, ivec2(0), ivec2(uSize - 1, uSize - 1));
  return texelFetch(uHeight, cc, 0).r;
}

void main() {
  ivec2 coord = ivec2(int(round(uv.x * float(uSize - 1))), int(round(uv.y * float(uSize - 1))));
  float h = sample(coord);
  float dx = (sample(coord + ivec2(1, 0)) - sample(coord - ivec2(1, 0)));
  float dy = (sample(coord + ivec2(0, 1)) - sample(coord - ivec2(0, 1)));
  float cell = ${PLANE.toFixed(1)} / float(uSize - 1);

  vec3 displaced = vec3(position.x, position.y, h * uAmp);
  vec3 n = normalize(vec3(-dx * uAmp / (2.0 * cell), -dy * uAmp / (2.0 * cell), 1.0));

  vHeight = h;
  vNormal = normalize(normalMatrix * n);
  vec4 mv = modelViewMatrix * vec4(displaced, 1.0);
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const FS = /* glsl */ `
precision highp float;
in vec3 vNormal;
in vec3 vView;
in float vHeight;
uniform float uPeak;
uniform vec3 uLight;
out vec4 outColor;

void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(vView);
  vec3 l = normalize(uLight);

  float t = clamp(vHeight / max(uPeak, 1e-3) * 0.5 + 0.5, 0.0, 1.0);
  vec3 deep = vec3(0.02, 0.10, 0.18);
  vec3 mid  = vec3(0.05, 0.42, 0.48);
  vec3 crest= vec3(0.62, 0.98, 0.82);
  vec3 base = t < 0.5 ? mix(deep, mid, t * 2.0) : mix(mid, crest, (t - 0.5) * 2.0);

  float diff = max(dot(n, l), 0.0);
  float spec = pow(max(dot(reflect(-l, n), v), 0.0), 48.0);
  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);

  vec3 col = base * (0.28 + 0.85 * diff) + vec3(0.55, 0.95, 1.0) * spec * 0.6
           + vec3(0.10, 0.35, 0.5) * fres;
  outColor = vec4(col, 1.0);
}
`;

export class WaveFieldScene implements SimScene {
  readonly simId = 'wavefield';
  readonly hint = 'clic sur la membrane : impulsion · glisser : orbite · shift+clic : impulsion negative';

  private ctx!: SceneContext;
  private api!: SceneApi;
  private group = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private wire: THREE.LineSegments | null = null;
  private texture: THREE.DataTexture | null = null;
  private size = 0;
  private material!: THREE.RawShaderMaterial;
  private raycaster = new THREE.Raycaster();
  private stats: Record<string, string> = {};

  opts = { amplitude: 1.0, wireframe: false, autorotate: false };

  init(ctx: SceneContext, api: SceneApi): void {
    this.ctx = ctx;
    this.api = api;
    ctx.scene.add(this.group);
    ctx.camera.position.set(0, 5.4, 7.4);
    ctx.controls.target.set(0, 0, 0);

    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(PLANE, 0.02, PLANE)),
      new THREE.LineBasicMaterial({ color: 0x1d4657, transparent: true, opacity: 0.6 }),
    );
    this.group.add(frame);

    const grid = new THREE.GridHelper(PLANE * 2, 24, 0x14303c, 0x0e222b);
    grid.position.y = -1.4;
    this.group.add(grid);

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VS,
      fragmentShader: FS,
      uniforms: {
        uHeight: { value: null },
        uAmp: { value: 1.0 },
        uSize: { value: 2 },
        uPeak: { value: 1.0 },
        uLight: { value: new THREE.Vector3(0.4, 0.9, 0.55) },
      },
      side: THREE.DoubleSide,
    });
  }

  private rebuild(size: number): void {
    this.size = size;
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    if (this.wire) {
      this.group.remove(this.wire);
      this.wire.geometry.dispose();
    }
    const geo = new THREE.PlaneGeometry(PLANE, PLANE, size - 1, size - 1);
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.rotation.x = -Math.PI / 2;
    this.group.add(this.mesh);

    this.texture?.dispose();
    const tex = new THREE.DataTexture(new Float32Array(size * size), size, size);
    tex.format = THREE.RedFormat;
    tex.type = THREE.FloatType;
    tex.internalFormat = 'R32F';
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    this.texture = tex;
    this.material.uniforms.uHeight.value = tex;
    this.material.uniforms.uSize.value = size;
  }

  frame(frame: SimFrame): void {
    const { header, data } = frame;
    if (header.kind !== 'height_grid' || !data) return;
    const size = header.shape[0] ?? 0;
    if (!size) return;
    if (size !== this.size) this.rebuild(size);

    const tex = this.texture as THREE.DataTexture;
    (tex.image.data as Float32Array).set(data);
    tex.needsUpdate = true;

    const peak = statNumber(header.stats, 'peak', 1);
    this.material.uniforms.uPeak.value = Math.max(peak, 0.05);
    this.material.uniforms.uAmp.value = this.opts.amplitude;
    this.material.wireframe = this.opts.wireframe;

    this.stats = {
      grille: `${size}² = ${(size * size).toLocaleString('fr-FR')} cellules`,
      'sous-pas': String(statNumber(header.stats, 'substeps', 1)),
      'Mcell/s': statNumber(header.stats, 'mcells_per_s').toFixed(1),
      amplitude: peak.toFixed(3),
      energie: statNumber(header.stats, 'energy').toExponential(2),
    };
  }

  update(dt: number): void {
    if (this.opts.autorotate) this.group.rotation.y += dt * 0.06;
  }

  pointer(event: PointerEvent, ndc: THREE.Vector2, kind: 'down' | 'move'): void {
    if (kind !== 'down' || !this.mesh || event.button !== 0) return;
    this.raycaster.setFromCamera(ndc, this.ctx.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit?.uv) return;
    // texture : u = colonne (axe j du serveur), v = ligne (axe i)
    this.api.command({
      type: 'pulse',
      x: hit.uv.y,
      y: hit.uv.x,
      amp: event.shiftKey ? -1.2 : 1.2,
    });
  }

  clientGui(gui: GUI): void {
    const folder = gui.addFolder('Rendu (client / GPU)');
    folder.add(this.opts, 'amplitude', 0.2, 3, 0.05).name('relief');
    folder.add(this.opts, 'wireframe').name('filaire');
    folder.add(this.opts, 'autorotate').name('rotation auto');
  }

  telemetry(): Record<string, string> {
    return this.stats;
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    const clear = document.createElement('button');
    clear.className = 'btn secondary';
    clear.textContent = 'calmer la membrane';
    clear.onclick = () => api.command({ type: 'clear' });

    const note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      'Raideur > 0 : le noyau tanh SPEAR remplace le ressort lineaire — les cretes '
      + 'se plafonnent et les fronts se raidissent au lieu de diverger.';
    host.append(clear, note);
  }

  dispose(): void {
    this.mesh?.geometry.dispose();
    this.wire?.geometry.dispose();
    this.texture?.dispose();
    this.material?.dispose();
    this.group.clear();
  }
}
