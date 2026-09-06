/**
 * Scene 1 — Champ de flux neuronal.
 *
 * Le serveur pousse un champ de vitesse 3D (rot du potentiel produit par un
 * MLP GELU). Le GPU s'occupe du reste : advection de 65 536 particules dans
 * une paire de render targets ping-pong, echantillonnage trilineaire du champ
 * via une texture 3D, rendu en points additifs colores par la vitesse.
 *
 * Repartition : le CPU SIMD fait l'algebre lineaire, le GPU fait la geometrie.
 */

import * as THREE from 'three';
import type { SimFrame } from '../net/protocol';
import { statNumber } from '../net/protocol';
import type GUI from 'lil-gui';
import type { SceneApi, SceneContext, SimScene } from '../core/types';

const QUAD_VS = /* glsl */ `
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ADVECT_FS = /* glsl */ `
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPos;
uniform sampler3D uVel;
uniform float uDt;
uniform float uSpeed;
uniform float uLifeRate;
uniform float uSeed;
uniform float uInit;

vec3 hash3(vec2 p, float s) {
  vec3 q = vec3(dot(p, vec2(127.1, 311.7)) + s,
                dot(p, vec2(269.5, 183.3)) - s,
                dot(p, vec2(419.2, 371.9)) + s * 0.7);
  return fract(sin(q) * 43758.5453);
}

void main() {
  vec4 state = texture(uPos, vUv);
  vec3 pos = state.xyz;
  float life = state.w;

  if (uInit > 0.5) {
    outColor = vec4(hash3(vUv, uSeed), hash3(vUv, uSeed + 3.1).x);
    return;
  }

  vec3 vel = texture(uVel, pos).xyz * 2.0 - 1.0;
  pos += vel * uDt * uSpeed;
  pos = fract(pos + 1.0);            // domaine periodique, sans couture
  life -= uDt * uLifeRate;

  if (life <= 0.0) {
    pos = hash3(vUv, uSeed);
    life = 1.0;
  }
  outColor = vec4(pos, life);
}
`;

const POINTS_VS = /* glsl */ `
precision highp float;
precision highp sampler3D;
in vec2 aRef;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D uPos;
uniform sampler3D uVel;
uniform float uSize;
uniform float uScale;
uniform float uPixelRatio;
out vec3 vColor;
out float vAlpha;

vec3 palette(float t) {
  vec3 a = vec3(0.08, 0.28, 0.42);
  vec3 b = vec3(0.20, 0.90, 0.70);
  vec3 c = vec3(0.95, 0.85, 0.45);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, (t - 0.5) * 2.0);
}

void main() {
  vec4 state = texture(uPos, aRef);
  vec3 world = (state.xyz - 0.5) * uScale;
  vec3 vel = texture(uVel, state.xyz).xyz * 2.0 - 1.0;
  float speed = clamp(length(vel) * 1.25, 0.0, 1.0);

  vColor = palette(speed);
  float life = state.w;
  vAlpha = min(smoothstep(0.0, 0.12, 1.0 - life), smoothstep(0.0, 0.3, life));

  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_PointSize = uSize * uPixelRatio * (6.0 / max(-mv.z, 0.1)) * (0.55 + speed);
  gl_Position = projectionMatrix * mv;
}
`;

const POINTS_FS = /* glsl */ `
precision highp float;
in vec3 vColor;
in float vAlpha;
out vec4 outColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float a = exp(-r2 * 9.0) * vAlpha;
  outColor = vec4(vColor * a, a);
}
`;

const COUNT_OPTIONS: Record<string, number> = {
  '16 k': 128,
  '65 k': 256,
  '262 k': 512,
};

export class FlowFieldScene implements SimScene {
  readonly simId = 'flowfield';
  readonly hint = 'clic-glisser : orbite · molette : zoom · bouton « nouveau reseau » : autres poids';

  private ctx!: SceneContext;
  private group = new THREE.Group();
  private velTexture: THREE.Data3DTexture | null = null;
  private velBytes: Uint8Array | null = null;
  private gridSize = 0;

  private rtA!: THREE.WebGLRenderTarget;
  private rtB!: THREE.WebGLRenderTarget;
  private advectMat!: THREE.RawShaderMaterial;
  private pointsMat!: THREE.RawShaderMaterial;
  private points!: THREE.Points;
  private quadScene = new THREE.Scene();
  private quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private quad!: THREE.Mesh;
  private texSize = 256;

  private opts = { speed: 0.35, size: 2.6, autorotate: true, count: '65 k' };
  private stats: Record<string, string> = {};
  private needsInit = true;

  init(ctx: SceneContext): void {
    this.ctx = ctx;
    ctx.scene.add(this.group);
    ctx.camera.position.set(4.4, 2.8, 5.2);

    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(6, 6, 6)),
      new THREE.LineBasicMaterial({ color: 0x1d4657, transparent: true, opacity: 0.55 }),
    );
    this.group.add(box);

    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quadScene.add(this.quad);

    this.buildTargets(COUNT_OPTIONS[this.opts.count]);
    this.buildPoints();
  }

  /* ---------------- GPGPU ---------------- */
  private makeTarget(size: number): THREE.WebGLRenderTarget {
    const floatOk = this.ctx.renderer.extensions.has('EXT_color_buffer_float');
    return new THREE.WebGLRenderTarget(size, size, {
      type: floatOk ? THREE.FloatType : THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
  }

  private buildTargets(size: number): void {
    this.rtA?.dispose();
    this.rtB?.dispose();
    this.texSize = size;
    this.rtA = this.makeTarget(size);
    this.rtB = this.makeTarget(size);

    this.advectMat?.dispose();
    this.advectMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: QUAD_VS,
      fragmentShader: ADVECT_FS,
      uniforms: {
        uPos: { value: this.rtA.texture },
        uVel: { value: this.velTexture },
        uDt: { value: 0.016 },
        uSpeed: { value: this.opts.speed },
        uLifeRate: { value: 0.22 },
        uSeed: { value: Math.random() * 100 },
        uInit: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.needsInit = true;
  }

  private buildPoints(): void {
    if (this.points) {
      this.group.remove(this.points);
      this.points.geometry.dispose();
    }
    const n = this.texSize * this.texSize;
    const refs = new Float32Array(n * 2);
    const dummy = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      refs[i * 2] = ((i % this.texSize) + 0.5) / this.texSize;
      refs[i * 2 + 1] = (Math.floor(i / this.texSize) + 0.5) / this.texSize;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(dummy, 3));
    geo.setAttribute('aRef', new THREE.BufferAttribute(refs, 2));

    this.pointsMat?.dispose();
    this.pointsMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POINTS_VS,
      fragmentShader: POINTS_FS,
      uniforms: {
        uPos: { value: this.rtA.texture },
        uVel: { value: this.velTexture },
        uSize: { value: this.opts.size },
        uScale: { value: 6.0 },
        uPixelRatio: { value: this.ctx.renderer.getPixelRatio() },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: true,
      depthWrite: false,
    });

    this.points = new THREE.Points(geo, this.pointsMat);
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }

  /* ---------------- flux serveur ---------------- */
  frame(frame: SimFrame): void {
    const { header, data } = frame;
    if (header.kind !== 'velocity_grid' || !data) return;
    const n = header.shape[0] ?? 0;
    if (!n) return;

    if (n !== this.gridSize) {
      this.gridSize = n;
      this.velBytes = new Uint8Array(n * n * n * 4);
      this.velTexture?.dispose();
      const tex = new THREE.Data3DTexture(this.velBytes, n, n, n);
      tex.format = THREE.RGBAFormat;
      tex.type = THREE.UnsignedByteType;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.wrapR = THREE.RepeatWrapping;
      tex.unpackAlignment = 4;
      tex.needsUpdate = true;
      this.velTexture = tex;
      this.advectMat.uniforms.uVel.value = tex;
      this.pointsMat.uniforms.uVel.value = tex;
    }

    // v in [-1,1] -> octet [0,255] : 8 bits suffisent, le champ est lisse
    const bytes = this.velBytes as Uint8Array;
    const total = n * n * n;
    for (let i = 0; i < total; i += 1) {
      const s = i * 3;
      const d = i * 4;
      bytes[d] = Math.max(0, Math.min(255, (data[s] * 0.5 + 0.5) * 255));
      bytes[d + 1] = Math.max(0, Math.min(255, (data[s + 1] * 0.5 + 0.5) * 255));
      bytes[d + 2] = Math.max(0, Math.min(255, (data[s + 2] * 0.5 + 0.5) * 255));
      bytes[d + 3] = 255;
    }
    if (this.velTexture) this.velTexture.needsUpdate = true;

    this.stats = {
      grille: `${n}³ = ${(header.shape[0] ** 3).toLocaleString('fr-FR')} pts`,
      'MLP cache': `${statNumber(header.stats, 'hidden')} × 2`,
      'MFLOP / tick': statNumber(header.stats, 'mflop_per_tick').toFixed(1),
      'GFLOPS noyau': statNumber(header.stats, 'gflops').toFixed(1),
      'vitesse moy.': statNumber(header.stats, 'speed_mean').toFixed(3),
      'div. relative': statNumber(header.stats, 'div_rel').toExponential(1),
    };
  }

  /* ---------------- boucle ---------------- */
  update(dt: number): void {
    if (!this.velTexture) return;
    const renderer = this.ctx.renderer;
    const prevTarget = renderer.getRenderTarget();

    const passes = this.needsInit ? 2 : 1;
    for (let p = 0; p < passes; p += 1) {
      this.advectMat.uniforms.uInit.value = this.needsInit && p === 0 ? 1 : 0;
      this.advectMat.uniforms.uPos.value = this.rtA.texture;
      this.advectMat.uniforms.uDt.value = Math.min(dt, 0.05);
      this.advectMat.uniforms.uSpeed.value = this.opts.speed;
      this.advectMat.uniforms.uSeed.value = Math.random() * 100;
      this.quad.material = this.advectMat;

      renderer.setRenderTarget(this.rtB);
      renderer.render(this.quadScene, this.quadCamera);
      const tmp = this.rtA;
      this.rtA = this.rtB;
      this.rtB = tmp;
    }
    this.needsInit = false;
    renderer.setRenderTarget(prevTarget);

    this.pointsMat.uniforms.uPos.value = this.rtA.texture;
    this.pointsMat.uniforms.uSize.value = this.opts.size;
    this.pointsMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
    if (this.opts.autorotate) this.group.rotation.y += dt * 0.05;
  }

  telemetry(): Record<string, string> {
    return { particules: (this.texSize * this.texSize).toLocaleString('fr-FR'), ...this.stats };
  }

  buildPanel(host: HTMLElement, api: SceneApi): void {
    const button = document.createElement('button');
    button.className = 'btn';
    button.textContent = 'nouveau reseau (reseed)';
    button.onclick = () => {
      api.command({ type: 'reseed' });
      api.toast('nouveaux poids tires — la topologie du flux change', 'ok');
    };
    const note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      'Le serveur evalue le MLP sur la grille et renvoie rot(A) quantifie en int16 ; '
      + "le GPU advecte les particules dans une texture 3D — aucune position n'est transmise.";
    host.append(button, note);
  }

  clientGui(gui: GUI): void {
    const folder = gui.addFolder('Rendu (client / GPU)');
    folder.add(this.opts, 'count', Object.keys(COUNT_OPTIONS)).name('particules').onChange((value: string) => {
      const size = COUNT_OPTIONS[value] ?? 256;
      if (size === this.texSize) return;
      this.buildTargets(size);
      this.buildPoints();
    });
    folder.add(this.opts, 'speed', 0.05, 1.5, 0.01).name('vitesse advection');
    folder.add(this.opts, 'size', 0.6, 8, 0.1).name('taille point');
    folder.add(this.opts, 'autorotate').name('rotation auto');
  }

  dispose(): void {
    this.rtA?.dispose();
    this.rtB?.dispose();
    this.velTexture?.dispose();
    this.pointsMat?.dispose();
    this.advectMat?.dispose();
    this.points?.geometry.dispose();
    this.quad.geometry.dispose();
    this.group.clear();
  }
}
