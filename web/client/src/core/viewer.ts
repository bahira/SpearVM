/**
 * Coquille de rendu : un seul WebGLRenderer/canvas partage par toutes les
 * scenes (evite les pertes de contexte au changement d'onglet), boucle
 * rAF unique, gestion du redimensionnement et de la visibilite de l'onglet.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneApi, SceneContext, SimScene } from './types';

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  scene: THREE.Scene;

  private active: SimScene | null = null;
  private clock = new THREE.Clock();
  private raf = 0;
  private frames = 0;
  private fpsTime = 0;
  fps = 0;
  private paused = false;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x05080c, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.05, 200);
    this.camera.position.set(4.2, 3.0, 5.4);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 40;

    window.addEventListener('resize', this.onResize);
    document.addEventListener('visibilitychange', () => {
      this.paused = document.hidden;
      if (!this.paused) this.clock.getDelta();
    });
    canvas.addEventListener('pointerdown', (e) => this.onPointer(e, 'down'));
    canvas.addEventListener('pointermove', (e) => this.onPointer(e, 'move'));
    this.onResize();
  }

  get context(): SceneContext {
    return {
      renderer: this.renderer,
      canvas: this.canvas,
      scene: this.scene,
      camera: this.camera,
      controls: this.controls,
      width: this.canvas.clientWidth,
      height: this.canvas.clientHeight,
    };
  }

  private onPointer = (event: PointerEvent, kind: 'down' | 'move'): void => {
    if (!this.active?.pointer) return;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.active.pointer(event, ndc, kind);
  };

  private onResize = (): void => {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    this.active?.resize?.(w, h);
  };

  async setScene(scene: SimScene, api: SceneApi): Promise<void> {
    this.active?.dispose();
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x05080c, 0.02);
    this.camera.position.set(4.2, 3.0, 5.4);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.active = scene;
    await scene.init(this.context, api);
    this.onResize();
  }

  get current(): SimScene | null {
    return this.active;
  }

  start(): void {
    const loop = (): void => {
      this.raf = requestAnimationFrame(loop);
      const dt = Math.min(this.clock.getDelta(), 0.1);
      if (this.paused) return;
      this.controls.update();
      this.active?.update(dt, this.clock.elapsedTime);
      this.renderer.render(this.scene, this.camera);

      this.frames += 1;
      this.fpsTime += dt;
      if (this.fpsTime >= 0.5) {
        this.fps = this.frames / this.fpsTime;
        this.frames = 0;
        this.fpsTime = 0;
      }
    };
    loop();
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
  }

  dispose(): void {
    this.stop();
    this.active?.dispose();
    this.renderer.dispose();
  }
}
