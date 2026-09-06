import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type GUI from 'lil-gui';
import type { SimFrame } from '../net/protocol';

export interface SceneContext {
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  width: number;
  height: number;
}

export interface SceneApi {
  /** Envoie une commande a la simulation (serveur ou moteur local). */
  command(message: Record<string, unknown>): void;
  /** Pousse de nouveaux parametres serveur. */
  setParams(params: Record<string, unknown>): void;
  toast(message: string, kind?: 'ok' | 'warn' | 'error'): void;
}

export interface SimScene {
  /** Identifiant de simulation serveur, ou null si la scene est autonome. */
  readonly simId: string | null;
  readonly hint: string;
  init(ctx: SceneContext, api: SceneApi): void | Promise<void>;
  frame?(frame: SimFrame): void;
  update(dt: number, elapsed: number): void;
  resize?(width: number, height: number): void;
  telemetry?(): Record<string, string>;
  buildPanel?(host: HTMLElement, api: SceneApi): void;
  /** Controles de rendu purement client (non transmis au serveur). */
  clientGui?(gui: GUI): void;
  pointer?(event: PointerEvent, ndc: THREE.Vector2, kind: 'down' | 'move'): void;
  dispose(): void;
}
