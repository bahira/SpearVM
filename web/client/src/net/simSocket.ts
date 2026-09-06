/**
 * Client WebSocket du serveur de simulation.
 *
 * Points "production" : reconnexion exponentielle bornee, mesure de latence
 * (ping/pong applicatif), compteur de bande passante, file de messages
 * sortants quand la socket n'est pas encore ouverte, et bascule automatique
 * vers un moteur local si le serveur reste injoignable.
 */

import { decodeFrame, type SimFrame } from './protocol';

export type LinkState = 'connecting' | 'live' | 'retrying' | 'local' | 'closed';

export interface SimDescription {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  kernels: string[];
  rate: number;
  params: ParamSpec[];
}

export interface ParamSpec {
  key: string;
  label: string;
  kind: 'range' | 'select' | 'bool';
  default: unknown;
  help?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: (string | number)[];
}

export interface HelloMessage {
  type: 'hello';
  sim: SimDescription;
  params: Record<string, unknown>;
  meta: Record<string, unknown>;
  rate: number;
  backend: Record<string, unknown>;
  protocol: string;
}

export interface SocketHandlers {
  onHello?(msg: HelloMessage): void;
  onFrame?(frame: SimFrame): void;
  onState?(state: LinkState, detail?: string): void;
  onParams?(params: Record<string, unknown>): void;
}

const MAX_BACKOFF = 8000;

export class SimSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private attempt = 0;
  private timer: number | null = null;
  private pingTimer: number | null = null;
  private queue: string[] = [];

  latencyMs = 0;
  bytesPerSecond = 0;
  framesPerSecond = 0;
  state: LinkState = 'connecting';

  private windowBytes = 0;
  private windowFrames = 0;
  private windowStart = performance.now();

  constructor(
    private readonly simId: string,
    private readonly handlers: SocketHandlers,
    private readonly params: Record<string, unknown> = {},
  ) {
    this.connect();
  }

  private url(): string {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const qs = new URLSearchParams();
    if (Object.keys(this.params).length) qs.set('params', JSON.stringify(this.params));
    const query = qs.toString();
    return `${proto}://${window.location.host}/ws/sim/${this.simId}${query ? `?${query}` : ''}`;
  }

  private setState(state: LinkState, detail?: string): void {
    if (this.state === state && !detail) return;
    this.state = state;
    this.handlers.onState?.(state, detail);
  }

  private connect(): void {
    if (this.closed) return;
    this.setState(this.attempt === 0 ? 'connecting' : 'retrying');

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url());
    } catch (err) {
      this.scheduleReconnect(String(err));
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.setState('live');
      while (this.queue.length) ws.send(this.queue.shift() as string);
      this.pingTimer = window.setInterval(() => {
        this.send({ type: 'ping', t: performance.now() });
      }, 3000);
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        this.handleText(event.data);
        return;
      }
      const frame = decodeFrame(event.data as ArrayBuffer);
      this.accountBandwidth(frame.bytes);
      this.handlers.onFrame?.(frame);
    };

    ws.onerror = () => {
      /* onclose fera le travail */
    };

    ws.onclose = (event) => {
      this.clearPing();
      if (this.closed) {
        this.setState('closed');
        return;
      }
      this.scheduleReconnect(event.reason || `code ${event.code}`);
    };
  }

  private handleText(raw: string): void {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.type === 'hello') this.handlers.onHello?.(msg as unknown as HelloMessage);
    else if (msg.type === 'params') this.handlers.onParams?.(msg.params as Record<string, unknown>);
    else if (msg.type === 'pong' && typeof msg.t === 'number') {
      this.latencyMs = performance.now() - msg.t;
    }
  }

  private accountBandwidth(bytes: number): void {
    this.windowBytes += bytes;
    this.windowFrames += 1;
    const now = performance.now();
    const dt = now - this.windowStart;
    if (dt >= 1000) {
      this.bytesPerSecond = (this.windowBytes * 1000) / dt;
      this.framesPerSecond = (this.windowFrames * 1000) / dt;
      this.windowBytes = 0;
      this.windowFrames = 0;
      this.windowStart = now;
    }
  }

  private scheduleReconnect(detail: string): void {
    this.ws = null;
    this.attempt += 1;
    const delay = Math.min(MAX_BACKOFF, 400 * 2 ** (this.attempt - 1));
    this.setState(this.attempt >= 3 ? 'local' : 'retrying', detail);
    this.timer = window.setTimeout(() => this.connect(), delay);
  }

  private clearPing(): void {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  send(message: Record<string, unknown>): void {
    const payload = JSON.stringify(message);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(payload);
    else if (this.queue.length < 32) this.queue.push(payload);
  }

  setParams(params: Record<string, unknown>): void {
    this.send({ type: 'params', params });
  }

  command(message: Record<string, unknown>): void {
    this.send({ type: 'command', ...message });
  }

  close(): void {
    this.closed = true;
    this.clearPing();
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
  }
}
