/**
 * Decodeur du protocole binaire `spearvm.sim.v1`.
 *
 *   u32 headerLen | JSON utf-8 (+ padding 4o) | payload (f32 | i16)
 *
 * Les charges i16 sont dequantifiees (`value * scale`) dans un Float32Array
 * recycle par taille : a 30 Hz on evite ainsi ~1.5 Mo/s d'allocations.
 */

export interface FrameHeader {
  magic: string;
  kind: string;
  tick: number;
  time: number;
  compute_ms: number;
  dtype: 'f32' | 'i16' | 'none';
  shape: number[];
  scale: number;
  stats: Record<string, unknown>;
}

export interface SimFrame {
  header: FrameHeader;
  /** Donnees dequantifiees. Attention : buffer recycle, ne pas conserver. */
  data: Float32Array | null;
  bytes: number;
}

const pool = new Map<number, Float32Array>();

function scratch(size: number): Float32Array {
  let buf = pool.get(size);
  if (!buf) {
    buf = new Float32Array(size);
    pool.set(size, buf);
  }
  return buf;
}

const decoder = new TextDecoder();

export function decodeFrame(buffer: ArrayBuffer): SimFrame {
  const view = new DataView(buffer);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(
    decoder.decode(new Uint8Array(buffer, 4, headerLen)),
  ) as FrameHeader;

  const start = 4 + headerLen + ((4 - (headerLen % 4)) % 4);
  let data: Float32Array | null = null;

  if (header.dtype === 'f32') {
    const count = (buffer.byteLength - start) >> 2;
    data = scratch(count);
    data.set(new Float32Array(buffer, start, count));
  } else if (header.dtype === 'i16') {
    const count = (buffer.byteLength - start) >> 1;
    const src = new Int16Array(buffer, start, count);
    data = scratch(count);
    const s = header.scale;
    for (let i = 0; i < count; i += 1) data[i] = src[i] * s;
  }

  return { header, data, bytes: buffer.byteLength };
}

export function statNumber(stats: Record<string, unknown>, key: string, fallback = 0): number {
  const v = stats[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

export function statString(stats: Record<string, unknown>, key: string, fallback = '—'): string {
  const v = stats[key];
  return typeof v === 'string' ? v : fallback;
}
