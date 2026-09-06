import { defineConfig } from 'vite';

// Cible du serveur de simulation Python (uvicorn).
const API_TARGET = process.env.SPEARVM_API ?? 'http://127.0.0.1:8000';

// En preview distante (sandbox/proxy HTTPS), le websocket HMR doit sortir en wss:443.
const proxiedHmr = process.env.VITE_PROXY_HMR === '1';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // le proxy de preview presente un hostname arbitraire : on l'autorise
    allowedHosts: true,
    hmr: proxiedHmr ? { protocol: 'wss', clientPort: 443 } : undefined,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
      '/ws': { target: API_TARGET, ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2020',
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: { three: ['three'] },
      },
    },
  },
});
