import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The frontend calls a same-origin '/api' by default. In dev we proxy that
// to the FastAPI backend so there are no CORS concerns and the demo works
// with `npm run dev` + `uvicorn backend.main:app`. Override the target with
// VITE_BACKEND_URL if the backend runs elsewhere.
const BACKEND = process.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'

function stalePwaFallback() {
  return {
    name: 'stale-pwa-fallback',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        if (req.url && req.url.includes('@vite-plugin-pwa')) {
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/javascript')
          res.end('export default {}; export const pwaInfo = {};')
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), stalePwaFallback()],
  server: {
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
})
