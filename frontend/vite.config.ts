import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/sim': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
        // Don't proxy page navigation (HTML requests) — let React Router handle them
        bypass(req) {
          if (req.headers.accept?.includes('text/html')) return req.url
        },
      },
    },
  },
})
