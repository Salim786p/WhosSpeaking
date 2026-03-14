import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/health': 'http://127.0.0.1:5000',
      '/train': 'http://127.0.0.1:5000',
      '/predict': 'http://127.0.0.1:5000',
    },
  },
})
