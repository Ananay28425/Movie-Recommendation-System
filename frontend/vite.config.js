// Purpose: route local frontend API calls to the FastAPI development server.
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
