import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc' // <-- THIS IS THE FIX

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
})