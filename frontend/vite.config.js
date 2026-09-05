import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Override if port 8000 is unavailable (e.g. a stuck/stale socket - a known
// Windows quirk where a port shows LISTENING in netstat with no real owning
// process, and only clears on reboot): BACKEND_PORT=8080 npm run dev
const backendPort = process.env.BACKEND_PORT || 8000;
const backendUrl = `http://localhost:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backendUrl,
      "/media": backendUrl,
    },
  },
});
