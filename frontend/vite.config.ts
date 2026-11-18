import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const proxy = {
  "/api": "http://127.0.0.1:8000",
  "/ws": { target: "http://127.0.0.1:8000", ws: true },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { proxy },
});
