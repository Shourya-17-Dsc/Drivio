import { defineConfig } from "@playwright/test";

// Runs against the simulated driver, so no camera or audio device is needed.
export default defineConfig({
  testDir: ".",
  timeout: 45_000,
  use: { baseURL: "http://127.0.0.1:5173" },
  webServer: [
    {
      command: "python -m uvicorn app.main:app --port 8000",
      cwd: "../backend",
      env: { DRIVIO_SIMULATE: "1", DRIVIO_AUDIO: "0", DRIVIO_DATA_DIR: "./data/e2e" },
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
    },
    { command: "npm run dev -- --host 127.0.0.1", cwd: "../frontend", url: "http://127.0.0.1:5173", reuseExistingServer: true },
  ],
});
