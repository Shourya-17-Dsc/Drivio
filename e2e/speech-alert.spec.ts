import { expect, test } from "@playwright/test";

test("speaks a caution prompt once when an alert starts", async ({ page }) => {
  await page.addInitScript(() => {
    const testWindow = window as Window & { __spokenAlerts: string[] };
    testWindow.__spokenAlerts = [];
    Object.defineProperty(window, "speechSynthesis", {
      configurable: true,
      value: {
        speak: (utterance: { text: string }) => testWindow.__spokenAlerts.push(utterance.text),
      },
    });
  });

  await page.routeWebSocket(/\/ws\/state$/, (socket) => {
    const state = {
      ts: Date.now() / 1000,
      mode: "simulation",
      recording: false,
      session_id: null,
      driver_id: null,
      ground_truth: null,
      score: 60,
      state: "fatigued",
      confidence: 1,
      factors: [{ name: "Eye closure", penalty: 5, detail: "Eyes closed", kind: "fatigue" }],
      metrics: {
        ear: 0.3,
        ear_threshold: 0.21,
        mar: 0.08,
        yaw: 0,
        pitch: 0,
        eyes_closed: false,
        closure_s: 0,
        blinks_total: 0,
        blink_rate_pm: 0,
        perclos: 0,
        yawning: false,
        yawns_recent: 0,
        head_turned_s: 0,
        face_present: true,
        calibrated: true,
        observed_s: 5,
        gaze: 0,
        gaze_off_s: 0,
      },
      level: 1,
      level_name: "caution",
      events: [],
      health: {
        camera_ok: true,
        vision_ok: true,
        audio_ok: true,
        phone_detection: false,
        fps: 20,
        latency_ms: 10,
        uptime_s: 5,
        error: null,
      },
      safety: {
        active: false,
        score: 100,
        drive_duration_s: 0,
        fatigue_duration_s: 0,
        distraction_duration_s: 0,
        critical_alerts: 0,
        breaks_taken: 0,
        break_due: false,
        recommendation: "",
      },
    };

    socket.send(JSON.stringify(state));
    setTimeout(() => socket.send(JSON.stringify(state)), 100);
  });

  await page.goto("/");
  await expect.poll(() => page.evaluate(
    () => (window as Window & { __spokenAlerts: string[] }).__spokenAlerts,
  )).toEqual(["Please be cautious!"]);
});
