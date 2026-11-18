import { useState } from "react";
import { api, post, useLoad } from "../api";
import type { DriverProfile, LiveState, SessionSummary } from "../types";
import Gauge from "./Gauge";
import { ErrorNote, LEVEL_LABEL, STATE_COLOR, STATE_LABEL, StatusDot, fmtDuration } from "./ui";

const LEVEL_TONE = ["", "border-caution bg-caution/15", "border-caution bg-caution/25", "border-danger bg-danger/25"];

export default function Live({ live, connected }: { live: LiveState | null; connected: boolean }) {
  const [label, setLabel] = useState("Morning commute");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [newDriver, setNewDriver] = useState("");
  const drivers = useLoad<DriverProfile[]>(() => api("/drivers"), []);

  if (!live) {
    return (
      <div className="panel p-10 text-center" role="status">
        <p className="font-display text-2xl font-semibold">{connected ? "Waiting for the first frame" : "Backend not reachable"}</p>
        <p className="mt-2 text-sm text-muted">
          {connected ? "Face tracking is starting up." : "Start it with: uvicorn app.main:app --port 8000 (inside the backend folder)."}
        </p>
      </div>
    );
  }

  const { metrics: m, health } = live;
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); setError(null); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const createDriver = () => act(async () => {
    const profile = await post<DriverProfile>("/drivers", { name: newDriver });
    await post(`/drivers/${profile.id}/select`);
    setNewDriver("");
    drivers.reload();
  });
  const lead = live.factors[0];
  const stats: [string, string][] = [
    ["Eye openness", m.ear.toFixed(2)],
    ["Counted as closed below", m.ear_threshold.toFixed(2)],
    ["Mouth opening", m.mar.toFixed(2)],
    ["Eyes closed, last minute", `${(m.perclos * 100).toFixed(0)}%`],
    ["Blinks per minute", m.observed_s < 15 ? "measuring" : m.blink_rate_pm.toFixed(0)],
    ["Yawns, last 10 min", String(m.yawns_recent)],
    ["Head turn", `${Math.abs(m.yaw).toFixed(0)} degrees`],
    ["Eye direction", `${m.gaze >= 0 ? "+" : ""}${m.gaze.toFixed(2)}`],
    ["Reading confidence", `${(live.confidence * 100).toFixed(0)}%`],
  ];

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
      <section aria-label="Camera" className="space-y-3">
        <div className={`panel relative overflow-hidden ${live.level === 3 ? "critical-edge" : ""}`}>
          <img src="/api/video" alt="Live driver camera with face tracking overlay" className="aspect-[4/3] w-full bg-night object-cover" />
          {live.level > 0 && (
            <div role="alert" className={`absolute inset-x-3 top-3 rounded-md border px-4 py-3 backdrop-blur ${LEVEL_TONE[live.level]}`}>
              <p className="font-display text-2xl font-semibold leading-none">{LEVEL_LABEL[live.level]}: {STATE_LABEL[live.state]}</p>
              {lead && <p className="mt-1 text-sm">{lead.detail}</p>}
            </div>
          )}
          {live.mode === "simulation" && (
            <p className="absolute bottom-3 left-3 rounded bg-night/80 px-2 py-1 text-xs">Simulated driver. No camera is in use.</p>
          )}
        </div>

        <div className="panel flex flex-wrap items-center gap-3 p-3">
          {live.recording ? (
            <button className="btn btn-danger" disabled={busy} onClick={() => act(() => post<SessionSummary>("/session/stop"))}>Stop recording</button>
          ) : (
            <>
              <label className="sr-only" htmlFor="session-label">Session name</label>
              <input id="session-label" value={label} onChange={(e) => setLabel(e.target.value)}
                     className="w-48 rounded-md border border-line bg-night px-3 py-2 text-sm" />
              <button className="btn btn-primary" disabled={busy} onClick={() => act(() => post("/session/start", { label, driver_id: live.driver_id }))}>Start recording</button>
            </>
          )}
          <button className="btn" disabled={busy} onClick={() => act(() => post("/calibrate", live.driver_id ? { driver_id: live.driver_id } : undefined))}>Recalibrate</button>
          <span className="ml-auto text-xs text-muted">
            {m.calibrated ? "Calibrated to this driver" : "Calibrating: look at the road"}
          </span>
        </div>
        <div className="panel flex flex-wrap items-center gap-2 p-3">
          <label className="text-xs text-muted" htmlFor="driver-profile">Driver profile</label>
          <select id="driver-profile" className="rounded-md border border-line bg-night px-3 py-2 text-sm" value={live.driver_id ?? ""}
                  onChange={(e) => act(() => post(e.target.value ? `/drivers/${e.target.value}/select` : "/drivers/clear"))} disabled={busy || live.recording}>
            <option value="">No profile selected</option>
            {drivers.data.map((driver) => <option key={driver.id} value={driver.id}>{driver.name}{driver.calibration_available ? " · calibrated" : ""}</option>)}
          </select>
          <input value={newDriver} onChange={(e) => setNewDriver(e.target.value)} placeholder="New driver name" aria-label="New driver name"
                 className="w-40 rounded-md border border-line bg-night px-3 py-2 text-sm" disabled={busy || live.recording} />
          <button className="btn" disabled={busy || live.recording || !newDriver.trim()} onClick={createDriver}>Create profile</button>
        </div>
        <ErrorNote message={error ?? drivers.error} />
      </section>

      <section aria-label="Driver state" className="space-y-5">
        <div className="panel p-4">
          <Gauge score={live.score} state={live.state} />
          <p className="text-center font-display text-3xl font-semibold" style={{ color: STATE_COLOR[live.state] }}>
            {STATE_LABEL[live.state]}
          </p>
        </div>

        <div className={`panel p-4 ${live.safety.break_due ? "border-caution bg-caution/10" : ""}`}>
          <div className="flex items-start justify-between gap-3">
            <div><h2 className="font-display text-xl font-semibold">Trip safety</h2><p className="text-xs text-muted">Session-level exposure score</p></div>
            <span className="font-display text-3xl font-semibold">{live.safety.score}</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div><p className="text-xs text-muted">Driving time</p><p>{fmtDuration(live.safety.drive_duration_s)}</p></div>
            <div><p className="text-xs text-muted">Fatigue exposure</p><p>{fmtDuration(live.safety.fatigue_duration_s)}</p></div>
            <div><p className="text-xs text-muted">Distraction exposure</p><p>{fmtDuration(live.safety.distraction_duration_s)}</p></div>
            <div><p className="text-xs text-muted">Breaks taken</p><p>{live.safety.breaks_taken}</p></div>
          </div>
          <p className={`mt-3 text-sm ${live.safety.break_due ? "text-caution" : "text-muted"}`}>{live.safety.recommendation}</p>
          {live.recording && <button className="btn mt-3 w-full" disabled={busy} onClick={() => act(() => post("/session/break"))}>Record break taken</button>}
        </div>

        <div className="panel p-4">
          <h2 className="font-display text-xl font-semibold">Why this score</h2>
          {live.factors.length === 0 ? (
            <p className="mt-2 text-sm text-muted">Nothing is lowering the score right now.</p>
          ) : (
            <ul className="mt-3 space-y-3">
              {live.factors.map((f) => (
                <li key={f.name}>
                  <div className="flex justify-between text-sm"><span>{f.name}</span><span className="text-muted">-{f.penalty.toFixed(0)}</span></div>
                  <div className="mt-1 h-1.5 rounded bg-night"><div className="h-full rounded bg-caution" style={{ width: `${Math.min(100, f.penalty * 2.5)}%` }} /></div>
                  <p className="mt-1 text-xs text-muted">{f.detail}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section aria-label="Measurements" className="panel p-4">
        <h2 className="font-display text-xl font-semibold">Measurements</h2>
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          {stats.map(([k, v]) => (
            <div key={k}><dt className="text-xs text-muted">{k}</dt><dd className="font-display text-2xl font-semibold">{v}</dd></div>
          ))}
        </dl>
      </section>

      <section aria-label="Alerts and system health" className="space-y-5">
        <div className="panel p-4">
          <h2 className="font-display text-xl font-semibold">Alert log</h2>
          {live.events.length === 0 ? (
            <p className="mt-2 text-sm text-muted">No alerts since the monitor started.</p>
          ) : (
            <ul className="mt-2 max-h-40 space-y-2 overflow-y-auto text-sm">
              {[...live.events].reverse().map((e) => (
                <li key={e.ts} className="flex gap-3">
                  <time className="shrink-0 text-muted">{new Date(e.ts * 1000).toLocaleTimeString()}</time>
                  <span>{e.kind === "recovery" ? "Recovered" : LEVEL_LABEL[e.level]}: {e.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="panel flex flex-wrap gap-x-5 gap-y-2 p-4">
          <StatusDot ok={health.camera_ok} label={live.mode === "simulation" ? "Simulator" : "Camera"} />
          <StatusDot ok={health.vision_ok} label="Face tracking" />
          <StatusDot ok={health.audio_ok} label="Audio alerts" />
          <StatusDot ok={health.phone_detection} label="Phone detection" />
          <span className="text-xs text-muted">{health.fps.toFixed(0)} fps, {health.latency_ms.toFixed(0)} ms per frame, up {fmtDuration(health.uptime_s)}{health.camera_backend ? ` · ${health.camera_backend}` : ""}</span>
          {health.error && <p className="w-full text-xs text-caution">{health.error}</p>}
        </div>
      </section>
    </div>
  );
}
