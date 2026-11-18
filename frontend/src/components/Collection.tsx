import { useEffect, useState } from "react";
import { api, post, useLiveState, useLoad } from "../api";
import type { TemporalDatasetStatus } from "../types";
import { ErrorNote } from "./ui";

const LABELS = [
  ["alert", "Alert / non-drowsy"],
  ["drowsy", "Drowsy"],
  ["fatigued", "Fatigued"],
  ["distracted", "Distracted"],
] as const;

const PROTOCOL = [
  { label: "alert", title: "Alert baseline", instruction: "Sit normally and look at the road/camera." },
  { label: "drowsy", title: "Fatigue demonstration", instruction: "Only while stationary: briefly demonstrate heavy eyelids or a slow blink. Never do this while driving." },
  { label: "distracted", title: "Distraction demonstration", instruction: "Only while stationary: briefly look away from the camera. Do not use a phone while driving." },
  { label: "alert", title: "Recovery baseline", instruction: "Return to a normal, attentive posture." },
] as const;

export default function Collection() {
  const { state: live } = useLiveState();
  const dataset = useLoad<TemporalDatasetStatus>(() => api<TemporalDatasetStatus>("/ml/temporal-dataset/status"), {
    labelled_samples: 0, labelled_sessions: 0, labels: {}, sessions_by_label: {},
    minimum_sessions_required: 6, ready_for_temporal_training: false,
    reason: "Collect and label samples before training",
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [exported, setExported] = useState<number | null>(null);
  const [phase, setPhase] = useState(0);
  const [phaseSeconds, setPhaseSeconds] = useState(0);

  useEffect(() => {
    if (!live?.recording) return;
    const timer = window.setInterval(() => setPhaseSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [live?.recording]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  return (
    <div className="space-y-5">
      <ErrorNote message={error} />
      <section className="panel p-5">
        <h2 className="font-display text-3xl font-semibold">Data collection</h2>
        <p className="mt-2 max-w-3xl text-sm text-muted">Collect labelled driver measurements for research and model evaluation. Drivio stores derived metrics and labels, not raw video.</p>
        <div className="mt-4 rounded-md border border-line bg-night/40 p-3 text-sm">
          <p className="font-medium">Safe collection protocol</p>
          <ul className="mt-2 list-inside list-disc space-y-1 text-muted"><li>Use a stationary camera or simulator.</li><li>Do not ask anyone to drive while drowsy or intentionally use a phone.</li><li>Obtain participant consent and use anonymous IDs.</li></ul>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {!live?.recording ? <button className="btn btn-primary" disabled={busy} onClick={() => act(async () => { await post("/session/start", { label: "Guided labelled collection" }); setPhase(0); setPhaseSeconds(0); await post("/session/label", { label: PROTOCOL[0].label }); })}>Start guided collection</button> : <button className="btn btn-danger" disabled={busy} onClick={() => act(async () => { await post("/session/stop"); setPhaseSeconds(0); dataset.reload(); })}>Stop collection</button>}
          <span className="text-sm text-muted">{live?.recording ? "Collection is recording once per second." : "No collection session active."}</span>
        </div>
      </section>

      <section className="panel p-5" aria-label="Temporal dataset readiness">
        <div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-display text-xl font-semibold">Temporal dataset readiness</h3><p className="mt-1 text-sm text-muted">The GRU evaluates complete sessions, so it needs multiple sessions across at least two conditions.</p></div><span className={`rounded-full px-3 py-1 text-xs ${dataset.data.ready_for_temporal_training ? "bg-ok/20 text-ok" : "bg-caution/20 text-caution"}`}>{dataset.data.ready_for_temporal_training ? "Ready to train" : "Collect more data"}</span></div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3"><div><p className="text-xs text-muted">Labelled samples</p><p className="font-display text-2xl font-semibold">{dataset.data.labelled_samples.toLocaleString()}</p></div><div><p className="text-xs text-muted">Labelled sessions</p><p className="font-display text-2xl font-semibold">{dataset.data.labelled_sessions} / {dataset.data.minimum_sessions_required}</p></div><div><p className="text-xs text-muted">Conditions</p><p className="font-display text-2xl font-semibold">{Object.keys(dataset.data.labels).length}</p></div></div>
        <p className="mt-3 text-sm text-muted">{dataset.data.reason}{Object.keys(dataset.data.labels).length > 0 ? ` · ${Object.entries(dataset.data.labels).map(([label, count]) => `${label}: ${count}`).join(", ")}` : ""}</p>
      </section>

      <section className="panel p-5" aria-label="Guided collection protocol">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h3 className="font-display text-xl font-semibold">Guided protocol</h3><p className="mt-1 text-sm text-muted">Use one phase at a time, collect at least 20–30 seconds per phase, then advance manually. This creates clean session-level labels for the temporal model.</p></div>
          {live?.recording && <span className="rounded-full bg-ok/20 px-3 py-1 text-xs text-ok">Phase time {Math.floor(phaseSeconds / 60)}:{String(phaseSeconds % 60).padStart(2, "0")}</span>}
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-4">
          {PROTOCOL.map((item, index) => <button key={`${item.title}-${index}`} className={`rounded-md border p-3 text-left text-sm ${index === phase ? "border-ok bg-ok/10" : "border-line"}`} disabled={!live?.recording || busy} onClick={() => act(async () => { await post("/session/label", { label: item.label }); setPhase(index); setPhaseSeconds(0); })}><span className="text-xs text-muted">Phase {index + 1}</span><span className="mt-1 block font-medium">{item.title}</span><span className="mt-1 block text-xs text-muted">{item.label}</span></button>)}
        </div>
        {live?.recording && <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-line bg-night/40 p-3 text-sm"><p><span className="font-medium">{PROTOCOL[phase].title}:</span> {PROTOCOL[phase].instruction}</p><button className="btn" disabled={busy || phase === PROTOCOL.length - 1} onClick={() => act(async () => { const next = phase + 1; await post("/session/label", { label: PROTOCOL[next].label }); setPhase(next); setPhaseSeconds(0); })}>Next phase</button></div>}
      </section>

      <section className="panel p-5">
        <h3 className="font-display text-xl font-semibold">Ground-truth label</h3>
        <p className="mt-1 text-sm text-muted">Choose the condition being demonstrated. Change it whenever the participant changes activity.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {LABELS.map(([value, label]) => <button key={value} className={`btn ${live?.ground_truth === value ? "border-ok bg-ok/20" : ""}`} disabled={busy || !live?.recording} onClick={() => act(() => post("/session/label", { label: value }))}>{label}</button>)}
          <button className="btn" disabled={busy || !live?.recording} onClick={() => act(() => post("/session/label", { label: null }))}>Clear label</button>
        </div>
        <p className="mt-3 text-sm">Current label: <span className="font-medium">{live?.ground_truth ?? "None"}</span></p>
      </section>

      <section className="panel p-5">
        <h3 className="font-display text-xl font-semibold">Export labelled measurements</h3>
        <p className="mt-1 text-sm text-muted">Exports only explicitly labelled samples using the feature columns expected by the existing ML experiments.</p>
        <button className="btn mt-4" disabled={busy} onClick={() => act(async () => { const result = await post<{ rows: number }>("/ml/datasets/from-labeled-sessions", { name: "labelled_sessions" }); setExported(result.rows); })}>Export labelled dataset</button>
        {exported !== null && <p className="mt-2 text-sm text-ok">Exported {exported.toLocaleString()} labelled samples.</p>}
      </section>
    </div>
  );
}
