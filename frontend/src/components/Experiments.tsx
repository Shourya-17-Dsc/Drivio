import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, post, useLoad } from "../api";
import type { Dataset, ImageDatasetAudit, ImageTrainingJob, ModelRun, TemporalTrainingJob } from "../types";
import { ErrorNote, chartTooltip } from "./ui";

const ALGO_LABEL: Record<string, string> = {
  random_forest: "Random forest", gradient_boosting: "Gradient boosting",
  logistic_regression: "Logistic regression", mlp_torch: "Neural network (PyTorch)", image_cnn: "Drowsiness image CNN",
};

export default function Experiments() {
  const imageAudit = useLoad<ImageDatasetAudit | null>(() => api("/ml/image-dataset/audit"), null);
  const datasets = useLoad<Dataset[]>(() => api("/ml/datasets"), []);
  const runs = useLoad<ModelRun[]>(() => api("/ml/compare"), []);
  const algos = useLoad<string[]>(() => api("/ml/algorithms"), []);
  const [dataset, setDataset] = useState("");
  const [algo, setAlgo] = useState("random_forest");
  const [selected, setSelected] = useState<ModelRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageJob, setImageJob] = useState<ImageTrainingJob | null>(null);
  const [temporalJob, setTemporalJob] = useState<TemporalTrainingJob | null>(null);

  useEffect(() => {
    if (!imageJob || !["queued", "split", "training"].includes(imageJob.status)) return;
    const timer = window.setInterval(() => {
      api<ImageTrainingJob>(`/ml/image-model/jobs/${imageJob.id}`).then((job) => {
        setImageJob(job);
        if (job.status === "complete") runs.reload();
        if (job.status === "failed") setError(job.error);
      }).catch((e: Error) => setError(e.message));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [imageJob?.id, imageJob?.status]);

  useEffect(() => {
    if (!temporalJob || !["queued", "training"].includes(temporalJob.status)) return;
    const timer = window.setInterval(() => {
      api<TemporalTrainingJob>(`/ml/temporal-model/jobs/${temporalJob.id}`).then((job) => {
        setTemporalJob(job);
        if (job.status === "complete") runs.reload();
        if (job.status === "failed") setError(job.error);
      }).catch((e: Error) => setError(e.message));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [temporalJob?.id, temporalJob?.status]);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); setError(null); datasets.reload(); runs.reload(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const current = dataset || datasets.data[0]?.name || "";
  const chart = runs.data.slice(0, 8).map((r) => ({ name: `#${r.id} ${ALGO_LABEL[r.algorithm] ?? r.algorithm}`, accuracy: r.accuracy, f1: r.f1_macro }));

  return (
    <div className="space-y-5">
      <ErrorNote message={error ?? datasets.error ?? imageAudit.error} />
      {imageAudit.data && (
        <section className="panel p-4" aria-label="Image dataset audit">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="font-display text-xl font-semibold">Drowsiness image dataset</h2>
              <p className="mt-1 text-sm text-muted">Validation checkpoint before image-model training.</p>
            </div>
            <span className={`rounded-full px-3 py-1 text-xs font-medium ${imageAudit.data.ready_for_training ? "bg-ok/20 text-ok" : "bg-danger/20 text-danger"}`}>
              {imageAudit.data.ready_for_training ? "Ready for audit review" : "Needs attention"}
            </span>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            <div><p className="text-xs text-muted">Readable images</p><p className="font-display text-2xl font-semibold">{imageAudit.data.total_readable.toLocaleString()}</p></div>
            <div><p className="text-xs text-muted">Drowsy</p><p className="font-display text-2xl font-semibold">{imageAudit.data.classes["Drowsy"]?.readable?.toLocaleString() ?? 0}</p></div>
            <div><p className="text-xs text-muted">Non-drowsy</p><p className="font-display text-2xl font-semibold">{imageAudit.data.classes["Non Drowsy"]?.readable?.toLocaleString() ?? 0}</p></div>
            <div><p className="text-xs text-muted">Detected groups</p><p className="font-display text-2xl font-semibold">{imageAudit.data.group_count}</p></div>
          </div>
          <div className="mt-4 rounded-md border border-line bg-night/40 p-3 text-sm">
            <p><span className="text-muted">Balance ratio:</span> {(imageAudit.data.class_balance_ratio * 100).toFixed(1)}% of the larger class</p>
            <p className="mt-1"><span className="text-muted">Most common dimensions:</span> {Object.entries(imageAudit.data.dimensions).slice(0, 3).map(([size, count]) => `${size} (${count.toLocaleString()})`).join(", ") || "none"}</p>
            {imageAudit.data.issues.length > 0 && <div className="mt-2 text-caution"><p className="font-medium">Issues to review</p><ul className="mt-1 list-inside list-disc">{imageAudit.data.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div>}
          </div>
        </section>
      )}
      <section className="panel p-4" aria-label="Image model training">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-xl font-semibold">Train drowsiness image model</h2>
            <p className="mt-1 text-sm text-muted">Uses grouped splits so images from one filename group do not cross evaluation sets.</p>
          </div>
          <div className="flex flex-wrap gap-2">
          {(["eyes", "full"] as const).map((focus) => <button key={focus} className="btn btn-primary" disabled={!imageAudit.data?.ready_for_training || ["queued", "split", "training"].includes(imageJob?.status ?? "")}
                  onClick={() => {
                    setError(null);
                    post<ImageTrainingJob>("/ml/image-model/train", { epochs: 3, batch_size: 32, image_size: 64, focus })
                      .then(setImageJob).catch((e: Error) => setError(e.message));
                  }}>
            {["queued", "split", "training"].includes(imageJob?.status ?? "") ? "Training..." : focus === "eyes" ? "Train eye-focused model" : "Train full-face baseline"}
          </button>)}
          </div>
        </div>
        {imageJob && (
          <div className="mt-4 rounded-md border border-line bg-night/40 p-3 text-sm">
            <p><span className="text-muted">Status:</span> {imageJob.status}</p>
            {imageJob.status === "training" && <p className="mt-1"><span className="text-muted">Epoch:</span> {imageJob.epoch ?? "starting"} · <span className="text-muted">Validation accuracy:</span> {imageJob.accuracy === undefined ? "measuring" : `${(imageJob.accuracy * 100).toFixed(1)}%`}</p>}
            {imageJob.result && <>
              <p className={`mt-1 ${imageJob.result.deployment_ready ? "text-ok" : "text-caution"}`}>{imageJob.result.deployment_ready ? "Passed live-use quality gate" : "Not ready for live alerts"} · Test accuracy {(imageJob.result.accuracy * 100).toFixed(1)}% · F1 {(imageJob.result.f1_macro * 100).toFixed(1)}% · Drowsy recall {(imageJob.result.drowsy_recall * 100).toFixed(1)}% · {imageJob.result.test_rows.toLocaleString()} held-out images</p>
              <p className="mt-1 text-xs text-muted">Validation-to-test F1 gap: {(imageJob.result.generalization_gap * 100).toFixed(1)} percentage points · {imageJob.result.quality_note}</p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {imageJob.result.labels.map((label, index) => <div key={label} className="rounded border border-line px-2 py-1 text-xs"><span className="font-medium">{label}</span><span className="ml-2 text-muted">Precision {(imageJob.result!.precision_by_class[index] * 100).toFixed(0)}% · Recall {(imageJob.result!.recall_by_class[index] * 100).toFixed(0)}% · F1 {(imageJob.result!.f1_by_class[index] * 100).toFixed(0)}% · n={imageJob.result!.support_by_class[index]}</span></div>)}
              </div>
              <details className="mt-2 text-xs"><summary className="cursor-pointer text-muted">View unseen-group results</summary>
                <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
                  {Object.entries(imageJob.result.test_group_metrics).sort(([, a], [, b]) => a.f1_macro - b.f1_macro).map(([group, metrics]) => <span key={group}>{group}: F1 {(metrics.f1_macro * 100).toFixed(0)}%, drowsy recall {(metrics.drowsy_recall * 100).toFixed(0)}%</span>)}
                </div>
              </details>
            </>}
            {imageJob.error && <p className="mt-1 text-danger">{imageJob.error}</p>}
          </div>
        )}
      </section>
      <section className="panel p-4" aria-label="Temporal model training">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h2 className="font-display text-xl font-semibold">Train temporal model</h2><p className="mt-1 text-sm text-muted">Uses labelled metric sequences and holds out complete sessions for testing.</p></div>
          <button className="btn btn-primary" disabled={["queued", "training"].includes(temporalJob?.status ?? "")}
                  onClick={() => { setError(null); post<TemporalTrainingJob>("/ml/temporal-model/train", { window_size: 10, stride: 5, epochs: 10, batch_size: 32 }).then(setTemporalJob).catch((e: Error) => setError(e.message)); }}>
            {["queued", "training"].includes(temporalJob?.status ?? "") ? "Training..." : "Train GRU from labelled sessions"}
          </button>
        </div>
        {temporalJob && <div className="mt-4 rounded-md border border-line bg-night/40 p-3 text-sm"><p><span className="text-muted">Status:</span> {temporalJob.status}</p>{temporalJob.status === "training" && <p className="mt-1"><span className="text-muted">Epoch:</span> {temporalJob.epoch ?? "starting"} · <span className="text-muted">Validation accuracy:</span> {temporalJob.accuracy === undefined ? "measuring" : `${(temporalJob.accuracy * 100).toFixed(1)}%`}</p>}{temporalJob.result && <><p className={`mt-1 ${temporalJob.result.deployment_ready ? "text-ok" : "text-caution"}`}>{temporalJob.result.deployment_ready ? "Passed quality gate" : "Not ready for live alerts"} · Test F1 {(temporalJob.result.f1_macro * 100).toFixed(1)}% · {temporalJob.result.groups} held-out sessions</p><div className="mt-3 grid gap-2 sm:grid-cols-2">{temporalJob.result.labels.map((label, index) => <div key={label} className="rounded border border-line px-2 py-1 text-xs"><span className="font-medium">{label}</span><span className="ml-2 text-muted">Precision {(temporalJob.result!.precision_by_class[index] * 100).toFixed(0)}% · Recall {(temporalJob.result!.recall_by_class[index] * 100).toFixed(0)}% · F1 {(temporalJob.result!.f1_by_class[index] * 100).toFixed(0)}% · n={temporalJob.result!.support_by_class[index]}</span></div>)}</div></>}{temporalJob.error && <p className="mt-1 text-danger">{temporalJob.error}</p>}</div>}
      </section>
      <section className="panel p-4">
        <h2 className="font-display text-xl font-semibold">Datasets</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          <button className="btn" disabled={busy} onClick={() => run(() => post("/ml/datasets/synthetic", { name: "synthetic_demo" }))}>Generate demo dataset</button>
          <button className="btn" disabled={busy} onClick={() => run(() => post("/ml/datasets/from-sessions", { name: "recorded_sessions" }))}>Export recorded sessions</button>
          <label className="btn cursor-pointer">
            Upload CSV
            <input type="file" accept=".csv" className="sr-only" onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) { const body = new FormData(); body.append("file", f); run(() => api("/ml/datasets/upload", { method: "POST", body })); }
              e.target.value = "";
            }} />
          </label>
        </div>
        {datasets.data.length === 0 ? (
          <p className="mt-3 text-sm text-muted">No datasets yet. Generate the demo dataset to try training. Uploaded files need the columns ear, mar, perclos, closure_s, blink_rate, yawns_recent, yaw, pitch, head_turned_s and state.</p>
        ) : (
          <ul className="mt-3 divide-y divide-line/50 text-sm">
            {datasets.data.map((d) => (
              <li key={d.name} className="flex flex-wrap justify-between gap-2 py-2">
                <span className="font-medium">{d.name}</span>
                <span className="text-muted">{d.rows} rows: {Object.entries(d.classes).map(([k, v]) => `${k} ${v}`).join(", ")}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel p-4">
        <h2 className="font-display text-xl font-semibold">Train a model</h2>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <label className="text-sm text-muted">Dataset
            <select className="mt-1 block rounded-md border border-line bg-night px-3 py-2 text-paper" value={current} onChange={(e) => setDataset(e.target.value)}>
              {datasets.data.map((d) => <option key={d.name}>{d.name}</option>)}
            </select>
          </label>
          <label className="text-sm text-muted">Algorithm
            <select className="mt-1 block rounded-md border border-line bg-night px-3 py-2 text-paper" value={algo} onChange={(e) => setAlgo(e.target.value)}>
              {algos.data.map((a) => <option key={a} value={a}>{ALGO_LABEL[a] ?? a}</option>)}
            </select>
          </label>
          <button className="btn btn-primary" disabled={busy || !current}
                  onClick={() => run(async () => setSelected(await post<ModelRun>("/ml/train", { dataset: current, algorithm: algo })))}>
            {busy ? "Working..." : "Train and evaluate"}
          </button>
        </div>
        <p className="mt-3 text-xs text-muted">Models remain research candidates by default. An administrator can approve only a run that passes its quality gate and has a saved artifact; approval does not replace the rule-based live monitor.</p>
      </section>

      {runs.data.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-2">
          <section className="panel p-4">
            <h2 className="font-display text-xl font-semibold">Model comparison</h2>
            <div className="mt-3 h-64">
              <ResponsiveContainer>
                <BarChart data={chart} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid stroke="#2b5a62" strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} stroke="#93b3b0" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="#93b3b0" fontSize={11} width={150} />
                  <Tooltip {...chartTooltip} /><Legend />
                  <Bar dataKey="accuracy" name="Accuracy" fill="#5fd3a8" /><Bar dataKey="f1" name="F1 (macro)" fill="#f4b73f" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
          <section className="panel overflow-x-auto p-4">
            <h2 className="font-display text-xl font-semibold">Runs, best first</h2>
            <table className="mt-2 w-full">
              <thead><tr><th className="th">Run</th><th className="th">Dataset</th><th className="th">Accuracy</th><th className="th">F1</th><th className="th">Governance</th><th className="th" /></tr></thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.id} className="border-t border-line/50">
                    <td className="td">#{r.id} {ALGO_LABEL[r.algorithm] ?? r.algorithm}</td><td className="td text-muted">{r.dataset}</td>
                    <td className="td">{(r.accuracy * 100).toFixed(1)}%</td><td className="td">{r.f1_macro.toFixed(3)}</td>
                    <td className="td"><span className={r.deployment_status === "approved" ? "text-ok" : r.deployment_status === "retired" ? "text-muted" : "text-caution"}>{r.deployment_status}</span>{r.deployment_status === "candidate" && <button className="btn ml-2 text-xs" disabled={busy || !r.deployment_ready} title={r.deployment_ready ? "Approve this candidate" : r.quality_note} onClick={() => run(() => post(`/ml/runs/${r.id}/approve`))}>Approve</button>}</td>
                    <td className="td"><button className="btn" onClick={() => setSelected(r)}>Matrix</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}

      {selected && (
        <section className="panel overflow-x-auto p-4" aria-label="Confusion matrix">
          <h2 className="font-display text-xl font-semibold">Run #{selected.id}: what the model got right and wrong</h2>
          <p className="mt-1 text-sm text-muted">Rows are the true state, columns are the prediction. Tested on {selected.test_rows} held-out rows.</p>
          <table className="mt-3">
            <thead><tr><th className="th" />{selected.labels.map((l) => <th key={l} className="th">{l}</th>)}</tr></thead>
            <tbody>
              {selected.confusion_matrix.map((row, i) => (
                <tr key={i}>
                  <th className="th">{selected.labels[i]}</th>
                  {row.map((v, j) => (
                    <td key={j} className="td text-center font-display text-lg"
                        style={{ background: `rgba(95,211,168,${i === j ? Math.min(0.55, v / 80) : 0})`, color: i !== j && v > 0 ? "#f0585c" : undefined }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {selected.feature_importance && selected.feature_importance.length > 0 && <div className="mt-5 max-w-2xl"><h3 className="font-display text-lg font-semibold">What influenced this model</h3><p className="mt-1 text-xs text-muted">Normalized global importance from the trained estimator; this describes overall influence, not an explanation for one individual prediction.</p><div className="mt-3 space-y-2">{selected.feature_importance.map((item) => <div key={item.feature} className="grid grid-cols-[120px_1fr_48px] items-center gap-2 text-xs"><span>{item.feature}</span><div className="h-2 rounded bg-line"><div className="h-2 rounded bg-ok" style={{ width: `${Math.max(1, item.importance * 100)}%` }} /></div><span className="text-right text-muted">{(item.importance * 100).toFixed(1)}%</span></div>)}</div></div>}
        </section>
      )}
    </div>
  );
}
