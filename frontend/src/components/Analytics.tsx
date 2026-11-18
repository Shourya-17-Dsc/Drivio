import { useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, useLoad } from "../api";
import type { SessionDetail, SessionSummary } from "../types";
import { Empty, ErrorNote, STATE_LABEL, chartTooltip, fmtDuration } from "./ui";

interface Trends { sessions: SessionSummary[]; state_distribution: Record<string, number> }
const axis = { stroke: "#93b3b0", fontSize: 12 };

export default function Analytics() {
  const trends = useLoad<Trends>(() => api("/analytics/trends?days=90"), { sessions: [], state_distribution: {} });
  const list = useLoad<SessionSummary[]>(() => api("/sessions"), []);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [replaySecond, setReplaySecond] = useState(0);

  const open = (id: number) => api<SessionDetail>(`/sessions/${id}`).then((data) => { setDetail(data); setReplaySecond(0); }).catch((e: Error) => setError(e.message));
  const remove = async (id: number) => {
    const session = list.data.find((item) => item.id === id);
    if (!window.confirm(`Delete the recorded session "${session?.label ?? id}" and its measurements?`)) return;
    setDeletingId(id);
    try {
      await api(`/sessions/${id}`, { method: "DELETE" });
      if (detail?.summary.id === id) setDetail(null);
      list.reload(); trends.reload();
    } catch (e) { setError((e as Error).message); }
    finally { setDeletingId(null); }
  };

  if (list.data.length === 0) {
    return <Empty title="No recorded sessions yet">Start recording on the Live tab. Sessions show up here with trends and reports.</Empty>;
  }

  const trend = trends.data.sessions.map((s) => ({ name: `#${s.id}`, average: s.avg_score ?? 0, lowest: s.min_score ?? 0, fatigue: s.fatigue_pct }));
  const dist = Object.entries(trends.data.state_distribution).map(([k, v]) => ({ name: STATE_LABEL[k as keyof typeof STATE_LABEL] ?? k, seconds: v }));
  const replayMax = detail?.samples.length ? detail.samples[detail.samples.length - 1].s : 0;
  const replaySample = detail?.samples.find((sample) => sample.s >= replaySecond) ?? detail?.samples[detail.samples.length - 1];
  const replayEvent = detail?.events.filter((event) => event.s <= replaySecond).slice(-1)[0];

  return (
    <div className="space-y-5">
      <ErrorNote message={error ?? list.error} />
      <div className="grid gap-5 lg:grid-cols-2">
        <section className="panel p-4">
          <h2 className="font-display text-xl font-semibold">Alertness per session</h2>
          <div className="mt-3 h-60">
            <ResponsiveContainer>
              <LineChart data={trend}>
                <CartesianGrid stroke="#2b5a62" strokeDasharray="3 3" />
                <XAxis dataKey="name" {...axis} /><YAxis domain={[0, 100]} {...axis} />
                <Tooltip {...chartTooltip} />
                <Line type="monotone" dataKey="average" name="Average score" stroke="#5fd3a8" strokeWidth={2} dot />
                <Line type="monotone" dataKey="lowest" name="Lowest score" stroke="#f0585c" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel p-4">
          <h2 className="font-display text-xl font-semibold">Time in each driver state</h2>
          <div className="mt-3 h-60">
            <ResponsiveContainer>
              <BarChart data={dist}>
                <CartesianGrid stroke="#2b5a62" strokeDasharray="3 3" />
                <XAxis dataKey="name" {...axis} /><YAxis {...axis} unit=" s" />
                <Tooltip {...chartTooltip} />
                <Bar dataKey="seconds" name="Seconds" fill="#f4b73f" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="panel overflow-x-auto">
        <table className="w-full min-w-[640px]">
          <caption className="sr-only">Recorded sessions</caption>
          <thead><tr className="border-b border-line">
            {["Session", "Driver", "Started", "Length", "Average", "Lowest", "Time fatigued", "Alerts", ""].map((h) => <th key={h} className="th">{h}</th>)}
          </tr></thead>
          <tbody>
            {list.data.map((s) => (
              <tr key={s.id} className="border-b border-line/50 last:border-0">
                <td className="td font-medium">{s.label}{s.mode === "simulation" && <span className="ml-2 text-xs text-muted">simulated</span>}</td>
                <td className="td">{s.driver_name ?? <span className="text-muted">Unassigned</span>}</td>
                <td className="td text-muted">{new Date(s.started_at + "Z").toLocaleString()}</td>
                <td className="td">{fmtDuration(s.duration_s)}</td>
                <td className="td">{s.avg_score ?? "-"}</td><td className="td">{s.min_score ?? "-"}</td>
                <td className="td">{s.fatigue_pct}%</td><td className="td">{s.alerts}</td>
                <td className="td whitespace-nowrap text-right">
                  <button className="btn mr-2" onClick={() => open(s.id)}>View</button>
                  <button className="btn" disabled={deletingId === s.id} onClick={() => remove(s.id)} aria-label={`Delete session ${s.label}`}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {detail && (
        <section className="panel p-4" aria-label="Session detail">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-display text-2xl font-semibold">{detail.summary.label}</h2>
            <a className="btn ml-auto" href={`/api/sessions/${detail.summary.id}/report.csv`}>Download CSV</a>
            <a className="btn" href={`/api/sessions/${detail.summary.id}/report.pdf`}>Download PDF</a>
          </div>
          <div className="mt-4 h-64">
            <ResponsiveContainer>
              <AreaChart data={detail.samples}>
                <CartesianGrid stroke="#2b5a62" strokeDasharray="3 3" />
                <XAxis dataKey="s" {...axis} unit=" s" /><YAxis domain={[0, 100]} {...axis} />
                <Tooltip {...chartTooltip} />
                <Area type="monotone" dataKey="score" name="Alertness" stroke="#5fd3a8" fill="#5fd3a8" fillOpacity={0.15} strokeWidth={2} />
                {detail.events.filter((e) => e.kind === "escalation").map((e, i) => (
                  <ReferenceLine key={i} x={e.s} stroke={e.level >= 3 ? "#f0585c" : "#f4b73f"} strokeDasharray="4 3" />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-5 border-t border-line pt-4" aria-label="Session replay">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-display text-xl font-semibold">Session replay</h3>
                <p className="text-xs text-muted">Move through the trip to inspect the recorded measurements and alerts.</p>
              </div>
              <span className="font-display text-lg font-semibold">{fmtDuration(replaySecond)} / {fmtDuration(replayMax)}</span>
            </div>
            <label className="sr-only" htmlFor="replay-position">Replay position</label>
            <input id="replay-position" type="range" min={0} max={Math.max(1, replayMax)} value={Math.min(replaySecond, Math.max(1, replayMax))}
                   onChange={(e) => setReplaySecond(Number(e.target.value))} className="mt-4 w-full accent-ok" disabled={!detail.samples.length} />
            {replaySample && <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-6">
              <div><p className="text-xs text-muted">State</p><p className="font-display text-lg font-semibold">{STATE_LABEL[replaySample.state]}</p></div>
              <div><p className="text-xs text-muted">Score</p><p className="font-display text-lg font-semibold">{replaySample.score}</p></div>
              <div><p className="text-xs text-muted">Eye openness</p><p className="font-display text-lg font-semibold">{replaySample.ear.toFixed(2)}</p></div>
              <div><p className="text-xs text-muted">PERCLOS</p><p className="font-display text-lg font-semibold">{(replaySample.perclos * 100).toFixed(0)}%</p></div>
              <div><p className="text-xs text-muted">Trip safety</p><p className="font-display text-lg font-semibold">{replaySample.safety_score}</p></div>
              <div><p className="text-xs text-muted">Nearest event</p><p className="text-sm font-medium">{replayEvent ? replayEvent.message : "None"}</p></div>
            </div>}
          </div>
          <h3 className="mt-4 font-display text-xl font-semibold">Event log</h3>
          {detail.events.length === 0 ? <p className="mt-2 text-sm text-muted">No alerts in this session.</p> : (
            <ul className="mt-2 space-y-1 text-sm">
              {detail.events.map((e, i) => <li key={i}><span className="text-muted">{fmtDuration(e.s)} in</span> &nbsp;{e.message}</li>)}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
