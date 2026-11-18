import { useEffect, useState } from "react";
import { api, post } from "../api";
import type { AuthResponse, AuthUser } from "../types";
import { Empty, ErrorNote, STATE_LABEL, fmtDuration } from "./ui";

interface FleetDriver {
  id: number; name: string; sessions: number; average_safety: number | null;
  lowest_safety: number | null; fatigue_pct: number; alerts: number;
  last_session: { started_at: string; duration_s: number } | null;
}
interface FleetOverview {
  days: number;
  totals: { drivers: number; sessions: number; alerts: number };
  active: { driver_id: number | null; driver_name: string | null; recording: boolean; state?: string; safety?: { score: number; break_due: boolean } };
  drivers: FleetDriver[];
}
interface PrivacyStatus { store_video: boolean; sessions: number; samples: number; alerts: number }
interface AuditEntry { id: number; actor: string; action: string; target: string; detail: string | null; created_at: string }

const EMPTY_OVERVIEW: FleetOverview = {
  days: 30, totals: { drivers: 0, sessions: 0, alerts: 0 },
  active: { driver_id: null, driver_name: null, recording: false }, drivers: [],
};

export default function Fleet() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [checked, setChecked] = useState(false);
  const [overview, setOverview] = useState(EMPTY_OVERVIEW);
  const [privacy, setPrivacy] = useState<PrivacyStatus>({ store_video: false, sessions: 0, samples: 0, alerts: 0 });
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [registerMode, setRegisterMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadOverview = () => api<FleetOverview>("/fleet/overview?days=30").then(setOverview).catch((e: Error) => setError(e.message));
  const loadPrivacy = () => api<PrivacyStatus>("/privacy/status").then(setPrivacy).catch((e: Error) => setError(e.message));
  const loadAudit = () => api<AuditEntry[]>("/audit").then(setAudit).catch((e: Error) => setError(e.message));

  useEffect(() => {
    if (!localStorage.getItem("drivio_token")) { setChecked(true); return; }
    api<AuthUser>("/auth/me").then((current) => { setUser(current); return Promise.all([loadOverview(), loadPrivacy(), loadAudit()]); })
      .catch(() => { localStorage.removeItem("drivio_token"); })
      .finally(() => setChecked(true));
  }, []);

  const authenticate = async () => {
    setBusy(true); setError(null);
    try {
      const response = await post<AuthResponse>(registerMode ? "/auth/register" : "/auth/login", { username, password });
      localStorage.setItem("drivio_token", response.token);
      setUser(response.user);
      setPassword("");
      await Promise.all([loadOverview(), loadPrivacy(), loadAudit()]);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const signOut = async () => {
    try { await post("/auth/logout"); } finally { localStorage.removeItem("drivio_token"); setUser(null); }
  };

  const clearRecordedData = async () => {
    if (!window.confirm("Delete all recorded sessions, measurements and alerts? This cannot be undone.")) return;
    setBusy(true); setError(null);
    try { await api("/privacy/recorded-data", { method: "DELETE" }); await Promise.all([loadOverview(), loadPrivacy(), loadAudit()]); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  if (!checked) return <div className="panel p-10 text-center" role="status">Checking Fleet access…</div>;
  if (!user) return (
    <section className="panel mx-auto max-w-md p-5">
      <h2 className="font-display text-3xl font-semibold">{registerMode ? "Create first administrator" : "Fleet administrator login"}</h2>
      <p className="mt-2 text-sm text-muted">Fleet data is protected. The first local account becomes the administrator.</p>
      <ErrorNote message={error} />
      <label className="mt-4 block text-sm text-muted" htmlFor="fleet-username">Username</label>
      <input id="fleet-username" value={username} onChange={(e) => setUsername(e.target.value)} className="mt-1 w-full rounded-md border border-line bg-night px-3 py-2" autoComplete="username" />
      <label className="mt-3 block text-sm text-muted" htmlFor="fleet-password">Password</label>
      <input id="fleet-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-1 w-full rounded-md border border-line bg-night px-3 py-2" autoComplete={registerMode ? "new-password" : "current-password"} />
      <button className="btn btn-primary mt-4 w-full" disabled={busy || !username.trim() || password.length < 8} onClick={authenticate}>{registerMode ? "Create administrator" : "Sign in"}</button>
      <button className="mt-3 w-full text-sm text-muted hover:text-paper" onClick={() => { setRegisterMode(!registerMode); setError(null); }}>{registerMode ? "Already have an account? Sign in" : "First time here? Create the administrator"}</button>
    </section>
  );

  return (
    <div className="space-y-5">
      <ErrorNote message={error} />
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h2 className="font-display text-3xl font-semibold">Fleet overview</h2><p className="mt-1 text-sm text-muted">Signed in as {user.username} · last {overview.days} days.</p></div>
        <div className="flex gap-2"><button className="btn" onClick={() => { loadOverview(); loadPrivacy(); loadAudit(); }}>Refresh</button><button className="btn" onClick={signOut}>Sign out</button></div>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="panel p-4"><p className="text-xs text-muted">Driver profiles</p><p className="font-display text-3xl font-semibold">{overview.totals.drivers}</p></div>
        <div className="panel p-4"><p className="text-xs text-muted">Recorded sessions</p><p className="font-display text-3xl font-semibold">{overview.totals.sessions}</p></div>
        <div className="panel p-4"><p className="text-xs text-muted">Escalated alerts</p><p className="font-display text-3xl font-semibold">{overview.totals.alerts}</p></div>
      </div>
      <section className="panel p-4">
        <h3 className="font-display text-xl font-semibold">Current monitor</h3>
        {overview.active.recording ? <div className="mt-3 flex flex-wrap items-center gap-4 text-sm"><span className="font-medium">{overview.active.driver_name ?? "Unassigned driver"}</span><span>State: {STATE_LABEL[overview.active.state as keyof typeof STATE_LABEL] ?? overview.active.state ?? "Unknown"}</span><span>Safety: {overview.active.safety?.score ?? "-"}</span>{overview.active.safety?.break_due && <span className="text-caution">Break recommended</span>}</div> : <p className="mt-2 text-sm text-muted">No driver session is currently recording.</p>}
      </section>
      {overview.drivers.length === 0 ? <Empty title="No driver profiles yet">Create a driver profile on the Live tab to populate this view.</Empty> : <section className="panel overflow-x-auto"><table className="w-full min-w-[820px]"><caption className="sr-only">Driver safety overview</caption><thead><tr className="border-b border-line">{["Driver", "Sessions", "Average safety", "Lowest safety", "Time fatigued", "Alerts", "Last session"].map((heading) => <th key={heading} className="th">{heading}</th>)}</tr></thead><tbody>{overview.drivers.map((driver) => <tr key={driver.id} className="border-b border-line/50 last:border-0"><td className="td font-medium">{driver.name}</td><td className="td">{driver.sessions}</td><td className="td">{driver.average_safety ?? "-"}</td><td className={`td ${driver.lowest_safety !== null && driver.lowest_safety < 60 ? "text-danger" : ""}`}>{driver.lowest_safety ?? "-"}</td><td className="td">{driver.fatigue_pct}%</td><td className="td">{driver.alerts}</td><td className="td text-muted">{driver.last_session ? `${new Date(driver.last_session.started_at + "Z").toLocaleString()} (${fmtDuration(driver.last_session.duration_s)})` : "-"}</td></tr>)}</tbody></table></section>}
      <section className="panel border-danger/50 p-4">
        <h3 className="font-display text-xl font-semibold">Privacy controls</h3>
        <p className="mt-1 text-sm text-muted">Video storage: {privacy.store_video ? "enabled" : "disabled"}. Recorded data currently includes {privacy.sessions} sessions, {privacy.samples} measurements and {privacy.alerts} alerts.</p>
        <button className="btn btn-danger mt-3" disabled={busy || privacy.sessions === 0} onClick={clearRecordedData}>Delete recorded data</button>
      </section>
      <section className="panel overflow-x-auto p-4">
        <h3 className="font-display text-xl font-semibold">Recent administrative activity</h3>
        {audit.length === 0 ? <p className="mt-2 text-sm text-muted">No administrative actions recorded yet.</p> : <table className="mt-3 w-full min-w-[620px] text-sm"><thead><tr>{["Time", "Actor", "Action", "Target", "Details"].map((heading) => <th key={heading} className="th">{heading}</th>)}</tr></thead><tbody>{audit.map((entry) => <tr key={entry.id} className="border-t border-line/50"><td className="td text-muted">{new Date(entry.created_at + "Z").toLocaleString()}</td><td className="td">{entry.actor}</td><td className="td">{entry.action}</td><td className="td text-muted">{entry.target}</td><td className="td text-muted">{entry.detail ?? "-"}</td></tr>)}</tbody></table>}
      </section>
    </div>
  );
}
