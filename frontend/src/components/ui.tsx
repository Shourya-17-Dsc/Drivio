import type { DriverState } from "../types";

export const STATE_LABEL: Record<DriverState, string> = {
  alert: "Alert", fatigued: "Fatigued", drowsy: "Drowsy", distracted: "Distracted", no_face: "Face not visible",
};
export const STATE_COLOR: Record<DriverState, string> = {
  alert: "#5fd3a8", fatigued: "#f4b73f", drowsy: "#f0585c", distracted: "#f4b73f", no_face: "#93b3b0",
};
export const LEVEL_LABEL = ["Clear", "Caution", "Warning", "Critical"];

export function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-ok" : "bg-danger"}`} aria-hidden />
      {label}
      <span className="sr-only">{ok ? "working" : "not working"}</span>
    </span>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line p-8 text-center">
      <p className="font-display text-xl font-semibold">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </div>
  );
}

export function ErrorNote({ message }: { message: string | null }) {
  if (!message) return null;
  return <p role="alert" className="rounded-md border border-danger/60 bg-danger/10 px-3 py-2 text-sm">{message}</p>;
}

export const chartTooltip = {
  contentStyle: { background: "#0d2b30", border: "1px solid #2b5a62", borderRadius: 6, color: "#e9f1ef" },
  labelStyle: { color: "#93b3b0" },
};

export const fmtDuration = (s: number) => {
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
};
