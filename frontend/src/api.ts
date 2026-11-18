import { useCallback, useEffect, useRef, useState } from "react";
import type { LiveState } from "./types";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const token = typeof localStorage !== "undefined" ? localStorage.getItem("drivio_token") : null;
  const headers = init?.body instanceof FormData
    ? { ...(init?.headers ?? {}), ...(token ? { Authorization: `Bearer ${token}` } : {}) }
    : { "Content-Type": "application/json", ...init?.headers, ...(token ? { Authorization: `Bearer ${token}` } : {}) };
  const res = await fetch(`/api${path}`, {
    ...init,
    headers,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const post = <T,>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

/** Live driver state over a WebSocket, reconnecting if the backend restarts. */
export function useLiveState() {
  const [state, setState] = useState<LiveState | null>(null);
  const [connected, setConnected] = useState(false);
  const retry = useRef<number>();

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/state`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data && data.metrics) setState(data as LiveState);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry.current = window.setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry.current);
      ws?.close();
    };
  }, []);

  return { state, connected };
}

/** Run an async loader on mount and on demand. */
export function useLoad<T>(loader: () => Promise<T>, initial: T) {
  const [data, setData] = useState<T>(initial);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    loader().then((d) => { setData(d); setError(null); }).catch((e: Error) => setError(e.message));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(load, [load]);
  return { data, error, reload: load };
}
