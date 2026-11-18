import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useLiveState } from "./api";

const Analytics = lazy(() => import("./components/Analytics"));
const Collection = lazy(() => import("./components/Collection"));
const Experiments = lazy(() => import("./components/Experiments"));
const Fleet = lazy(() => import("./components/Fleet"));
const Live = lazy(() => import("./components/Live"));

const TABS = ["Live", "Analytics", "Collection", "Fleet", "Experiments"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Live");
  const { state, connected } = useLiveState();
  const previousLevel = useRef(0);

  useEffect(() => {
    if (!state) return;
    const newAlert = previousLevel.current === 0 && state.level > 0;
    previousLevel.current = state.level;
    if (newAlert && "speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance("Please be cautious!");
      utterance.lang = "en-US";
      utterance.rate = 0.95;
      window.speechSynthesis.speak(utterance);
    }
  }, [state]);

  return (
    <div className="mx-auto max-w-7xl px-4 pb-12 sm:px-6">
      <header className="flex flex-wrap items-center gap-x-8 gap-y-3 py-5">
        <h1 className="font-display text-4xl font-bold tracking-wide">DRIVIO</h1>
        <nav aria-label="Sections" className="flex gap-1">
          {TABS.map((t) => (
            <button key={t} onClick={() => setTab(t)} aria-current={tab === t ? "page" : undefined}
                    className={`rounded-md px-4 py-2 text-sm font-medium ${tab === t ? "bg-raised text-paper" : "text-muted hover:text-paper"}`}>
              {t}
            </button>
          ))}
        </nav>
        <p className="ml-auto text-xs text-muted">
          {connected ? "Processing on this device. No video is stored." : "Reconnecting to the monitor..."}
        </p>
      </header>
      <main>
        <Suspense fallback={<div className="panel p-10 text-center text-sm text-muted" role="status">Loading section…</div>}>
          {tab === "Live" && <Live live={state} connected={connected} />}
          {tab === "Analytics" && <Analytics />}
          {tab === "Collection" && <Collection />}
          {tab === "Fleet" && <Fleet />}
          {tab === "Experiments" && <Experiments />}
        </Suspense>
      </main>
    </div>
  );
}
