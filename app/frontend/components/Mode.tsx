"use client";
import { useEffect, useState } from "react";
import { getJSON, Meta, Mode, pct } from "@/lib/api";

let cache: Promise<Meta> | null = null;
/** thresholds + operating points from /api/meta (fetched once per page load) */
export function useMeta() {
  const [m, setM] = useState<Meta | null>(null);
  useEffect(() => { (cache ??= getJSON<Meta>("/api/meta")).then(setM).catch(() => {}); }, []);
  return m;
}

/** Balanced (0.5) vs Screening (lower cut-off fixed on CrohnIPI to catch 99% of lesions) */
export function ModeToggle({ mode, setMode, meta, compact = false }: { mode: Mode; setMode: (m: Mode) => void; meta: Meta | null; compact?: boolean }) {
  const op = meta?.operating_points;
  const opts: [Mode, string, number | undefined][] = [["balanced", "Balanced", meta?.threshold], ["screening", "Screening", meta?.screening_threshold]];
  return (
    <div>
      <div className="inline-flex rounded-lg border hairline overflow-hidden" role="radiogroup" aria-label="Operating point">
        {opts.map(([m, label, t]) => (
          <button key={m} role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
            className={`px-3 py-1.5 text-sm ${mode === m ? "bg-brandsoft text-brand font-semibold" : "hover:bg-[var(--surface-2)]"}`}>
            {label}{t != null && <span className="text-xs text-muted ml-1.5 tnum">≥ {t.toFixed(2)}</span>}
          </button>))}
      </div>
      {!compact && op && (
        <p className="text-xs text-muted mt-2 leading-relaxed">
          {mode === "balanced"
            ? <>Balanced: fewest false alarms. On all 47,238 Kvasir frames (Norway) it flags {pct(op.balanced.kvasir_47k.sensitivity)} of lesion frames and {pct(1 - op.balanced.kvasir_47k.specificity, 1)} of normal frames.</>
            : <>Screening: cut-off fixed on CrohnIPI to catch 99% of lesions, before any external test. On Kvasir it flags {pct(op.screening.kvasir_47k.sensitivity)} of lesion frames and {pct(1 - op.screening.kvasir_47k.specificity)} of normal frames; on SEE-AI (Japan) {pct(op.screening.seeai.erosion)}–{pct(op.screening.seeai.stenosis)} of lesions.</>}
        </p>)}
    </div>
  );
}
