"use client";
import { useRef, useState } from "react";
import { TIER_COLORS, TIER_NAMES, pct } from "@/lib/api";

export function PageHead({ kicker, title, children }: { kicker: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-8 max-w-3xl">
      <div className="kicker mb-2">{kicker}</div>
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">{title}</h1>
      {children && <div className="text-ink2 text-[15px] leading-relaxed">{children}</div>}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card card-pad">
      <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-1">{label}</div>
      <div className="text-3xl font-bold">{value}</div>
      {sub && <div className="text-xs text-ink2 mt-1">{sub}</div>}
    </div>
  );
}

export function Note({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "warn" }) {
  return (
    <div className="rounded-xl px-4 py-3 text-sm leading-relaxed border hairline"
      style={{ background: tone === "warn" ? "color-mix(in srgb, var(--warning) 12%, var(--surface-1))" : "var(--brand-soft)" }}>
      {children}
    </div>
  );
}

export function ImageDrop({ onFiles, multiple = false, accept = "image/*", label, hint }: { onFiles: (f: File[]) => void; multiple?: boolean; accept?: string; label?: string; hint?: string }) {
  const ref = useRef<HTMLInputElement>(null); const [over, setOver] = useState(false);
  return (
    <div onClick={() => ref.current?.click()} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const fs = Array.from(e.dataTransfer.files); if (fs.length) onFiles(fs); }}
      className="rounded-xl border-2 border-dashed cursor-pointer text-center px-4 py-8 transition-colors"
      style={{ borderColor: over ? "var(--brand)" : "var(--axis)", background: over ? "var(--brand-soft)" : "transparent" }}
      role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") ref.current?.click(); }}>
      <div className="text-2xl mb-1">⇪</div>
      <div className="font-semibold text-sm">{label ?? (multiple ? "Drop frames or a .zip here, or click to choose" : "Drop an image here, or click to choose")}</div>
      <div className="text-xs text-muted mt-1">{hint ?? <>JPG / PNG / BMP{multiple ? " · up to 400 frames" : ""} · processed locally on this machine</>}</div>
      <input ref={ref} type="file" className="hidden" multiple={multiple} accept={multiple && accept === "image/*" ? `${accept},.zip` : accept}
        onChange={(e) => { const fs = Array.from(e.target.files ?? []); if (fs.length) onFiles(fs); e.target.value = ""; }} />
    </div>
  );
}

/** Horizontal probability bars - single series, top bar emphasised, value labels in ink. */
export function ProbBars({ items, highlight, sd }: { items: { key: string; label: string; p: number }[]; highlight?: string; sd?: Record<string, number> }) {
  return (
    <div className="grid gap-2">
      {items.map((it) => (
        <div key={it.key} className="grid grid-cols-[130px_1fr_52px] items-center gap-3 text-sm" title={sd ? `${it.label}: ${pct(it.p, 1)} ± ${pct(sd[it.key] ?? 0, 1)} across the 5 fold models` : undefined}>
          <span className={it.key === highlight ? "font-semibold" : "text-ink2"}>{it.label}</span>
          <div className="h-3 rounded-full relative" style={{ background: "var(--surface-2)" }}>
            <div className="h-3 rounded-full" style={{ width: `${Math.max(it.p * 100, 0.5)}%`, background: it.key === highlight ? "var(--series-1)" : "var(--seq-250)" }} />
            {sd && sd[it.key] > 0.005 && (
              <div className="absolute top-1/2 h-[2px] -translate-y-1/2" style={{ left: `${Math.max(0, it.p - sd[it.key]) * 100}%`, width: `${Math.min(1, 2 * sd[it.key]) * 100}%`, background: "var(--text-secondary)", opacity: .6 }} />
            )}
          </div>
          <span className="tnum text-right">{pct(it.p, it.p < 0.1 ? 1 : 0)}</span>
        </div>
      ))}
    </div>
  );
}

/** 4-step ordinal severity meter; status colours always paired with the text label. */
export function TierMeter({ probs, tier }: { probs: number[]; tier: number }) {
  return (
    <div>
      <div className="grid grid-cols-4 gap-[2px] rounded-lg overflow-hidden">
        {probs.map((p, i) => (
          <div key={i} className="h-9 flex items-center justify-center text-xs font-semibold relative"
            style={{ background: i === tier ? TIER_COLORS[i] : "var(--surface-2)", color: i === tier ? "#111" : "var(--text-secondary)" }}>
            {TIER_NAMES[i]}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-4 gap-[2px] mt-1 text-[11px] text-muted tnum text-center">{probs.map((p, i) => <span key={i}>{pct(p)}</span>)}</div>
    </div>
  );
}

export function Spinner({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-ink2">
      <span className="inline-block w-4 h-4 rounded-full border-2 border-[var(--brand)] border-t-transparent animate-spin" /> {text}
    </div>
  );
}
