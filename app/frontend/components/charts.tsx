"use client";
import { useState } from "react";

type Tip = { x: number; y: number; text: string } | null;
function useTip() { const [t, setT] = useState<Tip>(null); return { t, show: (e: React.MouseEvent, text: string) => setT({ x: e.clientX + 12, y: e.clientY + 12, text }), hide: () => setT(null) }; }
function TipBox({ t }: { t: Tip }) { return t ? <div className="tooltip" style={{ left: t.x, top: t.y, whiteSpace: "pre-line" }}>{t.text}</div> : null; }

const SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

/** Row-normalised confusion matrix (each true class sums to 100%), sequential single-hue ramp, counts on hover. */
export function ConfusionMatrix({ m, labels, colLabels, title }: { m: number[][]; labels: string[]; colLabels?: string[]; title?: string }) {
  const { t, show, hide } = useTip(); const n = labels.length; const cell = n > 5 ? 48 : 64; const L = 96, T = 8; const cols = colLabels ?? labels;
  const rows = m.map((r) => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / s); });
  return (
    <figure>
      {title && <figcaption className="text-sm font-semibold mb-2">{title}</figcaption>}
      <div className="overflow-x-auto">
        <svg width={L + n * cell + 4} height={T + n * cell + 56} role="img" aria-label={title ?? "confusion matrix"}>
          {rows.map((r, i) => r.map((v, j) => {
            const k = Math.min(SEQ.length - 1, Math.floor(v * (SEQ.length - 1) + 0.0001 + (v > 0 ? 0.5 : 0)));
            return (
              <g key={`${i}-${j}`} onMouseMove={(e) => show(e, `True: ${labels[i]}\nPredicted: ${labels[j]}\n${m[i][j]} images (${(v * 100).toFixed(1)}% of row)`)} onMouseLeave={hide}>
                <rect x={L + j * cell + 1} y={T + i * cell + 1} width={cell - 2} height={cell - 2} rx={4} fill={SEQ[k]} />
                <text x={L + j * cell + cell / 2} y={T + i * cell + cell / 2 + 4} textAnchor="middle" fontSize={11} fill={k >= 4 ? "#fff" : "var(--text-primary)"} className="tnum">
                  {v >= 0.005 ? Math.round(v * 100) : ""}
                </text>
              </g>
            );
          }))}
          {labels.map((l, i) => <text key={"r" + l} x={L - 8} y={T + i * cell + cell / 2 + 4} textAnchor="end" fontSize={11} fill="var(--text-secondary)">{l}</text>)}
          {cols.map((l, j) => <text key={"c" + l} x={L + j * cell + cell / 2} y={T + n * cell + 16} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">{l}</text>)}
          <text x={L + (n * cell) / 2} y={T + n * cell + 40} textAnchor="middle" fontSize={11} fill="var(--text-muted)">Predicted class →  (rows = true class, % of row)</text>
        </svg>
      </div>
      <TipBox t={t} />
    </figure>
  );
}

/** Horizontal bar comparison, one series; optional CI whiskers; reference line. */
export function BarCompare({ rows, max = 1, fmt = (v: number) => v.toFixed(3), refLine, refLabel }:
  { rows: { label: string; value: number; ci?: [number, number]; ours?: boolean; note?: string }[]; max?: number; fmt?: (v: number) => string; refLine?: number; refLabel?: string }) {
  const { t, show, hide } = useTip(); const maxc = Math.max(...rows.map((r) => r.label.length));
  const L = Math.min(330, Math.max(110, Math.round(maxc * 6.6) + 16)), W = L + 330, R = 60, H = 30;
  const x = (v: number) => L + (v / max) * (W - L - R);
  return (
    <div className="overflow-x-auto">
      <svg width={W} height={rows.length * H + 24} role="img">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => <line key={g} x1={x(g * max)} x2={x(g * max)} y1={0} y2={rows.length * H} stroke="var(--grid)" />)}
        {rows.map((r, i) => (
          <g key={r.label} onMouseMove={(e) => show(e, `${r.label}\n${fmt(r.value)}${r.ci ? `  (95% CI ${fmt(r.ci[0])}–${fmt(r.ci[1])})` : ""}${r.note ? `\n${r.note}` : ""}`)} onMouseLeave={hide}>
            <rect x={0} y={i * H} width={W} height={H} fill="transparent" />
            <text x={L - 10} y={i * H + H / 2 + 4} textAnchor="end" fontSize={12} fill={r.ours ? "var(--text-primary)" : "var(--text-secondary)"} fontWeight={r.ours ? 600 : 400}>{r.label}</text>
            <rect x={L} y={i * H + 8} width={Math.max(2, x(r.value) - L)} height={H - 16} rx={4} fill={r.ours ? "var(--series-1)" : "var(--seq-250)"} />
            {r.ci && <line x1={x(r.ci[0])} x2={x(r.ci[1])} y1={i * H + H / 2} y2={i * H + H / 2} stroke="var(--text-primary)" strokeWidth={1.5} />}
            <text x={x(r.value) + 6} y={i * H + H / 2 + 4} fontSize={12} fill="var(--text-primary)" className="tnum">{fmt(r.value)}</text>
          </g>
        ))}
        {refLine != null && (<g><line x1={x(refLine)} x2={x(refLine)} y1={0} y2={rows.length * H} stroke="var(--text-muted)" strokeDasharray="4 3" />
          <text x={x(refLine) + 4} y={rows.length * H + 16} fontSize={11} fill="var(--text-muted)">{refLabel}</text></g>)}
      </svg>
      <TipBox t={t} />
    </div>
  );
}

/** Two aligned panels sharing the x axis (no dual axis): HBI line with flare threshold, and predicted risk bars. */
export function FlareTimeline({ data }: { data: { week: number; hbi: number; risk: number | null; fc: number | null }[] }) {
  const { t, show, hide } = useTip(); const W = 760, L = 44, R = 16, H1 = 170, H2 = 110, G = 34;
  if (!data.length) return null;
  const w0 = Math.min(...data.map((d) => d.week)), w1 = Math.max(...data.map((d) => d.week)) || 1;
  const x = (w: number) => L + ((w - w0) / Math.max(1, w1 - w0)) * (W - L - R);
  const hmax = Math.max(10, ...data.map((d) => d.hbi)); const y1 = (v: number) => 10 + (1 - v / hmax) * (H1 - 20);
  const rmax = Math.max(0.3, ...data.map((d) => d.risk ?? 0)); const y2 = (v: number) => H1 + G + (1 - v / rmax) * (H2 - 10);
  const path = data.map((d, i) => `${i ? "L" : "M"}${x(d.week)},${y1(d.hbi)}`).join(" ");
  const bw = Math.max(4, Math.min(16, (W - L - R) / data.length - 4));
  return (
    <div className="overflow-x-auto">
      <svg width={W} height={H1 + G + H2 + 30} role="img" aria-label="HBI and predicted flare risk over time">
        <text x={L} y={8} fontSize={11} fill="var(--text-muted)">Harvey-Bradshaw Index (HBI)</text>
        {[0, 5, 10, 15, 20].filter((v) => v <= hmax).map((v) => (<g key={v}><line x1={L} x2={W - R} y1={y1(v)} y2={y1(v)} stroke="var(--grid)" />
          <text x={L - 6} y={y1(v) + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)" className="tnum">{v}</text></g>))}
        <line x1={L} x2={W - R} y1={y1(5)} y2={y1(5)} stroke="var(--critical)" strokeDasharray="5 4" />
        <text x={W - R} y={y1(5) - 5} fontSize={10} textAnchor="end" fill="var(--critical)">flare threshold HBI ≥ 5</text>
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} />
        {data.map((d) => (<circle key={"p" + d.week} cx={x(d.week)} cy={y1(d.hbi)} r={4.5} fill={d.hbi >= 5 ? "var(--critical)" : "var(--series-1)"} stroke="var(--surface-1)" strokeWidth={2}
          onMouseMove={(e) => show(e, `Week ${d.week}\nHBI ${d.hbi}${d.fc != null ? `\nLast calprotectin ${d.fc.toFixed(0)} µg/g` : ""}`)} onMouseLeave={hide} />))}
        <text x={L} y={H1 + G - 8} fontSize={11} fill="var(--text-muted)">Predicted risk of flare at the next visit (computed only while in remission; out-of-fold, recalibrated)</text>
        {[0, rmax / 2, rmax].map((v) => (<g key={"r" + v}><line x1={L} x2={W - R} y1={y2(v)} y2={y2(v)} stroke="var(--grid)" />
          <text x={L - 6} y={y2(v) + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)" className="tnum">{Math.round(v * 100)}%</text></g>))}
        {data.filter((d) => d.risk != null).map((d) => (
          <rect key={"b" + d.week} x={x(d.week) - bw / 2} y={y2(d.risk!)} width={bw} height={Math.max(1, y2(0) - y2(d.risk!))} rx={3}
            fill={d.risk! >= 0.25 ? "var(--critical)" : d.risk! >= 0.15 ? "var(--serious)" : "var(--seq-250)"}
            onMouseMove={(e) => show(e, `Week ${d.week}\nPredicted next-visit flare risk ${(d.risk! * 100).toFixed(0)}%`)} onMouseLeave={hide} />))}
        {data.filter((_, i) => i % Math.ceil(data.length / 12) === 0).map((d) => (
          <text key={"x" + d.week} x={x(d.week)} y={H1 + G + H2 + 18} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">wk {d.week}</text>))}
      </svg>
      <TipBox t={t} />
    </div>
  );
}

/** Calibration: predicted vs observed per bin, with y=x reference. */
export function Calibration({ bins }: { bins: { pred: number; obs: number; n: number }[] }) {
  const { t, show, hide } = useTip(); const S = 220, P = 34; const mx = Math.max(0.5, ...bins.map((b) => Math.max(b.pred, b.obs))) * 1.05;
  const sx = (v: number) => P + (v / mx) * (S - P - 8), sy = (v: number) => S - P + 8 - (v / mx) * (S - P);
  return (
    <div>
      <svg width={S + 10} height={S + 10} role="img" aria-label="calibration plot">
        <line x1={sx(0)} y1={sy(0)} x2={sx(mx)} y2={sy(mx)} stroke="var(--axis)" strokeDasharray="4 3" />
        <line x1={sx(0)} y1={sy(0)} x2={sx(mx)} y2={sy(0)} stroke="var(--axis)" /><line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(mx)} stroke="var(--axis)" />
        <path d={bins.map((b, i) => `${i ? "L" : "M"}${sx(b.pred)},${sy(b.obs)}`).join(" ")} fill="none" stroke="var(--series-1)" strokeWidth={2} />
        {bins.map((b, i) => <circle key={i} cx={sx(b.pred)} cy={sy(b.obs)} r={5} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2}
          onMouseMove={(e) => show(e, `Mean predicted ${(b.pred * 100).toFixed(0)}%\nObserved flare rate ${(b.obs * 100).toFixed(0)}%\n${b.n} visit pairs`)} onMouseLeave={hide} />)}
        <text x={S / 2 + 10} y={S + 6} fontSize={10} textAnchor="middle" fill="var(--text-muted)">predicted risk</text>
        <text x={10} y={S / 2} fontSize={10} textAnchor="middle" fill="var(--text-muted)" transform={`rotate(-90 10 ${S / 2})`}>observed rate</text>
      </svg>
      <TipBox t={t} />
    </div>
  );
}

/** Whole capsule video: max lesion score per bin along the frame index, small-bowel window, expert lesion ticks, top frames. */
export function VideoTimeline({ bins, total, expert, sb, top, selected, onSelect, thr = 0.5, hideExpert = false }: {
  bins: number[][]; total: number; expert: [number, string][]; sb?: [number, number];
  top: { frame: number; lesion: number }[]; selected?: number | null; onSelect?: (frame: number) => void; thr?: number; hideExpert?: boolean;
}) {
  const { t, show, hide } = useTip(); const W = 900, L = 40, R = 14, T = 14, H = 180, LANE = 26, B = 30;
  const x = (f: number) => L + (f / Math.max(1, total)) * (W - L - R); const y = (v: number) => T + (1 - v) * H;
  // bins: [frame, max, mean]; band = max over the stretch, line = mean (falls back to max for older files)
  const mean = (b: number[]) => (b.length > 2 ? b[2] : b[1]);
  const area = bins.length ? bins.map((b, i) => `${i ? "L" : "M"}${x(b[0])},${y(b[1])}`).join(" ") + " " + [...bins].reverse().map((b) => `L${x(b[0])},${y(mean(b))}`).join(" ") + " Z" : "";
  const line = bins.map((b, i) => `${i ? "L" : "M"}${x(b[0])},${y(mean(b))}`).join(" ");
  const step = total > 40000 ? 10000 : total > 8000 ? 5000 : total > 2000 ? 1000 : total > 400 ? 100 : 10;
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect(); const fx = ((e.clientX - r.left) / r.width) * total;
    let k = 0; for (let i = 0; i < bins.length; i++) if (bins[i][0] <= fx) k = i;
    const nxt = bins[k + 1]?.[0] ?? total; const near = expert.filter(([n]) => n >= bins[k][0] && n < nxt);
    show(e, `Frames ${bins[k][0].toLocaleString()}–${nxt.toLocaleString()}\nAverage lesion probability ${(mean(bins[k]) * 100).toFixed(0)}% · highest ${(bins[k][1] * 100).toFixed(0)}%${near.length ? `\nExpert labels here: ${near.length} (${Array.from(new Set(near.map((n) => n[1]))).join(", ")})` : ""}`);
  };
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${T + H + LANE + B}`} width="100%" style={{ minWidth: 620 }} role="img" aria-label="Lesion probability along the whole capsule video">
        {sb && <g><rect x={x(sb[0])} y={T} width={Math.max(0, x(sb[1]) - x(sb[0]))} height={H} fill="var(--brand-soft)" />
          <text x={x(sb[0]) + 6} y={T + H - 6} fontSize={10} fill="var(--brand)">small bowel (between expert pylorus and ileocecal-valve labels)</text></g>}
        {[0, 0.5, 1].map((v) => (<g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
          <text x={L - 6} y={y(v) + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)" className="tnum">{v * 100}%</text></g>))}
        <line x1={L} x2={W - R} y1={y(thr)} y2={y(thr)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        <path d={area} fill="var(--series-1)" opacity={0.18} />
        <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={1.6} />
        {!hideExpert && <text x={L} y={T + H + 14} fontSize={10} fill="var(--text-muted)">Expert-labelled ulcer / erosion / erythema frames</text>}
        {expert.map(([n], i) => <line key={i} x1={x(n)} x2={x(n)} y1={T + H + 18} y2={T + H + LANE} stroke="var(--critical)" strokeWidth={1.5} />)}
        {Array.from({ length: Math.floor(total / step) + 1 }, (_, i) => i * step).map((f) => (
          <text key={f} x={x(f)} y={T + H + LANE + 18} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">{f === 0 ? "frame 0" : f >= 1000 ? `${f / 1000}k` : String(f)}</text>))}
        <rect x={L} y={T} width={W - L - R} height={H} fill="transparent" onMouseMove={onMove} onMouseLeave={hide} />
        {top.map((p, i) => (
          <g key={p.frame} onClick={() => onSelect?.(p.frame)} style={{ cursor: "pointer" }} onMouseMove={(e) => show(e, `#${i + 1} · frame ${p.frame.toLocaleString()}\nscan lesion probability ${(p.lesion * 100).toFixed(0)}%`)} onMouseLeave={hide}>
            <circle cx={x(p.frame)} cy={y(p.lesion)} r={selected === p.frame ? 8 : 6} fill={selected === p.frame ? "var(--critical)" : "var(--surface-1)"} stroke="var(--critical)" strokeWidth={2} />
            <text x={x(p.frame)} y={y(p.lesion) + 3.5} fontSize={8.5} textAnchor="middle" fill={selected === p.frame ? "#fff" : "var(--critical)"} fontWeight={700}>{i + 1}</text>
          </g>))}
      </svg>
      <TipBox t={t} />
    </div>
  );
}
