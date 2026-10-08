"use client";
import { useMemo, useState } from "react";
import { getJSON, pct, postFiles, StudyFrame, StudySummary, TIER_COLORS, TIER_NAMES, CLASS_NAMES, Mode, modeThr } from "@/lib/api";
import { ModeToggle, useMeta } from "@/components/Mode";
import { ImageDrop, Note, PageHead, Spinner, Stat } from "@/components/ui";

type Out = { summary: StudySummary; frames: StudyFrame[]; source?: string; truth?: Record<string, string>; cached?: boolean };

export default function Study() {
  const [out, setOut] = useState<Out | null>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "flagged" | 0 | 1 | 2 | 3>("flagged"); const [sel, setSel] = useState<StudyFrame | null>(null);
  async function go(p: Promise<Out>) { setErr(null); setBusy(true); setOut(null); setSel(null); try { setOut(await p); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } }
  const ranked = useMemo(() => out ? [...out.frames].sort((a, b) => b.lesion_score - a.lesion_score) : [], [out]);
  const meta = useMeta(); const [mode, setMode] = useState<Mode>("balanced"); const thr = modeThr(meta, mode);
  const nFlag = out ? out.frames.filter((f) => f.lesion_score >= thr).length : 0;
  const shown = ranked.filter((f) => filter === "all" ? true : filter === "flagged" ? f.lesion_score >= thr : f.tier === filter);
  const truthEval = useMemo(() => {
    if (!out?.truth) return null; const isLes = (l?: string) => !!l && !/normal/i.test(l);
    const totalLes = out.frames.filter((f) => isLes(out.truth![f.name])).length; if (!totalLes) return null;
    const k = nFlag; const caught = ranked.slice(0, k).filter((f) => isLes(out.truth![f.name])).length;
    return { totalLes, k, caught };
  }, [out, ranked, nFlag]);

  return (
    <div>
      <PageHead kicker="Module 1b · clinical workflow" title="Capsule Study Triage">
        A capsule study is a video of the whole small bowel. CrohnSight scores every frame, then sorts the study so the reader starts with the frames most likely to show a Crohn's lesion, and gives a study-level severity summary.
      </PageHead>
      <div className="grid lg:grid-cols-[360px_1fr] gap-6 mb-6">
        <div className="grid gap-3 content-start">
          <ImageDrop multiple onFiles={(fs) => go(postFiles<Out>("/api/capsule/study", fs))} />
          <button className="btn btn-ghost justify-center" onClick={() => go(getJSON<Out>("/api/capsule/demo-study"))}>Load demo study (real capsule video excerpt)</button>
          <ModeToggle mode={mode} setMode={setMode} meta={meta} />
          <Note>The demo is an excerpt of labelled frames from one Kvasir-Capsule video (Norway, Olympus capsule) — a different hospital and device from the CrohnIPI training data.</Note>
        </div>
        <div className="grid gap-4 content-start">
          {busy && <div className="card card-pad"><Spinner text="Scoring every frame with the 10-model ensemble…" /></div>}
          {err && <Note tone="warn">{err}</Note>}
          {out && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Frames analysed" value={out.summary.frames.toString()} sub={out.cached ? `cached result · first run took ${out.summary.seconds}s` : `${out.summary.seconds}s on this CPU`} />
                <Stat label={`Flagged (${mode})`} value={`${nFlag}`} sub={`${pct(nFlag / Math.max(1, out.summary.frames))} of frames ≥ ${thr.toFixed(2)}`} />
                <Stat label="Worst finding" value={out.summary.worst_tier_name} sub="highest severity tier seen" />
                <Stat label="Burden index" value={out.summary.burden_index.toFixed(0)} sub="0 = all normal · 100 = all severe" />
              </div>
              <div className="card card-pad">
                <div className="text-sm font-semibold mb-2">Frames by severity tier</div>
                <div className="flex h-7 rounded-lg overflow-hidden gap-[2px]">
                  {out.summary.tier_counts.map((c, i) => c > 0 && (
                    <div key={i} style={{ width: `${(100 * c) / out.summary.frames}%`, background: TIER_COLORS[i] }} title={`${TIER_NAMES[i]}: ${c} frames`} />
                  ))}
                </div>
                <div className="flex flex-wrap gap-4 mt-2 text-xs text-ink2">
                  {out.summary.tier_counts.map((c, i) => (<span key={i} className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: TIER_COLORS[i] }} />{TIER_NAMES[i]} <b className="tnum">{c}</b></span>))}
                </div>
                <p className="text-sm text-ink2 mt-3">Reviewing only the flagged frames would cut reading to <b>{pct(nFlag / Math.max(1, out.summary.frames))}</b> of the study.
                  {truthEval && <> In this demo those {truthEval.k} top-ranked frames contain <b>{truthEval.caught} of {truthEval.totalLes}</b> frames that experts labelled as abnormal.</>}</p>
              </div>
            </>
          )}
        </div>
      </div>
      {out && (
        <div className="card card-pad">
          <div className="flex flex-wrap items-center gap-2 mb-4">
            <span className="text-sm font-semibold mr-2">Review queue (most suspicious first)</span>
            {(["flagged", "all", 3, 2, 1, 0] as const).map((f) => (
              <button key={String(f)} onClick={() => setFilter(f)} className={`chip ${filter === f ? "!bg-[var(--brand)] !text-white" : ""}`}>
                {f === "flagged" ? "Flagged" : f === "all" ? "All" : TIER_NAMES[f]}
              </button>
            ))}
            <span className="text-xs text-muted ml-auto">{shown.length} frames</span>
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-7 lg:grid-cols-9 gap-2">
            {shown.map((f, i) => (
              <button key={f.idx} onClick={() => setSel(f)} className="text-left">
                <div className="relative rounded-lg overflow-hidden" style={{ boxShadow: `0 0 0 3px ${TIER_COLORS[f.tier]}` }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={f.thumb} alt={f.name} className="w-full aspect-square object-cover" />
                  <span className="absolute top-1 left-1 text-[10px] font-bold px-1.5 rounded bg-black/70 text-white tnum">#{i + 1}</span>
                </div>
                <div className="text-[11px] mt-1.5 leading-tight"><b>{f.tier_name}</b> · <span className="tnum">{pct(f.lesion_score)}</span></div>
                <div className="text-[10px] text-muted truncate">{CLASS_NAMES[f.top_class]}</div>
              </button>
            ))}
          </div>
          {sel && (
            <div className="mt-6 grid md:grid-cols-[200px_1fr] gap-4 items-start border-t hairline pt-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={sel.thumb} alt={sel.name} className="w-full rounded-lg" />
              <div className="text-sm">
                <div className="font-semibold">{sel.name}</div>
                <div className="text-ink2 mb-2">{sel.top_name} · tier <b>{sel.tier_name}</b> · lesion probability {pct(sel.lesion_score, 1)}{out.truth?.[sel.name] && <> · dataset label: <b>{out.truth[sel.name]}</b></>}</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-1 text-xs tnum">
                  {Object.entries(sel.probs).map(([k, p]) => <span key={k}>{CLASS_NAMES[k]}: <b>{pct(p, 1)}</b></span>)}
                </div>
                <p className="text-xs text-muted mt-2">Open the frame in the Lesion Analyzer to see its heat-map.</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
