"use client";
import { useEffect, useState } from "react";
import { CapsuleResult, CLASS_NAMES, CLASS_ORDER, getJSON, pct, postFile, TIER_COLORS, TIER_NAMES, urlToFile } from "@/lib/api";
import { Note, PageHead, Spinner, Stat } from "@/components/ui";
import { VideoTimeline } from "@/components/charts";
import CamOverlay, { CamLegend } from "@/components/CamOverlay";
import OwnVideo from "@/components/OwnVideo";

type VRow = { id: string; total_frames: number; has_inflammatory: boolean; expert_lesion_frames: number; peak: number; flagged_pct: number | null; flagged_pct_sb: number | null; recall_top5: number | null; other_findings: string[] };
type Top = { frame: number; lesion: number; tier: number; top_class: number; thumb: string | null; expert_nearby: string | null };
type Video = { id: string; total_frames: number; bins: number[][]; expert: [number, string][]; sb_window?: [number, number]; top: Top[]; stats: any; has_inflammatory: boolean };
type Study = Record<string, number>;

export default function VideoScan() {
  const [list, setList] = useState<VRow[] | null>(null); const [study, setStudy] = useState<Study | null>(null);
  const [vid, setVid] = useState<Video | null>(null); const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<Top | null>(null); const [res, setRes] = useState<CapsuleResult | null>(null); const [busy, setBusy] = useState(false);
  useEffect(() => { getJSON<{ videos: VRow[]; study: Study }>("/api/videos").then((d) => { setList(d.videos); setStudy(d.study); if (d.videos.length) open(d.videos[0].id); }).catch((e) => setErr(e.message)); }, []);
  async function open(id: string) { setSel(null); setRes(null); try { setVid(await getJSON<Video>(`/api/videos/${id}`)); } catch (e: any) { setErr(e.message); } }
  async function pick(t: Top) {
    setSel(t); setRes(null); if (!t.thumb) return; setBusy(true);
    try { setRes(await postFile<CapsuleResult>("/api/capsule/analyze", await urlToFile(t.thumb, `frame_${t.frame}.jpg`))); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  const s = vid?.stats ?? {}; const sampled = s.sampled ?? (vid ? Math.ceil(vid.total_frames / 10) : 0);
  return (
    <div>
      <PageHead kicker="Module 1c · whole capsule study" title="Whole-Video Scan">
        Scan any capsule video below. Further down, each Kvasir-Capsule video is one complete capsule study of about 50,000 frames (Norway, Olympus capsule, never used for training).
        The scan scored every 10th frame with one fold model, then re-scored the 150 most suspicious frames with all 5 models (the EfficientNet-B0 v2 models, sized for a laptop CPU); clicking a frame runs the current 10-model ensemble on it live. Red ticks mark frames that experts labelled as ulcer, erosion or erythema.
      </PageHead>
      <OwnVideo />
      <div className="text-lg font-semibold mb-3">Kvasir-Capsule studies scanned offline</div>
      {err && <Note tone="warn">{err}</Note>}
      {!list && !err && <div className="card card-pad"><Spinner text="Loading scanned videos…" /></div>}
      {list && list.length === 0 && <Note>No scanned videos are bundled yet. Run ml_device/scan_kvasir_videos.py and analyze_kvasir_videos.py.</Note>}
      {list && list.length > 0 && (
        <div className="grid lg:grid-cols-[260px_1fr] gap-6">
          <div className="card card-pad content-start self-start">
            {[true, false].map((g) => (
              <div key={String(g)} className="mb-4">
                <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">{g ? "Expert-labelled inflammatory lesions" : "No inflammatory labels"}</div>
                <div className="grid gap-1">
                  {list.filter((v) => v.has_inflammatory === g).map((v) => (
                    <button key={v.id} onClick={() => open(v.id)} className={`text-left px-2.5 py-1.5 rounded-lg text-sm ${vid?.id === v.id ? "bg-brandsoft text-brand font-semibold" : "hover:bg-[var(--surface-2)]"}`}>
                      <span className="font-mono text-xs">{v.id.slice(0, 8)}</span>
                      <span className="text-xs text-muted ml-2 tnum">{(v.total_frames / 1000).toFixed(0)}k frames{g ? ` · ${v.expert_lesion_frames} lesion labels` : ""}</span>
                    </button>))}
                </div>
              </div>))}
          </div>
          {vid && (
            <div className="grid gap-4 content-start">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Frames in study" value={vid.total_frames.toLocaleString()} sub={`${sampled.toLocaleString()} scored (every 10th)`} />
                <Stat label="Flagged ≥ 50%" value={s.all ? `${s.all.flagged_pct.toFixed(0)}%` : "—"} sub={s.sb_filtered ? `${s.sb_filtered.flagged_pct.toFixed(0)}% inside the small bowel` : "of scored frames"} />
                <Stat label="Expert lesion frames" value={String(vid.expert.length)} sub={vid.has_inflammatory ? "ulcer · erosion · erythema" : "none labelled in this study"} />
                <Stat label="Found in top 5%" value={s.lesion_recall_top5pct != null ? pct(s.lesion_recall_top5pct) : "—"} sub="expert lesion frames within ±10 frames of the top-ranked 5%" />
              </div>
              <div className="card card-pad">
                <div className="text-sm font-semibold mb-1">Lesion probability along the study</div>
                <div className="text-xs text-muted mb-3">Line = average lesion probability in each stretch of frames · light band = up to the highest · dashed line = 50% · numbered circles = the 12 frames the scan ranks highest (click one)</div>
                <VideoTimeline bins={vid.bins} total={vid.total_frames} expert={vid.expert} sb={vid.sb_window} top={vid.top} selected={sel?.frame ?? null}
                  onSelect={(f) => { const t = vid.top.find((x) => x.frame === f); if (t) pick(t); }} />
              </div>
              <div className="card card-pad">
                <div className="text-sm font-semibold mb-3">Top-ranked frames (5-model scan)</div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
                  {vid.top.map((t, i) => (
                    <button key={t.frame} onClick={() => pick(t)} className="text-left">
                      <div className="relative rounded-lg overflow-hidden" style={{ boxShadow: `0 0 0 3px ${sel?.frame === t.frame ? "var(--brand)" : TIER_COLORS[t.tier]}` }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        {t.thumb ? <img src={t.thumb} alt={`frame ${t.frame}`} className="w-full aspect-square object-cover" /> : <div className="aspect-square bg-[var(--surface-2)]" />}
                        <span className="absolute top-1 left-1 text-[10px] font-bold px-1.5 rounded bg-black/70 text-white tnum">#{i + 1}</span>
                        {t.expert_nearby && <span className="absolute bottom-1 left-1 right-1 text-[9px] font-bold px-1 rounded text-white text-center" style={{ background: "var(--critical)" }}>expert: {t.expert_nearby}</span>}
                      </div>
                      <div className="text-[11px] mt-1.5 leading-tight"><b>{TIER_NAMES[t.tier]}</b> · <span className="tnum">{pct(t.lesion)}</span></div>
                      <div className="text-[10px] text-muted truncate">{CLASS_NAMES[CLASS_ORDER[t.top_class]]} · frame {t.frame.toLocaleString()}</div>
                    </button>))}
                </div>
                {sel && (
                  <div className="mt-5 grid md:grid-cols-[300px_1fr] gap-5 items-start border-t hairline pt-4">
                    <div>{sel.thumb && <CamOverlay src={sel.thumb} cam={res?.cam?.[res?.top_class === "N" ? "N" : "lesion"]} opacity={0.55} size={336} />}<div className="mt-2"><CamLegend /></div></div>
                    <div className="text-sm">
                      <div className="font-semibold mb-1">Frame {sel.frame.toLocaleString()}{sel.expert_nearby && <> · expert label within ±10 frames: <b>{sel.expert_nearby}</b></>}</div>
                      {busy && <Spinner text="Running the 10-model ensemble on this frame…" />}
                      {res && <>
                        <div className="text-ink2 mb-2">{res.top_name} · tier <b style={{ color: TIER_COLORS[res.tier] }}>{res.tier_name}</b> · lesion probability {pct(res.lesion_score, 1)} · {res.agree_count}/{res.n_models} models agree</div>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs tnum">{CLASS_ORDER.map((k) => <span key={k}>{CLASS_NAMES[k]}: <b>{pct(res.probs[k], 1)}</b></span>)}</div>
                      </>}
                    </div>
                  </div>)}
              </div>
              {study && (
                <Note tone="warn">
                  <b>What the whole-video test shows ({study.n_videos} studies).</b> Inside a study that contains lesions, the scan finds them: a median of <b>{pct(study.median_lesion_recall_top5pct ?? 0)}</b> of expert lesion frames sit within the top 5% of scored frames
                  ({pct(study.median_lesion_recall_top10pct ?? 0)} within the top 10%). Telling lesion studies from lesion-free ones does <b>not</b> work yet (AUROC {(study.auroc_all_p99 ?? 0).toFixed(2)}, chance = 0.50):
                  raw video contains bubbles, debris, stomach and colon frames that the curated CrohnIPI training frames never showed the model. Retraining with raw-video frames as extra negatives raised study ranking only by making the model miss Kvasir lesions (it learned the camera), so it was not adopted; this needs more labelled capsule studies.
                </Note>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
