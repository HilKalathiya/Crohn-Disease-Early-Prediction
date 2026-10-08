"use client";
import { useEffect, useRef, useState } from "react";
import { API, CapsuleResult, CLASS_NAMES, CLASS_ORDER, getJSON, Mode, modeThr, pct, postJSON, TIER_COLORS, TIER_NAMES, VideoJob, VideoTop } from "@/lib/api";
import { ImageDrop, Note, Spinner, Stat } from "@/components/ui";
import { VideoTimeline } from "@/components/charts";
import CamOverlay, { CamLegend } from "@/components/CamOverlay";
import { ModeToggle, useMeta } from "@/components/Mode";

type Local = { folder: string; videos: { name: string; mb: number; modified: number }[]; decoder: string | null };
const TARGETS: [number, string][] = [[500, "Quick · ~500 frames"], [2000, "Standard · ~2,000 frames"], [5000, "Detailed · ~5,000 frames"], [200000, "Every frame (slow)"]];
const ord = (n: number) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;
const mmss = (s: number | null) => (s == null ? "—" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);

/** upload with progress (fetch cannot report upload progress) */
function upload(file: File, target: number, onProgress: (p: number) => void): Promise<VideoJob> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest(); x.open("POST", `${API}/api/video/upload?target=${target}`);
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => { try { const d = JSON.parse(x.responseText); x.status < 300 ? resolve(d) : reject(new Error(d.detail ?? x.statusText)); } catch { reject(new Error(x.statusText)); } };
    x.onerror = () => reject(new Error("Upload failed")); const fd = new FormData(); fd.append("file", file, file.name); x.send(fd);
  });
}

export default function OwnVideo() {
  const meta = useMeta(); const [mode, setMode] = useState<Mode>("balanced");
  const [local, setLocal] = useState<Local | null>(null); const [target, setTarget] = useState(2000);
  const [job, setJob] = useState<VideoJob | null>(null); const [up, setUp] = useState<number | null>(null); const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<VideoTop | null>(null); const [res, setRes] = useState<CapsuleResult | null>(null); const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = () => getJSON<Local>("/api/video/local").then(setLocal).catch((e) => setErr(e.message));
  useEffect(() => { refresh(); return () => { if (timer.current) clearInterval(timer.current); }; }, []);

  function follow(j: VideoJob) {
    setJob(j); setSel(null); setRes(null); if (timer.current) clearInterval(timer.current);
    timer.current = setInterval(async () => {
      try {
        const d = await getJSON<VideoJob>(`/api/video/job/${j.id}`); setJob(d);
        if (d.status === "done" || d.status === "error") { if (timer.current) clearInterval(timer.current); refresh(); }
      } catch (e: any) { setErr(e.message); if (timer.current) clearInterval(timer.current); }
    }, 1500);
  }
  async function startUpload(f: File) {
    setErr(null); setUp(0);
    try { follow(await upload(f, target, setUp)); } catch (e: any) { setErr(e.message); } finally { setUp(null); }
  }
  async function startLocal(name: string) {
    setErr(null); try { follow(await postJSON<VideoJob>("/api/video/scan", { name, target })); } catch (e: any) { setErr(e.message); }
  }
  async function pick(t: VideoTop) {
    if (!job) return; setSel(t); setRes(null); setBusy(true);
    try { setRes(await getJSON<CapsuleResult>(`/api/video/job/${job.id}/frame/${t.frame}`)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  const r = job?.result; const thr = modeThr(meta, mode); const running = job && (job.status === "queued" || job.status === "running");
  const flaggedTop = r ? r.top.filter((t) => t.lesion >= thr).length : 0;
  return (
    <div className="card card-pad mb-6">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <div className="text-lg font-semibold">Scan your own capsule video</div>
          <p className="text-sm text-ink2 mt-1 max-w-3xl">Upload a video, or copy it into the videos folder and press Refresh (better for files of several GB). One network scores about {target >= 200000 ? "every" : `${target.toLocaleString()}`} frames spread over the video; the most suspicious frames are then re-scored by all {meta ? "10" : ""} networks of the ensemble. Everything runs on this computer.</p>
        </div>
        <ModeToggle mode={mode} setMode={setMode} meta={meta} compact />
      </div>
      <div className="grid lg:grid-cols-[340px_1fr] gap-6">
        <div className="grid gap-3 content-start">
          <label className="text-xs font-semibold text-muted uppercase tracking-wide">Frames to score</label>
          <select className="rounded-lg border hairline px-2 py-1.5 text-sm bg-[var(--surface-1)]" value={target} onChange={(e) => setTarget(Number(e.target.value))} disabled={!!running}>
            {TARGETS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <ImageDrop accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.wmv,.m4v,.webm" label="Drop a capsule video here, or click to choose" hint="MP4 / AVI / MOV / MKV · stays on this computer" onFiles={(fs) => fs[0] && startUpload(fs[0])} />
          {up != null && <div className="text-xs text-muted">Uploading… {pct(up)}<div className="h-1.5 rounded bg-[var(--surface-2)] mt-1"><div className="h-1.5 rounded" style={{ width: pct(up), background: "var(--brand)" }} /></div></div>}
          <div className="border-t hairline pt-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-muted uppercase tracking-wide">Videos folder</span>
              <button className="text-xs text-brand underline" onClick={refresh}>Refresh</button>
            </div>
            {local && <div className="text-[11px] text-muted break-all mb-2 font-mono">{local.folder}</div>}
            {local && !local.decoder && <Note tone="warn">No video decoder found. Run <code>pip install imageio-ffmpeg</code> and restart the app.</Note>}
            {local && local.videos.length === 0 && <div className="text-xs text-muted">No videos yet. Copy .mp4 / .avi files here.</div>}
            <div className="grid gap-1 max-h-56 overflow-auto">
              {local?.videos.map((v) => (
                <div key={v.name} className="flex items-center justify-between gap-2 text-sm px-2 py-1 rounded hover:bg-[var(--surface-2)]">
                  <span className="truncate" title={v.name}>{v.name} <span className="text-xs text-muted tnum">· {v.mb} MB</span></span>
                  <button className="btn btn-ghost text-xs px-2 py-1" disabled={!!running} onClick={() => startLocal(v.name)}>Scan</button>
                </div>))}
            </div>
          </div>
        </div>
        <div className="grid gap-4 content-start">
          {err && <Note tone="warn">{err}</Note>}
          {!job && !err && <div className="text-sm text-muted">Choose a video to start. A 50,000-frame capsule study takes a few minutes on a laptop CPU at the standard setting, much less with a GPU.</div>}
          {running && (
            <div className="card card-pad">
              <Spinner text={`${job!.name}: ${job!.stage}${job!.todo ? ` · ${job!.done.toLocaleString()} / ${job!.todo.toLocaleString()} frames` : ""}`} />
              {job!.todo > 0 && <div className="h-2 rounded bg-[var(--surface-2)] mt-3"><div className="h-2 rounded" style={{ width: `${Math.min(100, (100 * job!.done) / job!.todo)}%`, background: "var(--brand)" }} /></div>}
            </div>)}
          {job?.status === "error" && <Note tone="warn">Scan failed: {job.error}</Note>}
          {r && job?.status === "done" && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Video" value={mmss(r.duration_s)} sub={`${r.total_frames.toLocaleString()} frames · ${r.width}×${r.height}`} />
                <Stat label="Frames scored" value={r.scored.toLocaleString()} sub={`every ${r.stride === 1 ? "" : ord(r.stride) + " "}frame · ${r.seconds}s`} />
                <Stat label={`Flagged (${mode})`} value={`${(mode === "screening" ? r.flagged_pct.screening : r.flagged_pct.balanced).toFixed(0)}%`} sub={`of scored frames ≥ ${thr.toFixed(2)} (one network)`} />
                <Stat label="Top frames above cut-off" value={`${flaggedTop} / ${r.top.length}`} sub={`full ${r.n_models}-model ensemble`} />
              </div>
              <div className="card card-pad">
                <div className="text-sm font-semibold mb-1">Lesion probability along {job.name}</div>
                <div className="text-xs text-muted mb-3">Line = average in each stretch · band = highest · dashed line = {mode} cut-off ({thr.toFixed(2)}) · circles = the frames re-scored by the full ensemble (click one)</div>
                <VideoTimeline bins={r.bins} total={r.total_frames} expert={[]} hideExpert top={r.top.map((t) => ({ frame: t.frame, lesion: t.scan_lesion }))} selected={sel?.frame ?? null} thr={thr}
                  onSelect={(f) => { const t = r.top.find((x) => x.frame === f); if (t) pick(t); }} />
              </div>
              <div className="card card-pad">
                <div className="text-sm font-semibold mb-3">Frames to review first (ranked by the {r.n_models}-model ensemble)</div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
                  {r.top.map((t, i) => (
                    <button key={t.frame} onClick={() => pick(t)} className="text-left">
                      <div className="relative rounded-lg overflow-hidden" style={{ boxShadow: `0 0 0 3px ${sel?.frame === t.frame ? "var(--brand)" : t.lesion >= thr ? TIER_COLORS[t.tier] : "var(--axis)"}` }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={t.thumb} alt={`frame ${t.frame}`} className="w-full aspect-square object-cover" />
                        <span className="absolute top-1 left-1 text-[10px] font-bold px-1.5 rounded bg-black/70 text-white tnum">#{i + 1}</span>
                      </div>
                      <div className="text-[11px] mt-1.5 leading-tight"><b>{t.lesion >= thr ? t.tier_name : "Below cut-off"}</b> · <span className="tnum">{pct(t.lesion)}</span></div>
                      <div className="text-[10px] text-muted truncate">{t.top_name} · {t.time_s != null ? mmss(t.time_s) : `frame ${t.frame}`}</div>
                    </button>))}
                </div>
                {sel && (
                  <div className="mt-5 grid md:grid-cols-[300px_1fr] gap-5 items-start border-t hairline pt-4">
                    <div>{res?.image ? <CamOverlay src={res.image} cam={res.cam?.[res.top_class === "N" ? "N" : "lesion"]} opacity={0.55} size={336} /> : <img src={sel.thumb} alt="" className="w-full rounded-xl" />}<div className="mt-2"><CamLegend /></div></div>
                    <div className="text-sm">
                      <div className="font-semibold mb-1">Frame {sel.frame.toLocaleString()}{sel.time_s != null && <> · {mmss(sel.time_s)}</>}</div>
                      {busy && <Spinner text="Computing the heat-map with all models…" />}
                      {res && <>
                        <div className="text-ink2 mb-2">{res.top_name} · tier <b style={{ color: TIER_COLORS[res.tier] }}>{res.tier_name}</b> · lesion probability {pct(res.lesion_score, 1)} · {res.agree_count}/{res.n_models} models agree</div>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs tnum">{CLASS_ORDER.map((k) => <span key={k}>{CLASS_NAMES[k]}: <b>{pct(res.probs[k], 1)}</b></span>)}</div>
                      </>}
                    </div>
                  </div>)}
              </div>
              <Note><b>How to read this.</b> Use the ranked frames as a reading list. On Kvasir-Capsule studies the frames ranked highest are usually where the lesions are, but deciding whether a whole study has Crohn's lesions at all is not validated (study-level AUROC 0.57), so a clinician still reads the video. Tier mix of scored frames: {r.tier_counts.map((c, i) => `${TIER_NAMES[i]} ${c}`).join(" · ")}.</Note>
            </>)}
        </div>
      </div>
    </div>
  );
}
