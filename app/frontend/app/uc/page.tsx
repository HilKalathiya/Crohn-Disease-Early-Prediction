"use client";
import { useEffect, useState } from "react";
import { getJSON, pct, postFile, UcResult, urlToFile } from "@/lib/api";
import { ImageDrop, Note, PageHead, ProbBars, Spinner } from "@/components/ui";
import CamOverlay, { CamLegend } from "@/components/CamOverlay";

type Sample = { url: string; label: string; source: string };
const MAYO = ["Mayo 0 · normal", "Mayo 1 · mild", "Mayo 2 · moderate", "Mayo 3 · severe"];
const MCOL = ["var(--good)", "var(--warning)", "var(--serious)", "var(--critical)"];

export default function UC() {
  const [samples, setSamples] = useState<Sample[]>([]); const [img, setImg] = useState<string | null>(null);
  const [res, setRes] = useState<UcResult | null>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<string | null>(null);
  const [k, setK] = useState("0"); const [op, setOp] = useState(0.5); const [truth, setTruth] = useState<string | null>(null);
  useEffect(() => { getJSON<any>("/api/samples").then((s) => setSamples(s.uc ?? [])).catch(() => {}); }, []);
  async function run(f: File, label: string | null = null) {
    setErr(null); setBusy(true); setRes(null); setTruth(label); setImg(URL.createObjectURL(f));
    try { const r = await postFile<UcResult>("/api/uc/grade", f); setRes(r); setK(String(r.mayo)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  return (
    <div>
      <PageHead kicker="Module 2 · LIMUC → HyperKvasir" title="Ulcerative Colitis Mayo Grader">
        Crohn's and ulcerative colitis are the two main inflammatory bowel diseases. This extension grades colonoscopy images on the Mayo endoscopic score (0–3). It was trained on 11,276 images from 564 patients (LIMUC, Turkey) and tested on images from a Norwegian hospital it never saw (HyperKvasir).
      </PageHead>
      <div className="grid lg:grid-cols-[360px_1fr] gap-6">
        <div className="grid gap-4 content-start">
          <ImageDrop onFiles={(fs) => run(fs[0])} label="Drop a colonoscopy image, or click to choose" />
          {samples.length > 0 && (
            <div className="card card-pad">
              <div className="text-sm font-semibold mb-3">Sample images</div>
              <div className="grid grid-cols-4 gap-2">
                {samples.map((s) => (
                  <button key={s.url} title={`${s.label} · ${s.source}`} onClick={async () => run(await urlToFile(s.url, s.url.split("/").pop()!), `${s.label} (${s.source})`)}
                    className="rounded-lg overflow-hidden border hairline hover:ring-2 ring-[var(--brand)]">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={s.url} alt={s.label} className="w-full aspect-square object-cover" />
                  </button>
                ))}
              </div>
            </div>
          )}
          <Note tone="warn">This grader uses a frozen ImageNet backbone with a linear head (CPU budget). It separates remission from active disease well, but exact 4-level grades are less reliable — see Results.</Note>
        </div>
        <div className="card card-pad min-h-[420px]">
          {!img && <div className="h-full grid place-items-center text-ink2 text-sm">Choose an image to grade.</div>}
          {img && (
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <CamOverlay src={img} cam={res?.cam?.[k]} opacity={op} />
                {res && (<div className="mt-3 grid gap-2">
                  <div className="flex flex-wrap gap-1.5">{[0, 1, 2, 3].map((m) => <button key={m} onClick={() => setK(String(m))} className={`chip ${k === String(m) ? "!bg-[var(--brand)] !text-white" : ""}`}>Evidence for Mayo {m}</button>)}</div>
                  <label className="flex items-center gap-3 text-xs text-ink2">Overlay <input type="range" min={0} max={1} step={0.05} value={op} onChange={(e) => setOp(+e.target.value)} className="flex-1" /></label>
                  <CamLegend />
                </div>)}
              </div>
              <div className="grid gap-5 content-start">
                {busy && <Spinner text="Grading…" />}{err && <Note tone="warn">{err}</Note>}
                {res && (<>
                  <div>
                    <div className="text-xs text-muted uppercase font-semibold tracking-wide">Predicted grade</div>
                    <div className="flex items-center gap-3"><span className="w-3 h-3 rounded-full" style={{ background: MCOL[res.mayo] }} /><span className="text-2xl font-bold">{MAYO[res.mayo]}</span></div>
                    <div className="text-sm text-ink2 mt-1">{res.mayo_text}</div>
                    {truth && <div className="text-xs text-muted mt-1">Dataset label: {truth}</div>}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl p-3" style={{ background: "var(--surface-2)" }}><div className="text-xs text-muted">Active disease (Mayo 2–3)</div><div className="text-xl font-bold tnum">{pct(res.active_prob)}</div></div>
                    <div className="rounded-xl p-3" style={{ background: "var(--surface-2)" }}><div className="text-xs text-muted">Expected Mayo score</div><div className="text-xl font-bold tnum">{res.expected_mayo.toFixed(2)}</div></div>
                  </div>
                  <ProbBars items={res.probs.map((p, i) => ({ key: String(i), label: MAYO[i], p }))} highlight={String(res.mayo)} />
                </>)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
