"use client";
import { useEffect, useState } from "react";
import { CapsuleResult, CLASS_NAMES, CLASS_ORDER, getJSON, pct, postFile, urlToFile } from "@/lib/api";
import { ImageDrop, Note, PageHead, ProbBars, Spinner, TierMeter } from "@/components/ui";
import CamOverlay, { CamLegend } from "@/components/CamOverlay";

type Sample = { url: string; label: string; source: string; note?: string };
const FINDING_NAMES: Record<string, string> = { ulcer_erosion: "Ulcer / erosion", erythema: "Erythema", angiectasia: "Angiectasia", bleeding: "Bleeding", lymphangiectasia: "Lymphangiectasia", polyp: "Polyp-like", foreign_body: "Foreign body", stenosis: "Stenosis" };

export default function Capsule() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [img, setImg] = useState<string | null>(null);
  const [res, setRes] = useState<CapsuleResult | null>(null);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState<string | null>(null);
  const [camKey, setCamKey] = useState("lesion"); const [op, setOp] = useState(0.55); const [truth, setTruth] = useState<string | null>(null);
  useEffect(() => { getJSON<any>("/api/samples").then((s) => setSamples(s.capsule ?? [])).catch(() => {}); }, []);

  async function run(f: File, label: string | null = null) {
    setErr(null); setBusy(true); setRes(null); setTruth(label); setImg(URL.createObjectURL(f));
    try { const r = await postFile<CapsuleResult>("/api/capsule/analyze", f); setRes(r); setCamKey(r.top_class === "N" ? "N" : "lesion"); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  const items = CLASS_ORDER.map((c) => ({ key: c, label: CLASS_NAMES[c], p: res?.probs[c] ?? 0 }));

  return (
    <div>
      <PageHead kicker="Module 1 · CrohnIPI" title="Crohn's Lesion Analyzer">
        One small-bowel capsule frame in; lesion type, severity tier and a heat-map out. The model is an ensemble of ten EfficientNet-B2 networks at full 320 px resolution (5 folds × 2 training seeds), each fine-tuned on a different 80% of the 3,484 expert-labelled CrohnIPI frames.
      </PageHead>
      <div className="grid lg:grid-cols-[360px_1fr] gap-6">
        <div className="grid gap-4 content-start">
          <ImageDrop onFiles={(fs) => run(fs[0])} />
          {samples.length > 0 && (
            <div className="card card-pad">
              <div className="text-sm font-semibold mb-1">Sample frames</div>
              <div className="text-xs text-muted mb-3">Mostly from external datasets the model never trained on (Kvasir-Capsule, SEE-AI), plus four CrohnIPI frames that were in training (marked on hover).</div>
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
          <Note>Heat-maps are class-activation maps: they show which regions pushed the score up. They are a sanity check, not a lesion outline.</Note>
        </div>

        <div className="card card-pad min-h-[420px]">
          {!img && <div className="h-full grid place-items-center text-ink2 text-sm">Choose a frame or a sample to begin.</div>}
          {img && (
            <div className="grid md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-6">
              <div>
                <CamOverlay src={img} cam={res?.cam?.[camKey]} opacity={op} />
                {res?.cam && (
                  <div className="mt-3 grid gap-2">
                    <div className="flex flex-wrap gap-1.5">
                      {["lesion", ...CLASS_ORDER].map((k) => (
                        <button key={k} onClick={() => setCamKey(k)} className={`chip ${camKey === k ? "!bg-[var(--brand)] !text-white" : ""}`}>{k === "lesion" ? "All lesions" : CLASS_NAMES[k]}</button>
                      ))}
                    </div>
                    <label className="flex items-center gap-3 text-xs text-ink2">Overlay <input type="range" min={0} max={1} step={0.05} value={op} onChange={(e) => setOp(+e.target.value)} className="flex-1" /></label>
                    <CamLegend />
                  </div>
                )}
              </div>
              <div className="grid gap-5 content-start">
                {busy && <Spinner text="Running the 10-model ensemble…" />}
                {err && <Note tone="warn">{err}</Note>}
                {res && (
                  <>
                    <div>
                      <div className="text-xs text-muted uppercase font-semibold tracking-wide">Most likely finding</div>
                      <div className="text-2xl font-bold">{res.top_name}</div>
                      <div className="text-sm text-ink2 mt-1">Lesion probability <b className="tnum">{pct(res.lesion_score, 1)}</b> · {res.agree_count}/{res.n_models} models agree · {res.ms} ms</div>
                      <div className="text-xs text-muted mt-0.5">Balanced cut-off: <b>{res.lesion ? "lesion" : "no lesion"}</b> · Screening cut-off: <b>{res.lesion_screening ? "lesion" : "no lesion"}</b></div>
                      {truth && <div className="text-xs text-muted mt-1">Dataset label: {truth}</div>}
                    </div>
                    <div>
                      <div className="text-sm font-semibold mb-2">Severity tier</div>
                      <TierMeter probs={res.tier_probs} tier={res.tier} />
                      <p className="text-sm text-ink2 mt-2">{res.tier_text}</p>
                    </div>
                    <div>
                      <div className="text-sm font-semibold mb-2">Class probabilities <span className="text-xs text-muted font-normal">(line = ±1 SD across the {res.n_models} models)</span></div>
                      <ProbBars items={items} highlight={res.top_class} sd={res.probs_sd} />
                    </div>
                    {res.findings && (
                      <details className="text-sm">
                        <summary className="cursor-pointer font-semibold">Other small-bowel findings (experimental screen)</summary>
                        <p className="text-xs text-muted my-2">Separate one-vs-rest model trained on SEE-AI (Japan). On Kvasir-Capsule (Norway) its AUROC ranged 0.60–0.88 by finding, so treat these as hints only.</p>
                        <ProbBars items={Object.entries(res.findings.probs).map(([k, p]) => ({ key: k, label: FINDING_NAMES[k] ?? k, p }))} />
                      </details>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
