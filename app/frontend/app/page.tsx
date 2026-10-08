"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getJSON, pct } from "@/lib/api";
import { Stat, Note } from "@/components/ui";

const MODULES = [
  { href: "/capsule/", tag: "Main module · CrohnIPI", title: "Crohn's Lesion Analyzer", text: "Upload one capsule-endoscopy frame. A 10-model EfficientNet-B2 ensemble names the lesion (7 classes), grades severity on a 4-step scale, and shows a heat-map of where it looked.", },
  { href: "/study/", tag: "Clinical workflow", title: "Capsule Study Triage", text: "Upload a whole set of frames (or a .zip). Every frame is scored and the study is ranked so the reader sees the most severe frames first — capsule videos hold tens of thousands of frames.", },
  { href: "/video/", tag: "Whole studies · Kvasir-Capsule", title: "Whole-Video Scan", text: "All 43 labelled Kvasir-Capsule capsule studies (2 million frames) scanned end to end: lesion probability along each study, the expert lesion labels, and the top-ranked frames with heat-maps.", },
  { href: "/uc/", tag: "Extension · LIMUC", title: "Ulcerative Colitis Grader", text: "Mayo endoscopic score (0–3) from a colonoscopy image, trained on 564 patients and tested on a second hospital (HyperKvasir).", },
  { href: "/flare/", tag: "Add-on · HMP2", title: "Flare Risk Forecaster", text: "From a patient's recent symptom scores and labs, estimate the chance of a flare at the next visit, with the reasons behind each estimate.", },
];

export default function Home() {
  const [r, setR] = useState<any>(null);
  useEffect(() => { getJSON<any>("/api/results").then(setR).catch(() => {}); }, []);
  const ft = r?.crohnipi?.finetuned;
  return (
    <div>
      <section className="grid lg:grid-cols-[1.25fr_1fr] gap-8 items-center mb-12">
        <div>
          <div className="kicker mb-3">AI-based early prediction of Crohn's disease severity</div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight leading-[1.05] mb-4">See Crohn's damage early,<br />grade it consistently.</h1>
          <p className="text-ink2 text-[16px] leading-relaxed mb-6 max-w-xl">
            CrohnSight reads small-bowel capsule-endoscopy frames, finds Crohn's lesions — from the earliest aphthoid ulcers to large ulcers and
            strictures — and turns them into a severity grade a clinician can check. Two add-on modules extend the idea to ulcerative colitis and to
            forecasting flares from routine follow-up data.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/capsule/" className="btn">Try the Lesion Analyzer →</Link>
            <Link href="/results/" className="btn btn-ghost">See validated results</Link>
          </div>
        </div>
        <div className="card card-pad">
          <div className="text-sm font-semibold mb-3">Pipeline at a glance</div>
          <Pipeline />
        </div>
      </section>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
        <Stat label="Lesion vs normal (AUROC)" value={ft ? ft.binary.auroc.toFixed(3) : "…"} sub="CrohnIPI, official 5-fold CV" />
        <Stat label="Lesion vs normal accuracy" value={ft ? pct(ft.binary.crossfit.accuracy, 1) : "…"} sub="published benchmark: 93.7%" />
        <Stat label="Severity grade agreement" value={ft ? (ft.severity_crossfit?.qwk ?? ft.severity.qwk).toFixed(2) : "…"} sub="quadratic-weighted kappa, 4 tiers, cut-points set on other folds" />
        <Stat label="Images used across 5 datasets" value={r?.totals ? r.totals.images.toLocaleString() : "…"} sub={r?.totals?.videos_scanned ? `+ ${r.totals.videos_scanned} whole capsule videos (2.0 M frames) · all open` : "all open-access, no application needed"} />
      </section>

      <section className="grid md:grid-cols-2 gap-4 mb-12">
        {MODULES.map((m) => (
          <Link key={m.href} href={m.href} className="card card-pad hover:shadow-lg transition-shadow block">
            <div className="chip mb-3">{m.tag}</div>
            <div className="text-xl font-bold mb-2">{m.title} →</div>
            <p className="text-sm text-ink2 leading-relaxed">{m.text}</p>
          </Link>
        ))}
      </section>

      <section className="grid lg:grid-cols-3 gap-4">
        <div className="card card-pad lg:col-span-2 prose-sm">
          <div className="text-lg font-bold mb-2">Why this matters</div>
          <p>Crohn's disease is a lifelong inflammatory bowel disease that often starts in the small bowel, where a normal colonoscopy cannot reach. Capsule endoscopy films the whole small bowel, but a single study produces tens of thousands of frames that a specialist must read by eye — slow, tiring, and graded differently by different readers (in the CrohnIPI study, three experts agreed only moderately on the exact lesion type, κ = 0.57).</p>
          <p>Catching small lesions such as aphthoid ulcers early, and grading severity consistently, is exactly what treat-to-target care needs. CrohnSight automates the first pass and keeps the clinician in control: every prediction is shown with its uncertainty and a heat-map.</p>
        </div>
        <Note tone="warn"><b>Interim build.</b> The original plan (onset 5–6 years before diagnosis and disease-course prediction from hospital records) is <b>on hold, not dropped</b>: it needs MIMIC-IV, whose credentialed access is still pending. This build delivers the severity side of the project on five open datasets.</Note>
      </section>
    </div>
  );
}

function Pipeline() {
  const steps = [
    ["Capsule frame", "320×320 RGB"], ["Pre-process", "full 320 px, normalise"], ["EfficientNet-B2 ×10", "5 folds × 2 seeds, fine-tuned"],
    ["7 lesion classes", "N · E · O · AU · U3-10 · U>10 · S"], ["Severity tier", "None · Mild · Moderate · Severe"], ["Explain", "class-activation heat-map + model agreement"],
  ];
  return (
    <ol className="grid gap-2">
      {steps.map(([a, b], i) => (
        <li key={a} className="flex items-center gap-3">
          <span className="w-7 h-7 shrink-0 rounded-full grid place-items-center text-xs font-bold text-white" style={{ background: "var(--brand)" }}>{i + 1}</span>
          <div className="flex-1 rounded-lg px-3 py-2 text-sm" style={{ background: "var(--surface-2)" }}><b>{a}</b> <span className="text-ink2">— {b}</span></div>
        </li>
      ))}
    </ol>
  );
}
