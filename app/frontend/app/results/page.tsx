"use client";
import { useEffect, useState } from "react";
import { getJSON, pct, f3, CLASS_ORDER } from "@/lib/api";
import { PageHead, Stat, Note, Spinner } from "@/components/ui";
import { BarCompare, Calibration, ConfusionMatrix } from "@/components/charts";

const SHORT: Record<string, string> = { N: "Normal", E: "Erythema", O: "Oedema", AU: "Aphthoid", "U3-10": "Ulcer 3-10", "U>10": "Ulcer >10", S: "Stenosis" };

function Section({ id, title, kicker, children }: { id: string; title: string; kicker: string; children: React.ReactNode }) {
  return (<section id={id} className="mb-12 scroll-mt-20"><div className="kicker mb-1">{kicker}</div><h2 className="text-2xl font-bold mb-4">{title}</h2>{children}</section>);
}

export default function Results() {
  const [r, setR] = useState<any>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getJSON<any>("/api/results").then(setR).catch((e) => setErr(e.message)); }, []);
  if (err) return <Note tone="warn">Could not load results: {err}</Note>;
  if (!r) return <Spinner text="Loading results…" />;
  const c = r.crohnipi, ft = c.finetuned;
  return (
    <div>
      <PageHead kicker="Validation" title="Results">
        Every number below is measured on images the model did not train on. CrohnIPI uses its official 5-fold split; external tests use whole datasets from other hospitals and capsule brands. Hover any bar or cell for exact values.
      </PageHead>
      <nav className="flex flex-wrap gap-2 mb-10 text-sm">
        {[["crohn", "Crohn's lesions"], ["external", "External validation"], ["kvasir", "All 47k Kvasir frames"], ["video", "Whole-video scan"], ["leak", "Leakage check"], ["uc", "UC grader"], ["flare", "Flare risk"]].filter(([id]) => (id !== "kvasir" || r.kvasir_full) && (id !== "video" || r.video_study)).map(([id, l]) => <a key={id} href={`#${id}`} className="chip">{l}</a>)}
      </nav>

      <Section id="crohn" kicker="Module 1 · CrohnIPI · 3,484 frames · 5-fold CV" title="Crohn's lesion detection and severity">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Stat label="Lesion vs normal AUROC" value={f3(ft.binary.auroc)} sub={ft.binary.auroc_ci95 ? `95% CI ${f3(ft.binary.auroc_ci95[0])}–${f3(ft.binary.auroc_ci95[1])}` : undefined} />
          <Stat label="Lesion vs normal accuracy" value={pct(ft.binary.crossfit.accuracy, 1)} sub={`sens ${pct(ft.binary.crossfit.sensitivity, 1)} · spec ${pct(ft.binary.crossfit.specificity, 1)} · threshold set on other folds`} />
          <Stat label="7-class accuracy" value={pct(ft.multiclass.accuracy, 1)} sub={`balanced ${pct(ft.multiclass.balanced_accuracy, 1)} · macro-F1 ${f3(ft.multiclass.macro_f1, 2)}`} />
          <Stat label="Severity tier (4 levels)" value={ft.severity_crossfit ? `κ ${f3(ft.severity_crossfit.qwk, 3)}` : pct(ft.severity.accuracy, 1)}
            sub={ft.severity_crossfit ? `tier cut-points chosen on the other folds · argmax tiers: κ ${f3(ft.severity.qwk, 3)}, ${pct(ft.severity.accuracy, 1)} exact` : `quadratic-weighted κ ${f3(ft.severity.qwk, 2)}`} />
        </div>
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-1">Lesion vs normal accuracy against the published benchmark</div>
            <div className="text-xs text-muted mb-3">Benchmarks from de Maissin et al., Endoscopy International Open 2021 — same dataset, same 5-fold protocol. Ours: per-fold threshold chosen on the other folds (at a fixed 0.5 cut-off: {pct(ft.binary.accuracy, 1)}).</div>
            <BarCompare rows={c.benchmarks.map((b: any) => ({ label: b.label, value: b.accuracy, ours: b.ours }))} max={1} fmt={(v) => pct(v, 1)} />
          </div>
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-1">What fine-tuning adds (same backbone family)</div>
            <div className="text-xs text-muted mb-3">Frozen ImageNet features + logistic regression vs full fine-tuning. Metric: 7-class balanced accuracy.</div>
            <BarCompare rows={c.ablation.map((a: any) => ({ label: a.label, value: a.balanced_accuracy, ours: a.ours, note: `binary AUROC ${f3(a.auroc)}` }))} max={1} fmt={(v) => pct(v, 1)} />
          </div>
        </div>
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="card card-pad"><ConfusionMatrix m={ft.multiclass.confusion} labels={CLASS_ORDER.map((k) => SHORT[k])} colLabels={CLASS_ORDER} title="7-class confusion matrix (all 5 test folds)" /></div>
          <div className="card card-pad">
            <ConfusionMatrix m={ft.severity.confusion} labels={["None", "Mild", "Moderate", "Severe"]} title="Severity-tier confusion matrix" />
            <p className="text-xs text-muted mt-3">Most errors are between neighbouring tiers, which the quadratic-weighted κ treats as small mistakes. Human experts on this dataset agreed with κ = 0.57 on the 7 fine classes and 0.79 on lesion vs normal.</p>
          </div>
        </div>
      </Section>

      <Section id="external" kicker="Generalisation" title="External validation on unseen hospitals and capsules">
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-1">CrohnIPI model on other datasets — inflammatory lesion vs normal (AUROC)</div>
            <div className="text-xs text-muted mb-3">No re-training. Kvasir-Capsule: Norway, Olympus capsule. SEE-AI: Japan. Lesion score = 1 − P(normal).</div>
            <BarCompare rows={c.external_rows} max={1} refLine={0.5} refLabel="chance" />
            {c.pointing && <p className="text-sm text-ink2 mt-4">Heat-map check on expert boxes (pointing game): the hottest point fell inside the lesion box in <b>{pct(c.pointing.hit_rate)}</b> of {c.pointing.n.toLocaleString()} Kvasir ulcer/erosion/erythema frames (chance {pct(c.pointing.baseline)}), and on SEE-AI in <b>{pct(c.pointing.seeai.erosion.hit_rate)}</b> of erosion frames (chance {pct(c.pointing.seeai.erosion.baseline)}) and <b>{pct(c.pointing.seeai.stenosis.hit_rate)}</b> of stenosis frames (chance {pct(c.pointing.seeai.stenosis.baseline)}).</p>}
          </div>
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-1">General small-bowel screen: SEE-AI → Kvasir-Capsule (AUROC vs normal)</div>
            <div className="text-xs text-muted mb-3">Experimental one-vs-rest heads on frozen features, trained in Japan, tested in Norway.</div>
            <BarCompare rows={r.screen_rows} max={1} refLine={0.5} refLabel="chance" />
            {r.hyperkvasir_extra?.polyp_localisation && (() => { const pl = r.hyperkvasir_extra.polyp_localisation; return (
              <p className="text-sm text-ink2 mt-4">Colonoscopy check (HyperKvasir, {pl.n.toLocaleString()} polyps with expert masks): the polyp heat-map hits the mask in <b>{pct(pl.pointing_hit_rate)}</b> of images (chance {pct(pl.chance)}), but only {pct(pl.detected_at_0_5)} are detected at 0.5. Capsule-trained heads do not transfer well to colonoscopy.</p>); })()}
          </div>
        </div>
      </Section>

      {r.kvasir_full && (() => { const k = r.kvasir_full.A, b = r.kvasir_full.B; const iv = k.inflammatory_vs_normal; const pc = k.per_class;
        const rows = Object.entries(pc).filter(([n]) => n !== "Normal clean mucosa").map(([n, v]: any) => ({ label: `${n} (${v.n.toLocaleString()})`, value: v.auroc_vs_normal, ours: ["Ulcer", "Erosion", "Erythema"].includes(n), note: `flagged at 0.5: ${pct(v.flagged_at_0_5)}` })).sort((a, z) => z.value - a.value);
        const lv = b.videos.filter((v: any) => v.inflammatory_frames > 0 && v.prevalence < 0.5); /* skip videos whose labelled frames are almost all lesions (trivial) */ const rank1 = lv.filter((v: any) => v.rank_first_lesion === 1).length;
        return (
        <Section id="kvasir" kicker={`Kvasir-Capsule · all ${k.frames.toLocaleString()} expert-labelled frames · ${k.videos} videos`} title="The full Norwegian dataset, at its natural lesion rate">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Stat label="Inflammatory vs normal AUROC" value={f3(iv.auroc)} sub={`95% CI ${f3(iv.auroc_ci95_video_bootstrap[0], 2)}–${f3(iv.auroc_ci95_video_bootstrap[1], 2)} (videos resampled)`} />
            <Stat label="Ulcer vs normal AUROC" value={f3(pc["Ulcer"].auroc_vs_normal)} sub={`${pc["Ulcer"].n} ulcer frames · ${k.normal_frames.toLocaleString()} normal`} />
            <Stat label="Reading the top 20%" value={pct(k.pooled_triage.top_20pct)} sub={`of all lesion frames found · top 10%: ${pct(k.pooled_triage.top_10pct)} · top 30%: ${pct(k.pooled_triage.top_30pct)}`} />
            <Stat label="Best frame is a lesion" value={`${rank1} of ${lv.length}`} sub="videos where the single top-ranked labelled frame is a lesion (videos with <50% lesion frames)" />
          </div>
          <div className="grid lg:grid-cols-[1.2fr_1fr] gap-6">
            <div className="card card-pad">
              <div className="text-sm font-semibold mb-1">Every Kvasir-Capsule class against normal mucosa (AUROC)</div>
              <div className="text-xs text-muted mb-3">CrohnIPI model, unchanged. Blue = the Crohn&apos;s-type inflammatory findings it was built for. Hover for the share flagged at the app&apos;s 0.5 cut-off.</div>
              <BarCompare rows={rows} max={1} refLine={0.5} refLabel="chance" />
            </div>
            <div className="card card-pad prose-sm">
              <p><b>Why this matters.</b> The earlier external test used all abnormal frames but only 4,000 of the 34,338 normal ones. Here every labelled frame is scored, so lesions are about {pct(k.inflammatory_frames / k.frames, 1)} of frames, as in a real study. The AUROC holds ({f3(iv.auroc)} vs {f3(r.crohnipi.external.kvasir.inflammatory_vs_normal.auroc)} on the subset).</p>
              <p><b>Where it fails.</b> Anatomy it never saw in CrohnIPI is often flagged: pylorus {pct(pc["Pylorus"].flagged_at_0_5)}, ileocecal valve {pct(pc["Ileocecal valve"].flagged_at_0_5)} of frames. Lymphangiectasia and foreign bodies are not recognised as abnormal (AUROC {f3(pc["Lymphangiectasia"].auroc_vs_normal, 2)} and {f3(pc["Foreign Body"].auroc_vs_normal, 2)}): the model learned inflammation, not every capsule finding.</p>
              <p><b>Per video.</b> Ranking each video&apos;s labelled frames puts a median {pct(b.median_recall_top30pct)} of its lesion frames in its top 30%.</p>
            </div>
          </div>
        </Section>); })()}

      {r.video_study && (() => { const v = r.video_study; const q = r.quality_filter; const qr = r.quality_filter_rejected; const hn = r.hard_negative_head; const gpu = r.hard_negative_gpu; const h2 = r.hard_negative_head_v2;
        return (
        <Section id="video" kicker={`Whole capsule studies · ${v.n_videos} Kvasir-Capsule videos · ~50,000 frames each`} title="Whole-video scan">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Stat label="Lesions found in top 5%" value={pct(v.median_lesion_recall_top5pct)} sub={`median share of expert lesion frames within ±10 frames of the top-ranked 5% (top 10%: ${pct(v.median_lesion_recall_top10pct)}) · ${v.with_inflammatory} lesion studies`} />
            <Stat label="Lesion vs lesion-free study" value={`AUROC ${f3(v.auroc_all_p99, 2)}`} sub={`${v.with_inflammatory} vs ${v.without} studies · app model, near chance`} />
            {gpu ? <Stat label="Lesions caught with raw-video negatives" value={`${pct(gpu.labelled_frames.sensitivity_at_0_5_original)} → ${pct(gpu.labelled_frames.sensitivity_at_0_5_hard_negative)}`} sub={`held-out Kvasir lesion frames · study AUROC rose to ${f3(gpu.hard_negative.auroc_p99, 2)}${hn ? `–${f3(hn.summary.head_hn.study_auroc_p99, 2)}` : ""}, but the model learned the camera · not adopted`} />
                : <Stat label="Found in top 10%" value={pct(v.median_lesion_recall_top10pct)} sub={`${v.with_inflammatory} studies with expert inflammatory labels`} />}
            {qr ? <Stat label="Quality filter (rejected)" value={`${(100 - qr.inflammatory_kept_pct).toFixed(0)}% lesions lost`} sub={`it removed ${qr.landmark_and_poor_view_dropped_pct.toFixed(0)}% of landmark / poor-view frames on held-out videos, but also real lesions`} />
               : q ? <Stat label="Quality filter" value={`${q.landmark_and_poor_view_dropped_pct.toFixed(0)}% removed`} sub={`landmark / poor-view frames · ${q.inflammatory_kept_pct?.toFixed(0)}% of lesion frames kept (held-out videos)`} />
               : <Stat label="Frames scored" value="every 10th" sub="fold-1 model; top 150 re-scored by all 5 models" />}
          </div>
          <Note tone="warn"><b>Honest result.</b> Inside a study that has lesions, the scan finds them near the top of the reading list. With the app model it barely tells a lesion study from a lesion-free one, because raw video is full of bubbles, debris, stomach and colon frames that curated CrohnIPI frames never showed the model. Restricting to the small bowel between the expert landmarks did not help (AUROC {f3(v.auroc_sb_p99, 2)}), and a landmark / poor-view filter trained on Kvasir labels was rejected because on held-out videos it also removed real lesions. Adding raw-video frames of lesion-free studies as extra normal examples (trained on one half of the videos, tested on the other) raised study ranking only by losing lesions{gpu ? <>: after a full GPU retrain, sensitivity on held-out expert-labelled Kvasir lesion frames fell from {pct(gpu.labelled_frames.sensitivity_at_0_5_original)} to {pct(gpu.labelled_frames.sensitivity_at_0_5_hard_negative)} (AUROC {f3(gpu.labelled_frames.auroc_original_ensemble, 2)} → {f3(gpu.labelled_frames.auroc_hard_negative, 2)}). Every new negative came from the Kvasir capsule and every lesion from CrohnIPI, so the model learned the camera, not the disease{h2 ? <>; adding the training half{"'"}s Kvasir lesion frames did not repair it (study AUROC {f3(h2.head_bal.study_auroc_p99, 2)})</> : null}. The app keeps the original model: fixing study ranking needs more labelled capsule studies.</> : <>.</>} Open <a className="underline" href="/video/">Video Scan</a> to inspect each study.</Note>
        </Section>); })()}

      <Section id="leak" kicker="Methodology" title="Why we split by patient / video">
        <div className="grid lg:grid-cols-[1fr_1fr] gap-6">
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-3">Same model, same Kvasir-Capsule frames — two ways of splitting (balanced accuracy, 9 classes)</div>
            <BarCompare rows={[{ label: "Random frame split", value: r.leakage.random_frame_split.balanced_accuracy }, { label: "Split by capsule video", value: r.leakage.video_grouped_split.balanced_accuracy, ours: true }]} max={1} fmt={(v) => pct(v, 1)} refLine={1 / 9} refLabel="chance (1/9)" />
          </div>
          <div className="card card-pad prose-sm">
            <p><b>Neighbouring frames from one video are near-duplicates.</b> If they land on both sides of a random split, the model is tested on frames it has practically already seen, and the score looks excellent.</p>
            <p>In Kvasir-Capsule every abnormal class comes from only 1–9 videos (ulcer: {r.leakage.videos_per_class?.ulcer}, erythema: {r.leakage.videos_per_class?.erythema}, polyp: {r.leakage.videos_per_class?.polyp}). Splitting by video drops balanced accuracy from {pct(r.leakage.random_frame_split.balanced_accuracy)} to {pct(r.leakage.video_grouped_split.balanced_accuracy)}.</p>
            <p>That is why all our headline numbers use patient- or video-level splits or fully external datasets, and why very high accuracies from frame-level splits in the literature should be read with care.</p>
          </div>
        </div>
      </Section>

      <Section id="uc" kicker="Module 2 · LIMUC 11,276 images · 564 patients" title="Ulcerative colitis Mayo grader">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Stat label="Remission vs active AUROC" value={f3(r.uc.internal.remission_vs_active.auroc)} sub="LIMUC, patient-grouped 5-fold" />
          <Stat label="4-level Mayo κ (QWK)" value={f3(r.uc.internal.qwk, 2)} sub={`accuracy ${pct(r.uc.internal.accuracy, 1)} · macro-F1 ${f3(r.uc.internal.macro_f1, 2)}`} />
          <Stat label="External AUROC (HyperKvasir)" value={f3(r.uc.external.remission_vs_active.auroc)} sub={`${r.uc.external.remission_vs_active.n} images, Norway`} />
          {r.hyperkvasir_extra?.uc_video_test ? <Stat label="Whole colonoscopy videos" value={`AUROC ${f3(r.hyperkvasir_extra.uc_video_test.auroc_mean_active)}`} sub={`HyperKvasir: ${r.hyperkvasir_extra.uc_video_test.n_colitis} colitis vs ${r.hyperkvasir_extra.uc_video_test.n_clean} clean-bowel videos, 1 frame/s`} />
            : <Stat label="External rank correlation" value={f3(r.uc.external.spearman_expected_mayo_vs_grade, 2)} sub="Spearman ρ, predicted vs expert grade" />}
        </div>
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="card card-pad"><ConfusionMatrix m={r.uc.internal.confusion} labels={["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]} title="LIMUC confusion matrix (out-of-fold)" /></div>
          <div className="card card-pad prose-sm">
            <p><b>Reading the numbers.</b> Telling remission (Mayo 0–1) from active disease (Mayo 2–3) — the decision that changes treatment — works well inside LIMUC (AUROC {f3(r.uc.internal.remission_vs_active.auroc)}). Exact grades are harder: Mayo 1 vs 2 is also where human endoscopists disagree most.</p>
            <p>On HyperKvasir the AUROC falls to {f3(r.uc.external.remission_vs_active.auroc)}: different scopes, lighting and graders. This domain shift is the main limitation of the extension, and fine-tuning (as done for the Crohn's model) is the next step.</p>
          </div>
        </div>
      </Section>

      <Section id="flare" kicker="Module 3 · HMP2 · 66 patients · 629 visit pairs · 61 flares" title="Flare risk at the next visit">
        <div className="grid lg:grid-cols-[1.3fr_1fr] gap-6">
          <div className="card card-pad overflow-x-auto">
            <table className="data">
              <thead><tr><th>Model</th><th className="num">AUROC</th><th className="num">95% CI</th><th className="num">AUPRC</th><th className="num">Brier</th></tr></thead>
              <tbody>
                {Object.entries(r.flare.results).map(([k, v]: any) => (
                  <tr key={k}><td className={k === r.flare.best ? "font-semibold" : ""}>{k}</td><td className="num">{f3(v.auroc_mean)} ± {f3(v.auroc_sd, 3)}</td>
                    <td className="num">{f3(v.auroc_ci95[0], 2)}–{f3(v.auroc_ci95[1], 2)}</td><td className="num">{f3(v.auprc_mean)}</td><td className="num">{v.brier == null ? "—" : f3(v.brier)}</td></tr>))}
              </tbody>
            </table>
            <p className="text-xs text-muted mt-3">{r.flare.cv}. Base flare rate {pct(r.flare.summary.flare_rate, 1)} (AUPRC of a random model).</p>
          </div>
          <div className="card card-pad">
            <div className="text-sm font-semibold mb-2">Calibration after Platt recalibration</div>
            <Calibration bins={r.flare.calibration_recalibrated ?? r.flare.calibration} />
            <div className="text-xs text-muted mt-1">Brier score {f3(r.flare.results[r.flare.best].brier)} → {f3(r.flare.results[r.flare.best].brier_recalibrated)} after recalibration (lower is better).</div>
            <div className="text-xs text-muted mt-2">Top drivers: {r.flare.importance.slice(0, 4).map((x: any) => x[0]).join(" · ")}</div>
          </div>
        </div>
      </Section>
    </div>
  );
}
