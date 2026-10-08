import { PageHead, Note } from "@/components/ui";

const DATASETS = [
  { name: "CrohnIPI", role: "Main training set — Crohn's lesion type & severity", size: "3,484 labelled capsule frames · 63 Crohn's patients · 7 classes", where: "CHU Nantes, France · PillCam SB3",
    paper: "Vallée et al., SPIE Medical Imaging 2020; de Maissin et al., Endoscopy International Open 2021", licence: "CC BY-NC-ND 4.0 (research use)", link: "https://zenodo.org/records/14616507" },
  { name: "Kvasir-Capsule", role: "External test of the Crohn's model (every labelled frame + 43 whole videos) · demo study · leakage experiment", size: "All 47,238 labelled frames (14 classes) and all 43 labelled full videos (1,979,285 frames)", where: "Bærum Hospital, Norway · Olympus capsule",
    paper: "Smedsrud et al., Scientific Data 2021", licence: "CC BY 4.0", link: "https://osf.io/dv2ag/" },
  { name: "SEE-AI Project", role: "Second external test · trains the experimental findings screen", size: "18,481 capsule images with expert bounding boxes, 12 finding types", where: "Japan",
    paper: "Yokote et al., DEN Open 2024", licence: "Kaggle dataset licence (see dataset page)", link: "https://www.kaggle.com/datasets/capsuleyolo/kyucapsule" },
  { name: "LIMUC", role: "Trains the ulcerative-colitis Mayo grader", size: "11,276 colonoscopy images · 564 patients · Mayo 0–3", where: "Turkey (single centre)",
    paper: "Polat et al., Inflammatory Bowel Diseases 2022", licence: "CC BY 4.0", link: "https://zenodo.org/records/5827695" },
  { name: "HyperKvasir", role: "External test of the UC grader (images + whole videos) · polyp heat-map check", size: "851 UC images graded 0-1 to 3 · 46 colonoscopy videos · 1,000 polyp masks", where: "Bærum Hospital, Norway",
    paper: "Borgli et al., Scientific Data 2020", licence: "CC BY 4.0", link: "https://osf.io/mh9sj" },
  { name: "HMP2 / IBDMDB", role: "Flare-risk add-on", size: "66 Crohn's patients · 873 HBI visits · fecal calprotectin, CRP, medications", where: "US IBD centres (IBDMDB)",
    paper: "Lloyd-Price et al., Nature 2019", licence: "Public research data", link: "https://ibdmdb.org" },
];

const STEPS = [
  ["Data audit", "Counted patients, classes and duplicates in every dataset. The Kaggle 'MIMIC-IV transcripts' file turned out to hold 7 patients, 13 admissions and no Crohn's cases (209,951 of its 232,158 rows are exact duplicates), so it was not used."],
  ["Pre-processing", "Full frame at its native 320 × 320 (no cropping — lesions can sit at the edge of the circular capsule view), ImageNet normalisation."],
  ["Model", "EfficientNet-B2 (7.7 M parameters) pre-trained on ImageNet, fine-tuned end-to-end for 40 epochs on an RTX 3050 GPU: AdamW, cosine schedule, label smoothing 0.1, square-root class weights for the rare classes, exponential moving average of the weights. Two training seeds; the final model averages both (decided before scoring)."],
  ["Augmentation", "Random rotation (any angle — the capsule view is circular), flips, ±15% zoom, brightness/contrast/saturation jitter."],
  ["Validation", "Official CrohnIPI 5-fold split; one model per fold, never evaluated on its own training frames. Fixed number of epochs — no early stopping on test data. 8-view test-time augmentation (4 rotations × flip). Earlier versions (EfficientNet-B0 at 224 px, 10 and 15 epochs) are kept in the results for comparison."],
  ["Severity tiers", "7 classes mapped to 4 ordinal tiers following capsule activity scores (Lewis, CECDAI) and the CrohnIPI severity order: Normal → Mild (erythema, oedema) → Moderate (aphthoid, 3–10 mm ulcer) → Severe (> 10 mm ulcer, stenosis)."],
  ["Explainability", "Class-activation maps from the last convolutional layer, averaged across the 10 models; model agreement and ±1 SD shown with every prediction."],
  ["External tests", "Kvasir-Capsule and SEE-AI evaluated with no re-training. Heat-maps checked against expert lesion boxes (pointing game)."],
];

export default function Data() {
  return (
    <div>
      <PageHead kicker="Transparency" title="Data & methods">
        Six open datasets, no access applications needed. Each one has a single job, chosen so that the claims we make can actually be tested.
      </PageHead>
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4 mb-12">
        {DATASETS.map((d) => (
          <div key={d.name} className="card card-pad">
            <div className="text-lg font-bold">{d.name}</div>
            <div className="text-sm text-brand font-semibold mb-2">{d.role}</div>
            <dl className="text-xs grid gap-1.5 text-ink2">
              <div><dt className="inline font-semibold text-ink">Size: </dt><dd className="inline">{d.size}</dd></div>
              <div><dt className="inline font-semibold text-ink">Source: </dt><dd className="inline">{d.where}</dd></div>
              <div><dt className="inline font-semibold text-ink">Paper: </dt><dd className="inline">{d.paper}</dd></div>
              <div><dt className="inline font-semibold text-ink">Licence: </dt><dd className="inline">{d.licence}</dd></div>
              <div><a href={d.link} className="text-brand underline" target="_blank" rel="noreferrer">{d.link}</a></div>
            </dl>
          </div>
        ))}
      </div>
      <h2 className="text-2xl font-bold mb-4">Method, step by step</h2>
      <ol className="grid md:grid-cols-2 gap-3 mb-12">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="card card-pad flex gap-4">
            <span className="w-8 h-8 shrink-0 rounded-full grid place-items-center text-sm font-bold text-white" style={{ background: "var(--brand)" }}>{i + 1}</span>
            <div><div className="font-semibold mb-1">{t}</div><p className="text-sm text-ink2 leading-relaxed">{d}</p></div>
          </li>
        ))}
      </ol>
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card card-pad prose-sm">
          <div className="text-lg font-bold mb-2">Limitations</div>
          <p>• CrohnIPI has no patient IDs in its public release; frames from one patient may sit in different folds, so internal scores can be optimistic — the external tests are there to check this.</p>
          <p>• Frame-level labels: one frame, one label. Real frames can show several lesions.</p>
          <p>• The UC grader and the findings screen use frozen features (CPU budget) and lose accuracy on new hospitals.</p>
          <p>• The flare model has only 61 flare events; its confidence intervals are wide.</p>
          <p>• Datasets are European, Japanese, Turkish and US; performance on Indian patients is untested.</p>
        </div>
        <div className="grid gap-4 content-start">
          <Note><b>On hold, not dropped:</b> onset prediction (5–6 years before diagnosis) and disease-course prediction (B1 → stricturing/penetrating, surgery) from routine hospital records. These need MIMIC-IV (credentialed access pending). The web app and validation code are built so these modules can be added without changing the interface.</Note>
          <Note tone="warn">CrohnSight is a student research prototype. It is not a medical device and must not be used to make clinical decisions.</Note>
        </div>
      </div>
    </div>
  );
}
