"use client";
import { useEffect, useMemo, useState } from "react";
import { getJSON, pct, postJSON } from "@/lib/api";
import { Note, PageHead, Spinner } from "@/components/ui";
import { FlareTimeline } from "@/components/charts";

type Pt = { id: string; visits: number; max_hbi: number };
type Pred = { risk: number; relative: number; imputed: string[]; level: string; base_rate: number; contributions: { feature: string; key: string; value: number | null; contribution: number }[] };
const FIELDS: { key: string; label: string; min: number; max: number; step: number; def: number }[] = [
  { key: "hbi", label: "Current HBI", min: 0, max: 4, step: 1, def: 2 },
  { key: "hbi_prev", label: "Previous visit HBI", min: 0, max: 20, step: 1, def: 2 },
  { key: "hbi_mean3", label: "Mean HBI, last 3 visits", min: 0, max: 15, step: 0.5, def: 2 },
  { key: "hbi_max_all", label: "Highest HBI so far", min: 0, max: 25, step: 1, def: 5 },
  { key: "wellbeing", label: "General wellbeing (0 very well – 4 terrible)", min: 0, max: 4, step: 1, def: 0 },
  { key: "liquid_stools", label: "Liquid stools in last 24 h", min: 0, max: 10, step: 1, def: 1 },
  { key: "pain", label: "Abdominal pain (0 none – 3 severe)", min: 0, max: 3, step: 1, def: 0 },
  { key: "fc_last", label: "Last fecal calprotectin (µg/g)", min: 0, max: 1500, step: 10, def: 80 },
  { key: "age", label: "Age (years)", min: 6, max: 80, step: 1, def: 30 },
  { key: "disease_years", label: "Years since diagnosis", min: 0, max: 40, step: 1, def: 5 },
  { key: "male", label: "Sex (0 = female, 1 = male)", min: 0, max: 1, step: 1, def: 0 },
];
const PRESETS: Record<string, Record<string, number>> = {
  "Stable remission": { hbi: 1, hbi_prev: 1, hbi_mean3: 1, hbi_max_all: 3, wellbeing: 0, liquid_stools: 0, pain: 0, fc_last: 40 },
  "Smouldering": { hbi: 4, hbi_prev: 3, hbi_mean3: 3.5, hbi_max_all: 8, wellbeing: 1, liquid_stools: 3, pain: 1, fc_last: 300 },
  "Recently flared": { hbi: 4, hbi_prev: 9, hbi_mean3: 6.5, hbi_max_all: 12, wellbeing: 2, liquid_stools: 4, pain: 1, fc_last: 600 },
};

export default function Flare() {
  const [pts, setPts] = useState<Pt[]>([]); const [pid, setPid] = useState<string | null>(null); const [tl, setTl] = useState<any[]>([]);
  const [v, setV] = useState<Record<string, number>>(Object.fromEntries(FIELDS.map((f) => [f.key, f.def])));
  const [pred, setPred] = useState<Pred | null>(null); const [busy, setBusy] = useState(false);
  useEffect(() => { getJSON<{ patients: Pt[] }>("/api/flare/patients").then((r) => { const s = [...r.patients].sort((a, b) => b.max_hbi - a.max_hbi); setPts(s); setPid(s.find((p) => p.visits > 10)?.id ?? s[0]?.id); }).catch(() => {}); }, []);
  useEffect(() => { if (pid) getJSON<any>(`/api/flare/patient/${pid}`).then((r) => setTl(r.timeline)).catch(() => {}); }, [pid]);
  useEffect(() => {
    const t = setTimeout(async () => {
      setBusy(true);
      try { setPred(await postJSON<Pred>("/api/flare/predict", { features: { ...v, hbi_delta: v.hbi - v.hbi_prev, hbi_max3: Math.max(v.hbi, v.hbi_prev, v.hbi_mean3), fc_log: Math.log1p(v.fc_last), fc_weeks_since: 2, gap_weeks: 2 } })); }
      catch {} finally { setBusy(false); }
    }, 250); return () => clearTimeout(t);
  }, [v]);
  const maxC = useMemo(() => Math.max(0.01, ...(pred?.contributions ?? []).map((c) => Math.abs(c.contribution))), [pred]);

  return (
    <div>
      <PageHead kicker="Module 3 · HMP2 / IBDMDB (add-on)" title="Flare Risk Forecaster">
        For a Crohn's patient currently in remission (Harvey-Bradshaw Index &lt; 5), estimate the chance that the next visit (about two weeks later) shows a flare (HBI ≥ 5). Trained on 66 Crohn's patients followed for up to a year in the HMP2 study.
      </PageHead>
      <div className="grid lg:grid-cols-[1fr_380px] gap-6">
        <div className="card card-pad">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <span className="text-sm font-semibold">Patient timeline</span>
            <select value={pid ?? ""} onChange={(e) => setPid(e.target.value)} className="rounded-lg border hairline px-2 py-1 text-sm bg-[var(--surface-1)]">
              {pts.map((p) => <option key={p.id} value={p.id}>{p.id} · {p.visits} visits · max HBI {p.max_hbi}</option>)}
            </select>
          </div>
          <FlareTimeline data={tl} />
          <p className="text-xs text-muted mt-2">Risk bars are out-of-fold predictions: each patient's bars come from models that never saw that patient.</p>
        </div>
        <div className="card card-pad">
          <div className="text-sm font-semibold mb-1">What-if calculator</div>
          <div className="flex flex-wrap gap-1.5 mb-3">{Object.keys(PRESETS).map((k) => <button key={k} className="chip" onClick={() => setV({ ...v, ...PRESETS[k] })}>{k}</button>)}</div>
          <div className="grid gap-2.5">
            {FIELDS.map((f) => (
              <label key={f.key} className="grid grid-cols-[1fr_52px] items-center gap-x-3 text-xs">
                <span className="text-ink2">{f.label}</span><span className="tnum text-right font-semibold">{v[f.key]}</span>
                <input type="range" min={f.min} max={f.max} step={f.step} value={v[f.key]} onChange={(e) => setV({ ...v, [f.key]: +e.target.value })} className="col-span-2" />
              </label>
            ))}
          </div>
          <div className="mt-4 rounded-xl p-4" style={{ background: "var(--surface-2)" }}>
            {busy && !pred ? <Spinner text="…" /> : pred && (<>
              <div className="text-xs text-muted">Risk of flare at next visit</div>
              <div className="text-3xl font-bold tnum">{pct(pred.risk)} <span className="text-base font-semibold" style={{ color: pred.level === "Low" ? "var(--good)" : pred.level === "Elevated" ? "var(--serious)" : "var(--critical)" }}>{pred.level}</span></div>
              <div className="text-xs text-muted"><b className="tnum">{pred.relative.toFixed(1)}×</b> the average risk ({pct(pred.base_rate)} of remission visits flared at the next visit)</div>
            </>)}
          </div>
          {pred && (<div className="mt-4">
            <div className="text-xs font-semibold mb-2">What moved this estimate</div>
            {pred.contributions.slice(0, 7).map((c) => (
              <div key={c.key} className="grid grid-cols-[1fr_120px] gap-2 items-center text-[11px] mb-1">
                <span className="text-ink2 truncate" title={c.feature}>{c.feature}</span>
                <div className="relative h-3"><div className="absolute left-1/2 top-0 bottom-0 w-px" style={{ background: "var(--axis)" }} />
                  <div className="absolute top-0 h-3 rounded" style={{ background: c.contribution > 0 ? "var(--critical)" : "var(--series-1)", left: c.contribution > 0 ? "50%" : `${50 - (50 * Math.abs(c.contribution)) / maxC}%`, width: `${(50 * Math.abs(c.contribution)) / maxC}%` }} />
                </div>
              </div>))}
            <div className="text-[10px] text-muted mt-1">red = raises risk · blue = lowers risk (random-forest path attribution). Not entered, typical value used: {pred.imputed.join(", ") || "none"}.</div>
          </div>)}
        </div>
      </div>
      <div className="mt-6"><Note tone="warn"><b>Honest result.</b> With 61 flare events in 27 patients, the random forest (AUROC 0.71) only matches a one-number baseline — the current HBI on its own (AUROC 0.71). Recent symptom trajectory carries most of the signal; more patients, or MIMIC-IV once approved, are needed before extra variables add value.</Note></div>
    </div>
  );
}
