export const API = process.env.NEXT_PUBLIC_API ?? "";

export type Cam = Record<string, number[][]>;
export interface CapsuleResult {
  probs: Record<string, number>; probs_sd: Record<string, number>; top_class: string; top_name: string;
  lesion_score: number; lesion: boolean; tier: number; tier_name: string; tier_color: string; tier_text: string;
  tier_probs: number[]; agreement: number; n_models: number; agree_count: number; cam?: Cam; ms?: number; lesion_screening?: boolean; image?: string;
  findings?: { probs: Record<string, number>; top: string; abnormal_prob: number | null };
}
export interface StudyFrame {
  idx: number; name: string; thumb: string; lesion_score: number; top_class: string; top_name: string;
  tier: number; tier_name: string; tier_color: string; probs: Record<string, number>;
}
export interface StudySummary {
  frames: number; flagged: number; flagged_pct: number; worst_tier: number; worst_tier_name: string;
  tier_counts: number[]; class_counts: Record<string, number>; burden_index: number; review_order: number[];
  review_needed: number; reading_saved_pct: number; seconds: number;
}
export interface OpPoint { threshold: number; crohnipi: { sensitivity: number; specificity: number }; kvasir_47k: { sensitivity: number; specificity: number };
  seeai: { erosion: number; stenosis: number; erythema: number; no_finding_specificity: number } }
export interface Meta { threshold: number; screening_threshold: number; operating_points?: { balanced: OpPoint; screening: OpPoint } }
export type Mode = "balanced" | "screening";
export const modeThr = (m: Meta | null, mode: Mode) => (m ? (mode === "screening" ? m.screening_threshold : m.threshold) : 0.5);
export interface VideoTop { frame: number; time_s: number | null; scan_lesion: number; lesion: number; tier: number; tier_name: string; top_class: string; top_name: string; agree_count: number; n_models: number; thumb: string }
export interface VideoResult { total_frames: number; fps: number; duration_s: number | null; width: number; height: number; stride: number; scored: number; seconds: number;
  stage1_model: string; n_models: number; thresholds: { balanced: number; screening: number }; flagged_pct: { balanced: number; screening: number };
  tier_counts: number[]; burden_index: number; bins: number[][]; top: VideoTop[] }
export interface VideoJob { id: string; name: string; status: "queued" | "running" | "done" | "error"; stage: string; done: number; todo: number; error: string | null; result: VideoResult | null }
export interface UcResult { probs: number[]; mayo: number; mayo_text: string; expected_mayo: number; active_prob: number; remission: boolean; cam: Cam }

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail ?? m; } catch {} throw new Error(m); }
  return r.json() as Promise<T>;
}
export async function getJSON<T>(path: string): Promise<T> { return handle<T>(await fetch(API + path)); }
export async function postFile<T>(path: string, file: File | Blob, name = "image.jpg"): Promise<T> {
  const fd = new FormData(); fd.append("file", file, (file as File).name ?? name);
  return handle<T>(await fetch(API + path, { method: "POST", body: fd }));
}
export async function postFiles<T>(path: string, files: File[]): Promise<T> {
  const fd = new FormData(); files.forEach((f) => fd.append("files", f, f.name));
  return handle<T>(await fetch(API + path, { method: "POST", body: fd }));
}
export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
}
export async function urlToFile(url: string, name: string): Promise<File> {
  const b = await (await fetch(API + url)).blob(); return new File([b], name, { type: b.type || "image/jpeg" });
}

export const CLASS_NAMES: Record<string, string> = {
  N: "Normal mucosa", E: "Erythema", O: "Oedema", AU: "Aphthoid ulcer", "U3-10": "Ulcer 3–10 mm", "U>10": "Ulcer >10 mm", S: "Stenosis",
};
export const CLASS_ORDER = ["N", "E", "O", "AU", "U3-10", "U>10", "S"];
export const TIER_COLORS = ["var(--good)", "var(--warning)", "var(--serious)", "var(--critical)"];
export const TIER_NAMES = ["No lesion", "Mild", "Moderate", "Severe"];
export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
export const f3 = (x: number | null | undefined, d = 3) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d));
