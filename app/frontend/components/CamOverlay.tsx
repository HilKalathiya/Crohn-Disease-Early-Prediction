"use client";
import { useEffect, useRef } from "react";

// perceptually ordered colormap (dark blue -> cyan -> yellow -> red), alpha grows with evidence
const STOPS: [number, number[]][] = [[0, [13, 54, 107]], [0.35, [57, 135, 229]], [0.6, [27, 175, 122]], [0.8, [237, 161, 0]], [1, [227, 73, 72]]];
function cmap(v: number) {
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i]; const [t0, c0] = STOPS[i - 1];
    if (v <= t1) { const f = (v - t0) / (t1 - t0); return c0.map((c, k) => Math.round(c + f * (c1[k] - c))); }
  }
  return STOPS[STOPS.length - 1][1];
}
/** bilinear upsample of an h x w grid to S x S */
function upsample(g: number[][], S: number) {
  const h = g.length, w = g[0].length, out = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const gy = Math.min(h - 1, Math.max(0, ((y + 0.5) / S) * h - 0.5)), gx = Math.min(w - 1, Math.max(0, ((x + 0.5) / S) * w - 0.5));
    const y0 = Math.floor(gy), x0 = Math.floor(gx), y1 = Math.min(h - 1, y0 + 1), x1 = Math.min(w - 1, x0 + 1), fy = gy - y0, fx = gx - x0;
    out[y * S + x] = g[y0][x0] * (1 - fx) * (1 - fy) + g[y0][x1] * fx * (1 - fy) + g[y1][x0] * (1 - fx) * fy + g[y1][x1] * fx * fy;
  }
  return out;
}

export default function CamOverlay({ src, cam, opacity, size = 448 }: { src: string; cam?: number[][]; opacity: number; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current; if (!cv) return; const ctx = cv.getContext("2d")!; const img = new Image();
    img.onload = () => {
      cv.width = size; cv.height = size; ctx.drawImage(img, 0, 0, size, size);
      if (!cam || opacity <= 0) return;
      const S = 112, up = upsample(cam, S), id = ctx.createImageData(S, S);
      for (let i = 0; i < S * S; i++) { const v = Math.max(0, Math.min(1, up[i])), c = cmap(v); id.data.set([c[0], c[1], c[2], Math.round(255 * opacity * Math.pow(v, 0.8))], i * 4); }
      const off = document.createElement("canvas"); off.width = S; off.height = S; off.getContext("2d")!.putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(off, 0, 0, size, size);
    };
    img.src = src;
  }, [src, cam, opacity, size]);
  return <canvas ref={ref} className="w-full h-auto rounded-xl" style={{ aspectRatio: "1 / 1", background: "#000" }} aria-label="Frame with model attention heat-map" />;
}

export function CamLegend() {
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted">
      <span>low evidence</span>
      <span className="h-2 w-28 rounded-full" style={{ background: "linear-gradient(90deg,#0d366b,#3987e5,#1baf7a,#eda100,#e34948)" }} />
      <span>high evidence</span>
    </div>
  );
}
