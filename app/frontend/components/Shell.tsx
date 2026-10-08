"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/capsule/", label: "Lesion Analyzer" },
  { href: "/study/", label: "Study Triage" },
  { href: "/video/", label: "Video Scan" },
  { href: "/uc/", label: "UC Grader" },
  { href: "/flare/", label: "Flare Risk" },
  { href: "/results/", label: "Results" },
  { href: "/data/", label: "Data & Methods" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [theme, setTheme] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => { try { const t = localStorage.getItem("theme"); if (t) { document.documentElement.dataset.theme = t; setTheme(t); } } catch {} }, []);
  const toggle = () => {
    const cur = theme ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nx = cur === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = nx; setTheme(nx);
    try { localStorage.setItem("theme", nx); } catch {}
  };
  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-40 border-b hairline" style={{ background: "color-mix(in srgb, var(--page) 88%, transparent)", backdropFilter: "blur(8px)" }}>
        <div className="mx-auto max-w-7xl px-4 h-14 flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2 font-bold text-[17px] shrink-0">
            <Logo /> CrohnSight
          </Link>
          <nav className="hidden lg:flex items-center gap-1 ml-4">
            {NAV.map((n) => {
              const active = n.href === "/" ? path === "/" : path?.startsWith(n.href);
              return (
                <Link key={n.href} href={n.href} className={`px-3 py-1.5 rounded-lg text-sm font-medium ${active ? "bg-brandsoft text-brand" : "text-ink2 hover:text-ink"}`}>
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <span className="chip hidden sm:inline-flex">Research prototype</span>
            <button onClick={toggle} className="btn btn-ghost !px-2.5" aria-label="Toggle light / dark theme" title="Light / dark theme">
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M8 1.5a6.5 6.5 0 0 1 0 13z" fill="currentColor" /></svg>
            </button>
            <button onClick={() => setOpen(!open)} className="btn btn-ghost !px-2.5 lg:hidden" aria-label="Menu">☰</button>
          </div>
        </div>
        {open && (
          <nav className="lg:hidden border-t hairline px-4 py-2 grid gap-1">
            {NAV.map((n) => (<Link key={n.href} href={n.href} onClick={() => setOpen(false)} className="px-2 py-2 rounded-lg text-sm hover:bg-[var(--surface-2)]">{n.label}</Link>))}
          </nav>
        )}
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8">{children}</main>
      <footer className="border-t hairline">
        <div className="mx-auto max-w-7xl px-4 py-6 text-xs text-muted flex flex-wrap gap-x-6 gap-y-2">
          <span>CrohnSight · Minor Project 2026 · PDEU — Hil Kalathiya &amp; Sahaj Saliya · Guide: Prof. Dr. Babalu Kumar Singh</span>
          <span>Research prototype — not a medical device and not for clinical decisions.</span>
        </div>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="14" fill="var(--brand)" />
      <circle cx="16" cy="16" r="8.5" fill="none" stroke="#fff" strokeWidth="2.4" />
      <circle cx="19" cy="13" r="2.6" fill="#fff" />
    </svg>
  );
}
