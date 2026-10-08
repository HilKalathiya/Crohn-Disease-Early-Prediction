/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        surface: "var(--surface-1)", page: "var(--page)", ink: "var(--text-primary)", ink2: "var(--text-secondary)",
        muted: "var(--text-muted)", line: "var(--grid)", brand: "var(--brand)", brandsoft: "var(--brand-soft)",
        good: "var(--good)", warn: "var(--warning)", serious: "var(--serious)", critical: "var(--critical)",
      },
      fontFamily: { sans: ['system-ui', '-apple-system', '"Segoe UI"', 'Roboto', 'sans-serif'] },
      borderRadius: { xl2: "14px" },
    },
  },
  plugins: [],
};
