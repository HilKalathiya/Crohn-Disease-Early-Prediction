import "./globals.css";
import Shell from "@/components/Shell";

export const metadata = {
  title: "CrohnSight — AI Crohn's Severity Assessment",
  description: "Capsule-endoscopy lesion detection & severity grading for Crohn's disease, UC Mayo grading and flare-risk forecasting. Research prototype.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body><Shell>{children}</Shell></body>
    </html>
  );
}
