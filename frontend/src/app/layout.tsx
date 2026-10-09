import type { Metadata } from "next";
import "./globals.css";
import SmoothScroll from "@/components/SmoothScroll";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Complaint Intelligence & LLM Evaluation Pipeline",
  description:
    "Benchmarking classical ML, sentence embeddings, and generative LLMs on CFPB consumer complaints. Macro-F1, cost, calibration, and counterfactual bias probes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground antialiased selection:bg-brand-amber/30 selection:text-brand-amber">
        <SmoothScroll>
          <div className="min-h-screen flex flex-col bg-grid-pattern relative">
            <Navbar />
            <main className="flex-1">{children}</main>
          </div>
        </SmoothScroll>
      </body>
    </html>
  );
}
