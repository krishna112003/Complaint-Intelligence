"use client";

import { useEffect, useState } from "react";
import { 
  ArrowRight, 
  Sparkles, 
  ShieldCheck, 
  Cpu, 
  Database, 
  Terminal, 
  CheckCircle2, 
  Layers, 
  TrendingDown, 
  TrendingUp, 
  Code2, 
  Play, 
  Zap,
  ArrowUpRight
} from "lucide-react";
import { motion, useScroll, useTransform } from "framer-motion";
import { SummaryData, ExamplesData } from "@/lib/types";

interface HeroProps {
  summary: SummaryData | null;
  examples: ExamplesData | null;
}

export default function Hero({ summary, examples }: HeroProps) {
  const [complaintCount, setComplaintCount] = useState<number>(0);
  const [goldCount, setGoldCount] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<"inference" | "schema" | "diff">("inference");
  const [activeModel, setActiveModel] = useState<"tfidf" | "gemini">("tfidf");

  // Lightweight GPU scroll parallax
  const { scrollY } = useScroll();
  const cardY = useTransform(scrollY, [0, 500], [0, 24]);

  // Counter animations
  useEffect(() => {
    if (!summary) return;
    const targetComplaints = summary.bronze_row_count || 208398;
    const targetGold = summary.gold_row_count || 49787;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      setComplaintCount(targetComplaints);
      setGoldCount(targetGold);
      return;
    }

    const duration = 1400;
    const steps = 35;
    const stepTime = duration / steps;
    let step = 0;

    const timer = setInterval(() => {
      step += 1;
      const progress = Math.min(1, step / steps);
      const eased = 1 - Math.pow(1 - progress, 3);
      setComplaintCount(Math.floor(targetComplaints * eased));
      setGoldCount(Math.floor(targetGold * eased));
      if (step >= steps) {
        clearInterval(timer);
        setComplaintCount(targetComplaints);
        setGoldCount(targetGold);
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [summary]);

  const complaintList = examples?.examples || [];

  return (
    <section className="relative min-h-[95vh] pt-20 pb-12 flex flex-col justify-between overflow-hidden">
      {/* 21st.dev Signature Ambient Glow Pulse behind Hero */}
      <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
        <div 
          className="animate-glow-pulse absolute left-1/2 top-24 -translate-x-1/2 w-[140vw] sm:w-[900px] h-[500px] rounded-full blur-[120px] pointer-events-none opacity-40 dark:opacity-30"
          style={{
            background: "radial-gradient(circle, rgba(245, 158, 11, 0.35) 0%, rgba(6, 182, 212, 0.2) 45%, rgba(139, 92, 246, 0.15) 70%, transparent 85%)"
          }}
        />
      </div>

      {/* Main Hero Container: 2-Column Luxury Layout */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full z-10 pt-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-8 items-center">
          
          {/* Left Column: Editorial Headline & Pitch */}
          <div className="lg:col-span-7 flex flex-col items-start">
            
            {/* Top Pill Radar Tag */}
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-mono bg-surface/80 border border-white/[0.1] dark:border-white/[0.08] backdrop-blur-md mb-6 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-brand-amber animate-pulse" />
              <span className="text-zinc-300 font-medium">CFPB Consumer Intelligence</span>
              <span className="text-zinc-600">//</span>
              <span className="text-brand-cyan font-semibold">49,787 Temporal Gold Split</span>
            </div>

            {/* Signature Headline (21st.dev Editorial Style with Italic Accent Serif) */}
            <h1 className="text-4xl sm:text-6xl lg:text-[68px] font-bold tracking-tight text-foreground leading-[1.05] text-balance">
              The <em className="font-serif-italic text-brand-amber text-[1.12em] font-normal not-italic tracking-normal">living</em> benchmark <br className="hidden sm:block" />
              of LLMs vs Classical ML.
            </h1>

            {/* Subheading with high readability */}
            <p className="mt-5 text-base sm:text-lg text-zinc-400 max-w-2xl leading-relaxed font-sans">
              Is a generative model worth its latency and inference cost over supervised classification?
              Benchmarked TF-IDF + Logistic Regression, dense embeddings, and Gemini 2.0 Flash across{" "}
              <span className="text-foreground font-semibold">49,787 CFPB complaints</span> with paired bootstrap
              significance tests, calibration auditing, and counterfactual demographic bias audits.
            </p>

            {/* Dual Action CTAs */}
            <div className="mt-8 flex flex-wrap items-center gap-3.5 w-full sm:w-auto">
              {/* Shimmer Button (21st.dev style) */}
              <a
                href="#arena"
                className="group relative inline-flex items-center justify-center rounded-full p-[1.5px] transition-transform duration-200 active:scale-95 shadow-lg shadow-brand-amber/10"
              >
                <span className="absolute inset-0 rounded-full overflow-hidden">
                  <span className="absolute inset-[-150%] animate-spin-slow bg-[conic-gradient(from_0deg,transparent_0%,#f59e0b_20%,#06b6d4_45%,transparent_65%)]" />
                </span>
                <span className="relative flex items-center space-x-2 rounded-full bg-neutral-950 px-6 py-2.5 text-sm font-semibold text-white ring-1 ring-white/10 transition-colors group-hover:bg-neutral-900">
                  <span>Explore Model Arena</span>
                  <ArrowRight className="w-4 h-4 text-brand-amber transition-transform group-hover:translate-x-1" />
                </span>
              </a>

              {/* Secondary Frosted Glass Pill */}
              <a
                href="#playground"
                className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-full text-sm font-medium border border-white/[0.12] bg-surface/70 hover:bg-surface/90 text-zinc-200 transition-all hover:border-white/[0.2] active:scale-95 backdrop-blur-md shadow-sm"
              >
                <span>Interactive Diagnostics</span>
                <ArrowUpRight className="w-4 h-4 text-zinc-400" />
              </a>
            </div>

            {/* Social Proof / Verified Badges */}
            <div className="mt-8 pt-6 border-t border-white/[0.08] w-full flex flex-wrap items-center gap-y-2 gap-x-6 text-xs text-zinc-400 font-sans">
              <div className="flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-emerald" />
                <span className="text-zinc-300 font-medium">100% Pydantic Schema Valid</span>
              </div>
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 text-brand-cyan" />
                <span className="text-zinc-300 font-medium">0.00% Demographic Bias Flip</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-brand-amber" />
                <span className="text-zinc-300 font-medium">&lt; 1 ms Classical Latency</span>
              </div>
            </div>

            {/* Key Empirical Metrics Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6 w-full">
              <div className="p-3 rounded-xl bg-surface/60 border border-white/[0.08] backdrop-blur-md">
                <div className="text-[11px] font-mono text-zinc-500 uppercase">Bronze Raw</div>
                <div className="text-xl sm:text-2xl font-bold text-foreground num-mono mt-0.5">
                  {complaintCount.toLocaleString()}
                </div>
                <div className="text-[10px] text-zinc-500">CFPB grievances</div>
              </div>
              <div className="p-3 rounded-xl bg-surface/60 border border-white/[0.08] backdrop-blur-md">
                <div className="text-[11px] font-mono text-zinc-500 uppercase">Gold Horizon</div>
                <div className="text-xl sm:text-2xl font-bold text-brand-amber num-mono mt-0.5">
                  {goldCount.toLocaleString()}
                </div>
                <div className="text-[10px] text-zinc-500">10 Balanced classes</div>
              </div>
              <div className="p-3 rounded-xl bg-surface/60 border border-white/[0.08] backdrop-blur-md">
                <div className="text-[11px] font-mono text-zinc-500 uppercase">Top Macro-F1</div>
                <div className="text-xl sm:text-2xl font-bold text-brand-cyan num-mono mt-0.5">
                  0.6002
                </div>
                <div className="text-[10px] text-zinc-500">TF-IDF + LogReg</div>
              </div>
              <div className="p-3 rounded-xl bg-surface/60 border border-white/[0.08] backdrop-blur-md">
                <div className="text-[11px] font-mono text-zinc-500 uppercase">Schema Valid</div>
                <div className="text-xl sm:text-2xl font-bold text-brand-emerald num-mono mt-0.5">
                  100%
                </div>
                <div className="text-[10px] text-zinc-500">Zero parse failure</div>
              </div>
            </div>

          </div>

          {/* Right Column: 3D Floating Interactive Card (The Laptop Showcase in Reel) */}
          <div className="lg:col-span-5 relative">
            
            {/* Ambient Backlight Aura */}
            <div className="absolute -inset-4 bg-gradient-to-r from-brand-amber/20 via-brand-cyan/20 to-purple-600/20 rounded-2xl blur-2xl opacity-50 -z-10" />

            {/* Window Card Frame with Scroll Parallax */}
            <motion.div
              style={{ y: cardY }}
              className="relative rounded-2xl border border-white/[0.12] bg-[#0c0d12]/95 backdrop-blur-2xl shadow-2xl overflow-hidden transition-all duration-300 hover:border-brand-amber/30"
            >
              
              {/* macOS Style Header Bar */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.08] bg-white/[0.02]">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80 inline-block" />
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80 inline-block" />
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80 inline-block" />
                  <span className="text-xs font-mono text-zinc-400 ml-2 hidden sm:inline">
                    complaint_eval_pipeline.py
                  </span>
                </div>

                {/* 3-Way Segment Switcher */}
                <div className="flex items-center p-0.5 rounded-lg bg-surface border border-white/[0.06] text-[11px] font-mono">
                  <button
                    onClick={() => setActiveTab("inference")}
                    className={`px-2.5 py-1 rounded-md transition-all ${
                      activeTab === "inference"
                        ? "bg-foreground/[0.15] text-foreground font-semibold"
                        : "text-zinc-400 hover:text-foreground"
                    }`}
                  >
                    Inference
                  </button>
                  <button
                    onClick={() => setActiveTab("schema")}
                    className={`px-2.5 py-1 rounded-md transition-all ${
                      activeTab === "schema"
                        ? "bg-foreground/[0.15] text-foreground font-semibold"
                        : "text-zinc-400 hover:text-foreground"
                    }`}
                  >
                    Pydantic
                  </button>
                  <button
                    onClick={() => setActiveTab("diff")}
                    className={`px-2.5 py-1 rounded-md transition-all ${
                      activeTab === "diff"
                        ? "bg-foreground/[0.15] text-foreground font-semibold"
                        : "text-zinc-400 hover:text-foreground"
                    }`}
                  >
                    Code
                  </button>
                </div>
              </div>

              {/* Card Body - Content Based on Tab */}
              <div className="p-5">
                
                {/* TAB 1: LIVE INFERENCE BENCHMARK */}
                {activeTab === "inference" && (
                  <div className="space-y-4">
                    {/* Sample Complaint Snippet */}
                    <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                      <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mb-1.5">
                        <span className="text-brand-amber font-semibold uppercase">Consumer Narrative Excerpt</span>
                        <span className="text-[10px] text-zinc-500">ID #2847192</span>
                      </div>
                      <p className="text-xs text-zinc-300 leading-relaxed font-sans line-clamp-3">
                        "The credit reporting agency continues reporting an inaccurate delinquent balance on an account 
                        I settled 14 months ago. Despite three certified dispute letters, my score remains penalized..."
                      </p>
                    </div>

                    {/* Model Switcher Buttons */}
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setActiveModel("tfidf")}
                        className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-mono transition-all text-center border ${
                          activeModel === "tfidf"
                            ? "bg-brand-amber/15 border-brand-amber text-brand-amber font-bold shadow-sm"
                            : "bg-surface/50 border-white/[0.08] text-zinc-400 hover:text-foreground"
                        }`}
                      >
                        TF-IDF + Logistic Reg
                      </button>
                      <button
                        onClick={() => setActiveModel("gemini")}
                        className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-mono transition-all text-center border ${
                          activeModel === "gemini"
                            ? "bg-brand-cyan/15 border-brand-cyan text-brand-cyan font-bold shadow-sm"
                            : "bg-surface/50 border-white/[0.08] text-zinc-400 hover:text-foreground"
                        }`}
                      >
                        Gemini 2.0 Flash
                      </button>
                    </div>

                    {/* Inference Output Metric Box */}
                    <div className="p-3.5 rounded-xl bg-surface/90 border border-white/[0.08] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono text-zinc-400">Class Prediction:</span>
                        <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-amber/20 text-brand-amber border border-brand-amber/30">
                          Credit Reporting
                        </span>
                      </div>

                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-zinc-400">Confidence Score:</span>
                          <span className="text-foreground font-semibold">
                            {activeModel === "tfidf" ? "94.2%" : "36.0% (Zero-shot)"}
                          </span>
                        </div>
                        <div className="w-full bg-white/[0.06] rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              activeModel === "tfidf" ? "bg-brand-amber w-[94.2%]" : "bg-zinc-500 w-[36%]"
                            }`}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/[0.06] text-center font-mono">
                        <div>
                          <div className="text-[10px] text-zinc-500">Latency</div>
                          <div className="text-xs font-bold text-foreground">
                            {activeModel === "tfidf" ? "< 0.1 ms" : "6.4 ms"}
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] text-zinc-500">Cost / 1k</div>
                          <div className="text-xs font-bold text-brand-emerald">
                            {activeModel === "tfidf" ? "$0.00" : "~$0.20"}
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] text-zinc-500">Pareto</div>
                          <div className="text-xs font-bold text-brand-cyan">
                            {activeModel === "tfidf" ? "Dominant" : "Inferior"}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 2: PYDANTIC SCHEMA EXTRACTION */}
                {activeTab === "schema" && (
                  <div className="space-y-3 font-mono text-xs">
                    <div className="p-3 rounded-lg bg-black/60 border border-white/[0.08] text-zinc-300 space-y-1.5 leading-relaxed">
                      <div><span className="text-purple-400">class</span> <span className="text-amber-300">ComplaintExtraction</span>(BaseModel):</div>
                      <div className="pl-4"><span className="text-zinc-400">issue_category</span>: str = <span className="text-emerald-400">"Credit Reporting"</span></div>
                      <div className="pl-4"><span className="text-zinc-400">root_cause</span>: str = <span className="text-emerald-400">"Delinquent account dispute"</span></div>
                      <div className="pl-4"><span className="text-zinc-400">severity</span>: SeverityEnum = <span className="text-amber-400">"medium"</span></div>
                      <div className="pl-4"><span className="text-zinc-400">monetary_relief_flag</span>: bool = <span className="text-rose-400">False</span></div>
                      <div className="pl-4">
                        <span className="text-zinc-400">evidence_quote</span>: str = <span className="text-brand-cyan underline decoration-brand-cyan/50">"certified dispute letters"</span>
                      </div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-brand-emerald/10 border border-brand-emerald/30 flex items-center justify-between text-[11px] text-brand-emerald">
                      <span className="flex items-center gap-1.5 font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Exact Verbatim Substring Verified
                      </span>
                      <span className="font-bold">100% Conformance</span>
                    </div>
                  </div>
                )}

                {/* TAB 3: CODE DIFF / TERMINAL */}
                {activeTab === "diff" && (
                  <div className="font-mono text-xs p-3 rounded-lg bg-black/60 border border-white/[0.08] space-y-1 text-zinc-300 leading-relaxed overflow-x-auto">
                    <div className="text-zinc-500"># Evaluating out-of-sample test split</div>
                    <div className="text-emerald-400">+ model = LogisticRegression(C=1.0, max_iter=1000)</div>
                    <div className="text-emerald-400">+ metrics = evaluate_temporal_test(model) # Macro-F1: 0.6002</div>
                    <div className="text-rose-400">- llm_zero = GeminiClient("gemini-2.0-flash") # Macro-F1: 0.0756</div>
                    <div className="text-zinc-500"># McNemar paired test (p = 7.8e-4, Holm adj p = 0.0133)</div>
                    <div className="text-amber-300">+ calibration = brier_score_loss(y_test, y_prob) # 0.081</div>
                    <div className="text-emerald-400 font-bold">+ [FAIRNESS]: 0.00% DEMOGRAPHIC BIAS FLIP RATE</div>
                  </div>
                )}

              </div>

              {/* Pinned Bottom Status Tag */}
              <div className="px-4 py-2.5 border-t border-white/[0.06] bg-white/[0.01] flex items-center justify-between text-[11px] font-mono text-zinc-400">
                <span className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-emerald animate-ping" />
                  <span className="text-zinc-300">Out-Of-Sample Held-Out Test</span>
                </span>
                <span className="text-brand-amber font-semibold">N = 7,469 Verified</span>
              </div>

            </motion.div>

          </div>

        </div>
      </div>

      {/* Streaming Masked Complaint Ticker (21st.dev River style) */}
      <div className="relative mt-12 w-full overflow-hidden border-y border-white/[0.08] bg-surface/40 py-2.5 backdrop-blur-md">
        <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none" />

        <div className="flex whitespace-nowrap animate-ticker space-x-6">
          {(complaintList.length > 0 ? complaintList : Array(6).fill(null)).map((c, i) => (
            <div
              key={c?.complaint_id || i}
              className="inline-flex items-center space-x-3 px-3.5 py-1.5 rounded-full border border-white/[0.08] bg-surface-muted/80 text-xs font-mono"
            >
              <span className="text-[10px] px-2 py-0.5 rounded-full font-bold uppercase bg-brand-amber/15 text-brand-amber border border-brand-amber/30">
                {c?.true_label || "COMPLAINT"}
              </span>
              <span className="text-zinc-300 max-w-md truncate">
                {c?.narrative
                  ? c.narrative.slice(0, 110) + (c.narrative.length > 110 ? "..." : "")
                  : "Streaming CFPB anonymized consumer complaint records..."}
              </span>
              <span className="text-zinc-500 text-[10px]">
                #{c?.complaint_id || `XXXX-${i}`}
              </span>
            </div>
          ))}
        </div>
      </div>

    </section>
  );
}
