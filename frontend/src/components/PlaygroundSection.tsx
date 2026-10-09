"use client";

import { useEffect, useState } from "react";
import { ExamplesData, ComplaintExample } from "@/lib/types";
import { Play, Shuffle, CheckCircle2, ShieldCheck, Sparkles, Send, Activity } from "lucide-react";

interface PlaygroundSectionProps {
  examples: ExamplesData | null;
}

export default function PlaygroundSection({ examples }: PlaygroundSectionProps) {
  const exampleList: ComplaintExample[] = examples?.examples || [];
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [customText, setCustomText] = useState<string>("");
  const [isLiveOnline, setIsLiveOnline] = useState<boolean>(false);
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveResult, setLiveResult] = useState<any>(null);

  // Probe local FastAPI
  useEffect(() => {
    async function checkAPI() {
      try {
        const res = await fetch("http://localhost:8000/health", { method: "GET", mode: "cors" });
        if (res.ok) setIsLiveOnline(true);
      } catch {
        setIsLiveOnline(false);
      }
    }
    checkAPI();
  }, []);

  const currentExample = exampleList[selectedIndex] || exampleList[0];

  const handleRandom = () => {
    if (exampleList.length > 1) {
      const nextIdx = Math.floor(Math.random() * exampleList.length);
      setSelectedIndex(nextIdx);
      setLiveResult(null);
    }
  };

  const handleLivePredict = async () => {
    const textToSubmit = customText.trim() || currentExample?.narrative;
    if (!textToSubmit) return;

    setLiveLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ narrative: textToSubmit }),
      });
      if (res.ok) {
        const data = await res.json();
        setLiveResult(data);
      }
    } catch (err) {
      console.warn("Live API unavailable, continuing in replay mode:", err);
    } finally {
      setLiveLoading(false);
    }
  };

  if (!exampleList.length) {
    return (
      <section id="playground" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-foreground">06. Interactive Playground</h2>
        <p className="text-zinc-500 mt-2 font-mono text-xs">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  const activeNarrative = liveResult?.narrative || currentExample?.narrative || "";
  const activeTrueLabel = currentExample?.true_label || "Gold Label";
  const activePredictions = liveResult?.model_predictions || currentExample?.model_predictions || {};
  const activeExtraction = liveResult?.llm_extraction || currentExample?.llm_extraction;
  const evidenceQuote = activeExtraction?.evidence_quote || "";

  const renderHighlightedNarrative = () => {
    if (!evidenceQuote || !activeNarrative.includes(evidenceQuote)) {
      return <span>&ldquo;{activeNarrative}&rdquo;</span>;
    }
    const parts = activeNarrative.split(evidenceQuote);
    return (
      <span>
        &ldquo;{parts[0]}
        <mark className="bg-brand-amber/20 text-brand-amber font-semibold px-1 py-0.5 rounded border border-brand-amber/30">
          {evidenceQuote}
        </mark>
        {parts.slice(1).join(evidenceQuote)}&rdquo;
      </span>
    );
  };

  return (
    <section id="playground" className="py-24 bg-background relative overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute top-1/3 left-1/3 w-[600px] h-[600px] rounded-full bg-brand-amber/[0.04] blur-[140px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">06</span>
          <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between">
            <div>
              <div className="pill-badge bg-brand-amber/10 border-brand-amber/25 text-brand-amber mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-amber" />
                <span className="font-mono text-[11px] tracking-wider">EVIDENCE HIGHLIGHTING</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
                Interactive <span className="font-serif-italic text-gradient-amber">Playground</span>
              </h2>
            </div>

            <div className="mt-4 md:mt-0">
              <span
                className={`pill-badge text-[11px] font-mono ${
                  isLiveOnline
                    ? "bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald"
                    : "bg-white/[0.04] border-white/[0.08] text-zinc-400"
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>{isLiveOnline ? "LIVE ENGINE (:8000)" : "REPLAYING GOLD SET"}</span>
              </span>
            </div>
          </div>
        </div>

        {/* Example Selector Toolbar */}
        <div className="bento-card p-4 mb-8 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center space-x-3 flex-1 min-w-[280px]">
            <span className="text-zinc-500 font-semibold shrink-0">SELECT COMPLAINT:</span>
            <select
              value={selectedIndex}
              onChange={(e) => { setSelectedIndex(Number(e.target.value)); setLiveResult(null); setCustomText(""); }}
              className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-zinc-200 text-xs w-full focus:outline-none focus:border-brand-amber/50 transition-colors"
            >
              {exampleList.map((ex, idx) => (
                <option key={ex.complaint_id} value={idx}>
                  #{ex.complaint_id} • {ex.true_label} ({ex.word_count} words)
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleRandom}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] hover:border-white/[0.15] text-zinc-300 transition-all"
            >
              <Shuffle className="w-3.5 h-3.5 text-brand-amber" />
              <span>Random</span>
            </button>

            {isLiveOnline && (
              <button
                onClick={handleLivePredict}
                disabled={liveLoading}
                className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-brand-emerald text-black font-semibold hover:bg-emerald-400 transition-all disabled:opacity-50"
              >
                <Send className="w-3.5 h-3.5" />
                <span>{liveLoading ? "Running..." : "Live Inference"}</span>
              </button>
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Complaint Text */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bento-card p-6">
              <div className="flex items-center justify-between text-xs text-zinc-400 mb-3 pb-2 border-b border-white/[0.06]">
                <span className="flex items-center space-x-2">
                  <span className="font-semibold text-foreground">NARRATIVE</span>
                  <span className="text-zinc-600 font-mono">#{currentExample?.complaint_id}</span>
                </span>
                <span className="pill-badge bg-brand-cyan/10 border-brand-cyan/25 text-brand-cyan text-[10px]">
                  TRUE: {activeTrueLabel}
                </span>
              </div>

              <div className="text-sm text-zinc-300 leading-relaxed max-h-80 overflow-y-auto pr-2">
                {renderHighlightedNarrative()}
              </div>

              {evidenceQuote && (
                <div className="mt-4 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-start space-x-2 text-xs">
                  <Sparkles className="w-4 h-4 text-brand-amber shrink-0 mt-0.5" />
                  <div>
                    <span className="text-zinc-500 font-mono">EXTRACTED EVIDENCE:</span>
                    <div className="text-brand-amber font-medium mt-0.5">
                      &ldquo;{evidenceQuote}&rdquo;
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Structured LLM Extraction Card */}
            {activeExtraction && (
              <div className="bento-card p-6 text-xs">
                <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/[0.06]">
                  <span className="font-semibold text-brand-amber flex items-center space-x-1.5">
                    <ShieldCheck className="w-4 h-4" />
                    <span>PYDANTIC EXTRACTION (TASK B)</span>
                  </span>
                  <span className="text-[10px] text-zinc-600 font-mono">JSON-SCHEMA</span>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-zinc-500 font-mono">ISSUE</span>
                    <div className="text-foreground font-medium mt-0.5">{activeExtraction.issue_category}</div>
                  </div>
                  <div>
                    <span className="text-zinc-500 font-mono">RESOLUTION</span>
                    <div className="text-foreground font-medium mt-0.5 truncate">{activeExtraction.requested_resolution}</div>
                  </div>
                  <div>
                    <span className="text-zinc-500 font-mono">ROOT CAUSE</span>
                    <div className="text-zinc-300 mt-0.5 leading-snug">{activeExtraction.root_cause}</div>
                  </div>
                  <div>
                    <span className="text-zinc-500 font-mono">URGENCY • SENTIMENT</span>
                    <div className="mt-0.5 flex items-center space-x-2">
                      <span className="pill-badge bg-brand-cyan/10 border-brand-cyan/25 text-brand-cyan text-[10px] uppercase">{activeExtraction.urgency}</span>
                      <span className="pill-badge bg-white/[0.04] border-white/[0.08] text-zinc-300 text-[10px] uppercase">{activeExtraction.sentiment}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: Model Predictions */}
          <div className="lg:col-span-5 bento-card p-6 text-xs">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-4">
              <span className="font-semibold text-brand-cyan">ALL MODEL PREDICTIONS</span>
              <span className="text-[11px] text-zinc-500 font-mono">CONFIDENCE</span>
            </div>

            <div className="space-y-3">
              {Object.entries(activePredictions).map(([modelName, predData]: [string, any]) => {
                const isCorrect = predData.prediction === activeTrueLabel;
                const confPercent = ((predData.confidence ?? 0.85) * 100).toFixed(1);

                return (
                  <div key={modelName} className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-medium text-foreground truncate max-w-[180px]">{modelName}</span>
                      <span className={`pill-badge text-[10px] font-semibold ${
                        isCorrect
                          ? "bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald"
                          : "bg-brand-ruby/10 border-brand-ruby/25 text-brand-ruby"
                      }`}>
                        {isCorrect ? "CORRECT" : "MISS"}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-zinc-400 mt-1">
                      <span>PRED: <strong className="text-zinc-200">{predData.prediction}</strong></span>
                      <span className="num-mono text-brand-cyan">{confPercent}%</span>
                    </div>

                    <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden mt-2">
                      <div
                        className={`h-full rounded-full ${isCorrect ? "bg-brand-emerald" : "bg-brand-ruby"}`}
                        style={{ width: `${confPercent}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-6 pt-4 border-t border-white/[0.06] text-[11px] text-zinc-600">
              * Replaying out-of-sample evaluated test responses.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
