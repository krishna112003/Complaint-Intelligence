"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { PipelineData, PipelineStage } from "@/lib/types";
import {
  Database,
  Filter,
  CheckCircle2,
  Calendar,
  Layers,
  BarChart3,
  Scale,
  Sparkles,
  ShieldCheck,
  ArrowRight,
} from "lucide-react";

interface PipelineSectionProps {
  pipeline: PipelineData | null;
}

const STAGE_ICONS: Record<string, React.ReactNode> = {
  bronze: <Database className="w-5 h-5 text-brand-cyan" />,
  silver: <Filter className="w-5 h-5 text-zinc-400" />,
  gold: <Layers className="w-5 h-5 text-brand-amber" />,
  splits: <Calendar className="w-5 h-5 text-brand-emerald" />,
  models: <Layers className="w-5 h-5 text-brand-violet" />,
  eval: <BarChart3 className="w-5 h-5 text-brand-cyan" />,
  bias: <Scale className="w-5 h-5 text-brand-amber" />,
  topics: <Sparkles className="w-5 h-5 text-emerald-400" />,
  gate: <ShieldCheck className="w-5 h-5 text-brand-ruby" />,
  registry: <Database className="w-5 h-5 text-brand-emerald" />,
};

export default function PipelineSection({ pipeline }: PipelineSectionProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pinTargetRef = useRef<HTMLDivElement>(null);
  const [activeStageIndex, setActiveStageIndex] = useState<number>(0);

  const stages: PipelineStage[] = pipeline?.stages || [];

  useEffect(() => {
    if (!stages.length) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    gsap.registerPlugin(ScrollTrigger);

    const ctx = gsap.context(() => {
      // Pin the pipeline visualization while scrolling through stages
      ScrollTrigger.create({
        trigger: containerRef.current,
        start: "top top+=70",
        end: `+=${stages.length * 300}`,
        pin: pinTargetRef.current,
        scrub: 0.5,
        onUpdate: (self) => {
          const index = Math.min(
            stages.length - 1,
            Math.floor(self.progress * stages.length)
          );
          setActiveStageIndex(index);
        },
      });
    }, containerRef);

    return () => ctx.revert();
  }, [stages]);

  if (!stages.length) {
    return (
      <section id="pipeline" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-2xl font-bold">01. Pipeline Architecture</h2>
        <p className="text-zinc-500 mt-2">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  const currentStage = stages[activeStageIndex] || stages[0];
  const progressPercent = ((activeStageIndex + 1) / stages.length) * 100;

  return (
    <section
      id="pipeline"
      ref={containerRef}
      className="relative min-h-[220vh] bg-background"
    >
      {/* Ambient section glow */}
      <div className="pointer-events-none absolute top-20 left-1/4 w-[500px] h-[500px] rounded-full bg-brand-cyan/[0.04] blur-[100px]" />
      <div className="pointer-events-none absolute bottom-40 right-1/4 w-[400px] h-[400px] rounded-full bg-brand-amber/[0.04] blur-[100px]" />

      <div
        ref={pinTargetRef}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 h-screen flex flex-col justify-center"
      >
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">01</span>
          <div className="relative z-10">
            <div className="pill-badge bg-brand-cyan/10 border-brand-cyan/25 text-brand-cyan mb-3">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-cyan" />
              <span className="font-mono text-[11px] tracking-wider">DATA MEDALLION & GOVERNANCE</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              The Pipeline <span className="font-serif-italic text-gradient-amber">Lifecycle</span>
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-lg">
              Scroll to step through each refinement stage. Temporal splitting prevents data leakage.
            </p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 mb-2">
            <span>STAGE {activeStageIndex + 1} OF {stages.length}</span>
            <span>{Math.round(progressPercent)}% COMPLETE</span>
          </div>
          <div className="h-1 bg-white/[0.06] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-cyan via-brand-amber to-brand-emerald rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Interactive / Scrubbed Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left: Stage Progress Rail */}
          <div className="lg:col-span-5 space-y-2">
            {stages.map((stage, idx) => {
              const isActive = idx === activeStageIndex;
              const isPassed = idx < activeStageIndex;
              return (
                <button
                  key={stage.id}
                  onClick={() => setActiveStageIndex(idx)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl text-left transition-all text-xs border group ${
                    isActive
                      ? "glass-card !border-brand-amber/30 text-foreground shadow-lg shadow-brand-amber/[0.06]"
                      : isPassed
                      ? "bg-white/[0.02] border-white/[0.06] text-zinc-400 hover:text-foreground hover:border-white/[0.1]"
                      : "bg-transparent border-white/[0.04] text-zinc-600 hover:text-zinc-400 hover:border-white/[0.08]"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <span
                      className={`w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-bold transition-all ${
                        isActive
                          ? "bg-brand-amber text-black"
                          : isPassed
                          ? "bg-brand-emerald/20 text-brand-emerald border border-brand-emerald/30"
                          : "bg-white/[0.05] text-zinc-500 border border-white/[0.08]"
                      }`}
                    >
                      {isPassed ? "✓" : idx + 1}
                    </span>
                    <span className="font-medium">{stage.name}</span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="num-mono text-[11px] text-zinc-500">
                      {stage.count.toLocaleString()}
                    </span>
                    {isActive && <ArrowRight className="w-3.5 h-3.5 text-brand-amber" />}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right: Detailed Stage Inspector Card */}
          <div className="lg:col-span-7">
            <div className="relative bento-card p-6 sm:p-8 shadow-2xl">
              <div className="absolute top-4 right-4">
                <span className="pill-badge bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald text-[10px] font-mono uppercase">
                  {currentStage.status}
                </span>
              </div>

              {/* Icon & ID */}
              <div className="flex items-center space-x-3 mb-5">
                <div className="p-3 rounded-xl bg-white/[0.04] border border-white/[0.08]">
                  {STAGE_ICONS[currentStage.id] || <Layers className="w-5 h-5 text-brand-amber" />}
                </div>
                <div>
                  <div className="text-xs font-mono text-brand-amber/70 tracking-wider">
                    {currentStage.id.toUpperCase()}
                  </div>
                  <h3 className="text-2xl font-bold text-foreground">
                    {currentStage.name}
                  </h3>
                </div>
              </div>

              {/* Description */}
              <p className="text-zinc-300 text-sm sm:text-base leading-relaxed mb-6">
                {currentStage.description}
              </p>

              {/* Data metrics grid */}
              <div className="grid grid-cols-2 gap-4 pt-5 border-t border-white/[0.06]">
                <div>
                  <div className="text-[11px] text-zinc-500 font-mono tracking-wider">RECORD COUNT</div>
                  <div className="text-2xl sm:text-3xl font-bold text-gradient-cyan num-mono mt-0.5">
                    {currentStage.count.toLocaleString()}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-zinc-500 font-mono tracking-wider">ENGINE</div>
                  <div className="text-sm font-medium text-zinc-200 mt-1 truncate" title={currentStage.tool}>
                    {currentStage.tool}
                  </div>
                </div>
              </div>

              {/* Pipeline Safety Callout */}
              <div className="mt-6 p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-start space-x-3 text-xs text-zinc-400">
                <CheckCircle2 className="w-4 h-4 text-brand-emerald shrink-0 mt-0.5" />
                <span>
                  Anti-leakage guarantee: Temporal splitting boundary strictly isolates past training
                  dates from out-of-sample evaluation. No random shuffling or lookahead leakage.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
