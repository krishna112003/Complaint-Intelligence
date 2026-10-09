"use client";

import { useState } from "react";
import { BiasData } from "@/lib/types";
import { Scale, AlertTriangle, Check, RefreshCw } from "lucide-react";

interface BiasLabSectionProps {
  bias: BiasData | null;
}

export default function BiasLabSection({ bias }: BiasLabSectionProps) {
  const [selectedProbe, setSelectedProbe] = useState<string>("name_gender");
  const [exampleIndex, setExampleIndex] = useState<number>(0);

  const probes = bias?.probes || {};
  const probeKeys = Object.keys(probes);
  const examples = bias?.examples || [];
  const limitations = bias?.limitations || [];

  if (!probeKeys.length) {
    return (
      <section id="bias" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-foreground">04. Bias &amp; Robustness Lab</h2>
        <p className="text-zinc-500 mt-2 font-mono text-xs">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  // Filter examples matching the selected probe
  const filteredExamples = examples.filter((e) => e.probe === selectedProbe);
  const currentExample = filteredExamples[exampleIndex % (filteredExamples.length || 1)] || examples[0];

  return (
    <section id="bias" className="py-24 bg-background relative overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full bg-brand-amber/[0.04] blur-[120px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">04</span>
          <div className="relative z-10">
            <div className="pill-badge bg-brand-amber/10 border-brand-amber/25 text-brand-amber mb-3">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-amber" />
              <span className="font-mono text-[11px] tracking-wider">COUNTERFACTUAL FAIRNESS</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              Bias Lab: Counterfactual <span className="font-serif-italic text-gradient-amber">Sensitivity</span>
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-lg">
              Demographic perturbations, flip rate analysis, and noise robustness probes.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Aggregate Flip Rate Bars with CIs */}
          <div className="lg:col-span-5 bento-card p-6">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-4">
              <span className="text-xs font-semibold text-brand-amber">AGGREGATE FLIP RATES</span>
              <span className="text-[11px] text-zinc-500 font-mono">95% CI BARS</span>
            </div>

            <div className="space-y-3">
              {probeKeys.map((key) => {
                const p = probes[key];
                const isSelected = selectedProbe === key;
                const rate = p.flip_rate ?? 0.0;
                const percent = (rate * 100).toFixed(1);

                return (
                  <div
                    key={key}
                    onClick={() => {
                      setSelectedProbe(key);
                      setExampleIndex(0);
                    }}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "glass-card !border-brand-amber/30 text-foreground"
                        : "bg-white/[0.02] border-white/[0.06] hover:border-white/[0.12] text-zinc-400"
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-2">
                      <span className="font-medium text-foreground truncate max-w-[200px]">
                        {p.name.split("(")[0]}
                      </span>
                      <span className="num-mono font-semibold text-brand-amber text-[11px]">
                        {percent}% FLIP
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="relative w-full h-2 bg-white/[0.06] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-brand-amber to-amber-400 rounded-full transition-all duration-500"
                        style={{ width: `${Math.max(2, rate * 100)}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-zinc-500 mt-1.5 font-mono">
                      <span>{p.flips} / {p.total} flipped</span>
                      <span>CI: [{((p.ci_low ?? 0) * 100).toFixed(1)}%, {((p.ci_high ?? 0) * 100).toFixed(1)}%]</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Limitations */}
            <div className="mt-6 pt-4 border-t border-white/[0.06]">
              <div className="text-xs text-zinc-400 font-semibold mb-2 flex items-center space-x-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-brand-amber" />
                <span>LIMITATIONS</span>
              </div>
              <ul className="space-y-1.5 text-[11px] text-zinc-500">
                {limitations.map((lim, i) => (
                  <li key={i} className="flex items-start space-x-2">
                    <span className="text-brand-amber shrink-0">•</span>
                    <span>{lim}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Right: Interactive Before/After Counterfactual Probe Inspector */}
          <div className="lg:col-span-7 bento-card p-6">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-4">
              <div>
                <span className="text-xs font-semibold text-brand-cyan">
                  PERTURBATION INSPECTOR
                </span>
                <div className="text-xs text-zinc-500 mt-0.5">
                  Probing: <span className="text-foreground">{probes[selectedProbe]?.name}</span>
                </div>
              </div>

              {filteredExamples.length > 1 && (
                <button
                  onClick={() => setExampleIndex((prev) => prev + 1)}
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] hover:border-white/[0.15] text-xs text-zinc-300 transition-all"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Next</span>
                </button>
              )}
            </div>

            {currentExample ? (
              <div className="space-y-4">
                {/* Original Text Box */}
                <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                  <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
                    <span className="font-semibold text-foreground">ORIGINAL</span>
                    <span className="text-zinc-500">
                      PRED: <strong className="text-brand-cyan">{currentExample.original_pred}</strong>
                    </span>
                  </div>
                  <p className="text-xs text-zinc-300 leading-relaxed">
                    &ldquo;{currentExample.original_text}&rdquo;
                  </p>
                </div>

                {/* Perturbed Text Box */}
                <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                  <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
                    <span className="font-semibold text-brand-amber">PERTURBED</span>
                    <span className="text-zinc-500">
                      PRED: <strong className={currentExample.flipped ? "text-brand-ruby" : "text-brand-emerald"}>{currentExample.perturbed_pred}</strong>
                    </span>
                  </div>
                  <p className="text-xs text-zinc-300 leading-relaxed">
                    &ldquo;{currentExample.perturbed_text}&rdquo;
                  </p>
                </div>

                {/* Flip Result Badge */}
                <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-between text-xs">
                  <span className="text-zinc-500">OUTCOME:</span>
                  <div>
                    {currentExample.flipped ? (
                      <span className="pill-badge bg-brand-ruby/10 border-brand-ruby/25 text-brand-ruby text-[10px] font-semibold">
                        <AlertTriangle className="w-3 h-3" />
                        <span>FLIPPED (SENSITIVE)</span>
                      </span>
                    ) : (
                      <span className="pill-badge bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald text-[10px] font-semibold">
                        <Check className="w-3 h-3" />
                        <span>UNCHANGED (INVARIANT)</span>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-zinc-500 text-xs">
                No active counterfactual pairs recorded for this probe.
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
