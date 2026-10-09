"use client";

import { useState } from "react";
import { PairwiseData, CalibrationData, PairwiseTest } from "@/lib/types";
import { CheckCircle2, XCircle, BarChart2, Info } from "lucide-react";

interface StatsSectionProps {
  pairwise: PairwiseData | null;
  calibration: CalibrationData | null;
}

export default function StatsSection({ pairwise, calibration }: StatsSectionProps) {
  const [hoveredTest, setHoveredTest] = useState<PairwiseTest | null>(null);
  const [selectedModelCalib, setSelectedModelCalib] = useState<string>("TF-IDF + Logistic Reg");

  const pairs = pairwise?.pairs || [];
  const calibModels = calibration?.models || {};
  const calibModelNames = Object.keys(calibModels);

  if (!pairs.length) {
    return (
      <section id="stats" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-foreground">03. Is The Difference Real?</h2>
        <p className="text-zinc-500 mt-2 font-mono text-xs">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  // Calibration SVG dimensions
  const calibW = 480;
  const calibH = 340;
  const padding = { top: 30, right: 30, bottom: 50, left: 60 };
  const innerW = calibW - padding.left - padding.right;
  const innerH = calibH - padding.top - padding.bottom;

  const currentCalib = calibModels[selectedModelCalib] || calibModels[calibModelNames[0]];

  return (
    <section id="stats" className="py-24 bg-background relative overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute top-1/3 left-1/4 w-[500px] h-[500px] rounded-full bg-brand-emerald/[0.03] blur-[120px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">03</span>
          <div className="relative z-10">
            <div className="pill-badge bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald mb-3">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-emerald" />
              <span className="font-mono text-[11px] tracking-wider">STATISTICAL SIGNIFICANCE</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              Is the Difference <span className="font-serif-italic text-gradient-emerald">Real?</span>
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-lg">
              McNemar tests with Holm-Bonferroni correction. Calibration reliability diagrams.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Pairwise McNemar Heatmap / Matrix */}
          <div className="lg:col-span-7 bento-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Pairwise McNemar Matrix
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Hover to inspect effect size, CI, and adjusted p-values.
                </p>
              </div>
              <span className="pill-badge bg-white/[0.04] border-white/[0.08] text-zinc-400 text-[10px] font-mono">
                α = 0.05
              </span>
            </div>

            {/* Test Cards / Matrix */}
            <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
              {pairs.map((p, idx) => {
                const isSignificant = p.significant_after_holm;
                const isHovered =
                  hoveredTest?.model_a === p.model_a && hoveredTest?.model_b === p.model_b;

                return (
                  <div
                    key={idx}
                    onMouseEnter={() => setHoveredTest(p)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer text-xs ${
                      isHovered
                        ? "glass-card !border-brand-amber/30 text-foreground"
                        : "bg-white/[0.02] border-white/[0.06] hover:border-white/[0.12] text-zinc-300"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-medium text-foreground truncate max-w-[280px]">
                        {p.model_a} <span className="text-zinc-600 mx-1">vs</span> {p.model_b}
                      </div>

                      <div className="flex items-center space-x-3 shrink-0">
                        <span className="num-mono text-zinc-500">
                          ΔF1:{" "}
                          <span className={p.delta_f1 && p.delta_f1 > 0 ? "text-brand-emerald" : "text-zinc-300"}>
                            {p.delta_f1 ? (p.delta_f1 > 0 ? `+${p.delta_f1.toFixed(4)}` : p.delta_f1.toFixed(4)) : "0.0000"}
                          </span>
                        </span>

                        <span
                          className={`pill-badge text-[10px] font-semibold ${
                            isSignificant
                              ? "bg-brand-emerald/10 border-brand-emerald/25 text-brand-emerald"
                              : "bg-white/[0.04] border-white/[0.08] text-zinc-500"
                          }`}
                        >
                          {isSignificant ? (
                            <>
                              <CheckCircle2 className="w-3 h-3" />
                              <span>SIG</span>
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3 h-3" />
                              <span>NOT SIG</span>
                            </>
                          )}
                        </span>
                      </div>
                    </div>

                    {isHovered && (
                      <div className="mt-3 pt-3 border-t border-white/[0.06] grid grid-cols-3 gap-2 text-[11px] text-zinc-400">
                        <div>
                          <span className="text-zinc-600">95% CI:</span>
                          <div className="text-zinc-200 num-mono">
                            [{p.delta_f1_ci_lower?.toFixed(4)}, {p.delta_f1_ci_upper?.toFixed(4)}]
                          </div>
                        </div>
                        <div>
                          <span className="text-zinc-600">STAT:</span>
                          <div className="text-zinc-200 num-mono">{p.mcnemar_stat?.toFixed(2)}</div>
                        </div>
                        <div>
                          <span className="text-zinc-600">HOLM P:</span>
                          <div className="text-brand-cyan num-mono">
                            {p.adjusted_p_value && p.adjusted_p_value < 0.0001
                              ? "< 0.0001"
                              : p.adjusted_p_value?.toFixed(4)}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Calibration Reliability Diagram */}
          <div className="lg:col-span-5 bento-card p-6">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Expected Calibration Error
                </h3>
                <div className="text-xs text-zinc-500 mt-0.5">Reliability Diagram (10 Bins)</div>
              </div>

              {currentCalib && (
                <div className="text-right">
                  <span className="text-[10px] text-zinc-500 font-mono">ECE SCORE</span>
                  <div className="text-lg font-bold text-gradient-amber num-mono">
                    {currentCalib.ece.toFixed(4)}
                  </div>
                </div>
              )}
            </div>

            {/* Model Selector for Calibration */}
            <div className="mt-4 flex flex-wrap gap-1.5">
              {calibModelNames.map((name) => (
                <button
                  key={name}
                  onClick={() => setSelectedModelCalib(name)}
                  className={`px-2.5 py-1 rounded-full text-xs transition-all ${
                    selectedModelCalib === name
                      ? "bg-brand-emerald text-black font-semibold"
                      : "bg-white/[0.04] text-zinc-400 hover:text-foreground hover:bg-white/[0.06]"
                  }`}
                >
                  {name}
                </button>
              ))}
            </div>

            {/* Custom SVG Reliability Curve */}
            {currentCalib ? (
              <div className="mt-4 flex justify-center">
                <svg viewBox={`0 0 ${calibW} ${calibH}`} className="w-full h-auto select-none">
                  {/* Grid Lines */}
                  {[0.2, 0.4, 0.6, 0.8, 1.0].map((val) => {
                    const y = padding.top + innerH - val * innerH;
                    const x = padding.left + val * innerW;
                    return (
                      <g key={val}>
                        <line
                          x1={padding.left} y1={y} x2={calibW - padding.right} y2={y}
                          stroke="rgba(255, 255, 255, 0.05)" strokeDasharray="3 3"
                        />
                        <text x={padding.left - 8} y={y + 4} fill="#71717A" fontSize="10" textAnchor="end">
                          {val.toFixed(1)}
                        </text>
                        <text x={x} y={calibH - padding.bottom + 16} fill="#71717A" fontSize="10" textAnchor="middle">
                          {val.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}

                  {/* 45-degree Perfect Calibration Line */}
                  <line
                    x1={padding.left} y1={padding.top + innerH}
                    x2={calibW - padding.right} y2={padding.top}
                    stroke="#71717A" strokeWidth="1.5" strokeDasharray="4 4"
                  />

                  {/* Binned Empirical Dots */}
                  {currentCalib.bins.map((b, idx) => {
                    const cx = padding.left + b.confidence * innerW;
                    const cy = padding.top + innerH - b.accuracy * innerH;
                    return (
                      <g key={idx}>
                        <circle cx={cx} cy={cy} r="6" fill="#10B981" stroke="rgba(9,10,15,0.8)" strokeWidth="2" />
                      </g>
                    );
                  })}

                  {/* Axis Titles */}
                  <text x={calibW / 2} y={calibH - 12} fill="#A1A1AA" fontSize="11" textAnchor="middle" fontWeight="500">
                    Mean Predicted Confidence
                  </text>
                  <text x={-calibH / 2} y={16} transform="rotate(-90)" fill="#A1A1AA" fontSize="11" textAnchor="middle" fontWeight="500">
                    Empirical Accuracy
                  </text>
                </svg>
              </div>
            ) : (
              <div className="py-16 text-center text-zinc-500 text-xs">
                Calibration bin data NOT RUN.
              </div>
            )}

            <div className="mt-4 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-[11px] text-zinc-400 flex items-start space-x-2">
              <Info className="w-3.5 h-3.5 text-brand-emerald shrink-0 mt-0.5" />
              <span>
                Diagonal dashed line represents perfect calibration where predicted probability
                matches true accuracy rate. Lower ECE signifies greater calibration fidelity.
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
