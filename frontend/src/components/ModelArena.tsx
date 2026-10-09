"use client";

import { useState } from "react";
import { ModelsData, ModelMetric } from "@/lib/types";
import { TrendingUp, Clock, DollarSign, Award, HelpCircle, ArrowUpRight } from "lucide-react";

interface ModelArenaProps {
  data: ModelsData | null;
}

export default function ModelArena({ data }: ModelArenaProps) {
  const [viewMode, setViewMode] = useState<"cost" | "latency">("cost");
  const [hoveredModel, setHoveredModel] = useState<ModelMetric | null>(null);

  const models: ModelMetric[] = data?.models || [];
  const takeaway = data?.takeaway || "Classical algorithms dominate quality-cost efficiency.";

  if (!models.length) {
    return (
      <section id="arena" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-foreground">02. Model Arena &amp; Pareto Frontier</h2>
        <p className="text-zinc-500 mt-2 font-mono text-xs">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  // Find min/max values for scaling
  const maxF1 = 0.70;
  const minF1 = 0.0;

  // SVG dimensions
  const width = 800;
  const height = 440;
  const padding = { top: 40, right: 50, bottom: 60, left: 75 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // X scale mapping
  const getX = (m: ModelMetric) => {
    if (viewMode === "cost") {
      const cost = m.hypothetical_cost_per_1k ?? 0.0;
      return padding.left + (cost / 0.5) * chartW * 0.8 + 20;
    } else {
      const lat = m.latency_p95_ms ?? 0.0;
      return padding.left + Math.min(1, lat / 2.0) * chartW * 0.8 + 20;
    }
  };

  // Y scale mapping (Macro-F1)
  const getY = (m: ModelMetric) => {
    const f1 = m.macro_f1 ?? 0.0;
    return padding.top + chartH - ((f1 - minF1) / (maxF1 - minF1)) * chartH;
  };

  // Sort pareto models for drawing the frontier line
  const paretoModels = [...models]
    .filter((m) => m.is_pareto)
    .sort((a, b) => (getX(a) - getX(b)));

  return (
    <section id="arena" className="py-24 bg-background relative overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute top-0 right-1/4 w-[600px] h-[600px] rounded-full bg-brand-amber/[0.04] blur-[120px]" />
      <div className="pointer-events-none absolute bottom-20 left-1/3 w-[400px] h-[400px] rounded-full bg-brand-cyan/[0.03] blur-[100px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">02</span>
          <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between">
            <div>
              <div className="pill-badge bg-brand-amber/10 border-brand-amber/25 text-brand-amber mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-amber" />
                <span className="font-mono text-[11px] tracking-wider">EMPIRICAL BENCHMARK</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
                Model Arena & <span className="font-serif-italic text-gradient-amber">Pareto Frontier</span>
              </h2>
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center space-x-1 mt-4 md:mt-0 p-1 rounded-full bg-white/[0.04] border border-white/[0.08] backdrop-blur-sm">
              <button
                onClick={() => setViewMode("cost")}
                className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
                  viewMode === "cost"
                    ? "bg-brand-amber text-black font-semibold shadow-sm"
                    : "text-zinc-400 hover:text-foreground"
                }`}
              >
                <DollarSign className="w-3.5 h-3.5" />
                <span>F1 vs. Cost</span>
              </button>

              <button
                onClick={() => setViewMode("latency")}
                className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
                  viewMode === "latency"
                    ? "bg-brand-cyan text-black font-semibold shadow-sm"
                    : "text-zinc-400 hover:text-foreground"
                }`}
              >
                <Clock className="w-3.5 h-3.5" />
                <span>F1 vs. Latency</span>
              </button>
            </div>
          </div>
        </div>

        {/* Takeaway Banner */}
        <div className="bento-card p-4 mb-8 flex items-start space-x-3 text-sm text-zinc-300">
          <Award className="w-5 h-5 text-brand-amber shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-brand-amber mr-2">[DATA TAKEAWAY]</span>
            <span>{takeaway}</span>
            <div className="text-[11px] text-zinc-500 mt-1 font-mono">
              * Cost figures labeled: <em>{data?.reference_pricing_note || "Hypothetical cost, at reference pricing."}</em>
            </div>
          </div>
        </div>

        {/* Main Chart Card */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Custom SVG Scatter & Pareto Frontier Chart */}
          <div className="lg:col-span-8 bento-card p-4 sm:p-6">
            <div className="flex items-center justify-between text-xs font-mono text-zinc-500 mb-3">
              <span className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-brand-amber" />
                <span>⭐ Pareto Optimal Frontier</span>
              </span>
              <span>Error bars: 95% Bootstrap CI</span>
            </div>

            <div className="w-full overflow-x-auto">
              <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto min-w-[620px] select-none">
                {/* Grid Lines */}
                {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((tick) => {
                  const y = padding.top + chartH - ((tick - minF1) / (maxF1 - minF1)) * chartH;
                  return (
                    <g key={tick}>
                      <line
                        x1={padding.left}
                        y1={y}
                        x2={width - padding.right}
                        y2={y}
                        stroke="rgba(255, 255, 255, 0.05)"
                        strokeDasharray="4 4"
                      />
                      <text
                        x={padding.left - 12}
                        y={y + 4}
                        fill="#71717A"
                        fontSize="11"
                        fontFamily="ui-monospace"
                        textAnchor="end"
                      >
                        {tick.toFixed(2)}
                      </text>
                    </g>
                  );
                })}

                {/* Y-Axis Label */}
                <text
                  x={-height / 2}
                  y={22}
                  transform="rotate(-90)"
                  fill="#A1A1AA"
                  fontSize="12"
                  fontFamily="ui-monospace"
                  textAnchor="middle"
                  fontWeight="500"
                >
                  Macro-Averaged F1 Score (Out-of-sample)
                </text>

                {/* X-Axis Label */}
                <text
                  x={width / 2}
                  y={height - 15}
                  fill="#A1A1AA"
                  fontSize="12"
                  fontFamily="ui-monospace"
                  textAnchor="middle"
                  fontWeight="500"
                >
                  {viewMode === "cost"
                    ? "Inference Cost per 1k ($ USD, reference pricing)"
                    : "Inference Latency p95 (ms)"}
                </text>

                {/* Pareto Frontier Line */}
                {paretoModels.length > 0 && (
                  <path
                    d={`M ${padding.left} ${getY(paretoModels[0])} L ${getX(paretoModels[0])} ${getY(
                      paretoModels[0]
                    )}`}
                    fill="none"
                    stroke="#F59E0B"
                    strokeWidth="2.5"
                    strokeDasharray="5 5"
                    className="opacity-60"
                  />
                )}

                {/* Model Data Points and Error Bars */}
                {models.map((m) => {
                  const cx = getX(m);
                  const cy = getY(m);
                  const isHovered = hoveredModel?.id === m.id;
                  const isPareto = m.is_pareto;

                  // Bootstrap CI bounds
                  const yLow =
                    m.ci_low !== null
                      ? padding.top + chartH - ((m.ci_low - minF1) / (maxF1 - minF1)) * chartH
                      : cy;
                  const yHigh =
                    m.ci_high !== null
                      ? padding.top + chartH - ((m.ci_high - minF1) / (maxF1 - minF1)) * chartH
                      : cy;

                  return (
                    <g
                      key={m.id}
                      className="cursor-pointer transition-transform"
                      onMouseEnter={() => setHoveredModel(m)}
                      onMouseLeave={() => setHoveredModel(null)}
                    >
                      {/* Hover glow ring */}
                      {isHovered && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r="16"
                          fill="none"
                          stroke={isPareto ? "#F59E0B" : "#06B6D4"}
                          strokeWidth="1"
                          opacity="0.3"
                        />
                      )}

                      {/* CI Error Bar */}
                      <line
                        x1={cx}
                        y1={yLow}
                        x2={cx}
                        y2={yHigh}
                        stroke={isPareto ? "#F59E0B" : "#71717A"}
                        strokeWidth={isHovered ? "2.5" : "1.5"}
                        opacity={isHovered ? 1 : 0.5}
                      />
                      {/* CI Whiskers */}
                      <line
                        x1={cx - 4} y1={yLow} x2={cx + 4} y2={yLow}
                        stroke={isPareto ? "#F59E0B" : "#71717A"} strokeWidth="1.5"
                      />
                      <line
                        x1={cx - 4} y1={yHigh} x2={cx + 4} y2={yHigh}
                        stroke={isPareto ? "#F59E0B" : "#71717A"} strokeWidth="1.5"
                      />

                      {/* Main Scatter Dot */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isHovered ? 9 : isPareto ? 7.5 : 5.5}
                        fill={isPareto ? "#F59E0B" : isHovered ? "#06B6D4" : "#38BDF8"}
                        stroke="rgba(9, 10, 15, 0.8)"
                        strokeWidth="2"
                        className="transition-all duration-200"
                      />

                      {/* Label */}
                      <text
                        x={cx + 12}
                        y={cy + 4}
                        fill={isPareto ? "#FBBF24" : isHovered ? "#FFFFFF" : "#A1A1AA"}
                        fontSize={isHovered ? "12" : "11"}
                        fontFamily="ui-monospace"
                        fontWeight={isPareto || isHovered ? "700" : "400"}
                      >
                        {m.name} {isPareto ? "⭐" : ""}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Model Inspector / Details Card */}
          <div className="lg:col-span-4 bento-card p-6">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-4">
              <span className="text-xs font-semibold text-brand-amber">MODEL INSPECTOR</span>
              <span className="text-[11px] text-zinc-500">HOVER TO FOCUS</span>
            </div>

            {hoveredModel ? (
              <div className="mt-4 space-y-4 animate-fade-in-up" style={{ animationDuration: '0.3s' }}>
                <div>
                  <div className="text-xs text-zinc-500 font-mono">ARCHITECTURE</div>
                  <h3 className="text-xl font-bold text-foreground mt-0.5 flex items-center space-x-2">
                    <span>{hoveredModel.name}</span>
                    {hoveredModel.is_pareto && (
                      <span className="pill-badge bg-brand-amber/15 border-brand-amber/30 text-brand-amber text-[10px]">
                        PARETO ⭐
                      </span>
                    )}
                  </h3>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-3 border-t border-white/[0.06] text-xs">
                  <div>
                    <div className="text-zinc-500 font-mono">MACRO-F1</div>
                    <div className="text-lg font-bold text-gradient-emerald num-mono">
                      {hoveredModel.macro_f1?.toFixed(4) ?? "NOT RUN"}
                    </div>
                  </div>

                  <div>
                    <div className="text-zinc-500 font-mono">ACCURACY</div>
                    <div className="text-lg font-bold text-foreground num-mono">
                      {hoveredModel.accuracy ? `${(hoveredModel.accuracy * 100).toFixed(2)}%` : "NOT RUN"}
                    </div>
                  </div>

                  <div>
                    <div className="text-zinc-500 font-mono">95% BOOTSTRAP CI</div>
                    <div className="text-xs font-semibold text-zinc-300 num-mono mt-0.5">
                      [{hoveredModel.ci_low?.toFixed(3)}, {hoveredModel.ci_high?.toFixed(3)}]
                    </div>
                  </div>

                  <div>
                    <div className="text-zinc-500 font-mono">ECE</div>
                    <div className="text-xs font-semibold text-zinc-300 num-mono mt-0.5">
                      {hoveredModel.ece?.toFixed(4) ?? "NOT RUN"}
                    </div>
                  </div>

                  <div>
                    <div className="text-zinc-500 font-mono">P95 LATENCY</div>
                    <div className="text-xs font-semibold text-zinc-300 num-mono mt-0.5">
                      {hoveredModel.latency_p95_ms?.toFixed(2)} ms
                    </div>
                  </div>

                  <div>
                    <div className="text-zinc-500 font-mono">COST / 1K</div>
                    <div className="text-xs font-semibold text-zinc-300 num-mono mt-0.5">
                      ${hoveredModel.hypothetical_cost_per_1k?.toFixed(4)}
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-white/[0.06] text-[11px] text-zinc-500 font-mono">
                  Sample count: <span className="text-zinc-300">{hoveredModel.n_samples?.toLocaleString()} out-of-sample</span>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-zinc-500 text-xs">
                <HelpCircle className="w-8 h-8 mx-auto mb-3 opacity-30" />
                <p className="max-w-[200px] mx-auto leading-relaxed">
                  Hover over any model in the scatter plot to inspect confidence intervals and calibration.
                </p>
              </div>
            )}

            {/* Model Comparison Table */}
            <div className="mt-6 pt-4 border-t border-white/[0.06]">
              <div className="text-xs text-zinc-400 mb-2 font-semibold font-mono">ALL ARCHITECTURES</div>
              <div className="space-y-1 max-h-48 overflow-y-auto text-xs">
                {models.map((m) => (
                  <div
                    key={m.id}
                    onMouseEnter={() => setHoveredModel(m)}
                    className={`flex items-center justify-between p-2 rounded-lg cursor-pointer transition-all ${
                      hoveredModel?.id === m.id
                        ? "bg-white/[0.06] text-brand-amber font-semibold"
                        : "hover:bg-white/[0.03] text-zinc-400"
                    }`}
                  >
                    <span className="truncate max-w-[150px]">{m.name}</span>
                    <span className="num-mono text-zinc-500">
                      {m.macro_f1?.toFixed(4)} {m.is_pareto ? "⭐" : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
