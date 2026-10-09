"use client";

import { useState } from "react";
import { VolumeData, VolumeSummaryItem } from "@/lib/types";
import { TrendingUp, TrendingDown, Calendar, BarChart3, Layers, Filter } from "lucide-react";

interface VolumeSectionProps {
  volume: VolumeData | null;
}

export default function VolumeSection({ volume }: VolumeSectionProps) {
  const summaryTable = volume?.summary_table || [];
  const [selectedProduct, setSelectedProduct] = useState<string>(
    summaryTable[0]?.product || "Credit Reporting"
  );
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!summaryTable.length || !volume) {
    return (
      <section id="volume" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-foreground">05. Complaint Volume Trends</h2>
        <p className="text-zinc-500 mt-2 font-mono text-xs">Data NOT RUN / not yet loaded.</p>
      </section>
    );
  }

  const activeSummary: VolumeSummaryItem =
    summaryTable.find((s) => s.product === selectedProduct) || summaryTable[0];
  const series = volume.by_product[selectedProduct] || [];

  // SVG Chart Dimensions
  const chartW = 760;
  const chartH = 280;
  const padding = { top: 30, right: 30, bottom: 45, left: 60 };
  const innerW = chartW - padding.left - padding.right;
  const innerH = chartH - padding.top - padding.bottom;

  const maxVol = Math.max(10, ...series.map((s) => Math.max(s.monthly_volume, s.rolling_3m_avg || 0))) * 1.15;

  const points = series.map((s, idx) => {
    const x = padding.left + (idx / Math.max(1, series.length - 1)) * innerW;
    const yVol = padding.top + innerH - (s.monthly_volume / maxVol) * innerH;
    const yAvg = padding.top + innerH - ((s.rolling_3m_avg || 0) / maxVol) * innerH;
    return { x, yVol, yAvg, month: s.month, volume: s.monthly_volume, share: s.share_pct, ma: s.rolling_3m_avg, mom: s.mom_growth_pct, rank: s.monthly_rank };
  });

  const volPathD = points.length > 0 ? `M ${points.map((p) => `${p.x} ${p.yVol}`).join(" L ")}` : "";
  const avgPathD = points.length > 0 ? `M ${points.map((p) => `${p.x} ${p.yAvg}`).join(" L ")}` : "";
  const hoveredPoint = hoveredIndex !== null && points[hoveredIndex] ? points[hoveredIndex] : null;

  return (
    <section id="volume" className="py-24 bg-background relative overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute top-1/4 right-0 w-[500px] h-[500px] bg-brand-cyan/[0.04] rounded-full blur-[120px]" />
      <div className="pointer-events-none absolute bottom-10 left-10 w-[400px] h-[400px] bg-brand-amber/[0.03] rounded-full blur-[100px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="relative mb-10">
          <span className="section-number">05</span>
          <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between">
            <div>
              <div className="pill-badge bg-brand-cyan/10 border-brand-cyan/25 text-brand-cyan mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-cyan animate-pulse" />
                <span className="font-mono text-[11px] tracking-wider">DUCKDB ANALYTICS</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
                Complaint Volume <span className="font-serif-italic text-gradient-cyan">Trends</span>
              </h2>
            </div>
            <p className="text-xs font-mono text-zinc-500 mt-2 md:mt-0">
              {volume.time_range} • {volume.total_complaints.toLocaleString()} TOTAL
            </p>
          </div>
        </div>

        {/* Product selector pills */}
        <div className="flex items-center space-x-2 overflow-x-auto pb-3 mb-6 scrollbar-none">
          <span className="text-xs text-zinc-500 flex items-center pr-2 font-mono shrink-0">
            <Filter className="w-3 h-3 mr-1" /> PRODUCT:
          </span>
          {summaryTable.map((item) => {
            const isSelected = item.product === selectedProduct;
            return (
              <button
                key={item.product}
                onClick={() => { setSelectedProduct(item.product); setHoveredIndex(null); }}
                className={`px-3 py-1.5 rounded-full text-xs whitespace-nowrap transition-all border ${
                  isSelected
                    ? "bg-brand-cyan/15 text-brand-cyan border-brand-cyan/40 font-semibold"
                    : "bg-white/[0.03] text-zinc-400 border-white/[0.06] hover:border-white/[0.12] hover:text-zinc-200"
                }`}
              >
                {item.product}
                <span className="ml-1.5 text-[10px] opacity-60">
                  ({(item.total_volume / 1000).toFixed(1)}k)
                </span>
              </button>
            );
          })}
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: KPI Cards */}
          <div className="lg:col-span-4 space-y-4">
            {/* KPI 1 */}
            <div className="bento-card p-5 group">
              <div className="text-[11px] text-zinc-500 font-mono tracking-wider mb-1">
                CUMULATIVE INBOUND VOLUME
              </div>
              <div className="text-3xl font-bold text-foreground num-mono flex items-baseline justify-between">
                <span>{activeSummary.total_volume.toLocaleString()}</span>
                <span className="text-sm font-medium text-brand-cyan">
                  {activeSummary.avg_share_pct}% share
                </span>
              </div>
              <div className="mt-2 text-[11px] text-zinc-500 flex items-center justify-between border-t border-white/[0.06] pt-2 font-mono">
                <span>Category</span>
                <span className="text-zinc-300 font-medium truncate max-w-[180px]">
                  {activeSummary.product}
                </span>
              </div>
            </div>

            {/* KPI 2 */}
            <div className="bento-card p-5 group">
              <div className="text-[11px] text-zinc-500 font-mono tracking-wider mb-1">
                LATEST MONTH ({activeSummary.latest_month})
              </div>
              <div className="text-3xl font-bold text-foreground num-mono flex items-baseline justify-between">
                <span>{activeSummary.latest_volume.toLocaleString()}</span>
                <span
                  className={`pill-badge text-[10px] font-semibold ${
                    activeSummary.latest_mom_growth >= 0
                      ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-400"
                      : "bg-rose-500/10 border-rose-500/25 text-rose-400"
                  }`}
                >
                  {activeSummary.latest_mom_growth >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {activeSummary.latest_mom_growth > 0 ? "+" : ""}
                  {activeSummary.latest_mom_growth}%
                </span>
              </div>
              <div className="mt-2 text-[11px] text-zinc-500 flex items-center justify-between border-t border-white/[0.06] pt-2 font-mono">
                <span>3M Moving Avg</span>
                <span className="text-brand-amber font-medium num-mono">
                  {activeSummary.latest_rolling_3m_avg.toLocaleString()} / mo
                </span>
              </div>
            </div>

            {/* Explainer */}
            <div className="glass-card p-4 text-xs text-zinc-500 leading-relaxed">
              <div className="flex items-center space-x-2 text-zinc-300 font-semibold mb-1">
                <BarChart3 className="w-4 h-4 text-brand-cyan" />
                <span>Window Metrics</span>
              </div>
              <p className="text-[11px]">
                Computed via DuckDB window functions over gold-layer complaints.
                Tracks MoM differentials with a 3-month trailing rolling window.
              </p>
            </div>
          </div>

          {/* Right Column: Chart */}
          <div className="lg:col-span-8 bento-card p-6 font-mono">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-white/[0.06] gap-2">
              <div>
                <span className="text-xs text-brand-cyan font-semibold tracking-wide">
                  MONTHLY VOLUME & 3M ROLLING
                </span>
                <h3 className="text-sm font-semibold text-foreground mt-0.5">
                  {selectedProduct}
                </h3>
              </div>
              <div className="flex items-center space-x-4 text-xs">
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-brand-cyan inline-block rounded" />
                  <span className="text-zinc-500">Monthly</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-brand-amber inline-block rounded" />
                  <span className="text-zinc-500">3M Avg</span>
                </div>
              </div>
            </div>

            {/* Hover details */}
            <div className="h-9 mt-2 flex items-center text-xs text-zinc-400">
              {hoveredPoint ? (
                <div className="flex items-center space-x-4 bg-white/[0.03] px-3 py-1 rounded-full border border-white/[0.06] w-full">
                  <span className="font-semibold text-foreground">{hoveredPoint.month}</span>
                  <span>Vol: <strong className="text-brand-cyan">{hoveredPoint.volume.toLocaleString()}</strong></span>
                  <span>3M: <strong className="text-brand-amber">{hoveredPoint.ma.toLocaleString()}</strong></span>
                  <span>MoM: <strong className={hoveredPoint.mom >= 0 ? "text-emerald-400" : "text-rose-400"}>{hoveredPoint.mom > 0 ? "+" : ""}{hoveredPoint.mom}%</strong></span>
                </div>
              ) : (
                <span className="text-[11px] text-zinc-600 italic">Hover over data points to inspect metrics.</span>
              )}
            </div>

            <div className="mt-2 w-full overflow-x-auto">
              <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-auto min-w-[560px] select-none">
                {/* Grid */}
                {[0.25, 0.5, 0.75, 1.0].map((frac) => {
                  const val = maxVol * frac;
                  const y = padding.top + innerH - frac * innerH;
                  return (
                    <g key={frac}>
                      <line x1={padding.left} y1={y} x2={chartW - padding.right} y2={y} stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                      <text x={padding.left - 8} y={y + 3} fill="#52525B" fontSize="10" textAnchor="end">
                        {Math.round(val).toLocaleString()}
                      </text>
                    </g>
                  );
                })}

                {/* Area under volume curve */}
                {points.length > 0 && (
                  <path
                    d={`${volPathD} L ${points[points.length - 1].x} ${padding.top + innerH} L ${points[0].x} ${padding.top + innerH} Z`}
                    fill="rgba(6, 182, 212, 0.06)"
                  />
                )}

                {/* 3M MA line */}
                {avgPathD && <path d={avgPathD} fill="none" stroke="#F59E0B" strokeWidth="2" strokeDasharray="4 2" className="opacity-80" />}

                {/* Volume line */}
                {volPathD && <path d={volPathD} fill="none" stroke="#06B6D4" strokeWidth="2.5" />}

                {/* Data Points */}
                {points.map((p, idx) => {
                  const isHovered = hoveredIndex === idx;
                  return (
                    <g key={idx} className="cursor-pointer" onMouseEnter={() => setHoveredIndex(idx)} onMouseLeave={() => setHoveredIndex(null)}>
                      {isHovered && <line x1={p.x} y1={padding.top} x2={p.x} y2={padding.top + innerH} stroke="#06B6D4" strokeWidth="1" strokeDasharray="2 2" className="opacity-50" />}
                      <circle cx={p.x} cy={p.yVol} r={isHovered ? 6 : 3} fill="#06B6D4" stroke="rgba(9,10,15,0.8)" strokeWidth="2" className="transition-all duration-150" />
                      <circle cx={p.x} cy={p.yAvg} r={isHovered ? 4.5 : 2} fill="#F59E0B" stroke="rgba(9,10,15,0.8)" strokeWidth="1.5" />
                      {(idx % 5 === 0 || idx === points.length - 1) && (
                        <text x={p.x} y={chartH - 15} fill="#52525B" fontSize="10" textAnchor="middle">{p.month}</text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
