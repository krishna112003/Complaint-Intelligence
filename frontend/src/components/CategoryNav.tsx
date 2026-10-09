"use client";

import { useEffect, useState } from "react";
import { 
  Layers, 
  Cpu, 
  BarChart3, 
  TrendingUp, 
  ShieldAlert, 
  Sparkles, 
  Compass 
} from "lucide-react";

interface CategoryNavProps {
  onSelectCategory?: (id: string) => void;
}

export default function CategoryNav({ onSelectCategory }: CategoryNavProps) {
  const [activeTab, setActiveTab] = useState<string>("all");

  const categories = [
    { id: "all", label: "All Modules", icon: Compass, count: "7", target: "pipeline" },
    { id: "pipeline", label: "DuckDB Pipeline", icon: Layers, count: "9 Stages", target: "pipeline" },
    { id: "arena", label: "Model Zoo & Pareto", icon: Cpu, count: "5 Models", target: "arena" },
    { id: "stats", label: "McNemar Matrix", icon: BarChart3, count: "10 Pairs", target: "stats" },
    { id: "volume", label: "Volume Trends", icon: TrendingUp, count: "10 Products", target: "volume" },
    { id: "bias", label: "Fairness Probes", icon: ShieldAlert, count: "4 Audits", target: "bias" },
    { id: "playground", label: "Live Diagnostics", icon: Sparkles, count: "Interactive", target: "playground" },
  ];

  useEffect(() => {
    const targetSections = ["pipeline", "arena", "stats", "volume", "bias", "playground"];
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveTab(entry.target.id);
          }
        }
      },
      { rootMargin: "-15% 0px -65% 0px", threshold: 0 }
    );

    targetSections.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const handleClick = (id: string, target: string) => {
    setActiveTab(id);
    if (onSelectCategory) onSelectCategory(id);
    const el = document.getElementById(target);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="w-full border-y border-white/[0.08] dark:border-white/[0.06] bg-surface/50 backdrop-blur-md sticky top-14 z-40 py-2.5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-mono text-zinc-400 shrink-0 hidden md:flex">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-amber animate-pulse" />
            <span className="font-semibold text-foreground uppercase tracking-wider text-[11px]">System Registry:</span>
          </div>

          {/* Horizontal scrollable pills (Exact 21st.dev style) */}
          <div className="flex items-center space-x-2 overflow-x-auto no-scrollbar py-0.5 w-full md:w-auto">
            {categories.map((cat) => {
              const Icon = cat.icon;
              const isActive = activeTab === cat.id;
              return (
                <button
                  key={cat.id}
                  onClick={() => handleClick(cat.id, cat.target)}
                  className={`inline-flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all duration-200 outline-none ${
                    isActive
                      ? "bg-foreground text-background font-semibold shadow-sm scale-[1.02]"
                      : "bg-surface/80 border border-white/[0.08] text-zinc-300 hover:text-foreground hover:bg-surface hover:border-white/[0.15]"
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? "text-brand-amber" : "text-zinc-400"}`} />
                  <span>{cat.label}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                      isActive
                        ? "bg-background/20 text-background font-bold"
                        : "bg-white/[0.06] text-zinc-400"
                    }`}
                  >
                    {cat.count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
