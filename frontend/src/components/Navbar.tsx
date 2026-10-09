"use client";

import { useEffect, useState } from "react";
import { Terminal, Github, Sun, Moon, Activity, Sparkles, ExternalLink } from "lucide-react";

export default function Navbar() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isLiveLocal, setIsLiveLocal] = useState<boolean>(false);
  const [activeSection, setActiveSection] = useState<string>("pipeline");

  useEffect(() => {
    const checkLiveAPI = async () => {
      try {
        const res = await fetch("http://localhost:8000/health", { method: "GET", mode: "cors" });
        if (res.ok) setIsLiveLocal(true);
      } catch {
        setIsLiveLocal(false);
      }
    };
    checkLiveAPI();

    // Zero-overhead IntersectionObserver for active section highlighting
    const sections = ["pipeline", "arena", "stats", "volume", "bias", "playground"];
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      { rootMargin: "-15% 0px -65% 0px", threshold: 0 }
    );

    sections.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 transition-all duration-300 backdrop-blur-xl bg-background/80 border-b border-white/[0.08] dark:border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Brand */}
        <a href="#" className="flex items-center space-x-2.5 group outline-none">
          <div className="w-8 h-8 rounded-lg bg-surface border border-white/[0.1] flex items-center justify-center text-brand-amber transition-transform group-hover:scale-105 shadow-sm">
            <Terminal className="w-4 h-4" />
          </div>
          <div className="flex items-center space-x-2">
            <span className="font-semibold tracking-tight text-foreground text-sm flex items-center gap-1.5">
              <span>Complaint</span>
              <span className="text-brand-amber font-serif-italic text-base">Intelligence</span>
            </span>
            <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-mono font-medium bg-brand-amber/10 text-brand-amber border border-brand-amber/25">
              v2.4
            </span>
          </div>
        </a>

        {/* Center Pill Nav (21st.dev style) */}
        <nav className="hidden lg:flex items-center space-x-1 p-1 rounded-full bg-surface/70 border border-white/[0.08] backdrop-blur-md text-[13px] font-medium text-zinc-400">
          {[
            { id: "pipeline", label: "Pipeline" },
            { id: "arena", label: "Models" },
            { id: "stats", label: "Statistics" },
            { id: "volume", label: "Volume Trends" },
            { id: "bias", label: "Fairness" },
            { id: "playground", label: "Playground" },
          ].map((item) => {
            const isActive = activeSection === item.id;
            return (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`px-3 py-1 rounded-full transition-all duration-200 outline-none ${
                  isActive
                    ? "bg-foreground/[0.12] text-foreground font-semibold shadow-sm"
                    : "hover:text-foreground hover:bg-foreground/[0.05]"
                }`}
              >
                {item.label}
              </a>
            );
          })}
        </nav>

        {/* Right Action Utilities */}
        <div className="flex items-center space-x-2.5">
          {/* Engine indicator */}
          <div
            className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border backdrop-blur-sm transition-colors ${
              isLiveLocal
                ? "bg-brand-emerald/10 border-brand-emerald/30 text-brand-emerald"
                : "bg-surface/80 border-white/[0.08] text-zinc-400"
            }`}
            title={
              isLiveLocal
                ? "FastAPI inference engine online (http://localhost:8000)"
                : "Deterministic test evaluation artifacts (100% offline reproducible)"
            }
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isLiveLocal ? "bg-brand-emerald animate-pulse" : "bg-brand-amber"
              }`}
            />
            <span className="hidden sm:inline font-sans font-medium text-[12px]">
              {isLiveLocal ? "Live Engine" : "Verified Test Split"}
            </span>
          </div>

          {/* Theme switcher */}
          <button
            onClick={toggleTheme}
            className="w-8 h-8 rounded-full border border-white/[0.08] bg-surface/60 flex items-center justify-center text-zinc-400 hover:text-foreground hover:bg-surface transition-all active:scale-95"
            aria-label="Toggle visual theme"
          >
            {theme === "dark" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>

          {/* GitHub Link */}
          <a
            href="https://github.com/krishna112003/Complaint-Intelligence"
            target="_blank"
            rel="noopener noreferrer"
            className="w-8 h-8 rounded-full border border-white/[0.08] bg-surface/60 flex items-center justify-center text-zinc-400 hover:text-foreground hover:bg-surface transition-all active:scale-95"
            aria-label="GitHub Repository"
          >
            <Github className="w-3.5 h-3.5" />
          </a>

          {/* Quick Action Pill Button */}
          <a
            href="#playground"
            className="hidden sm:inline-flex items-center space-x-1.5 px-3.5 py-1 rounded-full text-[12px] font-medium bg-foreground text-background transition-transform hover:scale-[1.02] active:scale-[0.98] shadow-sm"
          >
            <Sparkles className="w-3 h-3 text-brand-amber" />
            <span>Try Pipeline</span>
          </a>
        </div>
      </div>
    </header>
  );
}
