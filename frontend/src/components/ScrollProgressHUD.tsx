"use client";

import { useEffect, useState } from "react";
import { ArrowUp, Compass } from "lucide-react";

export default function ScrollProgressHUD() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [activeSectionName, setActiveSectionName] = useState("Hero");

  useEffect(() => {
    let ticking = false;

    const updateScroll = () => {
      const totalScroll = document.documentElement.scrollHeight - window.innerHeight;
      const currentScroll = window.scrollY;

      if (totalScroll > 0) {
        setScrollProgress(Math.min(100, Math.max(0, Math.round((currentScroll / totalScroll) * 100))));
      }

      setIsVisible(currentScroll > 280);
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(updateScroll);
        ticking = true;
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    updateScroll();

    // Zero-cost IntersectionObserver for active section name
    const sectionNames: Record<string, string> = {
      pipeline: "01. Pipeline",
      arena: "02. Model Arena",
      stats: "03. McNemar Matrix",
      bias: "04. Fairness Lab",
      volume: "05. Volume Trends",
      playground: "06. Playground",
    };

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && sectionNames[entry.target.id]) {
            setActiveSectionName(sectionNames[entry.target.id]);
          }
        }
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 }
    );

    Object.keys(sectionNames).forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => {
      window.removeEventListener("scroll", onScroll);
      observer.disconnect();
    };
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <>
      {/* 1. Sleek Top-Edge Glowing Progress Bar */}
      <div className="fixed top-0 left-0 right-0 h-[2px] z-50 pointer-events-none bg-white/[0.04]">
        <div
          className="h-full bg-gradient-to-r from-brand-amber via-brand-cyan to-brand-emerald"
          style={{
            width: `${scrollProgress}%`,
            transform: "translate3d(0, 0, 0)",
            willChange: "width",
          }}
        />
      </div>

      {/* 2. Floating Bottom-Right 21st.dev Style Scroll HUD */}
      <div
        className={`fixed bottom-6 right-6 z-40 transition-all duration-300 transform ${
          isVisible ? "opacity-100 translate-y-0 pointer-events-auto" : "opacity-0 translate-y-4 pointer-events-none"
        }`}
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform, opacity" }}
      >
        <div className="flex items-center space-x-2 p-1.5 rounded-full bg-surface/90 backdrop-blur-xl border border-white/[0.12] shadow-2xl hover:border-white/[0.2] transition-colors">
          {/* Section Indicator Badge */}
          <div className="flex items-center space-x-1.5 px-3 py-1 text-[11px] font-mono text-zinc-300">
            <Compass className="w-3.5 h-3.5 text-brand-amber animate-spin-slow" />
            <span className="hidden sm:inline font-sans text-xs font-medium text-foreground">{activeSectionName}</span>
            <span className="text-zinc-500">•</span>
            <span className="text-brand-cyan font-bold num-mono">{scrollProgress}%</span>
          </div>

          {/* Quick Back to Top Pill Button */}
          <button
            onClick={scrollToTop}
            title="Scroll to top"
            aria-label="Scroll to top"
            className="w-8 h-8 rounded-full bg-foreground/[0.08] hover:bg-foreground hover:text-background text-zinc-300 flex items-center justify-center transition-all duration-200 group active:scale-90"
          >
            <ArrowUp className="w-4 h-4 transition-transform group-hover:-translate-y-0.5" />
          </button>
        </div>
      </div>
    </>
  );
}
