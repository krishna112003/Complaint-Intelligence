"use client";

import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  size: number;
  vx: number;
  vy: number;
  alpha: number;
  hue: number;
}

export default function CyberScrollBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    let animId: number;
    let width = 0;
    let height = 0;

    // Use DPR 1 for background particles — saves 75% fill-rate GPU bandwidth
    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };

    window.addEventListener("resize", resize, { passive: true });
    resize();

    // 24 elegant floating particles
    const particles: Particle[] = Array.from({ length: 24 }, () => ({
      x: Math.random() * (width || 1200),
      y: Math.random() * (height || 800),
      size: Math.random() * 1.5 + 0.8,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      alpha: Math.random() * 0.25 + 0.08,
      hue: Math.random() > 0.5 ? 42 : 195,
    }));

    let scrollY = window.scrollY;
    let targetScrollY = window.scrollY;

    const onScroll = () => {
      targetScrollY = window.scrollY;
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    const render = () => {
      scrollY += (targetScrollY - scrollY) * 0.1;
      const scrollDrift = (targetScrollY - scrollY) * 0.15;

      ctx.clearRect(0, 0, width, height);

      // Draw particles (ultra fast: single path or quick arcs)
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy - scrollDrift;

        if (p.x < -10) p.x = width + 10;
        else if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        else if (p.y > height + 10) p.y = -10;

        ctx.fillStyle = `hsla(${p.hue}, 85%, 65%, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden">
      {/* GPU Hardware-Accelerated Ambient Orbs (Zero CPU cost, 120 FPS native compositing) */}
      <div
        className="absolute -top-[10%] -right-[10%] w-[650px] h-[650px] rounded-full bg-brand-amber/[0.045] blur-[130px] animate-glow-pulse"
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform" }}
      />
      <div
        className="absolute top-[35%] -left-[12%] w-[600px] h-[600px] rounded-full bg-brand-cyan/[0.04] blur-[140px]"
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform" }}
      />
      <div
        className="absolute top-[65%] -right-[8%] w-[550px] h-[550px] rounded-full bg-purple-600/[0.035] blur-[150px]"
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform" }}
      />
      <div
        className="absolute -bottom-[5%] left-[20%] w-[500px] h-[500px] rounded-full bg-brand-emerald/[0.035] blur-[140px]"
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform" }}
      />

      {/* Lightweight Canvas for floating dynamic particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full opacity-70"
        style={{ transform: "translate3d(0, 0, 0)", willChange: "transform" }}
      />
    </div>
  );
}
