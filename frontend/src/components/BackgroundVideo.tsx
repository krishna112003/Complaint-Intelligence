"use client";

import { useEffect, useRef, useState } from "react";
import { Play, Pause, Video } from "lucide-react";

interface BackgroundVideoProps {
  videoSrc?: string;
  overlayOpacity?: number;
  className?: string;
}

export default function BackgroundVideo({
  videoSrc = "/videos/background.mp4",
  overlayOpacity = 0.55,
  className = "",
}: BackgroundVideoProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [hasVideoLoaded, setHasVideoLoaded] = useState<boolean>(true);
  const [hasError, setHasError] = useState<boolean>(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.play().then(() => {
      setIsPlaying(true);
      setHasVideoLoaded(true);
    }).catch((err) => {
      console.log("Autoplay notification:", err);
    });
  }, [videoSrc]);

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
      setIsPlaying(false);
    } else {
      video.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  return (
    <div className={`fixed inset-0 pointer-events-none -z-20 overflow-hidden ${className}`}>
      {/* 1. Hardware-Accelerated Ambient Looping Video */}
      {!hasError && (
        <video
          ref={videoRef}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          onLoadedData={() => setHasVideoLoaded(true)}
          onError={() => setHasError(true)}
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${
            hasVideoLoaded ? "opacity-75" : "opacity-0"
          }`}
          style={{
            transform: "translate3d(0, 0, 0)",
            willChange: "transform, opacity",
            filter: "brightness(0.9) contrast(1.1)",
          }}
        >
          <source src={videoSrc} type="video/mp4" />
        </video>
      )}

      {/* 2. Luxury Dark Frosted Vignette Overlay (Keeps all typography & cards crisp) */}
      <div
        className="absolute inset-0 bg-background/60 pointer-events-none"
        style={{ opacity: overlayOpacity }}
      />

      {/* 3. Radial Vignette (Clear center focus, cinematic dark edges) */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at 50% 30%, transparent 20%, rgba(9, 10, 15, 0.85) 85%)",
        }}
      />

      {/* 4. Interactive Video Pill Indicator (Bottom-Left) */}
      {!hasError && (
        <div className="fixed bottom-6 left-6 z-40 pointer-events-auto">
          <button
            onClick={togglePlayback}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-surface/90 hover:bg-surface border border-white/[0.12] text-xs text-zinc-300 hover:text-foreground backdrop-blur-xl transition-all shadow-xl active:scale-95 group"
            title={isPlaying ? "Click to pause video" : "Click to play video"}
            aria-label={isPlaying ? "Click to pause video" : "Click to play video"}
          >
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${isPlaying ? "bg-brand-emerald opacity-75" : "bg-zinc-500 opacity-20"}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isPlaying ? "bg-brand-emerald" : "bg-zinc-500"}`} />
            </span>
            <span className="text-[11px] font-mono flex items-center gap-1.5">
              <span>Dynamic Video</span>
              <span className="text-zinc-500">•</span>
              <span className={isPlaying ? "text-brand-amber font-semibold" : "text-zinc-400"}>
                {isPlaying ? "LIVE" : "PAUSED"}
              </span>
            </span>
            {isPlaying ? (
              <Pause className="w-3 h-3 text-zinc-400 group-hover:text-brand-amber transition-colors ml-1" />
            ) : (
              <Play className="w-3 h-3 text-zinc-400 group-hover:text-brand-amber transition-colors ml-1" />
            )}
          </button>
        </div>
      )}
    </div>
  );
}
