"use client";

import { useEffect, useRef, useState } from "react";
import { Play, Pause } from "lucide-react";

interface BackgroundVideoProps {
  videoSrc?: string;
  overlayOpacity?: number; // 0.0 to 1.0
  className?: string;
}

export default function BackgroundVideo({
  videoSrc = "/videos/background.mp4",
  overlayOpacity = 0.82,
  className = "",
}: BackgroundVideoProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [hasVideoLoaded, setHasVideoLoaded] = useState<boolean>(false);
  const [hasError, setHasError] = useState<boolean>(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Check if video can play
    const handleCanPlay = () => {
      setHasVideoLoaded(true);
      video.play().catch(() => {
        // Autoplay policy fallback
        setIsPlaying(false);
      });
    };

    const handleError = () => {
      setHasError(true);
    };

    video.addEventListener("canplay", handleCanPlay);
    video.addEventListener("error", handleError);

    return () => {
      video.removeEventListener("canplay", handleCanPlay);
      video.removeEventListener("error", handleError);
    };
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
      {/* 1. HTML5 Hardware-Accelerated Video Layer */}
      {!hasError && (
        <video
          ref={videoRef}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ${
            hasVideoLoaded ? "opacity-60 mix-blend-screen" : "opacity-0"
          }`}
          style={{ transform: "translate3d(0, 0, 0)", willChange: "transform, opacity" }}
        >
          <source src={videoSrc} type="video/mp4" />
          <source src={videoSrc.replace(".mp4", ".webm")} type="video/webm" />
        </video>
      )}

      {/* 2. Luxury Dark Vignette & Frosting (Keeps text and charts 100% crisp) */}
      <div
        className="absolute inset-0 bg-background pointer-events-none transition-colors duration-500"
        style={{ opacity: overlayOpacity }}
      />

      {/* 3. Subtle Gradient Depth Bleed */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-transparent to-background/90 pointer-events-none" />

      {/* 4. Interactive Video Control Pill (Bottom Left) */}
      {hasVideoLoaded && !hasError && (
        <div className="fixed bottom-6 left-6 z-40 pointer-events-auto">
          <button
            onClick={togglePlayback}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-surface/80 hover:bg-surface border border-white/[0.1] text-xs text-zinc-400 hover:text-foreground backdrop-blur-md transition-all shadow-lg active:scale-95"
            title={isPlaying ? "Pause ambient video" : "Resume ambient video"}
            aria-label={isPlaying ? "Pause ambient video" : "Resume ambient video"}
          >
            {isPlaying ? (
              <>
                <Pause className="w-3 h-3 text-brand-amber" />
                <span className="text-[11px] font-mono">Ambient Video ON</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 text-zinc-400" />
                <span className="text-[11px] font-mono">Ambient Video Paused</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
