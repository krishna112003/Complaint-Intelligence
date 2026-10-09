"use client";

import { useEffect, useState } from "react";
import Hero from "@/components/Hero";
import CategoryNav from "@/components/CategoryNav";
import PipelineSection from "@/components/PipelineSection";
import ModelArena from "@/components/ModelArena";
import StatsSection from "@/components/StatsSection";
import BiasLabSection from "@/components/BiasLabSection";
import VolumeSection from "@/components/VolumeSection";
import PlaygroundSection from "@/components/PlaygroundSection";
import Footer from "@/components/Footer";
import CyberScrollBackground from "@/components/CyberScrollBackground";
import BackgroundVideo from "@/components/BackgroundVideo";
import ScrollReveal from "@/components/ScrollReveal";
import ScrollProgressHUD from "@/components/ScrollProgressHUD";
import {
  getSummaryData,
  getPipelineData,
  getModelsData,
  getPairwiseData,
  getCalibrationData,
  getBiasData,
  getVolumeData,
  getExamplesData,
} from "@/lib/data";
import {
  SummaryData,
  PipelineData,
  ModelsData,
  PairwiseData,
  CalibrationData,
  BiasData,
  VolumeData,
  ExamplesData,
} from "@/lib/types";

export default function Home() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [pipeline, setPipeline] = useState<PipelineData | null>(null);
  const [models, setModels] = useState<ModelsData | null>(null);
  const [pairwise, setPairwise] = useState<PairwiseData | null>(null);
  const [calibration, setCalibration] = useState<CalibrationData | null>(null);
  const [bias, setBias] = useState<BiasData | null>(null);
  const [volume, setVolume] = useState<VolumeData | null>(null);
  const [examples, setExamples] = useState<ExamplesData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadAllData() {
      try {
        const [
          summaryRes,
          pipelineRes,
          modelsRes,
          pairwiseRes,
          calibRes,
          biasRes,
          volumeRes,
          examplesRes,
        ] = await Promise.all([
          getSummaryData(),
          getPipelineData(),
          getModelsData(),
          getPairwiseData(),
          getCalibrationData(),
          getBiasData(),
          getVolumeData(),
          getExamplesData(),
        ]);

        setSummary(summaryRes);
        setPipeline(pipelineRes);
        setModels(modelsRes);
        setPairwise(pairwiseRes);
        setCalibration(calibRes);
        setBias(biasRes);
        setVolume(volumeRes);
        setExamples(examplesRes);
      } catch (err) {
        console.error("Failed to load pipeline datasets:", err);
      } finally {
        setLoading(false);
      }
    }

    loadAllData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden">
        {/* Ambient loading glow */}
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full bg-brand-amber/[0.06] blur-[120px] animate-glow-pulse" />
        <div className="flex flex-col items-center space-y-4 relative z-10">
          <div className="w-10 h-10 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center">
            <span className="w-2.5 h-2.5 rounded-full bg-brand-amber animate-pulse" />
          </div>
          <div className="text-center">
            <span className="text-sm text-foreground font-medium">Loading <span className="font-serif-italic text-brand-amber">Pipeline</span></span>
            <p className="text-xs text-zinc-500 mt-1 font-mono">VERIFIED DATASETS</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col relative">
      {/* Dynamic Futuristic Video-like Canvas Scroll Background */}
      <CyberScrollBackground />

      {/* Dynamic Ambient Background Video (Plays seamlessly if video is present) */}
      <BackgroundVideo videoSrc="/videos/background.mp4" />

      {/* Scroll-driven Floating HUD & Progress Tracker */}
      <ScrollProgressHUD />

      {/* Hero Section */}
      <Hero summary={summary} examples={examples} />

      {/* 21st.dev Style Sticky Category Navigator Pills */}
      <CategoryNav />

      {/* 01. Pinned Pipeline Section */}
      <PipelineSection pipeline={pipeline} />

      {/* 02. Model Arena & Pareto Frontier */}
      <ScrollReveal delay={0.15}>
        <ModelArena data={models} />
      </ScrollReveal>

      {/* 03. Pairwise McNemar Matrix & Calibration */}
      <ScrollReveal delay={0.15}>
        <StatsSection pairwise={pairwise} calibration={calibration} />
      </ScrollReveal>

      {/* 04. Bias Lab & Counterfactual Probes */}
      <ScrollReveal delay={0.15}>
        <BiasLabSection bias={bias} />
      </ScrollReveal>

      {/* 05. Complaint Volume Trends (DuckDB Window SQL) */}
      <ScrollReveal delay={0.15}>
        <VolumeSection volume={volume} />
      </ScrollReveal>

      {/* 06. Interactive Try-It Playground */}
      <ScrollReveal delay={0.15}>
        <PlaygroundSection examples={examples} />
      </ScrollReveal>

      {/* Footer */}
      <Footer />
    </div>
  );
}
