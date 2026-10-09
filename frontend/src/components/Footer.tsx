import { Github, FileText, Terminal, Shield, ExternalLink, Heart } from "lucide-react";

export default function Footer() {
  return (
    <footer className="relative py-20 bg-background overflow-hidden">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] rounded-full bg-brand-amber/[0.03] blur-[100px]" />

      {/* Top divider with gradient */}
      <div className="h-px w-full bg-gradient-to-r from-transparent via-white/[0.08] to-transparent mb-16" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-14">
          {/* Col 1: Project Identity */}
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center space-x-2.5">
              <div className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-brand-amber">
                <Terminal className="w-4 h-4" />
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="font-semibold text-foreground text-sm">Complaint</span>
                <span className="font-serif-italic text-brand-amber text-base">Intelligence</span>
              </div>
            </div>
            <p className="text-zinc-500 text-xs leading-relaxed">
              Benchmarking classical ML, sentence transformers, and generative LLMs on CFPB consumer disputes.
            </p>
            <div className="text-[11px] text-zinc-600 font-mono">
              MIT License • Open Source
            </div>
          </div>

          {/* Col 2: Docs */}
          <div className="space-y-2.5">
            <div className="text-foreground font-semibold text-xs tracking-wider">DOCS & GOVERNANCE</div>
            <ul className="space-y-2 text-zinc-400 text-xs">
              {[
                { label: "Model Card", path: "docs/model_card.md" },
                { label: "System Architecture", path: "docs/architecture.md" },
                { label: "Evaluation Report", path: "reports/results.md" },
                { label: "Bias Report", path: "reports/bias_report.md" },
              ].map((doc) => (
                <li key={doc.path}>
                  <a
                    href={`https://github.com/krishna112003/Complaint-Intelligence/blob/main/${doc.path}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-brand-amber transition-colors flex items-center space-x-1.5 group"
                  >
                    <FileText className="w-3.5 h-3.5 text-zinc-600 group-hover:text-brand-amber transition-colors" />
                    <span>{doc.label}</span>
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Col 3: Honest Reporting Note */}
          <div className="space-y-2.5 md:col-span-1">
            <div className="text-foreground font-semibold text-xs flex items-center space-x-1.5 tracking-wider">
              <Shield className="w-3.5 h-3.5 text-brand-amber" />
              <span>HONEST REPORTING</span>
            </div>
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Every metric, bootstrap CI, p-value, and topic volume on this site is loaded
              directly from validated JSON exports produced by the Python pipeline. No hard-coded,
              fabricated, or cherry-picked numbers.
            </p>
          </div>

          {/* Col 4: Built With */}
          <div className="space-y-2.5">
            <div className="text-foreground font-semibold text-xs tracking-wider">BUILT WITH</div>
            <div className="flex flex-wrap gap-1.5 text-[10px]">
              {[
                "Next.js 14", "TypeScript", "Tailwind CSS", "Framer Motion",
                "GSAP", "DuckDB", "Scikit-Learn", "DSPy 2.6"
              ].map((tech) => (
                <span
                  key={tech}
                  className="px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.06] text-zinc-400 hover:text-foreground hover:border-white/[0.12] transition-all cursor-default"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/[0.06] to-transparent mb-6" />

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-zinc-600">
          <div className="flex items-center space-x-1">
            <span>&copy; 2026 Krishna</span>
            <span>•</span>
            <span className="flex items-center space-x-1">
              <span>Built with</span>
              <Heart className="w-3 h-3 text-brand-ruby inline" />
            </span>
          </div>
          <a
            href="https://github.com/krishna112003/Complaint-Intelligence"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] hover:border-brand-amber/30 text-zinc-400 hover:text-foreground transition-all"
          >
            <Github className="w-3.5 h-3.5" />
            <span>GitHub Repository</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>
    </footer>
  );
}
