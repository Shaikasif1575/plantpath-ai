import Link from "next/link";
import {
  ScanLine,
  ArrowRight,
  BrainCircuit,
  Microscope,
  Database,
  AlertTriangle,
  Leaf,
  FlaskConical,
  FileSearch,
  Layers,
} from "lucide-react";
import { api } from "@/lib/api";
import { SectionHeading, Disclaimer } from "@/components/ui";
import { ModelStateBadge, ModelUnavailableNotice } from "@/components/ModelStatus";
import { SupportedClasses } from "@/components/SupportedClasses";

const PIPELINE = [
  { icon: Leaf, label: "Leaf Image", desc: "Upload or capture" },
  { icon: FileSearch, label: "Image Validation", desc: "Format & quality checks" },
  { icon: BrainCircuit, label: "Disease Classification", desc: "Local ML inference" },
  { icon: Microscope, label: "Explainable AI", desc: "Grad-CAM attention maps" },
  { icon: Layers, label: "Severity Assessment", desc: "When a segmentation model is connected" },
  { icon: FlaskConical, label: "Analysis Report", desc: "Downloadable summary" },
];

const CAPABILITIES = [
  {
    title: "Local ML Inference",
    body: "Inference runs on-device from the connected PyTorch checkpoint. No paid or cloud AI APIs are used. If the checkpoint is not verified, the backend refuses to answer rather than returning a simulated disease name.",
  },
  {
    title: "Explainable AI (Grad-CAM)",
    body: "When the loaded model architecture supports it, Grad-CAM produces heatmaps showing which image regions influenced the prediction. These are interpretability aids — not biological causation proofs.",
  },
  {
    title: "Severity Assessment (extension point)",
    body: "The severity service is architected for a future U-Net / segmentation model. Until a validated mask model is connected, severity is reported as unavailable — never fabricated from classification confidence.",
  },
  {
    title: "Research Dashboard",
    body: "Model architecture, class mapping, training history, and evaluation metrics are surfaced only when they exist in project files. Missing metrics are shown as 'Not available' rather than invented.",
  },
];

const LIMITATIONS = [
  "Predictions are limited to the classes in the connected checkpoint's class mapping; an image outside those classes can only be reported as one of them.",
  "PlantVillage is a controlled benchmark dataset — field-collected imagery may behave differently.",
  "Model confidence is a probability distribution, not a guarantee of diagnostic correctness.",
  "Out-of-distribution images (unknown diseases, non-plant subjects) are not reliably detected without an explicit OOD model.",
  "Severity is not derived from classification confidence and requires a dedicated segmentation model.",
];

function HeroWorkflow() {
  return (
    <div
      className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6 shadow-sm"
      aria-label="Leaf analysis workflow"
    >
      <div className="text-xs font-semibold uppercase tracking-widest text-[var(--pp-muted)]">
        Analysis Workflow
      </div>
      <ol className="mt-4 space-y-0">
        {PIPELINE.map((step, i) => {
          const Icon = step.icon;
          return (
            <li key={step.label} className="relative flex gap-3 pb-4 last:pb-0">
              {i < PIPELINE.length - 1 ? (
                <span
                  className="absolute left-[15px] top-9 h-[calc(100%-22px)] w-px bg-forest-300 dark:bg-forest-700 animate-pulse-line"
                  aria-hidden
                />
              ) : null}
              <span className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-forest-300 bg-forest-50 text-forest-700 dark:border-forest-700 dark:bg-forest-900 dark:text-leaf-300">
                <Icon className="h-4 w-4" aria-hidden />
              </span>
              <div className="pt-1">
                <div className="text-sm font-semibold">{step.label}</div>
                <div className="text-xs text-[var(--pp-muted)]">{step.desc}</div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export const dynamic = "force-dynamic";

export default async function HomePage() {
  // Metrics are read from the backend, never hard-coded. On failure the
  // fallback text below is shown instead of guessing a value.
  let accuracy: number | null = null;
  let metricsNote: string | null = null;
  try {
    const info = await api.modelInfo();
    const m = (info.metrics || {}) as Record<string, unknown>;
    accuracy = typeof m.accuracy === "number" ? m.accuracy : null;
    metricsNote = typeof m.note === "string" ? m.note : null;
  } catch {
    accuracy = null;
  }
  return (
    <>
      {/* Hero */}
      <section className="hero-grid border-b border-[var(--pp-border)]">
        <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:px-8 lg:py-24">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-forest-300 bg-forest-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-forest-800 dark:border-forest-700 dark:bg-forest-900 dark:text-forest-100">
                <Leaf className="h-3.5 w-3.5" aria-hidden />
                Agricultural Computer Vision
              </span>
              <ModelStateBadge className="max-w-full sm:max-w-sm" />
            </div>
            <h1 className="mt-6 text-4xl font-bold tracking-tight sm:text-5xl">
              PlantPath <span className="text-forest-700 dark:text-leaf-400">AI</span>
            </h1>
            <p className="mt-3 text-xl font-medium text-[var(--pp-muted)]">
              Intelligent Plant Disease Detection from Leaf Images
            </p>
            <p className="mt-4 max-w-xl text-base leading-relaxed text-[var(--pp-muted)]">
              Analyze plant leaf images using deep learning, visualize model
              reasoning, and explore disease information through an interactive
              computer-vision platform.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                href="/scan"
                className="inline-flex items-center gap-2 rounded-md bg-forest-700 px-6 py-3 text-sm font-semibold text-white hover:bg-forest-800"
              >
                <ScanLine className="h-4 w-4" aria-hidden />
                Scan a Leaf
              </Link>
              <Link
                href="/research"
                className="inline-flex items-center gap-2 rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-6 py-3 text-sm font-semibold hover:bg-forest-50 dark:hover:bg-forest-900"
              >
                Explore the Research
                <ArrowRight className="h-4 w-4" aria-hidden />
              </Link>
            </div>
          </div>
          <HeroWorkflow />
        </div>
      </section>

      <div className="mx-auto max-w-7xl px-4 pt-8 sm:px-6 lg:px-8">
        <ModelUnavailableNotice />
      </div>

      {/* What it does */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Overview"
          title="What PlantPath AI Does"
          description="An end-to-end pipeline for leaf-image disease classification with scientific honesty about model limits."
        />
        <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {CAPABILITIES.map((c) => (
            <div
              key={c.title}
              className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-5"
            >
              <h3 className="font-semibold">{c.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-[var(--pp-muted)]">{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="border-y border-[var(--pp-border)] bg-[var(--pp-surface)]">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <SectionHeading
            eyebrow="Pipeline"
            title="How It Works"
            description="Each stage is isolated and replaceable — from image validation to explainability and (future) severity estimation."
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              ["Image Validation", "Format, size, dimensions, and basic visual-quality checks (dark, blurry, very small). Warnings never hard-reject valid images."],
              ["Local Inference", "Preprocessing matches the trained model's input contract. Softmax probabilities are mapped to class labels from the project's class mapping file."],
              ["Confidence Banding", "Predictions are banded low / medium / high using configurable thresholds — presented as relative model confidence, not certainty."],
              ["Grad-CAM", "Gradient-weighted class activation maps highlight influential regions when the model architecture supports it."],
              ["Severity (extension point)", "Segmentation-based severity is designed but not faked. The service returns available=false until a real mask model is connected."],
              ["History & Report", "SQLite-backed analysis history and a local PDF/HTML report — no cloud reporting services."],
            ].map(([title, body]) => (
              <div key={title} className="rounded-lg border border-[var(--pp-border)] p-5">
                <h3 className="font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--pp-muted)]">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported crops */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Coverage"
          title="Model Coverage"
          description="The class list below is whatever the connected model reports. Nothing is added from a reference dataset."
        />
        <div className="mt-8">
          <SupportedClasses title="Diseases This Model Can Detect" />
        </div>
        <p className="mt-4 text-sm text-[var(--pp-muted)]">
          The list above is served by the backend from the connected model&apos;s
          own class mapping. Separately, a PlantVillage-style reference library
          for 14 crop species and 38 classes is kept on the{" "}
          <Link href="/diseases" className="underline">Diseases</Link> page as
          educational material — it is labelled as a reference library, not as
          model output.
        </p>
      </section>

      {/* Explainable AI */}
      <section className="border-y border-[var(--pp-border)] bg-forest-900 text-forest-50">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="max-w-3xl">
            <div className="text-xs font-semibold uppercase tracking-widest text-leaf-300">
              Interpretability
            </div>
            <h2 className="mt-2 text-2xl font-bold sm:text-3xl">Explainable AI</h2>
            <p className="mt-4 leading-relaxed text-forest-100">
              Grad-CAM produces spatial heatmaps that indicate which image regions
              contributed strongly to the model&apos;s predicted class. The
              visualization is an interpretability aid for research and teaching —
              it does not establish biological causation or replace expert diagnosis.
              It is only rendered when the connected model reports that
              explainability is available; otherwise the page says so instead of
              drawing a placeholder heatmap.
            </p>
            <p className="mt-3 text-sm text-forest-200">
              Highlighted regions indicate image areas that contributed strongly to
              the model prediction. This visualization is an interpretability aid
              and does not establish biological causation.
            </p>
          </div>
        </div>
      </section>

      {/* Research methodology + limitations */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <SectionHeading
              eyebrow="Methodology"
              title="Research Approach"
              description="Transfer learning on a controlled plant-disease benchmark, with honest reporting of what the system can and cannot claim."
            />
            <div className="mt-6 space-y-4 text-sm leading-relaxed text-[var(--pp-muted)]">
              <p>
                The project is grounded in the PlantVillage dataset tradition:
                leaf images captured under controlled conditions, labelled by
                species and disease state, used to train a convolutional neural
                network via transfer learning.
              </p>
              <p>
                Evaluation metrics, when available in project files, are reported
                on the model page. They describe controlled-dataset performance
                only — not field deployment performance.
              </p>
              <p>
                See the{" "}
                <Link href="/research" className="font-medium text-forest-700 underline dark:text-leaf-400">
                  Research
                </Link>{" "}
                page for problem statement, gap analysis, objectives, and future work.
              </p>
            </div>
          </div>
          <div>
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-6 dark:border-amber-800 dark:bg-amber-950/20">
              <div className="flex items-center gap-2 font-semibold text-amber-900 dark:text-amber-100">
                <AlertTriangle className="h-5 w-5" aria-hidden />
                Model Limitations
              </div>
              <ul className="mt-4 space-y-3 text-sm text-amber-900 dark:text-amber-100">
                {LIMITATIONS.map((l) => (
                  <li key={l} className="flex gap-2">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-600 dark:bg-amber-400" aria-hidden />
                    <span>{l}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Model performance teaser */}
      <section className="border-y border-[var(--pp-border)] bg-[var(--pp-surface)]">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <SectionHeading
              eyebrow="Model"
              title="Model Performance"
              description="Metrics are shown only when present in project evaluation files."
            />
            <Link
              href="/model"
              className="inline-flex items-center gap-2 text-sm font-semibold text-forest-700 hover:text-forest-800 dark:text-leaf-400"
            >
              Open Model Dashboard
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Link>
          </div>
          <div className="mt-8 rounded-lg border border-dashed border-[var(--pp-border)] p-8 text-center">
            <Database className="mx-auto h-8 w-8 text-[var(--pp-muted)]" aria-hidden />
            {accuracy !== null ? (
              <>
                <p className="mt-3 font-mono text-3xl font-bold">
                  {(accuracy * 100).toFixed(2)}%
                </p>
                <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
                  Validation accuracy reported by the connected checkpoint
                </p>
                <p className="mt-3 text-sm text-[var(--pp-muted)]">
                  {metricsNote ||
                    "Read from the checkpoint's own metadata. Controlled-dataset validation only — not field performance."}{" "}
                  See the{" "}
                  <Link href="/model" className="underline">Model</Link>{" "}
                  dashboard for the full artifact record; precision, recall and F1
                  report &quot;Not available&quot; because no evaluation run produced them.
                </p>
              </>
            ) : (
              <p className="mt-3 text-sm text-[var(--pp-muted)]">
                No evaluation metrics are available. Accuracy, precision, recall and
                F1 are shown only when they exist in an evaluation file produced by an
                actual run — the{" "}
                <Link href="/model" className="underline">Model</Link>{" "}
                dashboard reports &quot;Not available&quot; until then, along with the
                connected checkpoint&apos;s architecture, class-mapping status, and the
                reason predictions are refused when they are.
              </p>
            )}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="rounded-2xl bg-forest-800 px-8 py-14 text-center text-white dark:bg-forest-900">
          <h2 className="text-3xl font-bold tracking-tight">Ready to analyze a leaf?</h2>
          <p className="mx-auto mt-4 max-w-xl text-forest-100">
            Upload an image or use your device camera. The platform will validate
            the image, send it to the backend, and either present a real prediction
            with confidence context, or state plainly why no prediction can be
            produced. Either way you are told which of the two happened.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link
              href="/scan"
              className="inline-flex items-center gap-2 rounded-md bg-white px-6 py-3 text-sm font-semibold text-forest-800 hover:bg-forest-50"
            >
              <ScanLine className="h-4 w-4" aria-hidden />
              Scan a Leaf
            </Link>
            <Link
              href="/about"
              className="inline-flex items-center gap-2 rounded-md border border-forest-600 px-6 py-3 text-sm font-semibold text-forest-100 hover:bg-forest-700"
            >
              About the Project
            </Link>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
        <Disclaimer />
      </div>
    </>
  );
}
