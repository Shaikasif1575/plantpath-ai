"use client";

import Link from "next/link";
import {
  Target,
  Search,
  FlaskConical,
  Database,
  Brain,
  Microscope,
  Layers,
  Rocket,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  LucideIcon,
} from "lucide-react";
import { SectionHeading, Disclaimer } from "@/components/ui";

const PIPELINE = [
  { icon: Database, title: "Dataset", desc: "PlantVillage controlled leaf images with species/disease labels." },
  { icon: FlaskConical, title: "Preprocessing", desc: "Resize, channel conversion, and normalisation matching the trained model contract." },
  { icon: Layers, title: "Augmentation", desc: "Training-time transformations to improve robustness within the dataset domain." },
  { icon: Brain, title: "Transfer Learning", desc: "Pretrained CNN backbone adapted to the plant-disease classification task." },
  { icon: Target, title: "Classification", desc: "Softmax over trained disease classes with calibrated probability reporting." },
  { icon: Microscope, title: "Explainability", desc: "Grad-CAM attention maps when the architecture supports gradient-based attribution." },
  { icon: Layers, title: "Severity", desc: "Segmentation-based severity estimation from project-trained U-Nets (never derived from classification confidence)." },
  { icon: Rocket, title: "Deployment", desc: "Local FastAPI inference with optional Docker packaging — no proprietary cloud AI." },
];

const SECTIONS: { icon: LucideIcon; title: string; body: React.ReactNode }[] = [
  {
    icon: Search,
    title: "Problem Statement",
    body: (
      <>
        <p>
          Early and accurate plant-disease identification supports timely
          intervention and reduces unnecessary chemical use. Deep learning can
          classify leaf images into disease categories, but research prototypes
          must communicate what models can and cannot claim — especially when
          trained on controlled datasets.
        </p>
      </>
    ),
  },
  {
    icon: Target,
    title: "Research Gap",
    body: (
      <>
        <p>
          Many public plant-disease classifiers report high accuracy on
          controlled benchmarks (e.g. PlantVillage) without transparent
          explainability, severity estimation, or explicit out-of-distribution
          limits. Field imagery introduces lighting, occlusion, mixed infections,
          and novel pathogens that benchmarks do not capture.
        </p>
      </>
    ),
  },
  {
    icon: Target,
    title: "Objectives",
    body: (
      <ul className="list-disc space-y-1 pl-5">
        <li>Build a working leaf-image classification pipeline with local inference.</li>
        <li>Integrate Grad-CAM for model interpretability where architecture permits.</li>
        <li>Surface model metadata, class mapping, and evaluation honestly.</li>
        <li>Implement segmentation-based severity (U-Net pair) with honest attribution; keep OOD detection as an extension point.</li>
        <li>Deliver a professional research-prototype web application.</li>
      </ul>
    ),
  },
  {
    icon: FlaskConical,
    title: "Methodology",
    body: (
      <p>
        Transfer learning from a pretrained convolutional backbone, fine-tuned on
        the PlantVillage class set. Inference runs locally through PyTorch. The
        application layer separates API routes, services,
        ML, and persistence so future modules (OOD detector)
        can be integrated without rewriting the product.
      </p>
    ),
  },
  {
    icon: Database,
    title: "Dataset",
    body: (
      <>
        <p>
          PlantVillage is a widely used controlled dataset of healthy and diseased
          leaf images across multiple crop species. It enables reproducible
          benchmarking but does not represent field photography conditions.
        </p>
        <p className="mt-2 text-sm text-[var(--pp-muted)]">
          Reference: Mohanty, Hughes, &amp; Salathé (2016),{" "}
          <em>Frontiers in Plant Science</em>.
        </p>
      </>
    ),
  },
  {
    icon: Brain,
    title: "Model Architecture",
    body: (
      <p>
        A convolutional neural network adapted via transfer learning (e.g.
        MobileNetV2 family) with a softmax classification head sized to the
        project&apos;s class mapping. Input resolution and preprocessing must
        match the trained weights; the Model page reflects whatever metadata
        exists in project files.
      </p>
    ),
  },
  {
    icon: BarChart3,
    title: "Evaluation Strategy",
    body: (
      <p>
        Accuracy, precision, recall, and F1 are reported only when generated from
        actual evaluation runs. Confusion matrices and class-wise metrics are
        optional artifacts. Controlled-dataset metrics are not presented as
        field-performance guarantees.
      </p>
    ),
  },
  {
    icon: Microscope,
    title: "Explainable AI",
    body: (
      <p>
        Grad-CAM computes gradient-weighted class activation maps over convolutional
        feature maps. Heatmaps are presented as interpretability aids. They do not
        prove biological causation and cannot substitute for expert diagnosis.
      </p>
    ),
  },
  {
    icon: AlertTriangle,
    title: "Real-World Limitations",
    body: (
      <>
        <p>
          The <strong>laboratory-to-field gap</strong> is central to this
          project&apos;s honesty. Controlled datasets use consistent backgrounds,
          lighting, and single-disease labels. Field images may contain multiple
          stresses, soil residue, variable focus, and diseases absent from training.
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-5">
          <li>Unknown diseases cannot be recognised by a fixed-class classifier.</li>
          <li>Confidence calibration on field data is not established here.</li>
          <li>Severity requires segmentation, not classification probability.</li>
          <li>Environmental and cultivar variation affect visual symptoms.</li>
        </ul>
      </>
    ),
  },
  {
    icon: Rocket,
    title: "Future Work",
    body: (
      <ul className="list-disc space-y-1 pl-5">
        <li>Human-annotated lesion masks to validate severity estimates against real labels</li>
        <li>Plant / non-plant validation classifier</li>
        <li>Out-of-distribution detector (e.g. energy-based or ensemble methods)</li>
        <li>Larger, community-labelled field evaluation set beyond Wikimedia Commons</li>
        <li>Multi-model comparison (ResNet50, EfficientNet, ONNX/TFLite)</li>
        <li>Batch analysis and mobile deployment</li>
      </ul>
    ),
  },
];

export default function ResearchPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Academic Project"
        title="Research"
        description="PlantPath AI: Explainable Deep Learning for Plant Disease Detection, Classification and Severity Assessment."
      />

      <div className="mt-8 rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
        <h2 className="text-lg font-bold">Research Pipeline</h2>
        <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map((step, i) => {
            const Icon = step.icon;
            return (
              <li key={step.title} className="relative rounded-lg border border-[var(--pp-border)] p-4">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-forest-700 text-xs font-bold text-white">
                    {i + 1}
                  </span>
                  <Icon className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
                </div>
                <div className="mt-2 text-sm font-semibold">{step.title}</div>
                <p className="mt-1 text-xs text-[var(--pp-muted)]">{step.desc}</p>
              </li>
            );
          })}
        </ol>
      </div>

      <div className="mt-8 space-y-6">
        {SECTIONS.map((s) => {
          const Icon = s.icon;
          return (
            <section
              key={s.title}
              className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6"
            >
              <h2 className="flex items-center gap-2 text-lg font-bold">
                <Icon className="h-5 w-5 text-forest-600 dark:text-leaf-400" aria-hidden />
                {s.title}
              </h2>
              <div className="mt-3 text-sm leading-relaxed text-[var(--pp-muted)]">
                {s.body}
              </div>
            </section>
          );
        })}
      </div>

      <div className="mt-8 rounded-xl border-2 border-forest-300 bg-forest-50 p-6 dark:border-forest-700 dark:bg-forest-900/30">
        <h2 className="flex items-center gap-2 text-lg font-bold text-forest-900 dark:text-forest-50">
          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" aria-hidden />
          Laboratory-to-Field Gap
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-forest-800 dark:text-forest-100">
          PlantVillage and similar controlled datasets capture leaves under
          laboratory-like conditions: uniform backgrounds, consistent lighting,
          and clear single-disease labels. Real agricultural fields present
          different lighting, occlusion by other leaves, soil and debris,
          multiple co-occurring stresses, and pathogen strains that may be
          visually novel. Performance measured on a controlled benchmark must not
          be represented as equivalent to field-deployment performance.
        </p>
        <div className="mt-4">
          <Link
            href="/model"
            className="inline-flex items-center gap-2 text-sm font-semibold text-forest-800 underline dark:text-leaf-300"
          >
            View model dashboard
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        </div>
      </div>

      <div className="mt-10">
        <Disclaimer />
      </div>
    </div>
  );
}
