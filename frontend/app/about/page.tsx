import Link from "next/link";
import { Leaf, ScanLine, Github, BookOpen } from "lucide-react";
import { SectionHeading, Disclaimer } from "@/components/ui";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="About"
        title="PlantPath AI"
        description="Explainable Deep Learning for Plant Disease Detection, Classification and Severity Assessment."
      />

      <div className="mt-8 space-y-6 text-sm leading-relaxed text-[var(--pp-muted)]">
        <p>
          PlantPath AI is a <strong>B.Tech Computer Science</strong> research
          prototype that demonstrates an end-to-end plant leaf analysis platform:
          image validation, local deep-learning inference, confidence
          communication, Grad-CAM explainability, disease knowledge base, and
          research documentation — built with free and open-source technologies.
        </p>
        <p>
          The project deliberately avoids proprietary cloud AI APIs. Inference
          runs locally from the connected PyTorch checkpoint. If that checkpoint is
          not verified, the backend refuses to answer and says why — it never
          substitutes a demonstration disease name for a real prediction.
        </p>
      </div>

      <section className="mt-8 rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
        <h2 className="text-lg font-bold">Technology Stack</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {[
            ["Frontend", "Next.js, TypeScript, Tailwind CSS, Lucide Icons, Recharts"],
            ["Backend", "Python, FastAPI, Uvicorn, Pydantic"],
            ["ML", "PyTorch, NumPy, Pillow, OpenCV"],
            ["Database", "SQLite"],
            ["Testing", "pytest, optional Playwright browser smoke tests"],
            ["Container", "Docker & Docker Compose"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg border border-[var(--pp-border)] p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
                {k}
              </div>
              <div className="mt-1 text-sm font-medium">{v}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-6 rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
        <h2 className="text-lg font-bold">Research Principles</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-[var(--pp-muted)]">
          <li>
            <strong>No fabricated results.</strong> Accuracy, precision, recall,
            F1, severity, and scientific claims are never invented.
          </li>
          <li>
            <strong>Confidence ≠ certainty.</strong> Model probabilities are
            presented as distributions, not diagnostic guarantees.
          </li>
          <li>
            <strong>Dataset honesty.</strong> PlantVillage is a controlled
            benchmark; field performance is not claimed from dataset metrics.
          </li>
          <li>
            <strong>Severity integrity.</strong> Severity is never derived from
            classification confidence; it requires a segmentation model.
          </li>
          <li>
            <strong>Explainability limits.</strong> Grad-CAM is an
            interpretability aid, not proof of biological causation.
          </li>
        </ul>
      </section>

      <section className="mt-6 rounded-xl border border-forest-300 bg-forest-50 p-6 dark:border-forest-700 dark:bg-forest-900/30">
        <h2 className="text-lg font-bold text-forest-900 dark:text-forest-50">Academic Disclaimer</h2>
        <p className="mt-3 text-sm leading-relaxed text-forest-800 dark:text-forest-100">
          PlantPath AI is a research and educational prototype. Predictions are
          limited to the classes represented by the connected model and should not
          be treated as definitive agricultural diagnosis. Field performance may
          differ from controlled-dataset performance.
        </p>
      </section>

      <section className="mt-6">
        <h2 className="text-lg font-bold">Documentation</h2>
        <div className="mt-3 flex flex-wrap gap-3">
          <Link
            href="/research"
            className="inline-flex items-center gap-2 rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-4 py-2 text-sm font-medium hover:bg-forest-50 dark:hover:bg-forest-900"
          >
            <BookOpen className="h-4 w-4" aria-hidden />
            Research Page
          </Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-4 py-2 text-sm font-medium hover:bg-forest-50 dark:hover:bg-forest-900"
          >
            API Docs (Swagger)
          </a>
          <Link
            href="/scan"
            className="inline-flex items-center gap-2 rounded-md bg-forest-700 px-4 py-2 text-sm font-semibold text-white hover:bg-forest-800"
          >
            <ScanLine className="h-4 w-4" aria-hidden />
            Scan a Leaf
          </Link>
        </div>
      </section>

      <div className="mt-10">
        <Disclaimer />
      </div>
    </div>
  );
}
