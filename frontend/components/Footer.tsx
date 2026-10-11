import Link from "next/link";
import { Leaf } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-[var(--pp-border)] bg-[var(--pp-surface)]">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid gap-8 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2 font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-forest-700 text-white">
                <Leaf className="h-3.5 w-3.5" aria-hidden />
              </span>
              PlantPath AI
            </div>
            <p className="mt-3 text-sm leading-relaxed text-[var(--pp-muted)]">
              Explainable deep learning for plant disease detection from leaf
              images. A research and educational prototype.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Explore
            </h3>
            <ul className="mt-3 space-y-2 text-sm">
              <li><Link className="hover:text-forest-600" href="/scan">Scan Leaf</Link></li>
              <li><Link className="hover:text-forest-600" href="/diseases">Disease Library</Link></li>
              <li><Link className="hover:text-forest-600" href="/model">Model Dashboard</Link></li>
              <li><Link className="hover:text-forest-600" href="/research">Research</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Disclaimer
            </h3>
            <p className="mt-3 text-sm leading-relaxed text-[var(--pp-muted)]">
              PlantPath AI is a research and educational prototype. Predictions
              are limited to the classes represented by the connected model and
              should not be treated as definitive agricultural diagnosis. Field
              performance may differ from controlled-dataset performance.
            </p>
          </div>
        </div>
        <div className="mt-8 border-t border-[var(--pp-border)] pt-6 text-xs text-[var(--pp-muted)]">
          © {new Date().getFullYear()} PlantPath AI — B.Tech Computer Science
          research prototype. No paid AI APIs are used.
        </div>
      </div>
    </footer>
  );
}
