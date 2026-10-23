"use client";

import { DiseaseInfo } from "@/lib/api";
import {
  AlertTriangle,
  FlaskConical,
  Leaf,
  ListChecks,
  ShieldCheck,
} from "lucide-react";

/**
 * Treatment + pesticide guidance for the predicted disease, straight from the
 * curated knowledge base. Renders nothing when the backend found no confident
 * record match — we never show another disease's advice.
 */
export function TreatmentPanel({ info }: { info?: DiseaseInfo | null }) {
  if (!info) return null;

  const treatment = info.treatment ?? [];
  const pesticides = info.pesticides ?? [];
  const isHealthy = info.condition === "healthy";

  return (
    <div
      className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6"
      data-testid="treatment-panel"
    >
      <h3 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
        <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-gradient-to-br from-[var(--pp-forest)] to-[var(--pp-leaf)] text-white shadow-sm" aria-hidden>
          <Leaf className="h-3 w-3" />
        </span>
        Treatment &amp; Control
      </h3>

      {isHealthy ? (
        <p className="mt-3 flex items-start gap-2 text-sm text-[var(--pp-muted)]">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-forest-600 dark:text-leaf-400" aria-hidden />
          No disease was detected, so no treatment or pesticides are indicated.
          Keep monitoring the crop and continue good cultural practices
          (clean seed, rotation, sanitation, balanced nutrition).
        </p>
      ) : (
        <>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="font-medium">
              {info.crop} — {info.disease}
            </span>
            {info.scientific_name ? (
              <span className="text-xs italic text-[var(--pp-muted)]">
                {info.scientific_name}
              </span>
            ) : null}
          </div>

          {info.incurable ? (
            <div
              className="mt-3 flex items-start gap-2 rounded-lg border border-amber-400 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100"
              role="note"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <span>
                No chemical cure exists for this disease. The steps below limit
                spread and protect healthy plants — remove and destroy infected
                plants where recommended, and manage the insect vector where one
                exists.
              </span>
            </div>
          ) : null}

          {treatment.length > 0 ? (
            <section className="mt-4">
              <h4 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
                <ListChecks className="h-4 w-4" aria-hidden />
                How to Treat It
              </h4>
              <ol className="mt-2 space-y-1.5 text-sm">
                {treatment.map((s, i) => (
                  <li key={s} className="flex gap-2">
                    <span className="mt-0.5 w-4 shrink-0 text-right font-mono text-xs text-[var(--pp-muted)]">
                      {i + 1}.
                    </span>
                    <span>{s}</span>
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          {pesticides.length > 0 ? (
            <section className="mt-5">
              <h4 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
                <FlaskConical className="h-4 w-4" aria-hidden />
                Recommended Pesticides
              </h4>
              <ul className="mt-3 space-y-3">
                {pesticides.map((p) => (
                  <li
                    key={p.category}
                    className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-bg)] p-4"
                  >
                    <div className="text-sm font-semibold">{p.category}</div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(p.active_ingredients ?? []).map((ing) => (
                        <span
                          key={ing}
                          className="inline-flex items-center rounded-full border border-[var(--pp-border-subtle)] bg-[rgba(107,168,110,0.08)] px-2.5 py-0.5 text-[11px] font-semibold text-[var(--pp-leaf)]"
                        >
                          {ing}
                        </span>
                      ))}
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-[var(--pp-muted)]">
                      {p.application}
                    </p>
                  </li>
                ))}
              </ul>
            </section>
          ) : treatment.length > 0 ? (
            <p className="mt-4 text-sm text-[var(--pp-muted)]">
              No pesticide is recommended for this result — the control measures
              above (sanitation, hygiene, vector management) are the effective
              approach.
            </p>
          ) : null}

          <div className="mt-5 rounded-lg border border-amber-300 bg-amber-50/70 p-3 text-xs leading-relaxed text-amber-900 dark:border-amber-700 dark:bg-amber-950/20 dark:text-amber-100">
            <AlertTriangle className="mr-1 inline h-3.5 w-3.5" aria-hidden />
            {info.disclaimer ??
              "Educational guidance only. Always follow the product label and consult your local agricultural extension service before applying any pesticide."}
          </div>

          {(info.sources ?? []).length > 0 ? (
            <p className="mt-3 text-xs text-[var(--pp-muted)]">
              Sources: {(info.sources ?? []).join(" · ")}
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}
