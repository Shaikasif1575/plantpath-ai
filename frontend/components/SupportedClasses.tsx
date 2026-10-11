"use client";

import { AlertCircle, Loader2 } from "lucide-react";
import { useApp } from "@/components/AppProvider";

/**
 * The model's supported disease/class list, always sourced from the API:
 * `supported_classes` on a predict response when available, otherwise the
 * `classes` reported by GET /api/model-info.
 *
 * Shows explicit loading / unavailable states — it never falls back to a
 * hard-coded class list.
 */
export function SupportedClasses({
  fromPrediction,
  title = "Diseases This Model Can Detect",
  className = "",
}: {
  fromPrediction?: string[] | null;
  title?: string;
  className?: string;
}) {
  const { modelInfo, loading, modelState } = useApp();

  const predicted = Array.isArray(fromPrediction)
    ? fromPrediction.filter((c): c is string => typeof c === "string" && c.length > 0)
    : [];
  const fromModel = modelInfo?.class_names?.length ? modelInfo.class_names : [];
  const classes = predicted.length > 0 ? predicted : fromModel;
  const source =
    predicted.length > 0
      ? "POST /api/predict · supported_classes"
      : "GET /api/model-info · classes";

  const status: "loading" | "ready" | "unavailable" | "unknown" =
    classes.length > 0
      ? "ready"
      : loading
        ? "loading"
        : modelState === "unreachable"
          ? "unknown"
          : "unavailable";

  return (
    <section
      className={`rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6 ${className}`}
      aria-label="Model supported classes"
    >
      <h2 className="flex items-center gap-2 text-lg font-bold tracking-tight">
        {title}
      </h2>
      <p className="mt-1 text-xs text-[var(--pp-muted)]">
        Reported by the connected API — this is the model&apos;s actual supported
        set, not a fixed reference list.
      </p>

      <div className="mt-4">
        {status === "loading" ? (
          <p
            className="flex items-center gap-2 text-sm text-[var(--pp-muted)]"
            role="status"
            aria-live="polite"
          >
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            Loading supported classes…
          </p>
        ) : status === "unavailable" || status === "unknown" ? (
          <div
            className="flex items-start gap-2 rounded-lg border border-dashed border-[var(--pp-border)] bg-[var(--pp-bg)] p-4 text-sm text-[var(--pp-muted)]"
            role="status"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {status === "unknown" ? (
              <span>
                Model classes are unknown — the API could not be reached, so no
                class list can be reported. Nothing is assumed in its place.
              </span>
            ) : (
              <span>
                No model classes are connected yet. The API reported no class
                mapping, so there is no supported set to list — the disease
                library on the{" "}
                <a href="/diseases" className="underline">
                  Diseases
                </a>{" "}
                page is an educational reference library, not this
                model&apos;s supported set.
              </span>
            )}
          </div>
        ) : (
          <>
            <p className="text-xs text-[var(--pp-muted)]" aria-live="polite">
              {classes.length} class{classes.length === 1 ? "" : "es"} reported
            </p>
            <ul className="mt-3 flex flex-wrap gap-1.5">
              {classes.map((c, i) => (
                <li
                  key={`${c}-${i}`}
                  className="rounded border border-[var(--pp-border)] bg-[var(--pp-bg)] px-2 py-1 font-mono text-[11px]"
                >
                  {c}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      {status === "ready" ? (
        <p className="mt-3 text-[11px] text-[var(--pp-muted)]">Source: {source}</p>
      ) : null}
    </section>
  );
}
