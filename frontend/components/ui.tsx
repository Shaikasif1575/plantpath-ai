"use client";

import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  Info,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";

/** Verbatim status strings. These are the only claims the UI is allowed to make. */
export const STATUS_LIVE = "LIVE MODEL";
export const STATUS_DEMO = "Demo mode — predictions are not real";
export const STATUS_UNAVAILABLE = "Model not available — no predictions can be generated";
export const STATUS_UNREACHABLE =
  "Backend unreachable — model state unknown, no predictions can be generated";

const pillBase =
  "inline-flex max-w-full items-start gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold leading-snug";

export function DemoBadge({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        pillBase,
        "border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-200",
        className,
      )}
      title={STATUS_DEMO}
    >
      <AlertTriangle className="mt-px h-3 w-3 shrink-0" aria-hidden />
      <span>{STATUS_DEMO}</span>
    </span>
  );
}

export function LiveModelBadge({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        pillBase,
        "border-emerald-500/40 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200",
        className,
      )}
      title="A verified model is loaded — predictions come from the loaded checkpoint"
    >
      <CheckCircle2 className="mt-px h-3 w-3 shrink-0" aria-hidden />
      <span>{STATUS_LIVE}</span>
    </span>
  );
}

export function ModelUnavailableBadge({
  className,
  label = STATUS_UNAVAILABLE,
}: {
  className?: string;
  label?: string;
}) {
  return (
    <span
      className={cn(
        pillBase,
        "border-rose-500/40 bg-rose-500/10 text-rose-800 dark:text-rose-200",
        className,
      )}
      title="Live mode, but no verified model is connected — predictions are unavailable rather than simulated"
    >
      <AlertCircle className="mt-px h-3 w-3 shrink-0" aria-hidden />
      <span>{label}</span>
    </span>
  );
}

export function BackendUnreachableBadge({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        pillBase,
        "border-[var(--pp-border)] bg-[var(--pp-bg)] text-[var(--pp-muted)]",
        className,
      )}
      title="The model-info endpoint could not be reached, so the model state is unknown"
    >
      <AlertCircle className="mt-px h-3 w-3 shrink-0" aria-hidden />
      <span>{STATUS_UNREACHABLE}</span>
    </span>
  );
}

/** Monospace block for the backend's own wording — surfaced verbatim, never paraphrased. */
export function BackendReason({ children }: { children: string }) {
  return (
    <pre
      className="mt-2 max-w-full overflow-x-auto whitespace-pre-wrap break-words rounded border border-[var(--pp-border)] bg-[var(--pp-bg)] p-2 font-mono text-[11px] leading-relaxed text-[var(--pp-muted)]"
      aria-label="Reason reported by the backend"
    >
      {children}
    </pre>
  );
}


export function Disclaimer({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "rounded-lg border border-forest-200 bg-forest-50 p-4 text-sm leading-relaxed text-forest-900 dark:border-forest-800 dark:bg-forest-900/40 dark:text-forest-100",
        className,
      )}
      role="note"
    >
      <p>
        <strong>Research disclaimer.</strong> PlantPath AI is a research and
        educational prototype. Predictions are limited to the classes represented
        by the connected model and should not be treated as definitive
        agricultural diagnosis. Field performance may differ from
        controlled-dataset performance (e.g. PlantVillage). Confidence scores
        reflect the model&apos;s probability distribution only.
      </p>
    </div>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
      <Loader2 className="h-8 w-8 animate-spin text-forest-600 dark:text-leaf-400" aria-hidden />
      <p className="text-sm text-[var(--pp-muted)]">{label}</p>
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      className="rounded-lg border border-rose-200 bg-rose-50 p-6 text-center dark:border-rose-900 dark:bg-rose-950/30"
      role="alert"
    >
      <AlertCircle className="mx-auto h-8 w-8 text-rose-600 dark:text-rose-400" aria-hidden />
      <h3 className="mt-3 text-base font-semibold text-rose-900 dark:text-rose-100">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-rose-800 dark:text-rose-200">{message}</p>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-md bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-700"
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-[var(--pp-border)] bg-[var(--pp-surface)] p-10 text-center">
      <Info className="mx-auto h-8 w-8 text-[var(--pp-muted)]" aria-hidden />
      <h3 className="mt-3 text-base font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-[var(--pp-muted)]">{description}</p>
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function QualityWarningList({
  warnings,
}: {
  warnings: { code: string; message: string; severity: string }[];
}) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div className="space-y-2" role="status" aria-live="polite">
      {warnings.map((w, i) => (
        <div
          key={`${w.code}-${i}`}
          className={cn(
            "flex items-start gap-2 rounded-md border px-3 py-2 text-sm",
            w.severity === "warning"
              ? "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100"
              : "border-forest-200 bg-forest-50 text-forest-800 dark:border-forest-800 dark:bg-forest-900/30 dark:text-forest-100",
          )}
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>{w.message}</span>
        </div>
      ))}
    </div>
  );
}

export function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number | null | undefined;
  hint?: string;
}) {
  const display =
    value === null || value === undefined || value === ""
      ? "Not available"
      : typeof value === "number"
        ? Number.isInteger(value)
          ? value.toLocaleString("en-US")
          : value.toFixed(4)
        : String(value);
  const missing = value === null || value === undefined || value === "";
  return (
    <div className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
        {label}
      </div>
      <div
        className={cn(
          "mt-1 font-mono text-xl font-semibold",
          missing ? "text-[var(--pp-muted)]" : "text-[var(--pp-text)]",
        )}
      >
        {display}
      </div>
      {hint ? <div className="mt-1 text-xs text-[var(--pp-muted)]">{hint}</div> : null}
    </div>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="max-w-3xl">
      {eyebrow ? (
        <div className="text-xs font-semibold uppercase tracking-widest text-forest-600 dark:text-leaf-400">
          {eyebrow}
        </div>
      ) : null}
      <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">{title}</h2>
      {description ? (
        <p className="mt-3 text-base leading-relaxed text-[var(--pp-muted)]">{description}</p>
      ) : null}
    </div>
  );
}

export function SuccessNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-100">
      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>{children}</div>
    </div>
  );
}
