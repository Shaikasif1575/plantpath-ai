"use client";

import { useApp } from "@/components/AppProvider";
import {
  BackendReason,
  BackendUnreachableBadge,
  DemoBadge,
  LiveModelBadge,
  ModelUnavailableBadge,
} from "@/components/ui";

/**
 * Global model-status badge. Source of truth is GET /api/model-info
 * (fetched by AppProvider); the state is derived, never guessed.
 *
 * - demo_mode=true                    → amber   "Demo mode — predictions are not real"
 * - demo_mode=false, model loaded     → green   "LIVE MODEL"
 * - demo_mode=false, model not loaded → rose    "Model not available — no predictions can be generated"
 * - backend unreachable               → neutral "Backend unreachable — model state unknown…"
 * - still loading                     → renders nothing, never guesses
 */
export function ModelStateBadge({ className }: { className?: string }) {
  const { modelState } = useApp();

  switch (modelState) {
    case "demo":
      return <DemoBadge className={className} />;
    case "live":
      return <LiveModelBadge className={className} />;
    case "unavailable":
      return <ModelUnavailableBadge className={className} />;
    case "unreachable":
      return <BackendUnreachableBadge className={className} />;
    default:
      return null;
  }
}

/**
 * Full-width explanation shown whenever predictions cannot be produced.
 * Renders the backend's own reason verbatim — it never substitutes a guess
 * and never names a disease.
 */
export function ModelUnavailableNotice({
  reason,
  className,
}: {
  reason?: string | null;
  className?: string;
}) {
  const { modelState, modelInfo, health, error } = useApp();
  if (modelState !== "unavailable" && modelState !== "unreachable") return null;

  const text =
    reason ??
    modelState === "unreachable"
      ? error
      : (modelInfo?.error ?? modelInfo?.status ?? health?.detail ?? null);

  return (
    <section
      className={`rounded-lg border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-100 ${className || ""}`}
      role="alert"
      aria-label="Model availability notice"
    >
      <p className="font-semibold">Why no prediction can be produced</p>
      <p className="mt-1">
        {modelState === "unreachable"
          ? "The model-info endpoint could not be reached, so the model state is unknown. Nothing is assumed in its place."
          : "No verified model is connected, so the backend refuses to serve predictions. Nothing is shown here because no result exists — a result would have to be invented."}
      </p>
      {text ? <BackendReason>{text}</BackendReason> : null}
      <p className="mt-2 text-xs">
        Uploads are still accepted and validated, so the image pipeline can still
        be checked. While the model is unavailable, no disease name, confidence,
        severity, or attention map is displayed anywhere in the app.
      </p>
    </section>
  );
}
