"use client";

import { PredictionResponse, PredictionItem, LOW_CONFIDENCE_FALLBACK } from "@/lib/api";
import {
  confidenceBandColor,
  confidenceLabel,
  formatPercent,
} from "@/lib/utils";
import { GradCamViewer } from "@/components/GradCamViewer";
import { SeverityCard } from "@/components/SeverityCard";
import { ReportButton } from "@/components/ReportButton";
import { QualityWarningList } from "@/components/ui";
import { ModelStateBadge } from "@/components/ModelStatus";
import { Activity, Brain, ShieldAlert, Stethoscope } from "lucide-react";

function Bar({ p, rank }: { p: number; rank: number }) {
  return (
    <div
      className={`bar-grow h-3 rounded-sm ${confidenceBandColor(rank === 1 ? "high" : rank === 2 ? "medium" : "low")}`}
      style={{ width: `${Math.max(p * 100, 2)}%` }}
      role="img"
      aria-label={`${(p * 100).toFixed(1)} percent`}
    />
  );
}

function LowConfidenceTag() {
  return (
    <span className="inline-flex shrink-0 items-center rounded-full border border-rose-300 bg-rose-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-rose-700 dark:border-rose-700 dark:bg-rose-900/50 dark:text-rose-200">
      Low confidence
    </span>
  );
}

export function PredictionResult({ result }: { result: PredictionResponse }) {
  const top = result.top_prediction ?? null;
  const topPredictions = Array.isArray(result.top_predictions)
    ? result.top_predictions
    : [];
  const band = result.confidence_band ?? "";
  const isDemo = result.demo_mode === true || result.model?.demo_mode === true;
  const lowConfidence =
    top !== null && result.low_confidence === true;
  const lowConfidenceMessage =
    (result.low_confidence_message || "").trim() || LOW_CONFIDENCE_FALLBACK;
  const hasPrediction = top !== null && topPredictions.length > 0;

  return (
    <div className="space-y-6">
      {isDemo ? (
        <div
          className="rounded-lg border border-amber-400 bg-amber-50 p-4 dark:border-amber-700 dark:bg-amber-950/30"
          role="status"
        >
          <div className="flex items-center gap-2 font-semibold text-amber-900 dark:text-amber-100">
            <ShieldAlert className="h-5 w-5 shrink-0" aria-hidden />
            DEMO MODE
          </div>
          <p className="mt-2 text-sm font-medium text-amber-800 dark:text-amber-200">
            Demo predictions — not a real diagnosis
          </p>
          <p className="mt-1 text-sm text-amber-800 dark:text-amber-200">
            No trained model is connected in this environment.
            {hasPrediction
              ? " The values below are simulated for interface demonstration only and must not be interpreted as plant-disease diagnosis."
              : " The backend returned no prediction, so none is displayed here."}
          </p>
        </div>
      ) : null}

      {lowConfidence ? (
        <div
          className="rounded-lg border border-rose-400 bg-rose-50 p-4 dark:border-rose-700 dark:bg-rose-950/30"
          role="alert"
        >
          <div className="flex items-center gap-2 font-semibold text-rose-900 dark:text-rose-100">
            <ShieldAlert className="h-5 w-5 shrink-0" aria-hidden />
            Low-confidence prediction
          </div>
          <p className="mt-2 text-sm font-medium text-rose-800 dark:text-rose-200">
            {lowConfidenceMessage}
          </p>
        </div>
      ) : null}

      <div className="rounded-xl border border-forest-300 bg-gradient-to-b from-forest-50 to-white p-6 dark:border-forest-700 dark:from-forest-900/40 dark:to-[var(--pp-surface)]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-lg font-bold tracking-tight">
            <Activity className="h-5 w-5 shrink-0 text-forest-600 dark:text-leaf-400" aria-hidden />
            {hasPrediction ? "Analysis Complete" : "No Prediction Returned"}
          </h2>
          <ModelStateBadge className="sm:max-w-[16rem]" />
        </div>

        {hasPrediction ? (
          <>
            <dl className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">Crop</dt>
                <dd className="mt-1 text-xl font-bold">{result.crop || top.crop}</dd>
              </div>
              <div>
                <dt className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">Condition</dt>
                <dd className="mt-1 text-xl font-bold capitalize">{result.condition || top.disease}</dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
                  Detected Disease
                </dt>
                <dd className="mt-1 text-xl font-bold">
                  {result.condition === "healthy"
                    ? "Healthy (no disease class predicted)"
                    : result.detected_disease || result.disease || top.disease}
                </dd>
              </div>
            </dl>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
                    Top Prediction Confidence
                  </div>
                  {lowConfidence ? <LowConfidenceTag /> : null}
                </div>
                <div
                  className={`mt-1 font-mono text-3xl font-bold ${lowConfidence ? "text-[var(--pp-muted)]" : ""}`}
                >
                  {formatPercent(top.probability)}
                </div>
                <div className="mt-1 text-sm text-[var(--pp-muted)]">
                  {band ? confidenceLabel(band) : "Band not reported"}
                </div>
                <div
                  className={`mt-3 h-2 w-full overflow-hidden rounded-full bg-[var(--pp-bg)] ${confidenceBandColor(band)}`}
                  aria-hidden
                >
                  <div
                    className="h-full rounded-full bg-white/30"
                    style={{ width: `${top.probability * 100}%` }}
                  />
                </div>
              </div>
              <div className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
                  Model
                </div>
                <div className="mt-1 text-lg font-semibold">
                  {result.model?.name || "Model name not reported"}
                </div>
                <div className="mt-1 text-sm text-[var(--pp-muted)]">
                  {result.model?.architecture || "Architecture not reported"}
                </div>
                <div className="mt-2 text-xs text-[var(--pp-muted)]">
                  {result.model?.version ? `v${result.model.version}` : "Version not reported"}
                  {result.model?.framework ? ` · ${result.model.framework}` : ""}
                </div>
              </div>
            </div>

            {result.confidence_note ? (
              <p className="mt-4 text-xs leading-relaxed text-[var(--pp-muted)]">
                {result.confidence_note}
              </p>
            ) : null}
          </>
        ) : (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-dashed border-[var(--pp-border)] bg-[var(--pp-surface)] p-5 text-sm text-[var(--pp-muted)]">
            <Stethoscope className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>
              The backend accepted the image but returned no prediction, so no
              crop, disease, or confidence is shown. There is no value to display
              rather than a value that is hidden.
            </span>
          </div>
        )}
      </div>

      {/* Top predictions */}
      {hasPrediction ? (
        <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
          <h3 className="flex items-center gap-2 font-semibold">
            <Brain className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
            Top Predictions
          </h3>
          <p className="mt-1 text-xs text-[var(--pp-muted)]">
            Confidence represents the model&apos;s predicted probability distribution
            and does not guarantee diagnostic correctness.
          </p>
          <ul className="mt-5 space-y-4" aria-label="Ranked predictions">
            {topPredictions.map((p: PredictionItem) => (
              <li key={p.class_name} className={lowConfidence ? "opacity-80" : undefined}>
                <div className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="flex flex-wrap items-center gap-2 font-medium">
                    <span>
                      {p.rank}. {p.disease}
                      <span className="ml-2 text-xs text-[var(--pp-muted)]">{p.crop}</span>
                    </span>
                    {lowConfidence ? <LowConfidenceTag /> : null}
                  </span>
                  <span className="font-mono text-xs text-[var(--pp-muted)]">
                    {formatPercent(p.probability, 2)}
                  </span>
                </div>
                <div className="mt-1.5">
                  <Bar p={p.probability} rank={p.rank} />
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {/* Quality warnings */}
      {(result.validation?.warnings?.length ?? 0) > 0 ? (
        <div>
          <h3 className="mb-2 font-semibold">Image Quality Notes</h3>
          <QualityWarningList warnings={result.validation!.warnings} />
        </div>
      ) : null}

      {/* Grad-CAM + Severity (stacked: results column is already half-width) */}
      <div className="grid gap-6">
        <GradCamViewer explainability={result.explainability} />
        <SeverityCard severity={result.severity} />
      </div>

      {/* Limitations */}
      {(result.limitations || []).length > 0 ? (
        <div className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-5">
          <h3 className="font-semibold">Limitations of This Result</h3>
          <ul className="mt-3 space-y-2 text-sm text-[var(--pp-muted)]">
            {(result.limitations || []).map((l) => (
              <li key={l} className="flex gap-2">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-forest-500" aria-hidden />
                {l}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <ReportButton result={result} />
    </div>
  );
}
