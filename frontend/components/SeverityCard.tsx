"use client";

import { SeverityResult } from "@/lib/api";
import { Layers, Info } from "lucide-react";

export function SeverityCard({
  severity,
}: {
  severity: SeverityResult | null | undefined;
}) {
  if (!severity || !severity.available) {
    const reason =
      severity?.unavailable_reason || severity?.reason || severity?.message || null;
    return (
      <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
        <h3 className="flex items-center gap-2 font-semibold">
          <Layers className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
          Severity Assessment
        </h3>
        <div className="mt-4 rounded-lg border border-dashed border-[var(--pp-border)] bg-[var(--pp-bg)] p-8 text-center">
          <Info className="mx-auto h-8 w-8 text-[var(--pp-muted)]" aria-hidden />
          <p className="mt-3 text-sm font-medium">
            Severity assessment is currently unavailable for this model.
          </p>
          {reason ? (
            <p className="mx-auto mt-1 max-w-sm text-xs text-[var(--pp-muted)]">
              {reason}
            </p>
          ) : null}
          <p className="mx-auto mt-3 max-w-sm text-xs text-[var(--pp-muted)]">
            Severity is never derived from classification confidence, so no
            percentage or severity level is shown. It will only appear once a
            validated segmentation model is connected.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
      <h3 className="flex items-center gap-2 font-semibold">
        <Layers className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
        Severity Assessment
      </h3>
      <dl className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
            Affected Leaf Area
          </dt>
          <dd className="mt-1 font-mono text-2xl font-bold">
            {severity.affected_area_percent != null
              ? `${severity.affected_area_percent}%`
              : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-[var(--pp-muted)]">
            Severity Level
          </dt>
          <dd className="mt-1 text-2xl font-bold">{severity.severity_level || "—"}</dd>
        </div>
      </dl>
      {severity.method ? (
        <p className="mt-3 text-xs text-[var(--pp-muted)]">Method: {severity.method}</p>
      ) : null}
      {(severity.original_image || severity.mask_image) && (
        <div className="mt-4 grid grid-cols-2 gap-2">
          {severity.original_image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={severity.original_image}
              alt="Original image for severity"
              className="aspect-square w-full rounded-md border object-cover"
            />
          ) : null}
          {severity.mask_image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={severity.mask_image}
              alt="Segmentation mask overlay"
              className="aspect-square w-full rounded-md border object-cover"
            />
          ) : null}
        </div>
      )}
    </div>
  );
}
