"use client";

import { GradCamResult } from "@/lib/api";
import { Microscope, Info } from "lucide-react";

export function GradCamViewer({
  explainability,
}: {
  explainability: GradCamResult | null | undefined;
}) {
  if (!explainability || !explainability.available) {
    return (
      <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
        <h3 className="flex items-center gap-2 font-semibold">
          <Microscope className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
          Explainable AI / Grad-CAM
        </h3>
        <div className="mt-4 rounded-lg border border-dashed border-[var(--pp-border)] bg-[var(--pp-bg)] p-8 text-center">
          <Info className="mx-auto h-8 w-8 text-[var(--pp-muted)]" aria-hidden />
          <p className="mt-3 text-sm font-medium">
            Explainability unavailable for this model.
          </p>
          <p className="mx-auto mt-1 max-w-sm text-xs text-[var(--pp-muted)]">
            {explainability?.message ||
              "Grad-CAM requires a loaded convolutional model. Connect a trained model to enable attention maps."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
      <h3 className="flex items-center gap-2 font-semibold">
        <Microscope className="h-4 w-4 text-forest-600 dark:text-leaf-400" aria-hidden />
        AI Attention / Grad-CAM
      </h3>
      <p className="mt-1 text-xs text-[var(--pp-muted)]">
        Target class:{" "}
        <span className="font-medium text-[var(--pp-text)]">
          {explainability.target_class || "top prediction"}
        </span>
      </p>
      <div className="mt-4 grid grid-cols-3 gap-2">
        {[
          { src: explainability.original_image, label: "Original" },
          { src: explainability.heatmap, label: "Heatmap" },
          { src: explainability.overlay, label: "Overlay" },
        ].map((img) => (
          <figure key={img.label} className="space-y-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={img.src || ""}
              alt={`${img.label} for Grad-CAM visualization`}
              className="aspect-square w-full rounded-md border border-[var(--pp-border)] object-cover"
            />
            <figcaption className="text-center text-[11px] text-[var(--pp-muted)]">
              {img.label}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-4 text-xs leading-relaxed text-[var(--pp-muted)]">
        Highlighted regions indicate image areas that contributed strongly to the
        model prediction. This visualization is an interpretability aid and does
        not establish biological causation.
      </p>
    </div>
  );
}
