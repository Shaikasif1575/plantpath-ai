import { Cpu, BarChart3, History, GitCompare, Info, LucideIcon } from "lucide-react";
import { api, ModelInfo } from "@/lib/api";
import { BackendReason, MetricCard, ErrorState, SectionHeading, Disclaimer } from "@/components/ui";
import { cn } from "@/lib/utils";

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: LucideIcon;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
      <h2 className="flex items-center gap-2 text-lg font-bold">
        <Icon className="h-5 w-5 text-forest-600 dark:text-leaf-400" aria-hidden />
        {title}
      </h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

export const dynamic = "force-dynamic";

export default async function ModelPage() {
  let info: ModelInfo | null = null;
  let error: string | null = null;
  try {
    info = await api.modelInfo();
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load model info.";
  }

  if (error || !info) return <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6"><ErrorState message={error || "Model info unavailable."} /></div>;

  const metrics = (info.metrics || {}) as Record<string, unknown>;
  const training = (info.training || {}) as Record<string, unknown>;
  const limitations = info.limitations || [];
  const classNames = info.class_names || [];
  const comparison = info.comparison || [];
  const severity = (info.severity || {}) as Record<string, unknown>;
  const inputSize = info.input_size;
  const inputLabel = inputSize
    ? `${inputSize.join("×")}${info.input_channels ? `×${info.input_channels}` : ""}`
    : null;
  // Predictions are refused whenever the backend reports no verified model.
  const servesPredictions = info.demo_mode === true || info.loaded === true;
  const severityReason =
    (typeof severity.unavailable_reason === "string" && severity.unavailable_reason) ||
    (typeof severity.note === "string" && severity.note) ||
    null;

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Research Dashboard"
        title="Model"
        description="Architecture, evaluation, and training artifacts are shown only when present in project files. Missing values are reported as unavailable — never invented."
      />

      {!servesPredictions ? (
        <div
          className="mt-8 rounded-lg border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-100"
          role="alert"
          aria-label="Model availability notice"
        >
          <p className="font-semibold">Model not available — no predictions can be generated</p>
          <p className="mt-1">
            The backend loaded the checkpoint but refuses to serve predictions from
            it, so the scan page returns HTTP 503 instead of a result. Nothing below
            is inferred, and no accuracy, confidence, severity, or attention map is
            displayed.
          </p>
          {info.error ? <BackendReason>{info.error}</BackendReason> : null}
        </div>
      ) : null}

      <div className="mt-8 space-y-6">
        <Section icon={Cpu} title="Model Overview">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Name" value={info.name} />
            <MetricCard label="Version" value={info.version} />
            <MetricCard label="Format" value={info.model_format} />
            <MetricCard label="Framework" value={info.framework} />
            <MetricCard label="Architecture" value={info.architecture} />
            <MetricCard label="Input Size" value={inputLabel} hint={inputSize ? "height × width × channels" : undefined} />
            <MetricCard label="Parameters" value={info.parameters} />
            <MetricCard
              label="Model File"
              value={info.model_file ? info.model_file.split(/[\\/]/).pop() : null}
              hint={info.loaded ? "Loaded" : "Not loaded"}
            />
            <MetricCard label="Loaded" value={info.loaded ? "Yes" : "No"} />
            <MetricCard label="Demo Mode" value={info.demo_mode ? "Yes" : "No"} />
            <MetricCard
              label="Grad-CAM"
              value={info.gradcam_available ? "Available" : "Unavailable"}
              hint="requires a verified architecture against a resolved class set"
            />
            <MetricCard
              label="Segmentation"
              value={info.segmentation_status || (info.segmentation_available ? "Available" : "Unavailable")}
            />
          </div>
          {info.preprocessing ? (
            <p className="mt-5 text-sm text-[var(--pp-muted)]">
              <strong className="text-[var(--pp-text)]">Preprocessing:</strong>{" "}
              {info.preprocessing}
              {info.preprocessing_key ? ` (key: ${info.preprocessing_key})` : ""}
            </p>
          ) : null}
          {limitations.length > 0 ? (
            <div className="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/20">
              <div className="flex items-center gap-2 font-semibold text-amber-900 dark:text-amber-100">
                <Info className="h-4 w-4" aria-hidden />
                Limitations
              </div>
              <ul className="mt-2 space-y-1.5 text-sm text-amber-900 dark:text-amber-100">
                {limitations.map((l) => (
                  <li key={l}>• {l}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </Section>

        <Section icon={Info} title="Prediction Availability">
          <p className="text-sm text-[var(--pp-muted)]">
            {servesPredictions
              ? info.demo_mode
                ? "Demo mode is enabled: the backend returns clearly-labelled demonstration values, not model outputs."
                : "A verified model is loaded and the backend serves real predictions."
              : "The backend refuses to serve predictions. The reason it reports is shown verbatim below."}
          </p>
          {info.error ? <BackendReason>{info.error}</BackendReason> : null}
          <p className="mt-3 text-sm text-[var(--pp-muted)]">
            <strong className="text-[var(--pp-text)]">Severity:</strong>{" "}
            {severityReason || (info.segmentation_available ? "Available" : "Unavailable")}
            {typeof severity.requires === "string" ? ` — requires ${severity.requires}` : ""}
          </p>
        </Section>

        <Section icon={BarChart3} title="Performance">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Accuracy" value={metrics.accuracy as number | null} />
            <MetricCard label="Precision" value={metrics.precision as number | null} />
            <MetricCard label="Recall" value={metrics.recall as number | null} />
            <MetricCard label="F1 Score" value={metrics.f1 as number | null} />
          </div>
          <p className="mt-3 text-xs text-[var(--pp-muted)]">
            {metrics.note ? String(metrics.note) : "Metrics are only displayed when present in evaluation files."}
          </p>
        </Section>

        <Section icon={History} title="Training">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Training Accuracy" value={training.training_accuracy as number | null} />
            <MetricCard label="Validation Accuracy" value={training.validation_accuracy as number | null} />
            <MetricCard label="Training Loss" value={training.training_loss as number | null} />
            <MetricCard label="Validation Loss" value={training.validation_loss as number | null} />
          </div>
          <p className="mt-3 text-xs text-[var(--pp-muted)]">
            {training.note ? String(training.note) : "No training history file is connected."}
          </p>
        </Section>

        <Section icon={GitCompare} title="Model Comparison">
          <p className="mb-4 text-sm text-[var(--pp-muted)]">
            Architecture slots are reserved for multi-model comparison. Architectures
            without trained weights are marked accordingly.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--pp-border)] text-xs uppercase tracking-wide text-[var(--pp-muted)]">
                  <th className="py-2 pr-4 font-semibold">Architecture</th>
                  <th className="py-2 pr-4 font-semibold">Status</th>
                  <th className="py-2 font-semibold">Metrics</th>
                </tr>
              </thead>
              <tbody>
                {comparison.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="py-2.5 text-[var(--pp-muted)]">
                      No architectures are reported for comparison.
                    </td>
                  </tr>
                ) : null}
                {comparison.map((row, i) => (
                  <tr key={i} className="border-b border-[var(--pp-border)]">
                    <td className="py-2.5 pr-4 font-medium">
                      {String(row.architecture || "—")}
                    </td>
                    <td className="py-2.5 pr-4">
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-[11px] font-semibold",
                          row.status === "available"
                            ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100"
                            : row.status === "demo"
                              ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100"
                              : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
                        )}
                      >
                        {row.status === "available"
                          ? "Available"
                          : row.status === "demo"
                            ? "Demo only"
                            : "Not trained yet"}
                      </span>
                    </td>
                    <td className="py-2.5 text-[var(--pp-muted)]">
                      {row.metrics && Object.keys(row.metrics as object).length > 0
                        ? JSON.stringify(row.metrics)
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section icon={Info} title="Class Mapping">
          {classNames.length > 0 ? (
            <>
              <p className="text-sm text-[var(--pp-muted)]">
                {classNames.length} classes reported by the connected class-name
                mapping. Class order must match the model output layer order.
              </p>
              <div className="mt-4 flex flex-wrap gap-1.5">
                {classNames.map((c, i) => (
                  <span
                    key={c}
                    className="rounded border border-[var(--pp-border)] bg-[var(--pp-bg)] px-2 py-1 font-mono text-[11px]"
                    title={`Index ${i}`}
                  >
                    {c}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <>
              <p className="text-sm text-[var(--pp-muted)]">
                Class names not resolved — the checkpoint&apos;s class mapping could
                not be determined, so no class index can be mapped to a disease name.
              </p>
              <p className="mt-2 text-sm text-[var(--pp-muted)]">
                <strong className="text-[var(--pp-text)]">Number of classes:</strong>{" "}
                {info.output_classes == null ? "Not available" : info.output_classes}
              </p>
            </>
          )}
          {info.class_mapping_provenance ? (
            <p className="mt-2 text-sm text-[var(--pp-muted)]">
              <strong className="text-[var(--pp-text)]">Mapping provenance:</strong>{" "}
              {info.class_mapping_provenance}
            </p>
          ) : null}
          {info.low_confidence_threshold != null ? (
            <p className="mt-2 text-sm text-[var(--pp-muted)]">
              <strong className="text-[var(--pp-text)]">Low-confidence threshold:</strong>{" "}
              {info.low_confidence_threshold}
            </p>
          ) : null}
        </Section>

        <Disclaimer />
      </div>
    </div>
  );
}
