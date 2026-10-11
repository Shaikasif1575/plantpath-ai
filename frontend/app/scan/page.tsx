"use client";

import { useCallback, useRef, useState } from "react";
import {
  ScanLine,
  Camera,
  Upload,
  Loader2,
  RotateCcw,
  AlertCircle,
} from "lucide-react";
import { api, ApiError, PredictionResponse } from "@/lib/api";
import { UploadDropzone, CameraCapture } from "@/components/Capture";
import { PredictionResult } from "@/components/PredictionResult";
import { ReportButton } from "@/components/ReportButton";
import { ModelStateBadge, ModelUnavailableNotice } from "@/components/ModelStatus";
import { SupportedClasses } from "@/components/SupportedClasses";
import {
  BackendReason,
  Disclaimer,
  ErrorState,
  LoadingState,
} from "@/components/ui";
import { useApp } from "@/components/AppProvider";

type Tab = "upload" | "camera";

export default function ScanPage() {
  const [tab, setTab] = useState<Tab>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const previewRef = useRef<string | null>(null);
  const { modelState, refresh } = useApp();

  const setPreview = useCallback((url: string | null) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = url;
    setPreviewUrl(url);
  }, []);

  const onFile = useCallback(
    (f: File) => {
      setFile(f);
      setResult(null);
      setError(null);
      setUnavailable(null);
      setPreview(URL.createObjectURL(f));
    },
    [setPreview],
  );

  const onClear = useCallback(() => {
    setFile(null);
    setResult(null);
    setError(null);
    setUnavailable(null);
    setPreview(null);
  }, [setPreview]);

  const onCapture = useCallback(
    (blob: Blob, filename: string) => {
      const f = new File([blob], filename, { type: "image/jpeg" });
      onFile(f);
      setTab("upload");
    },
    [onFile],
  );

  const analyze = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setUnavailable(null);
    try {
      // The real image is always sent, whatever the model state is.
      const res = await api.predict(file);
      setResult(res);
      await refresh();
      requestAnimationFrame(() => {
        document.getElementById("scan-results")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (e) {
      // 503 = the backend received the upload but refuses to invent a result.
      // Anything else (400 validation, network) is a request problem.
      if (e instanceof ApiError && e.status === 503) {
        setResult(null);
        setUnavailable(e.message);
      } else {
        setResult(null);
        setError(
          e instanceof Error ? e.message : "Prediction request failed.",
        );
      }
    } finally {
      setBusy(false);
    }
  }, [file, refresh]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Scan Leaf</h1>
          <p className="mt-2 max-w-2xl text-[var(--pp-muted)]">
            Upload a leaf image or capture one with your camera. The image is
            validated, sent to the backend, and — only if a verified model is
            connected — classified. If no model is connected, the upload still
            happens and the backend&apos;s reason is shown instead of a result.
          </p>
        </div>
        <ModelStateBadge className="sm:max-w-xs" />
      </div>

      <ModelUnavailableNotice className="mt-6" />

      <div className="mt-8 grid items-start gap-8 lg:grid-cols-2">
        {/* Input panel */}
        <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
          <div
            className="flex rounded-lg border border-[var(--pp-border)] bg-[var(--pp-bg)] p-1"
            role="tablist"
            aria-label="Image input method"
          >
            <button
              type="button"
              role="tab"
              aria-selected={tab === "upload"}
              onClick={() => setTab("upload")}
              className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                tab === "upload"
                  ? "bg-[var(--pp-surface)] text-forest-800 shadow dark:text-forest-100"
                  : "text-[var(--pp-muted)]"
              }`}
            >
              <Upload className="h-4 w-4" aria-hidden />
              Upload Image
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "camera"}
              onClick={() => setTab("camera")}
              className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                tab === "camera"
                  ? "bg-[var(--pp-surface)] text-forest-800 shadow dark:text-forest-100"
                  : "text-[var(--pp-muted)]"
              }`}
            >
              <Camera className="h-4 w-4" aria-hidden />
              Use Camera
            </button>
          </div>

          <div className="mt-5">
            {tab === "upload" ? (
              <UploadDropzone file={file} onFile={onFile} onClear={onClear} disabled={busy} />
            ) : (
              <CameraCapture onCapture={onCapture} disabled={busy} />
            )}
          </div>

          {previewUrl ? (
            <div className="mt-5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Selected leaf preview"
                className="max-h-72 w-full rounded-lg border border-[var(--pp-border)] object-contain bg-[var(--pp-bg)]"
              />
            </div>
          ) : null}

          <div className="mt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={analyze}
              disabled={!file || busy}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-forest-700 px-6 py-3 text-sm font-semibold text-white hover:bg-forest-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <ScanLine className="h-4 w-4" aria-hidden />
              )}
              {busy ? "Analyzing…" : "Analyze"}
            </button>
            {file ? (
              <button
                type="button"
                onClick={onClear}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-[var(--pp-border)] px-4 py-3 text-sm font-medium hover:bg-[var(--pp-bg)] disabled:opacity-50"
              >
                <RotateCcw className="h-4 w-4" aria-hidden />
                Reset
              </button>
            ) : null}
          </div>

          <p className="mt-3 text-xs text-[var(--pp-muted)]">
            Supported: JPG, JPEG, PNG, WEBP. Max size is configurable (default 10 MB).
            Images are processed locally and not permanently stored.
          </p>
          {modelState === "unavailable" || modelState === "demo" ? (
            <p className="mt-2 text-xs text-[var(--pp-muted)]">
              {modelState === "demo"
                ? "Demo mode is on: the backend returns no real prediction, and none is shown."
                : "No verified model is connected: Analyze still uploads the image, and the backend’s reason for refusing a prediction is shown instead of a result."}
            </p>
          ) : null}
        </div>

        {/* Results panel */}
        <div id="scan-results" className="scroll-mt-20">
          {busy && !result ? (
            <div className="rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6">
              <LoadingState label="Sending image to the backend…" />
              <div className="space-y-3">
                <div className="skeleton h-4 w-3/4 rounded" />
                <div className="skeleton h-4 w-1/2 rounded" />
                <div className="skeleton h-24 w-full rounded" />
              </div>
            </div>
          ) : unavailable ? (
            <div
              className="rounded-xl border border-rose-300 bg-rose-50 p-6 dark:border-rose-800 dark:bg-rose-950/30"
              role="alert"
              aria-label="Prediction unavailable"
            >
              <div className="flex items-center gap-2 font-semibold text-rose-900 dark:text-rose-100">
                <AlertCircle className="h-5 w-5 shrink-0" aria-hidden />
                Predictions are unavailable: {unavailable}
              </div>
              <p className="mt-3 text-sm text-rose-800 dark:text-rose-200">
                Your image reached the backend and was accepted for processing,
                but no prediction was returned, so there is nothing to show. No
                disease, confidence, severity, or attention map is displayed —
                any of those would have to be made up.
              </p>
              <BackendReason>{unavailable}</BackendReason>
              <div className="mt-4">
                <ReportButton
                  result={{
                    status: "unavailable",
                    demo_mode: false,
                    live_model: false,
                    top_prediction: null,
                    top_predictions: [],
                  }}
                  unavailableReason={unavailable}
                />
              </div>
            </div>
          ) : error ? (
            <ErrorState
              title="Analysis failed"
              message={error}
              onRetry={() => {
                setError(null);
                analyze();
              }}
            />
          ) : result ? (
            <PredictionResult result={result} />
          ) : (
            <div className="rounded-xl border border-dashed border-[var(--pp-border)] bg-[var(--pp-surface)] p-10 text-center">
              <ScanLine className="mx-auto h-10 w-10 text-forest-500 dark:text-leaf-400" aria-hidden />
              <h2 className="mt-4 text-lg font-semibold">No analysis yet</h2>
              <p className="mx-auto mt-2 max-w-sm text-sm text-[var(--pp-muted)]">
                Upload or capture a leaf image, then press{" "}
                <strong>Analyze</strong>. Either a real prediction appears here,
                or the backend states plainly why none can be produced.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-10 space-y-6">
        <SupportedClasses fromPrediction={result?.supported_classes} />
        <Disclaimer />
      </div>
    </div>
  );
}
