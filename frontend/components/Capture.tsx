"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, Camera, X, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function UploadDropzone({
  file,
  onFile,
  onClear,
  disabled,
}: {
  file: File | null;
  onFile: (f: File) => void;
  onClear: () => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFile(files[0]);
    },
    [onFile],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload leaf image"
        className={cn(
          "relative flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 text-center transition-colors",
          dragOver
            ? "border-forest-500 bg-forest-50 dark:bg-forest-900/40"
            : "border-[var(--pp-border)] bg-[var(--pp-surface)] hover:border-forest-400",
          disabled && "pointer-events-none opacity-50",
        )}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <UploadCloud className="h-10 w-10 text-forest-600 dark:text-leaf-400" aria-hidden />
        <div>
          <p className="text-sm font-semibold">Drag & drop a leaf image here</p>
          <p className="mt-1 text-xs text-[var(--pp-muted)]">
            or click to browse — JPG, JPEG, PNG, WEBP
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          className="sr-only"
          onChange={(e) => handleFiles(e.target.files)}
          aria-label="File input for leaf image"
        />
      </div>
      {file ? (
        <div className="mt-3 flex items-center justify-between rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-3 py-2 text-sm">
          <span className="flex items-center gap-2 truncate">
            <ImageIcon className="h-4 w-4 shrink-0 text-[var(--pp-muted)]" aria-hidden />
            <span className="truncate">{file.name}</span>
            <span className="shrink-0 text-xs text-[var(--pp-muted)]">
              {(file.size / 1024).toFixed(0)} KB
            </span>
          </span>
          <button
            type="button"
            onClick={onClear}
            className="rounded p-1 text-[var(--pp-muted)] hover:bg-[var(--pp-bg)]"
            aria-label="Remove selected file"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function CameraCapture({
  onCapture,
  disabled,
}: {
  onCapture: (blob: Blob, filename: string) => void;
  disabled?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [mode, setMode] = useState<"idle" | "starting" | "live" | "preview" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    setMode("starting");
    setError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API not supported in this browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      streamRef.current = stream;
      setMode("live");
      // attach after render
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      });
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Camera permission denied. Please allow camera access or use image upload instead."
          : err instanceof DOMException && err.name === "NotFoundError"
            ? "No camera was found on this device. Use image upload instead."
            : err instanceof Error
              ? err.message
              : "Camera unavailable.";
      setError(msg);
      setMode("error");
    }
  }, []);

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        stopStream();
        const url = URL.createObjectURL(blob);
        setCapturedBlob(blob);
        setCapturedUrl(url);
        setMode("preview");
      },
      "image/jpeg",
      0.92,
    );
  }, [stopStream]);

  const confirmCapture = useCallback(() => {
    if (!capturedBlob) return;
    onCapture(capturedBlob, "camera-capture.jpg");
    setCapturedBlob(null);
    setCapturedUrl(null);
    setMode("idle");
  }, [capturedBlob, onCapture]);

  const retake = useCallback(() => {
    if (capturedUrl) URL.revokeObjectURL(capturedUrl);
    setCapturedBlob(null);
    setCapturedUrl(null);
    startCamera();
  }, [capturedUrl, startCamera]);

  const close = useCallback(() => {
    stopStream();
    if (capturedUrl) URL.revokeObjectURL(capturedUrl);
    setCapturedBlob(null);
    setCapturedUrl(null);
    setMode("idle");
    setError(null);
  }, [stopStream, capturedUrl]);

  return (
    <div className="space-y-3">
      {mode === "idle" ? (
        <button
          type="button"
          onClick={startCamera}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-lg border-2 border-dashed border-[var(--pp-border)] bg-[var(--pp-surface)] p-8 text-sm font-semibold hover:border-forest-400 disabled:opacity-50"
        >
          <Camera className="h-5 w-5 text-forest-600 dark:text-leaf-400" aria-hidden />
          Use Camera
        </button>
      ) : null}

      {mode === "starting" ? (
        <div className="rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6 text-center text-sm text-[var(--pp-muted)]">
          Requesting camera access…
        </div>
      ) : null}

      {mode === "live" ? (
        <div className="overflow-hidden rounded-lg border border-[var(--pp-border)] bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="aspect-[4/3] w-full object-cover"
            aria-label="Camera preview"
          />
          <div className="flex gap-2 bg-[var(--pp-surface)] p-3">
            <button
              type="button"
              onClick={capture}
              className="flex-1 rounded-md bg-forest-700 px-4 py-2 text-sm font-semibold text-white hover:bg-forest-800"
            >
              Capture
            </button>
            <button
              type="button"
              onClick={close}
              className="rounded-md border border-[var(--pp-border)] px-4 py-2 text-sm font-medium"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {mode === "preview" && capturedUrl ? (
        <div className="overflow-hidden rounded-lg border border-[var(--pp-border)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={capturedUrl} alt="Captured leaf preview" className="aspect-[4/3] w-full object-cover" />
          <div className="flex gap-2 bg-[var(--pp-surface)] p-3">
            <button
              type="button"
              onClick={confirmCapture}
              className="flex-1 rounded-md bg-forest-700 px-4 py-2 text-sm font-semibold text-white hover:bg-forest-800"
            >
              Use This Photo
            </button>
            <button
              type="button"
              onClick={retake}
              className="rounded-md border border-[var(--pp-border)] px-4 py-2 text-sm font-medium"
            >
              Retake
            </button>
          </div>
        </div>
      ) : null}

      {mode === "error" ? (
        <div
          className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100"
          role="alert"
        >
          <p className="font-semibold">Camera unavailable</p>
          <p className="mt-1">{error}</p>
          <p className="mt-2 text-xs">You can still upload an image instead.</p>
        </div>
      ) : null}
    </div>
  );
}
