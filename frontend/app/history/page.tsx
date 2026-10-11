"use client";

import { useCallback, useEffect, useState } from "react";
import { History as HistoryIcon, Trash2, Loader2 } from "lucide-react";
import { api, HistoryItem, ApiError } from "@/lib/api";
import { formatDate, formatPercent } from "@/lib/utils";
import {
  EmptyState,
  ErrorState,
  SectionHeading,
  Disclaimer,
} from "@/components/ui";

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const [selected, setSelected] = useState<HistoryItem | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.history(100);
      setItems(Array.isArray(d.items) ? d.items : []);
      setTotal(typeof d.total === "number" ? d.total : 0);
      setError(null);
    } catch (e) {
      setItems([]);
      setTotal(0);
      setError(e instanceof ApiError ? e.message : "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const clear = useCallback(async () => {
    setClearing(true);
    try {
      await api.clearHistory();
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to clear history.");
    } finally {
      setClearing(false);
    }
  }, [load]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <SectionHeading
          eyebrow="Local Records"
          title="Analysis History"
          description="Prediction metadata is stored in SQLite on the backend. Uploaded images are not permanently stored."
        />
        <button
          type="button"
          onClick={clear}
          disabled={clearing || total === 0}
          className="inline-flex items-center gap-2 rounded-md border border-rose-300 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-800 hover:bg-rose-100 disabled:opacity-50 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-100"
        >
          {clearing ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <Trash2 className="h-4 w-4" aria-hidden />
          )}
          Clear History
        </button>
      </div>

      <div className="mt-8">
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton h-16 rounded-lg" />
            ))}
          </div>
        ) : error ? (
          <ErrorState message={error} onRetry={load} />
        ) : items.length === 0 ? (
          <EmptyState
            title="No predictions recorded"
            description="History is only written when the backend returns a real prediction. With no verified model connected, no prediction is generated, so this list stays empty — that is the correct outcome, not a fault."
          />
        ) : (
          <div className="overflow-x-auto rounded-xl border border-[var(--pp-border)] bg-[var(--pp-surface)]">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--pp-border)] bg-[var(--pp-bg)] text-xs uppercase tracking-wide text-[var(--pp-muted)]">
                  <th className="px-4 py-3 font-semibold">Date</th>
                  <th className="px-4 py-3 font-semibold">Crop</th>
                  <th className="px-4 py-3 font-semibold">Disease</th>
                  <th className="px-4 py-3 font-semibold">Confidence</th>
                  <th className="px-4 py-3 font-semibold">Model</th>
                  <th className="px-4 py-3 font-semibold">Mode</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.id}
                    className="cursor-pointer border-b border-[var(--pp-border)] hover:bg-forest-50 dark:hover:bg-forest-900/20"
                    onClick={() => setSelected(item)}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") setSelected(item);
                    }}
                    role="button"
                    aria-label={`Open history entry for ${item.disease || item.filename}`}
                  >
                    <td className="px-4 py-3 text-[var(--pp-muted)]">
                      {formatDate(item.timestamp)}
                    </td>
                    <td className="px-4 py-3 font-medium">{item.crop || "—"}</td>
                    <td className="px-4 py-3">{item.disease || "—"}</td>
                    <td className="px-4 py-3 font-mono">
                      {item.confidence == null ? "—" : formatPercent(item.confidence, 1)}
                    </td>
                    <td className="px-4 py-3 text-[var(--pp-muted)]">{item.model || "—"}</td>
                    <td className="px-4 py-3">
                      <span
                        className={
                          item.mode === "real"
                            ? "rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100"
                            : "rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-semibold text-amber-800 dark:bg-amber-900 dark:text-amber-100"
                        }
                      >
                        {item.mode === "real" ? "Real" : "Demo"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selected ? (
        <div
          className="fixed inset-0 z-[60] flex items-end justify-center bg-black/50 sm:items-center sm:p-4"
          role="dialog"
          aria-modal="true"
          aria-label="History details"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-t-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6 sm:rounded-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold">Analysis Details</h2>
            <dl className="mt-4 space-y-3 text-sm">
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Timestamp</dt>
                <dd>{formatDate(selected.timestamp)}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Image</dt>
                <dd>{selected.filename}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Crop</dt>
                <dd>{selected.crop || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Disease</dt>
                <dd>{selected.disease || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Condition</dt>
                <dd className="capitalize">{selected.condition || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Confidence</dt>
                <dd className="font-mono">
                  {selected.confidence == null ? "—" : formatPercent(selected.confidence, 2)}
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Model</dt>
                <dd>{selected.model || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-[var(--pp-muted)]">Mode</dt>
                <dd className="capitalize">{selected.mode || "—"}</dd>
              </div>
            </dl>
            {selected.top_predictions && selected.top_predictions.length > 0 ? (
              <div className="mt-5">
                <h3 className="text-sm font-semibold">Top Predictions</h3>
                <ul className="mt-2 space-y-1.5 text-sm">
                  {selected.top_predictions.map((p) => (
                    <li key={p.class_name} className="flex justify-between gap-3">
                      <span>
                        {p.rank}. {p.disease}
                      </span>
                      <span className="font-mono text-[var(--pp-muted)]">
                        {formatPercent(p.probability, 2)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="mt-6 w-full rounded-md border border-[var(--pp-border)] px-4 py-2 text-sm font-medium"
            >
              Close
            </button>
          </div>
        </div>
      ) : null}

      <div className="mt-10">
        <Disclaimer />
      </div>
    </div>
  );
}
