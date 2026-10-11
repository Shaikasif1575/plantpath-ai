"use client";

import { useEffect, useMemo, useState } from "react";
import { Search, Filter, X, BookOpen, Info } from "lucide-react";
import { api, CropsResponse, DiseaseRecord, CropInfo, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import { EmptyState, ErrorState, LoadingState, SectionHeading, Disclaimer } from "@/components/ui";

function DiseaseCard({
  disease,
  onOpen,
}: {
  disease: DiseaseRecord;
  onOpen: (d: DiseaseRecord) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onOpen(disease)}
      className={cn(
        "group rounded-lg border border-[var(--pp-border)] bg-[var(--pp-surface)] p-5 text-left transition-colors hover:border-forest-400",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-forest-600 dark:text-leaf-400">
            {disease.crop}
          </div>
          <h3 className="mt-1 font-semibold group-hover:text-forest-700 dark:group-hover:text-leaf-300">
            {disease.disease}
          </h3>
          <p className="mt-0.5 text-xs italic text-[var(--pp-muted)]">
            {disease.scientific_name}
          </p>
        </div>
        <span
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold",
            disease.condition === "healthy"
              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100"
              : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-100",
          )}
        >
          {disease.condition === "healthy" ? "Healthy" : "Disease"}
        </span>
      </div>
      <p className="mt-3 line-clamp-2 text-sm text-[var(--pp-muted)]">{disease.description}</p>
    </button>
  );
}

function DiseaseDetail({
  disease,
  onClose,
}: {
  disease: DiseaseRecord;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-[60] flex items-end justify-center bg-black/50 p-0 sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-label={disease.disease}
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-t-xl border border-[var(--pp-border)] bg-[var(--pp-surface)] p-6 sm:rounded-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-forest-600 dark:text-leaf-400">
              {disease.crop} · {disease.condition === "healthy" ? "Healthy" : "Diseased"}
            </div>
            <h2 className="mt-1 text-xl font-bold">{disease.disease}</h2>
            <p className="text-sm italic text-[var(--pp-muted)]">{disease.scientific_name}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-[var(--pp-border)] p-1.5"
            aria-label="Close disease details"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-4 text-sm leading-relaxed">{disease.description}</p>

        {disease.symptoms?.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Symptoms
            </h3>
            <ul className="mt-2 space-y-1.5 text-sm">
              {disease.symptoms.map((s) => (
                <li key={s} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-forest-500" aria-hidden />
                  {s}
                </li>
              ))}
            </ul>
          </section>
        )}

        {disease.visual_signs?.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Visual Signs
            </h3>
            <ul className="mt-2 space-y-1.5 text-sm">
              {disease.visual_signs.map((s) => (
                <li key={s} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-forest-500" aria-hidden />
                  {s}
                </li>
              ))}
            </ul>
          </section>
        )}

        {disease.prevention?.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Prevention
            </h3>
            <ul className="mt-2 space-y-1.5 text-sm">
              {disease.prevention.map((s) => (
                <li key={s} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-forest-500" aria-hidden />
                  {s}
                </li>
              ))}
            </ul>
          </section>
        )}

        {disease.management?.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Management
            </h3>
            <ul className="mt-2 space-y-1.5 text-sm">
              {disease.management.map((s) => (
                <li key={s} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-forest-500" aria-hidden />
                  {s}
                </li>
              ))}
            </ul>
          </section>
        )}

        {disease.sources?.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--pp-muted)]">
              Sources
            </h3>
            <ul className="mt-2 space-y-1 text-sm text-[var(--pp-muted)]">
              {disease.sources.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </section>
        )}

        <p className="mt-6 text-xs text-[var(--pp-muted)]">
          Disease information is for educational reference. Verify treatment
          decisions against local agricultural advisories.
        </p>
      </div>
    </div>
  );
}

export default function DiseasesPage() {
  const [items, setItems] = useState<DiseaseRecord[]>([]);
  const [modelCrops, setModelCrops] = useState<CropInfo[]>([]);
  const [library, setLibrary] = useState<CropsResponse["reference_library"]>(null);
  const [supportedClasses, setSupportedClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [crop, setCrop] = useState<string>("");
  const [condition, setCondition] = useState<string>("");
  const [selected, setSelected] = useState<DiseaseRecord | null>(null);

  useEffect(() => {
    api
      .crops()
      .then((d) => {
        setModelCrops(d.items);
        setLibrary(d.reference_library);
        setSupportedClasses(d.supported_classes);
      })
      .catch(() => {});
  }, []);

  // The filter and the index below are the educational reference library, never
  // the model's supported set (that list is shown separately, from the API).
  const libraryCrops: CropInfo[] =
    library && library.items.length > 0 ? library.items : modelCrops;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .diseases({
        crop: crop || undefined,
        condition: condition || undefined,
        search: search.trim() || undefined,
      })
      .then((d) => {
        if (!cancelled) {
          setItems(d.items);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : "Failed to load diseases.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [search, crop, condition]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Knowledge Base"
        title="Disease Library"
        description="Browse plant disease information compiled from public agricultural extension sources. Search by name, symptom, or crop."
      />

      <div
        className="mt-5 flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/20 dark:text-amber-100"
        role="note"
      >
        <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <div>
          <p className="font-semibold">
            Reference library (educational) — not the model&apos;s supported set
          </p>
          <p className="mt-1">
            Everything on this page is reference material. It does not describe
            what the connected model can predict.{" "}
            {supportedClasses.length > 0
              ? `The model's reported classes: ${supportedClasses.join(", ")}.`
              : "No model classes are connected yet, so the model cannot predict any of these classes."}
          </p>
        </div>
      </div>

      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--pp-muted)]" aria-hidden />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search diseases, symptoms, crops…"
            className="w-full rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] py-2.5 pl-10 pr-3 text-sm"
            aria-label="Search diseases"
          />
        </div>
        <div className="flex gap-2">
          <select
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            className="rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-3 py-2.5 text-sm"
            aria-label="Filter by crop"
          >
            <option value="">All crops</option>
            {libraryCrops.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
          <select
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
            className="rounded-md border border-[var(--pp-border)] bg-[var(--pp-surface)] px-3 py-2.5 text-sm"
            aria-label="Filter by condition"
          >
            <option value="">All conditions</option>
            <option value="diseased">Diseased</option>
            <option value="healthy">Healthy</option>
          </select>
        </div>
      </div>

      <div className="mt-6">
        {loading ? (
          <LoadingState label="Loading disease library…" />
        ) : error ? (
          <ErrorState message={error} />
        ) : items.length === 0 ? (
          <EmptyState
            title="No diseases match"
            description="Try a different search term or clear the filters."
          />
        ) : (
          <>
            <p className="mb-4 text-sm text-[var(--pp-muted)]" aria-live="polite">
              {items.length} record{items.length === 1 ? "" : "s"}
            </p>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {items.map((d) => (
                <DiseaseCard key={d.id} disease={d} onOpen={setSelected} />
              ))}
            </div>
          </>
        )}
      </div>

      {/* Crop explorer */}
      <section className="mt-14">
        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5 text-forest-600 dark:text-leaf-400" aria-hidden />
          <h2 className="text-xl font-bold">Reference crop index</h2>
        </div>
        <p className="mt-2 text-sm text-[var(--pp-muted)]">
          {library?.note ||
            "Reference library (educational) — not the model’s supported set. Counts describe the library index only."}
          {library?.total ? ` ${library.total} crops indexed.` : ""}
        </p>
        {libraryCrops.length === 0 ? (
          <p className="mt-5 text-sm text-[var(--pp-muted)]">
            The reference crop index is not available from the API.
          </p>
        ) : (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--pp-border)] text-xs uppercase tracking-wide text-[var(--pp-muted)]">
                  <th className="py-2 pr-4 font-semibold">Crop</th>
                  <th className="py-2 pr-4 font-semibold">Library classes</th>
                  <th className="py-2 pr-4 font-semibold">Healthy</th>
                  <th className="py-2 font-semibold">Diseases</th>
                </tr>
              </thead>
              <tbody>
                {libraryCrops.map((c) => (
                  <tr key={c.name} className="border-b border-[var(--pp-border)]">
                    <td className="py-2.5 pr-4 font-medium">{c.name}</td>
                    <td className="py-2.5 pr-4 font-mono">{c.class_count}</td>
                    <td className="py-2.5 pr-4">
                      {c.healthy_available ? (
                        <span className="text-emerald-600 dark:text-emerald-400">Yes</span>
                      ) : (
                        <span className="text-[var(--pp-muted)]">No</span>
                      )}
                    </td>
                    <td className="py-2.5 text-[var(--pp-muted)]">{c.diseases.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="mt-10">
        <Disclaimer />
      </div>

      {selected ? (
        <DiseaseDetail disease={selected} onClose={() => setSelected(null)} />
      ) : null}
    </div>
  );
}
