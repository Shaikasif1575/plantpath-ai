/**
 * Centralized API client for PlantPath AI.
 * All backend calls go through this module.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface QualityWarning {
  code: string;
  message: string;
  severity: string;
}

export interface ImageValidation {
  valid: boolean;
  width: number | null;
  height: number | null;
  format: string | null;
  mode: string | null;
  file_size_bytes: number | null;
  warnings: QualityWarning[];
  error: string | null;
}

export interface PredictionItem {
  class_name: string;
  crop: string;
  disease: string;
  probability: number;
  rank: number;
}

export interface GradCamResult {
  available: boolean;
  message: string | null;
  original_image: string | null;
  heatmap: string | null;
  overlay: string | null;
  target_class: string | null;
  target_layer?: string | null;
}

export interface SeverityResult {
  available: boolean;
  message: string | null;
  affected_area_percent: number | null;
  severity_level: string | null;
  method: string | null;
  original_image: string | null;
  mask_image: string | null;
  /** Reported by the backend when the severity service cannot run. */
  reason?: string | null;
  unavailable_reason?: string | null;
}

export interface ModelInfo {
  name: string | null;
  version: string;
  architecture: string | null;
  framework: string | null;
  input_size: number[] | null;
  output_classes: number | null;
  parameters: number | null;
  model_file: string | null;
  loaded: boolean;
  demo_mode: boolean;
  status?: string;
  metrics?: Record<string, unknown>;
  training?: Record<string, unknown>;
  limitations?: string[];
  class_names?: string[];
  comparison?: Array<Record<string, unknown>>;
  // Live-model contract fields (optional — parsed liberally).
  model_format?: string | null;
  classes?: string[];
  num_classes?: number | null;
  input_channels?: number | null;
  gradcam_available?: boolean;
  segmentation_available?: boolean;
  segmentation_status?: string | null;
  low_confidence_threshold?: number | null;
  /** Human-readable reason the backend gives for refusing to serve predictions. */
  error?: string | null;
  preprocessing?: string | null;
  preprocessing_key?: string | null;
  class_mapping_provenance?: string | null;
  /** model-info `severity` capability descriptor. */
  severity?: Record<string, unknown>;
}

export interface PredictionResponse {
  status: string;
  demo_mode: boolean;
  // Live-model additions (optional — fall back to current behavior when absent).
  live_model?: boolean;
  low_confidence?: boolean;
  low_confidence_message?: string | null;
  supported_classes?: string[];
  leaf_validation_available?: boolean;
  validation?: ImageValidation;
  crop?: string | null;
  disease?: string | null;
  condition?: string | null;
  detected_disease?: string | null;
  /** Null when the backend returns no prediction (e.g. demo mode). */
  top_prediction?: PredictionItem | null;
  top_predictions: PredictionItem[];
  confidence_band?: string;
  confidence_note?: string;
  model?: ModelInfo | null;
  explainability?: GradCamResult;
  severity?: SeverityResult;
  limitations?: string[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  demo_mode: boolean;
  model_name?: string;
  model_version?: string;
  gradcam_available: boolean;
  severity_available: boolean;
  leaf_validation_available?: boolean;
  class_count: number | null;
  timestamp: string;
  detail: string | null;
}

export interface DiseaseRecord {
  id: string;
  crop: string;
  disease: string;
  scientific_name: string;
  condition: string;
  description: string;
  symptoms: string[];
  visual_signs: string[];
  prevention: string[];
  management: string[];
  sources: string[];
}

export interface CropInfo {
  name: string;
  class_count: number;
  healthy_available: boolean;
  diseases: string[];
}

export interface CropsResponse {
  items: CropInfo[];
  total: number;
  /** Classes the connected model can actually predict (empty when none are resolved). */
  supported_classes: string[];
  /** Educational crop/disease index — explicitly not the model's supported set. */
  reference_library: {
    note: string | null;
    items: CropInfo[];
    total: number;
  } | null;
}

export interface HistoryItem {
  id: number;
  timestamp: string;
  filename: string;
  crop: string | null;
  disease: string | null;
  condition: string | null;
  confidence: number | null;
  model: string | null;
  mode: string | null;
  top_predictions: PredictionItem[] | null;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let message = "Request failed";
  try {
    const data = await res.json();
    if (data && typeof data.detail === "string") {
      message = data.detail;
    } else if (data && typeof data.detail === "object") {
      message = JSON.stringify(data.detail);
    }
  } catch {
    message = `Request failed with status ${res.status}`;
  }
  return new ApiError(message, res.status);
}

/** Fallback copy used when the API flags a low-confidence result but omits its message. */
export const LOW_CONFIDENCE_FALLBACK =
  "Low-confidence prediction — please upload a clearer image or an image from a supported plant/disease category.";

function asString(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string")
    : [];
}

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : undefined;
}

/**
 * Accepts both the live-model /api/model-info payload and the legacy shape,
 * mapping whichever fields are present. Absent fields fall back to neutral
 * defaults — nothing is invented.
 */
export function normalizeModelInfo(raw: unknown): ModelInfo {
  const r = asRecord(raw) || {};
  const classes =
    asStringArray(r.class_names).length > 0
      ? asStringArray(r.class_names)
      : asStringArray(r.classes);
  const comparison = Array.isArray(r.comparison)
    ? (r.comparison as Array<Record<string, unknown>>)
    : undefined;
  return {
    name: asString(r.name),
    version: asString(r.version) || "",
    architecture: asString(r.architecture),
    framework: asString(r.framework),
    input_size: Array.isArray(r.input_size)
      ? (r.input_size as number[])
      : null,
    output_classes: asNumber(r.output_classes) ?? asNumber(r.num_classes),
    parameters: asNumber(r.parameters),
    model_file: asString(r.model_file),
    loaded: r.model_loaded === true || r.loaded === true,
    demo_mode: r.demo_mode === true,
    status: asString(r.status) ?? undefined,
    metrics: asRecord(r.metrics),
    training: asRecord(r.training),
    limitations: Array.isArray(r.limitations)
      ? asStringArray(r.limitations)
      : undefined,
    class_names: classes,
    comparison,
    model_format: asString(r.model_format),
    classes,
    num_classes: asNumber(r.num_classes),
    input_channels: asNumber(r.input_channels),
    gradcam_available:
      typeof r.gradcam_available === "boolean"
        ? r.gradcam_available
        : undefined,
    segmentation_available:
      typeof r.segmentation_available === "boolean"
        ? r.segmentation_available
        : undefined,
    segmentation_status: asString(r.segmentation_status),
    low_confidence_threshold: asNumber(r.low_confidence_threshold),
    error: asString(r.error),
    preprocessing: asString(r.preprocessing),
    preprocessing_key: asString(r.preprocessing_key),
    class_mapping_provenance: asString(r.class_mapping_provenance),
    severity: asRecord(r.severity),
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    throw new ApiError(
      "Cannot reach the PlantPath AI backend. Make sure the API server is running.",
      0,
    );
  }
  if (!res.ok) {
    throw await parseError(res);
  }
  return (await res.json()) as T;
}

export const api = {
  async health(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
  },

  async modelInfo(): Promise<ModelInfo> {
    return normalizeModelInfo(await request<unknown>("/api/model-info"));
  },

  async crops(): Promise<CropsResponse> {
    const raw = await request<unknown>("/api/crops");
    const r = asRecord(raw) || {};
    const lib = asRecord(r.reference_library);
    const libItems = lib && Array.isArray(lib.items) ? (lib.items as CropInfo[]) : [];
    return {
      items: Array.isArray(r.items) ? (r.items as CropInfo[]) : [],
      total: asNumber(r.total) ?? 0,
      supported_classes: asStringArray(r.supported_classes),
      reference_library: lib
        ? {
            note: asString(lib.note),
            items: libItems,
            total: asNumber(lib.total) ?? libItems.length,
          }
        : null,
    };
  },

  async diseases(params?: {
    crop?: string;
    condition?: string;
    search?: string;
  }): Promise<{ items: DiseaseRecord[]; total: number; meta: Record<string, unknown> }> {
    const qs = new URLSearchParams();
    if (params?.crop) qs.set("crop", params.crop);
    if (params?.condition) qs.set("condition", params.condition);
    if (params?.search) qs.set("search", params.search);
    const q = qs.toString();
    return request(`/api/diseases${q ? `?${q}` : ""}`);
  },

  async disease(id: string): Promise<DiseaseRecord> {
    return request(`/api/diseases/${encodeURIComponent(id)}`);
  },

  async predict(file: File): Promise<PredictionResponse> {
    const form = new FormData();
    form.append("image", file);
    return request<PredictionResponse>("/api/predict", {
      method: "POST",
      body: form,
    });
  },

  async explain(
    file: File,
    targetClass?: string,
  ): Promise<GradCamResult> {
    const form = new FormData();
    form.append("image", file);
    if (targetClass) form.append("target_class", targetClass);
    return request<GradCamResult>("/api/explain", {
      method: "POST",
      body: form,
    });
  },

  async history(limit = 50): Promise<{ items: HistoryItem[]; total: number }> {
    return request(`/api/history?limit=${limit}`);
  },

  async clearHistory(): Promise<{ status: string; deleted: number }> {
    return request("/api/history", { method: "DELETE" });
  },
};
