"use client";

import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from "react";
import { HealthResponse, ModelInfo, api } from "@/lib/api";

/**
 * The three states the UI is allowed to claim, plus the two states in which
 * nothing can be claimed at all.
 *
 * - `live`        demo_mode === false && loaded === true
 * - `demo`        demo_mode === true
 * - `unavailable` demo_mode === false && loaded === false
 * - `loading`     request in flight — render nothing rather than guess
 * - `unreachable` backend could not be asked — model state is unknown
 */
export type ModelState = "loading" | "live" | "demo" | "unavailable" | "unreachable";

interface AppState {
  /** Source of truth for LIVE / DEMO / model-loaded status (GET /api/model-info). */
  modelInfo: ModelInfo | null;
  health: HealthResponse | null;
  /** Derived from modelInfo/error/loading — never inferred from a prediction. */
  modelState: ModelState;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const Ctx = createContext<AppState>({
  modelInfo: null,
  health: null,
  modelState: "loading",
  loading: true,
  error: null,
  refresh: async () => {},
});

function deriveState(
  modelInfo: ModelInfo | null,
  loading: boolean,
  error: string | null,
): ModelState {
  if (modelInfo) {
    if (modelInfo.demo_mode) return "demo";
    return modelInfo.loaded ? "live" : "unavailable";
  }
  if (!loading && error) return "unreachable";
  return "loading";
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    // Both calls are independent: health is supplementary context only, so a
    // failure there must not hide the model-info state (and vice versa).
    const [infoResult, healthResult] = await Promise.allSettled([
      api.modelInfo(),
      api.health(),
    ]);

    if (infoResult.status === "fulfilled") {
      setModelInfo(infoResult.value);
      setError(null);
    } else {
      setModelInfo(null);
      setError(
        infoResult.reason instanceof Error
          ? infoResult.reason.message
          : "Backend unavailable",
      );
    }

    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setHealth(null);

    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <Ctx.Provider
      value={{ modelInfo, health, modelState: deriveState(modelInfo, loading, error), loading, error, refresh }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useApp() {
  return useContext(Ctx);
}
