import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function confidenceLabel(band: string): string {
  if (band === "high") return "High relative confidence";
  if (band === "medium") return "Medium relative confidence";
  if (band === "low") return "Low relative confidence";
  return band;
}

export function confidenceBandColor(band: string): string {
  if (band === "high") return "bg-emerald-600";
  if (band === "medium") return "bg-amber-500";
  if (band === "low") return "bg-rose-500";
  return "bg-gray-400";
}
