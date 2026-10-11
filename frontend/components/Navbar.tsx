"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X, Leaf, Sun, Moon, ScanLine } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";
import { ModelStateBadge } from "@/components/ModelStatus";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/scan", label: "Scan Leaf" },
  { href: "/diseases", label: "Diseases" },
  { href: "/model", label: "Model" },
  { href: "/research", label: "Research" },
  { href: "/history", label: "History" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const { toggle, theme, mounted } = useTheme();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--pp-border)] bg-[var(--pp-bg)]/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="flex items-center gap-2 font-semibold tracking-tight"
          aria-label="PlantPath AI home"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-forest-700 text-white">
            <Leaf className="h-4 w-4" aria-hidden />
          </span>
          <span className="text-lg">
            PlantPath <span className="text-forest-600 dark:text-leaf-400">AI</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 lg:flex" aria-label="Primary">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-forest-100 text-forest-800 dark:bg-forest-800 dark:text-forest-100"
                    : "text-[var(--pp-muted)] hover:bg-[var(--pp-surface)] hover:text-[var(--pp-text)]",
                )}
                aria-current={active ? "page" : undefined}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          {/* Hidden below md so the header keeps its single row on phones and
              small tablets; the full badge is repeated inside the menu. */}
          <div className="hidden md:block">
            <ModelStateBadge className="max-w-[15rem]" />
          </div>
          <button
            type="button"
            onClick={toggle}
            className="rounded-md border border-[var(--pp-border)] p-2 text-[var(--pp-muted)] hover:bg-[var(--pp-surface)]"
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {mounted && theme === "dark" ? (
              <Sun className="h-4 w-4" aria-hidden />
            ) : (
              <Moon className="h-4 w-4" aria-hidden />
            )}
          </button>
          <Link
            href="/scan"
            className="hidden items-center gap-2 rounded-md bg-forest-700 px-4 py-2 text-sm font-semibold text-white hover:bg-forest-800 sm:inline-flex"
          >
            <ScanLine className="h-4 w-4" aria-hidden />
            Scan a Leaf
          </Link>
          <button
            type="button"
            className="rounded-md border border-[var(--pp-border)] p-2 lg:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? "Close menu" : "Open menu"}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {open ? (
        <nav
          id="mobile-nav"
          className="border-t border-[var(--pp-border)] bg-[var(--pp-bg)] px-4 py-3 lg:hidden"
          aria-label="Mobile"
        >
          <ul className="flex flex-col gap-1">
            {LINKS.map((l) => {
              const active = pathname === l.href;
              return (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className={cn(
                      "block rounded-md px-3 py-2.5 text-sm font-medium",
                      active
                        ? "bg-forest-100 text-forest-800 dark:bg-forest-800"
                        : "text-[var(--pp-muted)] hover:bg-[var(--pp-surface)]",
                    )}
                  >
                    {l.label}
                  </Link>
                </li>
              );
            })}
            <li className="pt-2">
              <div className="pb-2">
                <ModelStateBadge />
              </div>
              <Link
                href="/scan"
                onClick={() => setOpen(false)}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-forest-700 px-4 py-2.5 text-sm font-semibold text-white"
              >
                <ScanLine className="h-4 w-4" aria-hidden />
                Scan a Leaf
              </Link>
            </li>
          </ul>
        </nav>
      ) : null}
    </header>
  );
}
