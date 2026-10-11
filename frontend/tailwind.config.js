/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        forest: {
          50: "#f0f7f1",
          100: "#dcecdf",
          200: "#b8d9c2",
          300: "#8bbf9b",
          400: "#5da074",
          500: "#3d8258",
          600: "#2d6745",
          700: "#245238",
          800: "#1c402c",
          900: "#142e1f",
          950: "#0c1b12",
        },
        leaf: {
          400: "#6fbf73",
          500: "#4caf50",
          600: "#3d8b40",
        },
        cream: {
          50: "#fafaf7",
          100: "#f3f3ee",
          200: "#e8e8e0",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: "72ch",
          },
        },
      },
    },
  },
  plugins: [],
};
