import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppProvider } from "@/components/AppProvider";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import "@/styles/globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: {
    default: "PlantPath AI — Plant Disease Detection",
    template: "%s · PlantPath AI",
  },
  description:
    "Explainable deep learning for plant disease detection, classification, and severity assessment from leaf images.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('plantpath-theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){document.documentElement.classList.add('dark')}}catch(e){}})();`,
          }}
        />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <AppProvider>
          <div className="flex min-h-screen flex-col">
            <Navbar />
            <main className="flex-1">{children}</main>
            <Footer />
          </div>
        </AppProvider>
      </body>
    </html>
  );
}
