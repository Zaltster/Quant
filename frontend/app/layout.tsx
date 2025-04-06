// app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css"; // Keep default global styles
import NavBar from "../components/NavBar"; // Import the NavBar

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Quant Game Platform",
  description: "Practice Quant Trading Games",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <NavBar /> {/* Add the NavBar here */}
        <main style={{ padding: '2rem' }}> {/* Add padding for content */}
          {children}
        </main>
      </body>
    </html>
  );
}