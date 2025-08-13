import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
// import { ThemeProvider } from "@/components/theme-provider"
import Header from "@/app/common/header"
import Footer from "@/app/common/footer"
import { Toaster } from "react-hot-toast"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "HepaPredict - Hepatitis Prediction Tool",
  description: "A friendly tool to help identify potential hepatitis types based on your symptoms",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {/* <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange> */}
          <Header />
          {children}
          <Footer />
          <Toaster position="top-right" />
        {/* </ThemeProvider> */}
      </body>
    </html>
  )
}
