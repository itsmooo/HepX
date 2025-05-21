"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { HeartPulse, ShieldCheck, ChevronDown, Sparkles } from "lucide-react"

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [])

  const scrollToPredict = () => {
    const predictSection = document.getElementById("predict")
    if (predictSection) {
      predictSection.scrollIntoView({ behavior: "smooth" })
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  }

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: (i: number) => ({
      opacity: 1,
      scale: 1,
      transition: {
        delay: i * 0.1,
        duration: 0.5,
        type: "spring",
        stiffness: 100,
      },
    }),
  }

  const floatAnimation = {
    y: [0, -10, 0],
    transition: {
      duration: 3,
      repeat: Number.POSITIVE_INFINITY,
      repeatType: "reverse",
      ease: "easeInOut",
    },
  }

  return (
    <section className="relative py-24 px-6 sm:px-10 overflow-hidden bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-950">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute top-60 -left-20 w-60 h-60 bg-indigo-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-20 w-40 h-40 bg-emerald-500/10 rounded-full blur-3xl" />
      </div>

      <motion.div
        className="max-w-6xl mx-auto relative z-10"
        initial="hidden"
        animate={isVisible ? "visible" : "hidden"}
        variants={containerVariants}
      >
        <div className="flex flex-col md:flex-row items-center gap-12 lg:gap-16">
          <motion.div className="md:w-1/2" variants={itemVariants}>
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Sparkles className="h-4 w-4" />
              <span className="text-sm font-medium">Simple & Caring Health Tool</span>
            </motion.div>

            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-slate-900 dark:text-white mb-6 leading-tight"
              variants={itemVariants}
            >
              Your{" "}
              <span className="text-blue-600 dark:text-blue-400 relative">
                Friendly
                <motion.span
                  className="absolute bottom-1 left-0 w-full h-2 bg-blue-200 dark:bg-blue-900/50 rounded-full -z-10"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ delay: 0.8, duration: 0.6 }}
                />
              </span>{" "}
              Hepatitis Prediction Tool
            </motion.h1>

            <motion.p className="text-lg text-slate-600 dark:text-slate-300 mb-8" variants={itemVariants}>
              Wondering if your symptoms might be related to Hepatitis? HepaPredict helps identify whether you may have
              Hepatitis B or C based on your symptoms - all in a simple, caring way.
            </motion.p>

            <motion.div className="flex flex-col sm:flex-row gap-4" variants={itemVariants}>
              <Button
                onClick={scrollToPredict}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-1"
              >
                Check Symptoms
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-blue-500/50 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-500 transition-all duration-300 hover:-translate-y-1"
              >
                Learn More
              </Button>
            </motion.div>

            <motion.div
              className="mt-12 hidden md:flex items-center gap-2 text-slate-500 dark:text-slate-400"
              variants={itemVariants}
              animate={{ y: [0, -8, 0], transition: { duration: 2, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" } }}
            >
              <span className="text-sm">Scroll to explore</span>
              <ChevronDown className="h-4 w-4 animate-bounce" />
            </motion.div>
          </motion.div>

          <motion.div className="md:w-1/2 relative" variants={itemVariants}>
            <motion.div
              className="relative bg-gradient-to-br from-blue-500/20 via-blue-500/10 to-transparent rounded-3xl p-10 backdrop-blur-sm border border-blue-500/20 shadow-xl"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
            >
              <motion.div
                className="absolute -top-6 -left-6 bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-4 border border-slate-200 dark:border-slate-700"
                whileHover={{ scale: 1.05, rotate: -5 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <HeartPulse className="h-8 w-8 text-red-500" />
              </motion.div>

              <motion.div
                className="absolute -bottom-6 -right-6 bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-4 border border-slate-200 dark:border-slate-700"
                whileHover={{ scale: 1.05, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <ShieldCheck className="h-8 w-8 text-blue-500" />
              </motion.div>

              <div className="space-y-6">
                {[
                  {
                    title: "Fast & Caring",
                    description: "Get insights based on real medical data without stress or waiting.",
                  },
                  {
                    title: "Not a Diagnosis",
                    description: "A smart first step toward proper medical care and attention.",
                  },
                  {
                    title: "Educational",
                    description: "Learn about different types of hepatitis while you use the tool.",
                  },
                ].map((card, i) => (
                  <motion.div
                    key={i}
                    custom={i}
                    variants={cardVariants}
                    whileHover={{ scale: 1.03, backgroundColor: "rgba(255,255,255,0.1)" }}
                    className="p-6 rounded-xl backdrop-blur-sm border border-blue-500/10 transition-all duration-300 hover:border-blue-500/30 bg-white/5 dark:bg-slate-900/30"
                  >
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">{card.title}</h3>
                    <p className="text-slate-600 dark:text-slate-300">{card.description}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Decorative elements */}
            <motion.div
              className="absolute -z-10 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 rounded-full bg-blue-500/5 blur-3xl"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.5, 0.3],
              }}
              transition={{
                duration: 8,
                repeat: Number.POSITIVE_INFINITY,
                repeatType: "reverse",
              }}
            />
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}

export default Hero
