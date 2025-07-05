"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Sparkles, ChevronDown, HeartPulse, ShieldCheck } from "lucide-react"

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
    <section className="relative py-24 px-6 sm:px-10 overflow-hidden bg-gradient-to-b from-white via-blue-50/30 to-slate-50 dark:from-slate-900 dark:via-slate-800/50 dark:to-slate-900">
      {/* Cool background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-40 right-20 w-96 h-96 bg-indigo-500/8 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-20 left-1/4 w-64 h-64 bg-emerald-500/8 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        
        {/* Geometric shapes */}
        <div className="absolute top-1/4 right-1/4 w-32 h-32 border border-blue-300/20 rounded-full" />
        <div className="absolute bottom-1/3 left-1/3 w-24 h-24 border border-indigo-300/20 rotate-45" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-40 h-40 border border-emerald-300/20 rounded-full" />
      </div>

      <motion.div
        className="max-w-6xl mx-auto relative z-10"
        initial="hidden"
        animate={isVisible ? "visible" : "hidden"}
        variants={containerVariants}
      >
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-blue-600 dark:text-blue-400 mb-8 border border-blue-200/50 dark:border-blue-700/50"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Sparkles className="h-4 w-4" />
            <span className="text-sm font-medium">AI-Powered Health Prediction</span>
          </motion.div>

          <motion.h1
            className="text-5xl md:text-6xl lg:text-7xl font-bold text-slate-900 dark:text-white mb-8 leading-tight"
            variants={itemVariants}
          >
            Your{" "}
            <span className="relative">
              <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Smart
              </span>
              <motion.div
                className="absolute -bottom-2 left-0 w-full h-3 bg-gradient-to-r from-blue-200 to-indigo-200 dark:from-blue-800 dark:to-indigo-800 rounded-full -z-10"
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ delay: 0.8, duration: 0.6 }}
              />
            </span>{" "}
            <br />
            <span className="bg-gradient-to-r from-slate-700 to-slate-900 dark:from-slate-200 dark:to-slate-100 bg-clip-text text-transparent">
              Hepatitis Predictor
            </span>
          </motion.h1>

          <motion.p className="text-xl text-slate-600 dark:text-slate-300 mb-12 max-w-3xl mx-auto leading-relaxed" variants={itemVariants}>
            Experience the future of health screening with our advanced AI model. 
            Get instant insights about Hepatitis B & C based on your symptoms - 
            <span className="font-semibold text-blue-600 dark:text-blue-400"> fast, accurate, and confidential.</span>
          </motion.p>

          <motion.div className="flex flex-col sm:flex-row gap-6 justify-center mb-16" variants={itemVariants}>
            <Button
              onClick={scrollToPredict}
              size="lg"
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-xl shadow-blue-500/25 transition-all duration-300 hover:shadow-2xl hover:shadow-blue-500/40 hover:-translate-y-1 px-8 py-6 text-lg font-semibold"
            >
              Start Prediction Now
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="border-2 border-blue-500/30 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-500 transition-all duration-300 hover:-translate-y-1 px-8 py-6 text-lg font-semibold"
            >
              Learn More
            </Button>
          </motion.div>
        </div>

        {/* Feature cards */}
        <motion.div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto" variants={itemVariants}>
          {[
            {
              icon: HeartPulse,
              title: "Instant Results",
              description: "Get predictions in seconds with our advanced machine learning model",
              color: "from-red-500 to-pink-500"
            },
            {
              icon: ShieldCheck,
              title: "Privacy First",
              description: "Your data stays secure and confidential throughout the process",
              color: "from-blue-500 to-indigo-500"
            },
            {
              icon: Sparkles,
              title: "AI Powered",
              description: "Built with cutting-edge artificial intelligence for accuracy",
              color: "from-emerald-500 to-teal-500"
            }
          ].map((feature, i) => (
            <motion.div
              key={i}
              className="group relative p-8 rounded-2xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-2"
              whileHover={{ scale: 1.02 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
            >
              <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">{feature.title}</h3>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          className="mt-16 flex items-center justify-center gap-2 text-slate-500 dark:text-slate-400"
          variants={itemVariants}
          animate={{ y: [0, -8, 0], transition: { duration: 2, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" } }}
        >
          <span className="text-sm">Scroll to explore</span>
          <ChevronDown className="h-4 w-4 animate-bounce" />
        </motion.div>
      </motion.div>
    </section>
  )
}

export default Hero
