"use client";

import { useState, useEffect } from "react";


import {
  Award,
  Heart,
  HeartPulse,
  Microscope,
  Sparkles,
  ExternalLink,
  Brain,
  Shield,
  Activity,
  Zap,
  Target,
  BarChart3,
} from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const AboutPage = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const systemFeatures = [
    {
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms trained on comprehensive medical datasets to provide accurate hepatitis predictions.",
      icon: <Brain className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "95%+ Accuracy",
    },
    {
      title: "Real-Time Processing",
      description: "Instant analysis of symptoms and risk factors with immediate results and recommendations.",
      icon: <Zap className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "< 30 Seconds",
    },
    {
      title: "Secure & Private",
      description: "Enterprise-grade security with end-to-end encryption to protect your sensitive health information.",
      icon: <Shield className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "Bank-Level Security",
    },
    {
      title: "Comprehensive Assessment",
      description: "Evaluates multiple symptoms, risk factors, and medical history for a complete health picture.",
      icon: <Activity className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "15+ Data Points",
    },
    {
      title: "Evidence-Based",
      description: "Built on peer-reviewed medical research and validated clinical data from leading hepatology institutions.",
      icon: <Target className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "Clinical Validated",
    },
    {
      title: "Detailed Reports",
      description: "Comprehensive health reports with actionable insights and recommendations for next steps.",
      icon: <BarChart3 className="h-8 w-8 text-blue-600 dark:text-blue-400" />,
      stats: "Full Analysis",
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  };

  return (
    <div className="min-h-screen">
        {/* Hero Section */}
        <section className="py-24 px-6 sm:px-10 relative overflow-hidden bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-950">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl" />
            <div className="absolute top-60 -left-20 w-60 h-60 bg-indigo-500/10 rounded-full blur-3xl" />
          </div>

          <motion.div
            className="max-w-6xl mx-auto text-center relative z-10"
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
            variants={containerVariants}
          >
            <motion.div variants={itemVariants}>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-6">
                <Sparkles className="h-4 w-4" />
                <span className="text-sm font-medium">About The System</span>
              </div>
            </motion.div>

            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-slate-900 dark:text-white mb-6 leading-tight"
              variants={itemVariants}
            >
              Advanced AI-Powered{" "}
              <span className="text-blue-600 dark:text-blue-400 relative">
                HepaPredict
                <motion.span
                  className="absolute bottom-1 left-0 w-full h-2 bg-blue-200 dark:bg-blue-900/50 rounded-full -z-10"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ delay: 0.8, duration: 0.6 }}
                />
              </span>
            </motion.h1>

            <motion.p
              className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-8"
              variants={itemVariants}
            >
              A cutting-edge hepatitis prediction system that uses machine learning
              and artificial intelligence to provide accurate, fast, and reliable
              health assessments for Hepatitis A and C detection.
            </motion.p>

            <motion.div variants={itemVariants}>
              <Button
                className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-1"
                size="lg"
              >
                Try HepaPredict
              </Button>
            </motion.div>
          </motion.div>
        </section>

        {/* System Features Section */}
        <section className="py-20 px-6 sm:px-10 relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl" />
            <div className="absolute -bottom-20 -left-20 w-60 h-60 bg-indigo-500/5 rounded-full blur-3xl" />
          </div>

          <div className="max-w-6xl mx-auto relative z-10">
            <motion.div
              className="text-center mb-16"
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={containerVariants}
            >
              <motion.div variants={itemVariants}>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
                  <Brain className="h-4 w-4" />
                  <span className="text-sm font-medium">System Capabilities</span>
                </div>
              </motion.div>
              <motion.h2 
                className="text-4xl font-bold text-slate-900 dark:text-white mb-6"
                variants={itemVariants}
              >
                Why Choose HepaPredict?
              </motion.h2>
              <motion.p 
                className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto"
                variants={itemVariants}
              >
                Our advanced AI system combines cutting-edge technology with medical expertise
                to deliver reliable hepatitis predictions and health insights.
              </motion.p>
            </motion.div>

            <motion.div
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={containerVariants}
            >
              {systemFeatures.map((feature, index) => (
                <motion.div
                  key={index}
                  className="group"
                  variants={itemVariants}
                  custom={index}
                  whileHover={{ y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden border border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-300 h-full">
                    <div className="p-6 flex flex-col text-center">
                      <div className="flex justify-center mb-6">
                        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-full">
                          {feature.icon}
                        </div>
                      </div>
                      <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-3">
                        {feature.title}
                      </h3>
                      <p className="text-slate-600 dark:text-slate-300 mb-4 flex-grow">
                        {feature.description}
                      </p>
                      <div className="mt-auto">
                        <span className="inline-block px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-sm font-medium rounded-full">
                          {feature.stats}
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Mission Section */}
        <section className="py-20 px-6 sm:px-10 bg-gradient-to-b from-slate-50 to-white dark:from-slate-950 dark:to-slate-900 relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl" />
            <div className="absolute bottom-20 -left-20 w-60 h-60 bg-indigo-500/10 rounded-full blur-3xl" />
          </div>

          <motion.div
            className="max-w-6xl mx-auto relative z-10"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={containerVariants}
          >
            <motion.div className="text-center mb-16" variants={itemVariants}>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
                <Award className="h-4 w-4" />
                <span className="text-sm font-medium">Our Technology</span>
              </div>
              <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-6">
                How HepaPredict Works
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
                Our system combines advanced machine learning algorithms with medical expertise
                to provide accurate hepatitis predictions through comprehensive health assessments.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: (
                    <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  ),
                  title: "Data Collection",
                  description:
                    "Comprehensive symptom assessment and risk factor analysis through our intelligent questionnaire system.",
                },
                {
                  icon: (
                    <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  ),
                  title: "AI Processing",
                  description:
                    "Advanced machine learning algorithms analyze your data and compare it against validated medical datasets.",
                },
                {
                  icon: (
                    <BarChart3 className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  ),
                  title: "Results & Insights",
                  description:
                    "Instant, accurate predictions with detailed reports and actionable health recommendations.",
                },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  whileHover={{ y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 border border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-300 h-full">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-full w-16 h-16 flex items-center justify-center mb-6">
                      {item.icon}
                    </div>
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                      {item.title}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-300">
                      {item.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div
              className="mt-16 text-center"
              variants={itemVariants}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6 }}
            >
              <Button
                variant="outline"
                className="border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-500 transition-all duration-300 hover:-translate-y-1 group"
              >
                View Technical Documentation
                <ExternalLink className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Button>
            </motion.div>
          </motion.div>
        </section>

        {/* Get Started Section */}
        <section className="py-20 px-6 sm:px-10 bg-blue-600 dark:bg-blue-900 relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-white/10 rounded-full blur-3xl" />
            <div className="absolute bottom-20 -left-20 w-60 h-60 bg-white/10 rounded-full blur-3xl" />
          </div>

          <motion.div
            className="max-w-4xl mx-auto text-center relative z-10"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Ready to Get Started?
            </h2>
            <p className="text-blue-100 text-lg mb-8 max-w-2xl mx-auto">
              Experience the power of AI-driven hepatitis prediction. Get instant,
              accurate health assessments and take control of your health journey
              with HepaPredict.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="bg-white text-blue-600 hover:bg-blue-50 shadow-lg shadow-blue-800/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-800/30 hover:-translate-y-1"
              >
                Start Health Assessment
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-white/50 text-white hover:bg-blue-700/50 hover:border-white transition-all duration-300 hover:-translate-y-1"
              >
                Learn More
              </Button>
            </div>
          </motion.div>
        </section>
    </div>
  );
};

export default AboutPage;
