"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import {
  ShieldCheck,
  AlertCircle,
  BookOpen,
  Pill,
  HeartPulse,
  Bug,
  Sparkles,
} from "lucide-react";
import { motion } from "framer-motion";

const Education = () => {
  const [activeTab, setActiveTab] = useState("hepatitis-a");

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
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
    <section
      id="education"
      className="py-20 px-6 sm:px-10 bg-gradient-to-b from-white to-blue-50 dark:from-slate-900 dark:to-slate-900/80 relative overflow-hidden"
    >
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl" />
      </div>

      <motion.div
        className="max-w-5xl mx-auto relative z-10"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={containerVariants}
      >
        <motion.div className="text-center mb-12" variants={itemVariants}>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
            <Sparkles className="h-4 w-4" />
            <span className="text-sm font-medium">Educational Resources</span>
          </div>
          <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Learn About Hepatitis
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Understanding hepatitis types, symptoms, and treatments can help you
            make better healthcare decisions.
          </p>
        </motion.div>

        <motion.div variants={itemVariants}>
          <Tabs
            defaultValue={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <div className="flex justify-center mb-8">
              <TabsList className="grid grid-cols-2 bg-blue-100/50 dark:bg-blue-900/20 p-1 rounded-xl">
                <TabsTrigger
                  value="hepatitis-a"
                  className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  Hepatitis A
                </TabsTrigger>
                <TabsTrigger
                  value="hepatitis-c"
                  className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  Hepatitis C
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="hepatitis-a">
              <Card className="border-blue-200 dark:border-blue-900 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
                    <motion.div
                      className="p-8 bg-gradient-to-br from-blue-50 to-white dark:from-slate-900 dark:to-slate-800"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <Bug className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                          Hepatitis A Overview
                        </h3>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 mb-4">
                        Hepatitis A is a viral infection that causes liver inflammation and is typically spread through contaminated food and water. Unlike other forms of hepatitis, it is usually acute and rarely becomes chronic.
                      </p>
                      <p className="text-slate-700 dark:text-slate-300 mb-6">
                        Most people with Hepatitis A recover completely and develop lifelong immunity. It does not cause chronic liver disease and is rarely fatal.
                      </p>
                      <div className="flex items-center gap-3 mb-4 mt-8">
                        <div className="p-2 bg-amber-100 dark:bg-amber-900/50 rounded-lg">
                          <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
                        </div>
                        <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
                          Specific Symptoms
                        </h4>
                      </div>
                      <ul className="space-y-3 text-slate-700 dark:text-slate-300">
                        {[
                          "Sudden onset of nausea and vomiting",
                          "Fatigue and weakness",
                          "Abdominal pain and discomfort",
                          "Low-grade fever",
                          "Loss of appetite",
                          "Jaundice (yellowing of skin and eyes)",
                        ].map((symptom, i) => (
                          <motion.li
                            key={i}
                            className="flex items-start"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.5 + i * 0.1 }}
                          >
                            <span className="text-blue-500 mr-2">•</span>
                            {symptom}
                          </motion.li>
                        ))}
                      </ul>
                    </motion.div>

                    <motion.div
                      className="p-8 bg-gradient-to-br from-slate-50 to-white dark:from-slate-800 dark:to-slate-900 border-t md:border-t-0 md:border-l border-blue-100 dark:border-blue-900/50"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <ShieldCheck className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                          Prevention & Treatment
                        </h3>
                      </div>

                      <motion.div
                        className="p-5 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 mb-6"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                      >
                        <h4 className="font-semibold mb-3 text-slate-900 dark:text-white">
                          Prevention
                        </h4>
                        <ul className="space-y-2 text-slate-700 dark:text-slate-300 text-sm">
                          {[
                            "Hepatitis A vaccine is highly effective",
                            "Practice good hygiene and handwashing",
                            "Avoid contaminated food and water",
                            "Be cautious when traveling to high-risk areas",
                          ].map((item, i) => (
                            <li key={i} className="flex items-start">
                              <span className="text-blue-500 mr-2">•</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </motion.div>

                      <div className="flex items-center gap-3 mb-4 mt-8">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <Pill className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
                          Treatment Options
                        </h4>
                      </div>

                      <div className="space-y-4">
                        <p className="text-slate-700 dark:text-slate-300">
                          There is no specific treatment for Hepatitis A. Management focuses on supportive care and symptom relief:
                        </p>

                        {[
                          {
                            title: "Rest and Recovery",
                            description:
                              "Get plenty of rest and avoid strenuous activities while your body fights the infection.",
                          },
                          {
                            title: "Hydration",
                            description:
                              "Drink plenty of fluids to prevent dehydration, especially if experiencing vomiting.",
                          },
                          {
                            title: "Avoid Alcohol and Medications",
                            description:
                              "Avoid alcohol and unnecessary medications that could stress the liver during recovery.",
                          },
                        ].map((treatment, i) => (
                          <motion.div
                            key={i}
                            className="p-4 rounded-xl bg-white dark:bg-slate-800 shadow-sm border border-blue-100 dark:border-blue-900/50 hover:shadow-md transition-all duration-300"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.7 + i * 0.1 }}
                          >
                            <h5 className="font-semibold mb-2 text-slate-900 dark:text-white text-sm">
                              {treatment.title}
                            </h5>
                            <p className="text-slate-700 dark:text-slate-300 text-sm">
                              {treatment.description}
                            </p>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>



            <TabsContent value="hepatitis-c">
              <Card className="border-blue-200 dark:border-blue-900 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
                    <motion.div
                      className="p-8 bg-gradient-to-br from-blue-50 to-white dark:from-slate-900 dark:to-slate-800"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <Bug className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                          Hepatitis C Overview
                        </h3>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 mb-4">
                        Hepatitis C is a viral infection caused by the Hepatitis
                        C virus (HCV) that primarily affects the liver. It is
                        often referred to as a "silent epidemic" because many
                        people don't know they're infected.
                      </p>
                      <p className="text-slate-700 dark:text-slate-300 mb-6">
                        Hepatitis C is primarily spread through contact with
                        blood from an infected person. It can cause both acute
                        and chronic hepatitis, ranging in severity from a mild
                        illness lasting a few weeks to a serious, lifelong
                        condition.
                      </p>

                      <div className="flex items-center gap-3 mb-4 mt-8">
                        <div className="p-2 bg-amber-100 dark:bg-amber-900/50 rounded-lg">
                          <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
                        </div>
                        <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
                          Specific Symptoms
                        </h4>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 mb-4">
                        Many people with Hepatitis C don't experience symptoms
                        until liver damage has occurred, which may be many years
                        after infection. When symptoms do appear, they may
                        include:
                      </p>
                      <ul className="space-y-3 text-slate-700 dark:text-slate-300">
                        {[
                          "Fatigue (often the most prominent symptom)",
                          "Mild abdominal pain",
                          "Loss of appetite",
                          "Dark urine",
                          "Jaundice (less common than in Hepatitis B)",
                        ].map((symptom, i) => (
                          <motion.li
                            key={i}
                            className="flex items-start"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.5 + i * 0.1 }}
                          >
                            <span className="text-blue-500 mr-2">•</span>
                            {symptom}
                          </motion.li>
                        ))}
                      </ul>
                    </motion.div>

                    <motion.div
                      className="p-8 bg-gradient-to-br from-slate-50 to-white dark:from-slate-800 dark:to-slate-900 border-t md:border-t-0 md:border-l border-blue-100 dark:border-blue-900/50"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <ShieldCheck className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                          Prevention & Treatment
                        </h3>
                      </div>

                      <motion.div
                        className="p-5 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 mb-6"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                      >
                        <h4 className="font-semibold mb-3 text-slate-900 dark:text-white">
                          Prevention
                        </h4>
                        <ul className="space-y-2 text-slate-700 dark:text-slate-300 text-sm">
                          {[
                            "No vaccine is available for Hepatitis C",
                            "Avoid sharing needles or personal items",
                            "Ensure proper sterilization of medical equipment",
                            "Practice safe sex, especially if you have multiple partners",
                          ].map((item, i) => (
                            <li key={i} className="flex items-start">
                              <span className="text-blue-500 mr-2">•</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </motion.div>

                      <div className="flex items-center gap-3 mb-4 mt-8">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                          <HeartPulse className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
                          Treatment Success
                        </h4>
                      </div>

                      <p className="text-slate-700 dark:text-slate-300 mb-6">
                        Unlike Hepatitis B, Hepatitis C is usually curable with
                        antiviral medications. Current treatments are highly
                        effective, with cure rates of over 95%.
                      </p>

                      <div className="space-y-4">
                        {[
                          {
                            title: "Direct-Acting Antivirals (DAAs)",
                            description:
                              "Modern treatments like sofosbuvir, ledipasvir, and others can cure Hepatitis C in 8-12 weeks with minimal side effects.",
                          },
                          {
                            title: "Regular Monitoring",
                            description:
                              "During and after treatment, your healthcare provider will likely monitor your viral load and liver function.",
                          },
                          {
                            title: "Lifestyle Changes",
                            description:
                              "Avoiding alcohol, maintaining a healthy diet, and getting regular exercise can help support liver health.",
                          },
                        ].map((treatment, i) => (
                          <motion.div
                            key={i}
                            className="p-4 rounded-xl bg-white dark:bg-slate-800 shadow-sm border border-blue-100 dark:border-blue-900/50 hover:shadow-md transition-all duration-300"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.7 + i * 0.1 }}
                          >
                            <h5 className="font-semibold mb-2 text-slate-900 dark:text-white text-sm">
                              {treatment.title}
                            </h5>
                            <p className="text-slate-700 dark:text-slate-300 text-sm">
                              {treatment.description}
                            </p>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Education;
