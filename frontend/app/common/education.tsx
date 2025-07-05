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
  const [activeTab, setActiveTab] = useState("what-is-hepatitis");

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
              <TabsList className="grid grid-cols-3 bg-blue-100/50 dark:bg-blue-900/20 p-1 rounded-xl">
                <TabsTrigger
                  value="what-is-hepatitis"
                  className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  What is Hepatitis?
                </TabsTrigger>
                <TabsTrigger
                  value="hepatitis-b"
                  className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  Hepatitis B
                </TabsTrigger>
                <TabsTrigger
                  value="hepatitis-c"
                  className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-sm rounded-lg transition-all duration-200"
                >
                  Hepatitis C
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="what-is-hepatitis">
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
                          Understanding Hepatitis
                        </h3>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 mb-4">
                        Hepatitis is an inflammation of the liver. It can be
                        caused by various factors, including viral infections,
                        alcohol consumption, certain medications, and toxins.
                      </p>
                      <p className="text-slate-700 dark:text-slate-300 mb-6">
                        The liver is vital for digesting food, filtering toxins
                        from the blood, and storing energy. When it becomes
                        inflamed, these functions may be affected.
                      </p>
                      <div className="flex items-center gap-3 mb-4 mt-8">
                        <div className="p-2 bg-amber-100 dark:bg-amber-900/50 rounded-lg">
                          <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
                        </div>
                        <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
                          Common Symptoms
                        </h4>
                      </div>
                      <ul className="space-y-3 text-slate-700 dark:text-slate-300">
                        {[
                          "Fatigue and general weakness",
                          "Jaundice (yellowing of skin and eyes)",
                          "Abdominal pain, especially in the liver area",
                          "Nausea and vomiting",
                          "Dark urine and pale stool",
                          "Loss of appetite",
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
                          <BookOpen className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                          Types of Viral Hepatitis
                        </h3>
                      </div>
                      <div className="space-y-4">
                        {[
                          {
                            title: "Hepatitis A",
                            description:
                              "Typically spread through contaminated food or water. Usually acute and resolves without treatment.",
                          },
                          {
                            title: "Hepatitis B",
                            description:
                              "Spread through blood, semen, and other body fluids. Can be both acute and chronic.",
                          },
                          {
                            title: "Hepatitis C",
                            description:
                              "Primarily spread through contact with infected blood. Often becomes chronic.",
                          },
                          {
                            title: "Hepatitis D & E",
                            description:
                              "Less common types. Hepatitis D only occurs with Hepatitis B infection. Hepatitis E is typically spread through contaminated water.",
                          },
                        ].map((type, i) => (
                          <motion.div
                            key={i}
                            className="p-4 rounded-xl bg-white dark:bg-slate-800 shadow-sm border border-blue-100 dark:border-blue-900/50 hover:shadow-md transition-all duration-300"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 + i * 0.1 }}
                          >
                            <h4 className="font-semibold mb-2 text-slate-900 dark:text-white">
                              {type.title}
                            </h4>
                            <p className="text-slate-700 dark:text-slate-300 text-sm">
                              {type.description}
                            </p>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="hepatitis-b">
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
                          Hepatitis B Overview
                        </h3>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 mb-4">
                        Hepatitis B is a viral infection that attacks the liver
                        and can cause both acute and chronic disease. It is
                        transmitted through contact with the blood or other body
                        fluids of an infected person.
                      </p>
                      <p className="text-slate-700 dark:text-slate-300 mb-6">
                        Hepatitis B is a major global health problem and can
                        cause chronic infection, leading to a high risk of death
                        from cirrhosis and liver cancer.
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
                          "Jaundice (often more pronounced)",
                          "Fatigue that may last for weeks or months",
                          "Abdominal pain, especially in the right upper quadrant",
                          "Joint pain (may be more common in Hepatitis B)",
                          "Fever and chills",
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
                            "Hepatitis B vaccine is highly effective",
                            "Avoid sharing needles or personal items",
                            "Practice safe sex",
                            "Be cautious about body piercing and tattoos",
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
                          Acute Hepatitis B usually doesn't require specific
                          treatment except for supportive care. For chronic
                          Hepatitis B, treatments include:
                        </p>

                        {[
                          {
                            title: "Antiviral Medications",
                            description:
                              "Medications like entecavir and tenofovir can help fight the virus and slow liver damage.",
                          },
                          {
                            title: "Immune System Modulators",
                            description:
                              "Peginterferon alfa-2a boosts your immune system to fight the virus.",
                          },
                          {
                            title: "Regular Monitoring",
                            description:
                              "Regular liver function tests and possibly liver ultrasounds are important.",
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
