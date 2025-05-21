// import React from 'react'
import { motion } from "framer-motion";
import { Award, Heart, HeartPulse, Microscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const OurMission = () => {
  return (
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
            <span className="text-sm font-medium">Our Purpose</span>
          </div>
          <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-6">
            Our Mission
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            We're committed to providing accessible tools and information
            that help identify potential hepatitis cases early and connect
            people with the care they need.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              icon: (
                <Heart className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              ),
              title: "Care",
              description:
                "Supporting patients through their diagnosis journey with compassion and understanding.",
            },
            {
              icon: (
                <HeartPulse className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              ),
              title: "Prevention",
              description:
                "Promoting awareness and education to prevent the spread of hepatitis in communities.",
            },
            {
              icon: (
                <Microscope className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              ),
              title: "Research",
              description:
                "Advancing scientific understanding of hepatitis through rigorous research and data analysis.",
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
            Learn More About Our Work
            <ExternalLink className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
          </Button>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default OurMission;
