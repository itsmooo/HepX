"use client";

import { useState, useEffect } from "react";
import Header from "../common/header";
import Footer from "../common/footer";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Award,
  Heart,
  HeartPulse,
  Microscope,
  Sparkles,
  ExternalLink,
  Github,
  Linkedin,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const AboutPage = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const teamMembers = [
    {
      name: "Dr. Amina Hassan",
      role: "Medical Director",
      bio: "Hepatologist with 15 years of experience in liver disease research and treatment.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AH",
      social: {
        linkedin: "#",
        github: "#",
        email: "amina@hepapredict.com",
      },
    },
    {
      name: "Dr. Mohamed Abdi",
      role: "Research Lead",
      bio: "Specializes in viral hepatitis epidemiology and public health interventions.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "MA",
      social: {
        linkedin: "#",
        github: "#",
        email: "mohamed@hepapredict.com",
      },
    },
    {
      name: "Fartun Omar",
      role: "Data Scientist",
      bio: "Expert in machine learning algorithms for medical diagnostics and prediction models.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "FO",
      social: {
        linkedin: "#",
        github: "#",
        email: "fartun@hepapredict.com",
      },
    },
    {
      name: "Ahmed Jama",
      role: "Patient Advocate",
      bio: "Former hepatitis patient dedicated to improving awareness and support resources.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AJ",
      social: {
        linkedin: "#",
        github: "#",
        email: "ahmed@hepapredict.com",
      },
    },
    {
      name: "Dr. Hodan Farah",
      role: "Clinical Researcher",
      bio: "Focused on developing new treatment protocols and clinical trials for hepatitis.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "HF",
      social: {
        linkedin: "#",
        github: "#",
        email: "hodan@hepapredict.com",
      },
    },
    {
      name: "Abdikarim Mohamud",
      role: "Technology Director",
      bio: "Leads our digital health initiatives and ensures data security and accessibility.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AM",
      social: {
        linkedin: "#",
        github: "#",
        email: "abdikarim@hepapredict.com",
      },
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
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
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
                <span className="text-sm font-medium">Meet Our Team</span>
              </div>
            </motion.div>

            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-slate-900 dark:text-white mb-6 leading-tight"
              variants={itemVariants}
            >
              The People Behind{" "}
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
              We're a dedicated group of Somali medical professionals,
              researchers, and patient advocates working together to improve
              hepatitis awareness, prediction, and care.
            </motion.p>

            <motion.div variants={itemVariants}>
              <Button
                className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-1"
                size="lg"
              >
                Join Our Team
              </Button>
            </motion.div>
          </motion.div>
        </section>

        {/* Team Members Section */}
        <section className="py-20 px-6 sm:px-10 relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl" />
            <div className="absolute -bottom-20 -left-20 w-60 h-60 bg-indigo-500/5 rounded-full blur-3xl" />
          </div>

          <div className="max-w-6xl mx-auto relative z-10">
            <motion.div
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={containerVariants}
            >
              {teamMembers.map((member, index) => (
                <motion.div
                  key={index}
                  className="group"
                  variants={itemVariants}
                  custom={index}
                  whileHover={{ y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden border border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-300 h-full">
                    <div className="p-6 flex flex-col items-center text-center">
                      <div className="relative mb-6">
                        <Avatar className="h-28 w-28 border-4 border-blue-100 dark:border-blue-900 group-hover:border-blue-300 dark:group-hover:border-blue-700 transition-all duration-300">
                          <AvatarImage
                            src={member.image || "/placeholder.svg"}
                            alt={member.name}
                          />
                          <AvatarFallback className="bg-blue-600 text-white text-xl">
                            {member.initials}
                          </AvatarFallback>
                        </Avatar>
                        <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white text-xs px-3 py-1 rounded-full">
                          {member.role}
                        </div>
                      </div>
                      <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-3">
                        {member.name}
                      </h3>
                      <p className="text-slate-600 dark:text-slate-300 mb-6 flex-grow">
                        {member.bio}
                      </p>

                      <div className="flex justify-center space-x-3 mt-auto">
                        <a
                          href={member.social.linkedin}
                          className="p-2 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-blue-100 dark:hover:bg-blue-900/30 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                        >
                          <Linkedin className="h-4 w-4" />
                        </a>
                        <a
                          href={member.social.github}
                          className="p-2 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-blue-100 dark:hover:bg-blue-900/30 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                        >
                          <Github className="h-4 w-4" />
                        </a>
                        <a
                          href={`mailto:${member.social.email}`}
                          className="p-2 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-blue-100 dark:hover:bg-blue-900/30 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                        >
                          <Mail className="h-4 w-4" />
                        </a>
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

        {/* Join Us Section */}
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
              Join Our Mission
            </h2>
            <p className="text-blue-100 text-lg mb-8 max-w-2xl mx-auto">
              We're always looking for passionate individuals to join our team.
              Whether you're a medical professional, researcher, developer, or
              advocate, there's a place for you in our mission.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="bg-white text-blue-600 hover:bg-blue-50 shadow-lg shadow-blue-800/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-800/30 hover:-translate-y-1"
              >
                View Open Positions
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-white/50 text-white hover:bg-blue-700/50 hover:border-white transition-all duration-300 hover:-translate-y-1"
              >
                Volunteer Opportunities
              </Button>
            </div>
          </motion.div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default AboutPage;
