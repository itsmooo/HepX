"use client";

import {
  Activity,
  Heart,
  ChevronRight,
  Mail,
  Phone,
  MapPin,
  Github,
  Twitter,
  Facebook,
} from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

const Footer = () => {
  const currentYear = new Date().getFullYear();

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
    <footer className="bg-slate-900 text-white py-16 px-6 sm:px-10 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-20 -left-20 w-60 h-60 bg-indigo-500/10 rounded-full blur-3xl" />
      </div>

      <motion.div
        className="max-w-6xl mx-auto relative z-10"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={containerVariants}
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
          <motion.div className="md:col-span-1" variants={itemVariants}>
            <div className="flex items-center space-x-2 mb-6">
              <Activity className="h-7 w-7 text-blue-400" />
              <h2 className="text-2xl font-bold">
                <span className="text-blue-400">Hepa</span>Predict
              </h2>
            </div>
            <p className="text-slate-300 mb-6">
              A friendly tool to help identify potential hepatitis types based
              on your symptoms - designed with care for your health journey.
            </p>
            <div className="flex items-center text-sm text-slate-300">
              <Heart className="h-4 w-4 mr-2 text-red-400 animate-pulse" />
              <span>Made with care for your health</span>
            </div>
          </motion.div>

          <motion.div variants={itemVariants}>
            <h3 className="font-semibold mb-5 text-lg text-blue-400">
              Quick Links
            </h3>
            <ul className="space-y-3">
              {[
                { label: "Home", href: "/" },
                { label: "About", href: "#about" },
                { label: "Symptoms", href: "#symptoms" },
                { label: "Predict", href: "#predict" },
                { label: "Education", href: "#education" },
              ].map((link, i) => (
                <li key={i}>
                  <Link
                    href={link.href}
                    className="text-slate-300 hover:text-blue-400 transition-colors flex items-center group"
                  >
                    <ChevronRight className="h-3 w-3 mr-2 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div variants={itemVariants}>
            <h3 className="font-semibold mb-5 text-lg text-blue-400">
              Resources
            </h3>
            <ul className="space-y-3">
              {[
                { label: "Hepatitis Information", href: "#" },
                { label: "Prevention Tips", href: "#" },
                { label: "Finding Care", href: "#" },
                { label: "FAQ", href: "/faq" },
                { label: "Medical Disclaimer", href: "#" },
              ].map((link, i) => (
                <li key={i}>
                  <Link
                    href={link.href}
                    className="text-slate-300 hover:text-blue-400 transition-colors flex items-center group"
                  >
                    <ChevronRight className="h-3 w-3 mr-2 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div variants={itemVariants}>
            <h3 className="font-semibold mb-5 text-lg text-blue-400">
              Important Notice
            </h3>
            <div className="p-4 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700">
              <p className="text-sm text-slate-300 mb-4">
                HepaPredict is not a medical diagnosis tool. Always consult with
                a healthcare professional for proper diagnosis and treatment.
              </p>
              <p className="text-sm text-slate-300">
                If you're experiencing severe symptoms, please seek immediate
                medical attention.
              </p>
            </div>

            <div className="mt-6 space-y-3">
              <div className="flex items-center text-slate-300 text-sm">
                <Mail className="h-4 w-4 mr-3 text-blue-400" />
                <span>contact@hepapredict.com</span>
              </div>
              <div className="flex items-center text-slate-300 text-sm">
                <Phone className="h-4 w-4 mr-3 text-blue-400" />
                <span>+1 (555) 123-4567</span>
              </div>
              <div className="flex items-center text-slate-300 text-sm">
                <MapPin className="h-4 w-4 mr-3 text-blue-400" />
                <span>Health Tech Center, CA</span>
              </div>
            </div>
          </motion.div>
        </div>

        <motion.div
          className="border-t border-slate-700/50 mt-10 pt-8 flex flex-col md:flex-row justify-between items-center"
          variants={itemVariants}
        >
          <p className="text-sm text-slate-400 mb-6 md:mb-0">
            &copy; {currentYear} HepaPredict. All rights reserved.
          </p>

          <div className="flex flex-col sm:flex-row gap-6">
            <div className="flex space-x-4">
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                Privacy Policy
              </Link>
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                Terms of Service
              </Link>
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                Contact
              </Link>
            </div>

            <div className="flex space-x-4 items-center">
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                <Github className="h-5 w-5" />
              </Link>
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                <Twitter className="h-5 w-5" />
              </Link>
              <Link
                href="#"
                className="text-slate-400 hover:text-blue-400 transition-colors"
              >
                <Facebook className="h-5 w-5" />
              </Link>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </footer>
  );
};

export default Footer;
