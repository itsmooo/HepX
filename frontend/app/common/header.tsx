"use client";

import { useState, useEffect } from "react";
import {
  Activity,
  Bell,
  Users,
  ChevronDown,
  Menu,
  X,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "../hooks/use-mobile";
import Link from "next/link";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { motion, AnimatePresence } from "framer-motion";

const Header = () => {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const isMobile = useIsMobile();

  const navLinks = [
    // { href: "/#about", label: "About" },
    { href: "/symptoms", label: "Symptoms" },
    { href: "/predict", label: "Predict" },
    { href: "/#education", label: "Education" },
    { href: "/about", label: "Our Team" },

    { href: "/faq", label: "FAQ" },
  ];

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 20) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Handle drawer close and scroll to section
  const handleNavClick = (href: string) => {
    setOpen(false);

    // Smooth scroll to the section if it's a hash link
    if (href.startsWith("/#")) {
      const element = document.querySelector(href.substring(1));
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }
  };

  return (
    <header
      className={`py-4 sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg shadow-md"
          : "bg-transparent"
      }`}
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
        <motion.div
          className="flex items-center space-x-2"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Activity className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            <span className="text-blue-600 dark:text-blue-400">Hepa</span>
            Predict
          </h1>
        </motion.div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-6">
          <motion.nav
            className="flex items-center space-x-6"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {navLinks.map((link) =>
              link.href.startsWith("/#") ? (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 font-medium relative group"
                >
                  {link.label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-600 dark:bg-blue-400 transition-all duration-300 group-hover:w-full"></span>
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 font-medium relative group"
                >
                  {link.label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-600 dark:bg-blue-400 transition-all duration-300 group-hover:w-full"></span>
                </Link>
              )
            )}
          </motion.nav>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Button className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5">
              <Users className="h-4 w-4 mr-2" />
              Register
            </Button>
          </motion.div>
        </div>

        {/* Mobile Navigation */}
        {isMobile && (
          <div className="flex items-center gap-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <Button
                variant="outline"
                size="icon"
                className="rounded-full bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-600 hover:text-white mr-2"
              >
                <Bell className="h-4 w-4" />
              </Button>
            </motion.div>
            <Drawer open={open} onOpenChange={setOpen}>
              <DrawerTrigger asChild>
                <Button
                  variant="ghost"
                  className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                  aria-label="Menu"
                >
                  <Menu className="h-6 w-6" />
                </Button>
              </DrawerTrigger>
              <DrawerContent className="bg-white dark:bg-slate-900 border-t-4 border-blue-600 dark:border-blue-500">
                <DrawerHeader className="border-b border-slate-100 dark:border-slate-800">
                  <DrawerTitle className="text-center text-slate-900 dark:text-white">
                    <span className="text-blue-600 dark:text-blue-400">
                      Hepa
                    </span>
                    Predict Menu
                  </DrawerTitle>
                </DrawerHeader>
                <div className="flex flex-col items-center gap-6 py-8">
                  {navLinks.map((link) => (
                    <button
                      key={link.href}
                      onClick={() => handleNavClick(link.href)}
                      className="text-lg font-semibold text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors flex items-center"
                    >
                      {link.href.startsWith("/") &&
                      !link.href.startsWith("/#") ? (
                        <ExternalLink className="h-4 w-4 mr-2 opacity-70" />
                      ) : (
                        <ChevronDown className="h-4 w-4 mr-2 opacity-70" />
                      )}
                      {link.label}
                    </button>
                  ))}
                  <Button className="mt-6 bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30">
                    <Users className="h-4 w-4 mr-2" />
                    Register
                  </Button>
                </div>
                <DrawerFooter className="border-t border-slate-100 dark:border-slate-800">
                  <DrawerClose asChild>
                    <Button
                      variant="outline"
                      className="w-full border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                    >
                      <X className="h-4 w-4 mr-2" />
                      Close Menu
                    </Button>
                  </DrawerClose>
                </DrawerFooter>
              </DrawerContent>
            </Drawer>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
