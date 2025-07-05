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
  Sparkles,
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
import LoginForm from "./login-form";
import RegisterForm from "./register-form";
import UserProfile from "./user-profile";

const Header = () => {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [user, setUser] = useState<any>(null);
  const isMobile = useIsMobile();

  // Check for logged in user on mount and when localStorage changes
  useEffect(() => {
    const checkUser = () => {
      const storedUser = localStorage.getItem("user");
      if (storedUser) {
        try {
          setUser(JSON.parse(storedUser));
        } catch (error) {
          console.error("Error parsing user data:", error);
          localStorage.removeItem("user");
          localStorage.removeItem("token");
        }
      } else {
        setUser(null);
      }
    };

    checkUser();

    // Listen for storage changes
    window.addEventListener("storage", checkUser);
    window.addEventListener("focus", checkUser);

    return () => {
      window.removeEventListener("storage", checkUser);
      window.removeEventListener("focus", checkUser);
    };
  }, []);

  const navLinks = [
    { href: "/symptoms", label: "Symptoms" },
    { href: "/predict", label: "Predict" },
    { href: "/#education", label: "Education" },
    { href: "/about", label: "Our Team" },
    { href: "/faq", label: "FAQ" },
    ...(user ? [{ href: "/profile", label: "Profile" }] : []),
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

  const handleLoginClick = () => {
    setShowLogin(true);
    setOpen(false);
  };

  const handleRegisterClick = () => {
    setShowRegister(true);
    setOpen(false);
  };

  const handleCloseForms = () => {
    setShowLogin(false);
    setShowRegister(false);
  };

  const handleSwitchToRegister = () => {
    setShowLogin(false);
    setShowRegister(true);
  };

  const handleSwitchToLogin = () => {
    setShowRegister(false);
    setShowLogin(true);
  };

  // Handle logout
  const handleLogout = () => {
    setUser(null);
    setOpen(false);
  };

  return (
    <>
      <header
        className={`py-4 sticky top-0 z-50 transition-all duration-500 ${
          scrolled
            ? "bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl shadow-lg border-b border-slate-200/50 dark:border-slate-700/50"
            : "bg-transparent"
        }`}
      >
        {/* Background gradient for scrolled state */}
        {scrolled && (
          <div className="absolute inset-0 bg-gradient-to-r from-blue-50/50 via-white/50 to-indigo-50/50 dark:from-slate-900/50 dark:via-slate-800/50 dark:to-slate-900/50" />
        )}

        <div className="container mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center relative z-10">
          <motion.div
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Activity className="h-6 w-6 text-white" />
              </div>
              <motion.div
                className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <Sparkles className="h-2 w-2 text-white" />
              </motion.div>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
                <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Hepa
                </span>
                <span className="text-slate-700 dark:text-slate-300">Predict</span>
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 -mt-1">AI Health Prediction</p>
            </div>
          </motion.div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <motion.nav
              className="flex items-center space-x-8"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              {navLinks.map((link, index) => (
                <motion.div
                  key={link.href}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
                >
                  {link.href.startsWith("/#") ? (
                    <a
                      href={link.href}
                      className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 font-medium relative group px-3 py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20"
                    >
                      {link.label}
                      <span className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0.5 bg-gradient-to-r from-blue-600 to-indigo-600 transition-all duration-300 group-hover:w-full rounded-full"></span>
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 font-medium relative group px-3 py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20"
                    >
                      {link.label}
                      <span className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0.5 bg-gradient-to-r from-blue-600 to-indigo-600 transition-all duration-300 group-hover:w-full rounded-full"></span>
                    </Link>
                  )}
                </motion.div>
              ))}
            </motion.nav>
            
            <motion.div
              className="flex items-center space-x-3"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <Button
                variant="outline"
                size="icon"
                className="rounded-full border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-300 hover:scale-105"
              >
                <Bell className="h-4 w-4" />
              </Button>
              
              {user ? (
                <UserProfile user={user} onLogout={handleLogout} />
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={handleLoginClick}
                    className="border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-300 hover:scale-105 px-4 py-2 rounded-xl"
                  >
                    Login
                  </Button>
                  <Button 
                    onClick={handleRegisterClick}
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg shadow-blue-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/40 hover:-translate-y-0.5 px-6 py-2 rounded-xl"
                  >
                    <Users className="h-4 w-4 mr-2" />
                    Get Started
                  </Button>
                </>
              )}
            </motion.div>
          </div>

          {/* Mobile Navigation */}
          {isMobile && (
            <motion.div
              className="flex items-center gap-3"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <Button
                variant="outline"
                size="icon"
                className="rounded-full border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-300 hover:scale-105"
              >
                <Bell className="h-4 w-4" />
              </Button>
              <Drawer open={open} onOpenChange={setOpen}>
                <DrawerTrigger asChild>
                  <Button
                    variant="ghost"
                    className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-xl transition-all duration-300"
                    aria-label="Menu"
                  >
                    <Menu className="h-6 w-6" />
                  </Button>
                </DrawerTrigger>
                <DrawerContent className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border-t-4 border-gradient-to-r from-blue-600 to-indigo-600">
                  <DrawerHeader className="border-b border-slate-100 dark:border-slate-800 pb-6">
                    <DrawerTitle className="text-center text-slate-900 dark:text-white">
                      <div className="flex items-center justify-center space-x-2 mb-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                          <Activity className="h-5 w-5 text-white" />
                        </div>
                        <span className="text-xl font-bold">
                          <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                            Hepa
                          </span>
                          <span className="text-slate-700 dark:text-slate-300">Predict</span>
                        </span>
                      </div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">AI Health Prediction</p>
                    </DrawerTitle>
                  </DrawerHeader>
                  <div className="flex flex-col items-center gap-4 py-8">
                    {navLinks.map((link, index) => (
                      <motion.button
                        key={link.href}
                        onClick={() => handleNavClick(link.href)}
                        className="w-full text-left px-6 py-4 text-lg font-semibold text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 flex items-center justify-between rounded-xl hover:bg-blue-50 dark:hover:bg-blue-900/20"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        <span>{link.label}</span>
                        {link.href.startsWith("/") && !link.href.startsWith("/#") ? (
                          <ExternalLink className="h-4 w-4 opacity-70" />
                        ) : (
                          <ChevronDown className="h-4 w-4 opacity-70" />
                        )}
                      </motion.button>
                    ))}
                    <motion.div
                      className="w-full px-6 pt-4 space-y-3"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: 0.5 }}
                    >
                      {user ? (
                        <div className="space-y-3">
                          <div className="flex items-center space-x-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-sm font-bold text-white">
                              {user.firstName.charAt(0)}{user.lastName.charAt(0)}
                            </div>
                            <div>
                              <p className="font-semibold text-slate-900 dark:text-white">
                                {user.firstName} {user.lastName}
                              </p>
                              <p className="text-xs text-slate-600 dark:text-slate-400">
                                {user.email}
                              </p>
                            </div>
                          </div>
                          <Button 
                            onClick={handleLogout}
                            variant="outline"
                            className="w-full border-red-200 dark:border-red-700 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-xl py-3 transition-all duration-300"
                          >
                            Sign Out
                          </Button>
                        </div>
                      ) : (
                        <>
                          <Button 
                            onClick={handleLoginClick}
                            variant="outline"
                            className="w-full border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-xl py-3 transition-all duration-300"
                          >
                            Login
                          </Button>
                          <Button 
                            onClick={handleRegisterClick}
                            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg shadow-blue-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/40 rounded-xl py-3"
                          >
                            <Users className="h-4 w-4 mr-2" />
                            Get Started
                          </Button>
                        </>
                      )}
                    </motion.div>
                  </div>
                  <DrawerFooter className="border-t border-slate-100 dark:border-slate-800 pt-6">
                    <DrawerClose asChild>
                      <Button
                        variant="outline"
                        className="w-full border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-xl transition-all duration-300"
                      >
                        <X className="h-4 w-4 mr-2" />
                        Close Menu
                      </Button>
                    </DrawerClose>
                  </DrawerFooter>
                </DrawerContent>
              </Drawer>
            </motion.div>
          )}
        </div>
      </header>

      {/* Auth Forms */}
      <AnimatePresence>
        {showLogin && (
          <LoginForm
            onSwitchToRegister={handleSwitchToRegister}
            onClose={handleCloseForms}
          />
        )}
        {showRegister && (
          <RegisterForm
            onSwitchToLogin={handleSwitchToLogin}
            onClose={handleCloseForms}
          />
        )}
      </AnimatePresence>
    </>
  );
};

export default Header;
