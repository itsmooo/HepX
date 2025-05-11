"use client";

import { useState } from "react";
import { Activity, Bell, Users } from "lucide-react";
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

const Header = () => {
  const [open, setOpen] = useState(false);
  const isMobile = useIsMobile();

  const navLinks = [
    { href: "/#about", label: "About" },
    { href: "/#symptoms", label: "Symptoms" },
    { href: "/predict", label: "Predict" },
    { href: "/#education", label: "Education" },
    { href: "/about", label: "Our Team" },
    { href: "/resources", label: "Resources" },
    { href: "/faq", label: "FAQ" },
  ];

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
    <header className="py-4 px-6 sm:px-10 flex justify-between items-center border-b border-hepa-teal/10 sticky top-0 bg-white/80 backdrop-blur-sm z-50 shadow-sm">
      <div className="flex items-center space-x-2">
        <Activity className="h-8 w-8 text-hepa-teal" />
        <h1 className="text-2xl font-bold text-hepa-navy">
          <span className="text-hepa-teal">Hepa</span>Predict
        </h1>
      </div>

      {/* Desktop Navigation */}
      <div className="hidden md:flex items-center space-x-6">
        <nav className="flex items-center space-x-6">
          {navLinks.map((link) =>
            link.href.startsWith("/#") ? (
              <a
                key={link.href}
                href={link.href}
                className="text-hepa-navy/80 hover:text-hepa-teal transition-colors duration-200 font-medium"
              >
                {link.label}
              </a>
            ) : (
              <Link
                key={link.href}
                href={link.href}
                className="text-hepa-navy/80 hover:text-hepa-teal transition-colors duration-200 font-medium"
              >
                {link.label}
              </Link>
            )
          )}
        </nav>
        <Button
          variant="outline"
          className="bg-hepa-teal text-white hover:bg-hepa-teal/90 border-none shadow-sm"
        >
          <Users className="h-4 w-4 mr-2" />
          Register
        </Button>
      </div>

      {/* Mobile Navigation */}
      {isMobile && (
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            className="rounded-full bg-hepa-lightTeal/10 border-hepa-teal text-hepa-teal hover:bg-hepa-teal hover:text-white mr-2"
          >
            <Bell className="h-4 w-4" />
          </Button>
          <Drawer open={open} onOpenChange={setOpen}>
            <DrawerTrigger asChild>
              <Button 
                variant="ghost" 
                className="text-hepa-navy hover:text-hepa-teal hover:bg-hepa-lightTeal/10" 
                aria-label="Menu"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </Button>
            </DrawerTrigger>
            <DrawerContent className="bg-white border-t-4 border-hepa-teal">
              <DrawerHeader>
                <DrawerTitle className="text-center text-hepa-navy">
                  <span className="text-hepa-teal">Hepa</span>Predict Menu
                </DrawerTitle>
              </DrawerHeader>
              <div className="flex flex-col items-center gap-6 py-8">
                {navLinks.map((link) => (
                  <button
                    key={link.href}
                    onClick={() => handleNavClick(link.href)}
                    className="text-lg font-semibold text-hepa-navy/80 hover:text-hepa-teal transition-colors"
                  >
                    {link.label}
                  </button>
                ))}
                <Button
                  className="mt-4 bg-hepa-teal text-white hover:bg-hepa-teal/90 border-none shadow-sm"
                  variant="outline"
                >
                  <Users className="h-4 w-4 mr-2" />
                  Register
                </Button>
              </div>
              <DrawerFooter>
                <DrawerClose asChild>
                  <Button 
                    variant="outline" 
                    className="w-full border-hepa-teal text-hepa-teal hover:bg-hepa-teal hover:text-white"
                  >
                    Close
                  </Button>
                </DrawerClose>
              </DrawerFooter>
            </DrawerContent>
          </Drawer>
        </div>
      )}
    </header>
  )
}

export default Header
