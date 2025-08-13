"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useRouter } from "next/navigation";
import { 
  User, 
  Settings, 
  LogOut, 
  ChevronDown, 
  Activity,
  Shield,
  Heart,
  Calendar,
  TrendingUp
} from "lucide-react";

interface User {
  _id: string;
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  token: string;
}

interface UserProfileProps {
  user: User;
  onLogout: () => void;
}

const UserProfile = ({ user, onLogout }: UserProfileProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    onLogout();
    setIsOpen(false);
    router.push("/login");
  };

  const handleViewProfile = () => {
    router.push("/profile");
    setIsOpen(false);
  };

  const getInitials = (firstName: string, lastName: string) => {
    const firstInitial = firstName && firstName.length > 0 ? firstName.charAt(0) : '';
    const lastInitial = lastName && lastName.length > 0 ? lastName.charAt(0) : '';
    return `${firstInitial}${lastInitial}`.toUpperCase();
  };

  return (
    <div className="relative">
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg shadow-blue-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/40 hover:-translate-y-0.5 px-4 py-2 rounded-xl"
      >
        <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-sm font-semibold">
          {getInitials(user.firstName, user.lastName)}
        </div>
        <span className="hidden sm:block font-medium">
          {user.firstName} {user.lastName}
        </span>
        <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </Button>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-40"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              className="absolute right-0 top-full mt-2 w-80 z-50"
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            >
              <Card className="border-0 shadow-2xl bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl">
                <CardContent className="p-6">
                  {/* User Header */}
                  <div className="flex items-center space-x-4 mb-6">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-xl font-bold text-white shadow-lg">
                      {getInitials(user.firstName, user.lastName)}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                        {user.firstName} {user.lastName}
                      </h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        {user.email}
                      </p>
                      <div className="flex items-center space-x-2 mt-1">
                        <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                        <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                          Online
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                      <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-800 flex items-center justify-center mx-auto mb-2">
                        <Activity className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-400">Predictions</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">12</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20">
                      <div className="w-8 h-8 rounded-full bg-emerald-100 dark:bg-emerald-800 flex items-center justify-center mx-auto mb-2">
                        <Heart className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-400">Health Score</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">85%</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                      <div className="w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-800 flex items-center justify-center mx-auto mb-2">
                        <TrendingUp className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-400">Accuracy</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">92%</p>
                    </div>
                  </div>

                  {/* Menu Items */}
                  <div className="space-y-2">
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
                      onClick={handleViewProfile}
                    >
                      <User className="h-4 w-4 mr-3" />
                      View Profile
                    </Button>
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
                    >
                      <Settings className="h-4 w-4 mr-3" />
                      Settings
                    </Button>
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
                    >
                      <Calendar className="h-4 w-4 mr-3" />
                      History
                    </Button>
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
                    >
                      <Shield className="h-4 w-4 mr-3" />
                      Privacy
                    </Button>
                  </div>

                  {/* Logout Button */}
                  <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-700">
                    <Button
                      onClick={handleLogout}
                      variant="ghost"
                      className="w-full justify-start text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
                    >
                      <LogOut className="h-4 w-4 mr-3" />
                      Sign Out
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default UserProfile; 